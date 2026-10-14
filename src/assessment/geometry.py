from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import cv2
import numpy as np


def finite(value, name="value"):
    if not np.isfinite(np.asarray(value, dtype=float)).all():
        raise ValueError(f"{name} must be finite")


def polygon_array(points):
    q = np.asarray(points, dtype=np.float32)
    if q.shape != (4, 2):
        raise ValueError("expected four ordered image corners")
    finite(q, "polygon")
    if not cv2.isContourConvex(q) or abs(cv2.contourArea(q)) < 4:
        raise ValueError("polygon is degenerate or self-intersecting")
    return q


def polygon_center(points):
    q = polygon_array(points).astype(float)
    a, b = q[2] - q[0], q[3] - q[1]
    matrix = np.column_stack((a, -b))
    if abs(np.linalg.det(matrix)) < 1e-8:
        raise ValueError("diagonals are degenerate")
    t, _ = np.linalg.solve(matrix, q[1] - q[0])
    return tuple(map(float, q[0] + t * a))


@dataclass(frozen=True)
class Camera:
    matrix: tuple
    distortion: tuple = (0., 0., 0., 0., 0.)
    height_m: float = 0.0

    def __post_init__(self):
        k = np.asarray(self.matrix, dtype=float)
        distortion = np.asarray(self.distortion, dtype=float)
        finite(k, "intrinsics")
        finite(distortion, "distortion")
        finite(self.height_m, "camera height")
        if (k.shape != (3, 3) or k[0, 0] <= 0 or k[1, 1] <= 0
                or not np.allclose(k[2], (0, 0, 1)) or np.linalg.slogdet(k)[0] == 0):
            raise ValueError("invalid camera matrix")
        # OpenCV accepts a flat, row or column vector, never a coefficient
        # matrix/tensor. An empty nested array otherwise acts as zero distortion.
        if (distortion.ndim not in (1, 2) or distortion.size not in (4, 5, 8, 12, 14)
                or (distortion.ndim == 2 and min(distortion.shape) != 1)):
            raise ValueError("invalid distortion vector")

    @property
    def k(self):
        return np.asarray(self.matrix, dtype=np.float64)

    def optical_to_level(self, point):
        """Explicit level camera only: world X forward, Y left, Z up."""
        x, y, z = point
        return np.array((z, -x, self.height_m - y), dtype=float)

    def level_to_pixel(self, point):
        x, y, z = point
        optical = np.array((-y, self.height_m - z, x), dtype=float)
        if optical[2] <= 0:
            return None
        p, _ = cv2.projectPoints(optical.reshape(1, 3), np.zeros(3), np.zeros(3), self.k,
                                 np.asarray(self.distortion, dtype=float))
        return tuple(map(float, p.reshape(2)))


@dataclass(frozen=True)
class PlateGeometry:
    # These are the lamp-axis reference dimensions, NOT a network bbox's size.
    lamp_spacing_m: float = .135
    lamp_length_m: float = .055
    hit_width_m: float = .135
    hit_height_m: float = .055

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("physical dimensions must be positive")

    @property
    def reference_points(self):
        w, h = self.lamp_spacing_m / 2, self.lamp_length_m / 2
        return np.array(((-w, -h, 0), (w, -h, 0), (w, h, 0), (-w, h, 0)), dtype=np.float64)


@dataclass(frozen=True)
class PlatePose:
    timestamp: float
    position: tuple
    outward_normal: tuple
    reprojection_px: float
    ambiguity: bool
    source: str


def estimate_pose(corners, timestamp, camera, geometry, source="classical", previous_normal=None,
                  max_error_px=3.0, all_hypotheses=False, upright=False):
    finite(timestamp, "timestamp")
    q = polygon_array(corners).astype(np.float64)
    if upright:
        result = upright_solutions(q, camera, geometry)
    else:
        result = cv2.solvePnPGeneric(geometry.reference_points, q, camera.k,
                                    np.asarray(camera.distortion, dtype=float), flags=cv2.SOLVEPNP_IPPE)
    candidates = []
    if not result[0]:
        return [] if all_hypotheses else None
    for rvec, tvec in zip(result[1], result[2]):
        position = tvec.reshape(3)
        rotation, _ = cv2.Rodrigues(rvec)
        normal = -rotation[:, 2]
        if not np.isfinite(position).all() or position[2] <= 0 or np.dot(normal, -position) <= 0:
            continue
        projected, _ = cv2.projectPoints(geometry.reference_points, rvec, tvec, camera.k,
                                         np.asarray(camera.distortion, dtype=float))
        error = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - q) ** 2, axis=1))))
        if error <= max_error_px:
            candidates.append((error, position, normal))
    if not candidates:
        return [] if all_hypotheses else None
    candidates.sort(key=lambda item: item[0])
    if all_hypotheses:
        return [PlatePose(float(timestamp), tuple(map(float,camera.optical_to_level(p))),
                    (float(n[2]),float(-n[0]),float(-n[1])), float(e), True, source) for e,p,n in candidates]
    ambiguous = (len(candidates) > 1 and candidates[1][0] - candidates[0][0] < .25
                 and np.dot(candidates[0][2], candidates[1][2]) < math.cos(math.radians(12)))
    chosen = candidates[0]
    if ambiguous and previous_normal is not None:
        nearby = [c for c in candidates if c[0] <= candidates[0][0] + .25]
        chosen = max(nearby, key=lambda c: np.dot(c[2], previous_normal))
        ambiguous = np.dot(chosen[2], previous_normal) < math.cos(math.radians(35))
    error, position, normal = chosen
    world_position = camera.optical_to_level(position)
    world_normal = np.array((normal[2], -normal[0], -normal[1]))
    return PlatePose(float(timestamp), tuple(map(float, world_position)), tuple(map(float, world_normal)),
                     error, bool(ambiguous), source)


def upright_solutions(corners, camera, geometry):
    """Fit translation/yaw only for known vertical plates and a level camera.

    This scene constraint removes spurious out-of-plane tilt solutions. Both
    remaining yaw hypotheses are retained for independent visual disambiguation.
    """
    q = np.asarray(corners,dtype=float)
    height = (np.linalg.norm(q[3]-q[0])+np.linalg.norm(q[2]-q[1]))/2
    if height < 1:
        return (False,[],[])
    depth = camera.k[1,1]*geometry.lamp_length_m/height
    center = np.mean(q,axis=0)
    initial = ((center[0]-camera.k[0,2])*depth/camera.k[0,0],
               (center[1]-camera.k[1,2])*depth/camera.k[1,1], depth)
    distortion = np.asarray(camera.distortion,dtype=float)
    solutions=[]
    def project(parameters):
        return cv2.projectPoints(geometry.reference_points,np.array((0.,parameters[3],0.)),
                                 parameters[:3],camera.k,distortion)
    for angle in (-1.2,-.6,0.,.6,1.2):
        parameters=np.array((*initial,angle))
        for _ in range(30):
            pixels,jacobian=project(parameters)
            residual=(q-pixels.reshape(4,2)).reshape(-1)
            jac=np.column_stack((jacobian[:,3:6],jacobian[:,1]))
            step=np.linalg.lstsq(jac,residual,rcond=None)[0]
            improved=False
            for scale in (1.,.5,.25,.125):
                proposal=parameters+scale*step
                if proposal[2]<=.1 or abs(proposal[3])>=math.pi/2:
                    continue
                trial,_=project(proposal)
                if np.sum((q-trial.reshape(4,2))**2)<np.sum(residual**2):
                    parameters=proposal
                    improved=True
                    break
            if not improved or np.linalg.norm(scale*step)<1e-7:
                break
        pixels,_=project(parameters)
        error=float(np.linalg.norm(q-pixels.reshape(4,2)))
        if not any(abs(parameters[3]-p[3])<.01 for _,p in solutions):
            solutions.append((error,parameters))
    solutions.sort(key=lambda item:item[0])
    return (bool(solutions),[np.array((0.,p[3],0.)) for _,p in solutions],
            [p[:3] for _,p in solutions])


def impact_offset(corners, point, uncertainty_px=0.0, boundary_margin_px=2.0):
    q = polygon_array(corners)
    finite(point, "impact point")
    if len(point) != 2 or not math.isfinite(uncertainty_px) or uncertainty_px < 0:
        raise ValueError("invalid impact measurement")
    if not math.isfinite(boundary_margin_px) or boundary_margin_px < 0:
        raise ValueError("invalid boundary margin")
    target = polygon_center(q)
    distance = float(cv2.pointPolygonTest(q, tuple(map(float, point)), True))
    du, dv = target[0] - point[0], target[1] - point[1]
    boundary = abs(distance) <= uncertainty_px + boundary_margin_px
    location = "boundary_uncertain" if boundary else ("inside" if distance > 0 else "outside")
    starts = q.astype(float)
    segments = np.roll(starts,-1,axis=0)-starts
    lengths2 = np.sum(segments*segments,axis=1)
    fractions = np.clip(np.sum((np.asarray(point,dtype=float)-starts)*segments,axis=1)/lengths2,0.,1.)
    edge_distances = np.linalg.norm(starts+fractions[:,None]*segments-np.asarray(point,dtype=float),axis=1)
    edge = ("top","right","bottom","left")[int(np.argmin(edge_distances))]
    return dict(target=target, impact=tuple(map(float, point)), delta_u=float(du), delta_v=float(dv),
                signed_edge_distance_px=distance, location=location, uncertainty_px=float(uncertainty_px),
                nearest_edge=edge, impact_offset_u=float(-du), impact_offset_v=float(-dv),
                coordinate_convention="image_u_right_v_down",
                horizontal="right" if du > 0 else ("left" if du < 0 else "center"),
                vertical="down" if dv > 0 else ("up" if dv < 0 else "center"),
                scope="desired_image_impact_displacement; not actuator command")
