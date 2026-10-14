"""Camera-frame PnP with explicit provenance for supplied or assumed intrinsics."""
import math
import cv2
import numpy as np
from .geometry import Camera, PlateGeometry, polygon_array

PROFILES={"rm_vision_small":(.135,.055),"rm_vision_large":(.225,.055)}
PROFILE_SOURCE="https://github.com/chenjunnn/rm_auto_aim/blob/main/armor_detector/include/armor_detector/pnp_solver.hpp"


def default_calibration(width,height,profile="rm_vision_small"):
    """Preview assumption, not an RM/camera specification or a calibration."""
    if any(isinstance(v,bool) or not isinstance(v,int) or not 8<=v<=16384 for v in (width,height)):
        raise ValueError("无效的视频分辨率")
    focal=width/(2*math.tan(math.radians(60)/2))
    return validate_calibration(dict(image_width=width,image_height=height,
        camera_matrix=[[focal,0.,width/2],[0.,focal,height/2],[0.,0.,1.]],
        distortion_coefficients=[0.]*5,reference_profile=profile,max_reprojection_px=3.,
        intrinsics_source="assumed_fov",assumed_horizontal_fov_deg=60.))


def validate_calibration(value):
    if value is None:
        return None
    if not isinstance(value,dict):
        raise ValueError("相机标定须为 JSON 对象")
    try:
        width,height=value["image_width"],value["image_height"]
        if any(isinstance(v,bool) or not isinstance(v,int) or not 8<=v<=16384 for v in (width,height)):
            raise ValueError("invalid resolution")
        matrix=np.asarray(value["camera_matrix"],dtype=float)
        distortion=np.asarray(value["distortion_coefficients"],dtype=float)
        camera=Camera(tuple(map(tuple,matrix)),tuple(distortion))
        if matrix[0,1]!=0 or matrix[1,0]!=0 or not 0<=matrix[0,2]<width or not 0<=matrix[1,2]<height:
            raise ValueError("invalid principal point or skew")
        profile=value["reference_profile"]
        if profile=="custom":
            dimensions=np.asarray(value["lamp_reference_mm"],dtype=float)
            if dimensions.shape!=(2,) or not np.isfinite(dimensions).all() or np.any(dimensions<=0) or np.any(dimensions>1000):
                raise ValueError("invalid measured lamp dimensions")
            source="user_measured_lamp_reference"
        else:
            dimensions=np.asarray(PROFILES[profile])*1000
            source=PROFILE_SOURCE
        error=float(value.get("max_reprojection_px",3.))
        if not math.isfinite(error) or not 0<error<=10:
            raise ValueError("invalid reprojection limit")
        intrinsics_source=value.get("intrinsics_source","provided")
        if intrinsics_source not in ("provided","assumed_fov"):
            raise ValueError("invalid intrinsics source")
        assumed_fov=None
        if intrinsics_source=="assumed_fov":
            assumed_fov=float(value["assumed_horizontal_fov_deg"])
            if not math.isfinite(assumed_fov) or not 1<assumed_fov<179:
                raise ValueError("invalid assumed field of view")
    except (KeyError,TypeError,ValueError,OverflowError) as exc:
        raise ValueError("标定参数无效：需要图像尺寸、3×3 内参、畸变系数及灯条规格") from exc
    return dict(image_width=width,image_height=height,camera_matrix=camera.k.tolist(),
        distortion_coefficients=distortion.tolist(),reference_profile=profile,lamp_reference_mm=dimensions.tolist(),
        reference_source=source,max_reprojection_px=error,
        intrinsics_source=intrinsics_source,assumed_horizontal_fov_deg=assumed_fov,
        frame="camera_optical: x right, y down, z forward",corner_order="TL TR BR BL",
        rotation="Rodrigues axis-angle vector in radians; not Euler angles")


def observation_pose(observation,calibration,image_size,solver=None):
    corners=observation.get('corners')
    # A full model ROI can expose truncated lamp cores. Keep its detection and
    # measured corners, but do not call a tiny surviving fragment a full lamp.
    if calibration and corners is not None and observation.get('source')!='classical':
        q=np.asarray(corners,dtype=float)
        if q.shape==(4,2) and np.isfinite(q).all():
            left=q[3]-q[0];right=q[2]-q[1];axis=left+right;norm=np.linalg.norm(axis)
            if norm>0:
                box=np.asarray(observation['bbox']);extent=float(np.abs(axis/norm)@(box[2:]-box[:2]))
                fraction=float(min(np.linalg.norm(left),np.linalg.norm(right))/max(extent,1.))
                if fraction<.25:
                    return dict(status='incomplete_lamp_extent',lamp_extent_fraction=fraction,min_extent_fraction=.25)
    return (solver or camera_pose)(corners,calibration,image_size)


def camera_pose(corners,calibration,image_size):
    if calibration is None:
        return dict(status="calibration_required")
    if tuple(image_size)!=(calibration["image_width"],calibration["image_height"]):
        return dict(status="resolution_mismatch")
    if corners is None:
        return dict(status="lamp_corners_required")
    try:
        q=polygon_array(corners).astype(np.float64)
    except ValueError:
        return dict(status="invalid_corners")
    if np.any(q<0) or np.any(q[:,0]>=image_size[0]) or np.any(q[:,1]>=image_size[1]):
        return dict(status="corners_outside_image")
    if cv2.contourArea(q.astype(np.float32),oriented=True)<=0:
        return dict(status="invalid_corner_order")
    width,height=calibration["lamp_reference_mm"]
    geometry=PlateGeometry(lamp_spacing_m=width/1000,lamp_length_m=height/1000)
    k=np.asarray(calibration["camera_matrix"],dtype=np.float64)
    distortion=np.asarray(calibration["distortion_coefficients"],dtype=np.float64)
    try:
        solved=cv2.solvePnPGeneric(geometry.reference_points,q,k,distortion,flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return dict(status="solver_failed")
    candidates=[]
    if solved[0]:
        for rvec,tvec in zip(solved[1],solved[2]):
            if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
                continue
            rotation,_=cv2.Rodrigues(rvec)
            points=(rotation@geometry.reference_points.T).T+tvec.reshape(3)
            if np.any(points[:,2]<=0) or np.dot(-rotation[:,2],-tvec.reshape(3))<=0:
                continue
            projected,_=cv2.projectPoints(geometry.reference_points,rvec,tvec,k,distortion)
            rms=float(np.sqrt(np.mean(np.sum((projected.reshape(4,2)-q)**2,axis=1))))
            if rms>calibration["max_reprojection_px"]:
                continue
            # Decompose R = Rz(z) @ Ry(y) @ Rx(x); rvec itself is never treated
            # as Euler angles. At gimbal lock choose z=0 and flag ambiguity.
            sy=math.hypot(rotation[0,0],rotation[1,0])
            singular=sy<1e-6
            angles=(math.atan2(-rotation[1,2],rotation[1,1]) if singular else math.atan2(rotation[2,1],rotation[2,2]),
                    math.atan2(-rotation[2,0],sy),0. if singular else math.atan2(rotation[1,0],rotation[0,0]))
            axis_length=height/1000
            axes=np.vstack((np.zeros(3),np.eye(3)*axis_length))
            axes_pixels,_=cv2.projectPoints(axes,rvec,tvec,k,distortion)
            candidates.append(dict(tvec_m=tvec.reshape(3).tolist(),rvec_rad=rvec.reshape(3).tolist(),
                rotation_matrix=rotation.tolist(),reprojection_px=rms,range_m=float(np.linalg.norm(tvec)),
                depth_m=float(tvec.reshape(3)[2]),euler_xyz_deg=np.degrees(angles).tolist(),euler_singular=singular,
                corners_camera_m=points.tolist(),axes_image_px=axes_pixels.reshape(4,2).tolist(),
                axes_camera_m=((rotation@axes.T).T+tvec.reshape(3)).tolist(),
                outward_normal=(-rotation[:,2]).tolist()))
    candidates.sort(key=lambda c:c["reprojection_px"])
    if not candidates:
        return dict(status="reprojection_rejected")
    ambiguous=(len(candidates)>1 and candidates[1]["reprojection_px"]-candidates[0]["reprojection_px"]<.25
        and np.dot(candidates[0]["outward_normal"],candidates[1]["outward_normal"])<math.cos(math.radians(12)))
    approximate=calibration.get("intrinsics_source")=="assumed_fov"
    return dict(status="ambiguous" if ambiguous else "estimated",**candidates[0],hypotheses=candidates,
        reference_profile=calibration["reference_profile"],lamp_reference_mm=calibration["lamp_reference_mm"],
        reference_source=calibration["reference_source"],coordinate_frame="camera_optical",
        frame_axes="x right, y down, z forward",angle_representation="Rodrigues radians",
        euler_convention="R = Rz(z) @ Ry(y) @ Rx(x); angles in degrees",
        intrinsics_source=calibration.get("intrinsics_source","provided"),approximate=approximate,
        assumed_horizontal_fov_deg=calibration.get("assumed_horizontal_fov_deg"),
        pose_kind="assumed_intrinsics_image_time_estimate" if approximate else "raw_image_time_measurement",
        metric_valid=not approximate,orientation_reliable=not ambiguous and not approximate)
