from __future__ import annotations

from dataclasses import dataclass
import copy
from bisect import bisect_left
import math
import numpy as np

from .geometry import finite


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class SpinConfig:
    plate_count: int = 4
    radius_m: float = .25
    position_std_m: float = .025
    yaw_std_rad: float = .06
    max_gap_s: float = .25
    min_observations: int = 6
    min_duration_s: float = .25
    projectile_speed_mps: float = 15.
    execution_delay_s: float = .03
    hit_width_m: float = .135
    max_face_angle_deg: float = 55.
    switch_margin: float = .12
    max_angular_speed_rad_s: float = 16.
    motion_model: str = "ca"
    angular_jerk_std: float = 8.
    unknown_acceleration_bound_rad_s2: float = 10.
    max_prediction_horizon_s: float = .8
    history_s: float = 1.

    def __post_init__(self):
        if isinstance(self.plate_count, bool) or not isinstance(self.plate_count, int) or self.plate_count < 2:
            raise ValueError("invalid plate count")
        for name in ("radius_m", "position_std_m", "yaw_std_rad", "max_gap_s", "min_duration_s",
                     "projectile_speed_mps", "hit_width_m", "max_angular_speed_rad_s", "angular_jerk_std",
                     "max_prediction_horizon_s", "history_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"invalid {name}")
        if not math.isfinite(self.execution_delay_s) or self.execution_delay_s < 0 or not 0 < self.max_face_angle_deg < 90:
            raise ValueError("invalid decision window")
        if not math.isfinite(self.switch_margin) or self.switch_margin < 0 or self.min_observations < 2:
            raise ValueError("invalid stability gates")
        if self.motion_model not in ("cv","ca") or not math.isfinite(self.unknown_acceleration_bound_rad_s2) or self.unknown_acceleration_bound_rad_s2<0:
            raise ValueError("invalid angular motion model")


def flight_time(target, speed):
    """Low-arc vacuum flight from the origin, using estimated target coordinates."""
    x, y, z = target
    horizontal = math.hypot(x, y)
    g = 9.81
    if horizontal < 1e-6:
        return None
    disc = speed ** 4 - g * (g * horizontal ** 2 + 2 * z * speed ** 2)
    if disc <= 0:
        return None
    pitch = math.atan2(speed ** 2 - math.sqrt(disc), g * horizontal)
    return horizontal / (speed * math.cos(pitch))


class RotationTracker:
    """Stationary center + short-horizon angular CA/CV EKF at acquisition time."""

    def __init__(self, config=None):
        self.config = config or SpinConfig()
        self.reset()

    def reset(self):
        self.state = None
        self.covariance = None
        self.timestamp = None
        self.last_observation = None
        self.started = None
        self.observations = 0
        self.frames = 0
        self.selected = None
        self.observed_this_frame = False

    def update(self, poses, timestamp):
        finite(timestamp)
        if timestamp<0 or any(abs(p.timestamp-timestamp)>1e-6 for p in poses):
            raise ValueError("pose/image timestamps differ or invalid time")
        for pose in poses:
            finite(pose.timestamp)
            finite(pose.position);finite(pose.outward_normal);finite(pose.reprojection_px)
        if self.timestamp is not None and timestamp <= self.timestamp:
            if timestamp < self.timestamp:
                self.reset()
            else:
                raise ValueError("duplicate rotation timestamp")
        usable = [p for p in poses if not p.ambiguity and p.reprojection_px <= 3]
        cfg = self.config
        if self.last_observation is not None and timestamp - self.last_observation > cfg.max_gap_s:
            self.reset()
        if self.last_observation is not None and (timestamp - self.last_observation) * cfg.max_angular_speed_rad_s >= math.pi / cfg.plate_count:
            # Missing reliable poses count as a sampling gap even if RGB frames
            # keep arriving; identical plates can otherwise alias across the gap.
            self.reset()
        self.observed_this_frame = False
        if self.state is not None:
            dt = timestamp - self.timestamp
            self.state,self.covariance=self._propagate(self.state,self.covariance,dt)
        self.timestamp = float(timestamp)
        used_slots = set()
        accepted = []
        for pose in sorted(usable, key=lambda p: p.reprojection_px):
            p = np.array(pose.position)
            n = np.array(pose.outward_normal)
            finite(p)
            finite(n)
            if np.linalg.norm(n[:2]) < .8:
                continue
            yaw = math.atan2(n[1], n[0])
            if self.state is None:
                self.state = np.array((p[0] - cfg.radius_m * math.cos(yaw),
                                       p[1] - cfg.radius_m * math.sin(yaw), p[2], yaw, 0.,0.))
                self.covariance = np.diag((.0025, .0025, .0025, .01, 100.,100. if cfg.motion_model=="ca" else 0.))
                self.started = float(timestamp)
                slot = 0
            else:
                yaw_residuals = [wrap(yaw - (self.state[3] + i * 2 * math.pi / cfg.plate_count)) for i in range(cfg.plate_count)]
                slot = min(range(cfg.plate_count), key=lambda i: abs(yaw_residuals[i]))
                if slot in used_slots or abs(yaw_residuals[slot]) > .6:
                    continue
                theta = self.state[3] + slot * 2 * math.pi / cfg.plate_count
                predicted = np.array((self.state[0] + cfg.radius_m * math.cos(theta),
                                      self.state[1] + cfg.radius_m * math.sin(theta), self.state[2], theta))
                residual = np.array((*p, yaw)) - predicted
                residual[3] = wrap(residual[3])
                h = np.zeros((4, 6))
                h[:3, :3] = np.eye(3)
                h[0, 3] = -cfg.radius_m * math.sin(theta)
                h[1, 3] = cfg.radius_m * math.cos(theta)
                h[3, 3] = 1
                r = np.diag((cfg.position_std_m ** 2,) * 3 + (cfg.yaw_std_rad ** 2,))
                innovation = h @ self.covariance @ h.T + r
                if float(residual @ np.linalg.solve(innovation, residual)) > 25:
                    continue
                gain = np.linalg.solve(innovation, h @ self.covariance).T
                self.state += gain @ residual
                a = np.eye(6) - gain @ h
                self.covariance = a @ self.covariance @ a.T + gain @ r @ gain.T
                self.covariance=(self.covariance+self.covariance.T)/2
            used_slots.add(slot)
            accepted.append(slot)
            self.observations += 1
            self.last_observation = float(timestamp)
        if accepted:
            self.frames += 1
            self.observed_this_frame = True
        return dict(timestamp=self.timestamp, state_timestamp=self.timestamp, accepted_slots=accepted, observations=self.observations,
                    motion_model=cfg.motion_model,state_fields=["center_x","center_y","center_z","phase","angular_speed","angular_acceleration"],
                    state=self.state.tolist() if self.state is not None else None,
                    covariance=self.covariance.tolist() if self.covariance is not None else None,
                    coordinate_frame="level_world; X forward Y left Z up; launcher at origin")

    def _propagate(self,state,covariance,dt):
        f=np.eye(6);f[3,4]=dt
        q=np.zeros((6,6));q[:3,:3]=np.eye(3)*1e-6*dt
        if self.config.motion_model=="ca":
            f[3,5]=dt*dt/2;f[4,5]=dt
            q[3:,3:]=self.config.angular_jerk_std**2*np.array([
                [dt**5/20,dt**4/8,dt**3/6],[dt**4/8,dt**3/3,dt**2/2],[dt**3/6,dt**2/2,dt]])
        else:
            q[3,3]=1e-4*dt;q[4,4]=.2*dt
        return f@state,f@covariance@f.T+q

    def predict_at(self,timestamp):
        """Copy-only forecast: repeated render/decision calls never advance the EKF."""
        finite(timestamp)
        if self.state is None:
            return None
        horizon=float(timestamp-self.timestamp)
        if horizon<0:
            raise ValueError("prediction predates filter state")
        state,covariance=self._propagate(self.state,self.covariance,horizon)
        # This bound encloses any unobserved acceleration in [-A,A], measured
        # around the CA forecast. It is additional to measurement uncertainty.
        bound=self.config.unknown_acceleration_bound_rad_s2
        envelope=float(.5*(bound+abs(self.state[5]))*horizon*horizon) if bound else 0.
        return dict(state_timestamp=self.timestamp,target_timestamp=float(timestamp),prediction_horizon_s=horizon,
            state=state.tolist(),covariance=covariance.tolist(),phase_std_rad=float(math.sqrt(max(covariance[3,3],0))),
            unobserved_acceleration_envelope_rad=envelope,motion_model=self.config.motion_model,
            reliable_horizon=horizon<=self.config.max_prediction_horizon_s)

    def decide(self, now, camera=None):
        finite(now)
        cfg = self.config
        result = dict(timestamp=float(now), allow_fire=False, selected_plate=None, candidates=[],
                      reason="waiting_for_observations", scope="simulated_window_recommendation")
        if self.state is None:
            return result
        age = now - self.last_observation
        if age < 0:
            raise ValueError("decision predates observation")
        if now<self.timestamp:
            raise ValueError("decision predates filter state")
        result.update(observation_age_s=float(age), state_timestamp=self.timestamp,motion_model=cfg.motion_model,
            angular_speed_rad_s=float(self.state[4]),angular_acceleration_rad_s2=float(self.state[5]),
            current_prediction=self.predict_at(now),execution_delay_s=cfg.execution_delay_s)
        if age > cfg.max_gap_s:
            result["reason"] = "stale_observation"
            return result
        if not self.observed_this_frame:
            result["reason"] = "current_frame_unobserved"
            return result
        if (self.frames < cfg.min_observations or self.last_observation - self.started < cfg.min_duration_s
                or math.sqrt(self.covariance[4, 4]) > 1.0):
            result["reason"] = "rotation_not_stable"
            return result
        base_delay = now - self.timestamp + cfg.execution_delay_s
        for slot in range(cfg.plate_count):
            horizon = base_delay
            position = None
            for _ in range(5):
                predicted=self.predict_at(self.timestamp+horizon)
                theta = predicted["state"][3] + slot * 2 * math.pi / cfg.plate_count
                position = self.state[:3] + np.array((cfg.radius_m * math.cos(theta), cfg.radius_m * math.sin(theta), 0))
                tof = flight_time(position, cfg.projectile_speed_mps)
                if tof is None:
                    break
                horizon = base_delay + tof
            if tof is None:
                result["candidates"].append(dict(plate=slot, valid=False, reason="unreachable_ballistic_target"))
                continue
            # Recompute at the final horizon; all times reference the filter state timestamp.
            predicted=self.predict_at(self.timestamp+horizon)
            theta = predicted["state"][3] + slot * 2 * math.pi / cfg.plate_count
            normal = np.array((math.cos(theta), math.sin(theta), 0.))
            position = self.state[:3] + cfg.radius_m * normal
            view = -position / np.linalg.norm(position)
            facing = float(np.dot(normal, view))
            projected_covariance=np.asarray(predicted["covariance"])
            phase_variance=projected_covariance[3,3]
            sigma = math.sqrt(max(0., max(projected_covariance[0,0],projected_covariance[1,1]) + cfg.radius_m ** 2 * phase_variance))
            maneuver_margin=cfg.radius_m*predicted["unobserved_acceleration_envelope_rad"]
            available = cfg.hit_width_m * max(facing, 0.) / 2
            valid = bool(predicted["reliable_horizon"] and facing >= math.cos(math.radians(cfg.max_face_angle_deg)) and 2*sigma+maneuver_margin < available)
            score = facing - (2*sigma+maneuver_margin) / cfg.hit_width_m
            result["candidates"].append(dict(plate=slot, position=position.tolist(), facing_cos=facing,
                uncertainty_m=sigma, prediction_horizon_s=horizon, flight_time_s=horizon-base_delay,
                state_timestamp=self.timestamp,target_timestamp=self.timestamp+horizon,
                observation_age_s=age,execution_delay_s=cfg.execution_delay_s,maneuver_margin_m=maneuver_margin,
                phase_std_rad=predicted["phase_std_rad"],unobserved_acceleration_envelope_rad=predicted["unobserved_acceleration_envelope_rad"],
                predicted_pixel=camera.level_to_pixel(position) if camera else None, score=score, valid=valid,
                reason="ready" if valid else ("back_or_oblique_face" if facing < math.cos(math.radians(cfg.max_face_angle_deg)) else "prediction_uncertain")))
        viable = [c for c in result["candidates"] if c["valid"]]
        if not viable:
            result["reason"] = "waiting_for_window"
            return result
        best = max(viable, key=lambda c: c["score"])
        current = next((c for c in viable if c["plate"] == self.selected), None)
        if current and best["score"] < current["score"] + cfg.switch_margin:
            best = current
        self.selected = best["plate"]
        result.update(allow_fire=True, selected_plate=self.selected, reason="ready")
        return result


class DelayedRotationTracker:
    """Bounded out-of-order replay of complete states, covariances and observations.

    Receipt time is validated but never substituted for pose acquisition time.
    No prediction or previously issued decision is replayed as a measurement.
    """
    def __init__(self,config=None):
        self.filter=RotationTracker(config)
        self.anchor=copy.deepcopy(self.filter)
        self.entries=[]
        self.delivery_timestamp=None
        self.replayed=0;self.dropped=0

    @staticmethod
    def _validate_numeric(value,shape,name):
        array=np.asarray(value)
        if array.shape!=shape or array.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be real numeric data with shape {shape}")
        # NumPy's EKF linear solves do not support extended floating dtypes.
        if array.dtype.kind=="f" and array.dtype.itemsize>8:
            raise ValueError(f"{name} uses an unsupported extended-precision floating dtype")
        finite(array,name)

    def update(self,poses,timestamp,received_at):
        self._validate_numeric(timestamp,(),"timestamp")
        self._validate_numeric(received_at,(),"received_at")
        if timestamp<0 or received_at<timestamp or (self.delivery_timestamp is not None and received_at<self.delivery_timestamp):
            raise ValueError("invalid acquisition/delivery clock")
        # Validate the complete owned batch before changing clocks, history or
        # the filter. A malformed entry must never survive into later replay.
        try:
            poses=copy.deepcopy(list(poses))
        except TypeError as exc:
            raise ValueError("poses must be an iterable of observations") from exc
        try:
            for pose in poses:
                self._validate_numeric(pose.timestamp,(),"pose timestamp")
                self._validate_numeric(pose.position,(3,),"pose position")
                self._validate_numeric(pose.outward_normal,(3,),"pose outward normal")
                self._validate_numeric(pose.reprojection_px,(),"pose reprojection error")
                if not isinstance(pose.ambiguity,(bool,np.bool_)):
                    raise ValueError("pose ambiguity must be boolean")
                if abs(pose.timestamp-timestamp)>1e-6:
                    raise ValueError("pose/image timestamps differ")
        except AttributeError as exc:
            raise ValueError("invalid pose fields") from exc
        self.delivery_timestamp=float(received_at)
        if self.anchor.timestamp is not None and timestamp<=self.anchor.timestamp:
            self.dropped+=1
            return dict(status="outside_history",replayed=0,state_timestamp=self.filter.timestamp)
        index=bisect_left([e["timestamp"] for e in self.entries],timestamp)
        if index<len(self.entries) and self.entries[index]["timestamp"]==timestamp:
            return dict(status="duplicate_ignored",replayed=0,state_timestamp=self.filter.timestamp)
        entry=dict(timestamp=float(timestamp),poses=poses)
        self.entries.insert(index,entry)
        count=len(self.entries)-index
        if count>1:
            selected=self.filter.selected
            self.filter=copy.deepcopy(self.anchor if index==0 else self.entries[index-1]["after"])
            self.filter.selected=selected
            self.replayed+=1
        for item in self.entries[index:]:
            result=self.filter.update(item["poses"],item["timestamp"])
            item["after"]=copy.deepcopy(self.filter)
        cutoff=self.filter.timestamp-self.filter.config.history_s
        while self.entries and self.entries[0]["timestamp"]<cutoff:
            self.anchor=self.entries.pop(0)["after"]
        return dict(result,status="replayed" if count>1 else "updated",replayed=count if count>1 else 0,
            received_at=float(received_at),delivered_observation_age_s=float(received_at-timestamp))

    def predict_at(self,timestamp):
        return self.filter.predict_at(timestamp)

    def decide(self,now,camera=None):
        result=self.filter.decide(now,camera)
        result["delivery"]=dict(replayed_measurements=self.replayed,dropped_measurements=self.dropped,history_s=self.filter.config.history_s)
        return result
