"""Isolated stationary-rotor adapter. Only raw, admitted PnP enters the EKF.

The camera is assumed level and fixed; optical coordinates are explicitly
converted to forward/left/up. Unknown camera calibration never authorizes a
window. Plate slot numbers are model slots, not verified physical identities.
"""
import copy
import math
import numpy as np
from astra_competition.geometry import Camera, PlatePose
from astra_competition.spin import RotationTracker, SpinConfig


def options(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value)-{'plate_count','radius_m','projectile_speed_mps','execution_delay_s','target_track','target_robot'}:
        raise ValueError('旋转参数无效')
    result = dict(plate_count=4, radius_m=.25, projectile_speed_mps=15., execution_delay_s=.03, target_track=None,target_robot=None)
    result.update(value)
    if isinstance(result['plate_count'], bool) or not isinstance(result['plate_count'],int) or result['plate_count'] not in (2,3,4):
        raise ValueError('装甲板数量须为2～4')
    for key, lo, hi in [('radius_m',.05,1.),('projectile_speed_mps',1.,100.),('execution_delay_s',0.,.5)]:
        v=result[key]
        if isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or not lo<=v<=hi:
            raise ValueError('旋转参数超出范围')
    tid=result['target_track']
    if tid is not None and (isinstance(tid,bool) or not isinstance(tid,int) or not 1<=tid<=1000000000):
        raise ValueError('旋转目标ID无效')
    rid=result['target_robot']
    if rid is not None and (isinstance(rid,bool) or not isinstance(rid,int) or not 1<=rid<=1000000000):raise ValueError('车体目标ID无效')
    return result


class RotationPreview:
    def __init__(self, settings):
        self.settings=options(settings)
        self.config=SpinConfig(**{k:v for k,v in self.settings.items() if k not in ('target_track','target_robot')})
        self.tracker=RotationTracker(self.config)
        self.color=None
        self.anchor=None
        self.last_timestamp=None
        self.last_rotation=None
        self.input_count=0
        self.metric_valid=False

    def update(self, observations, tracks, timestamp, calibration):
        if self.last_timestamp==timestamp:
            return
        if self.last_timestamp is not None and timestamp<self.last_timestamp:
            raise ValueError('旋转时间必须递增')
        self.metric_valid=calibration.get('intrinsics_source')=='provided'
        camera=Camera(tuple(map(tuple,calibration['camera_matrix'])),tuple(calibration['distortion_coefficients']))
        rows=[]
        for index,o in enumerate(observations):
            pose=o.get('pose') or {}
            if not o.get('geometry_valid') or pose.get('status')!='estimated' or pose.get('reprojection_px',math.inf)>3:
                continue
            p=np.asarray(pose['tvec_m'],float); n=np.asarray(pose['outward_normal'],float)
            level=np.array([p[2],-p[0],-p[1]])
            normal=np.array([n[2],-n[0],-n[1]])
            if np.linalg.norm(normal[:2])<.8:
                continue
            center=level-self.config.radius_m*normal
            tid=next((t['track_id'] for t in tracks if t.get('observed') and t.get('observation_index')==index),None)
            area=(o['bbox'][2]-o['bbox'][0])*(o['bbox'][3]-o['bbox'][1])
            rows.append((o,PlatePose(timestamp,tuple(level),tuple(normal),pose['reprojection_px'],False,o['source']),center,tid,area))
        if self.anchor is None:
            seed=[r for r in rows if self.settings['target_track'] is None or r[3]==self.settings['target_track']]
            if seed:
                seed=max(seed,key=lambda r:r[4]); self.anchor=seed[2].copy(); self.color=seed[0]['color']
        center=np.asarray(self.tracker.state[:3]) if self.tracker.state is not None else self.anchor
        poses=[] if center is None else [r[1] for r in rows if r[0]['color']==self.color and
            np.linalg.norm(r[2]-center)<=max(.08,self.config.radius_m*.5)]
        self.input_count=len(poses)
        self.last_rotation=self.tracker.update(poses,timestamp)
        self.last_timestamp=timestamp
        self.camera=camera
        self.calibration=copy.deepcopy(calibration)

    def publish(self, now, enabled, horizon):
        if self.last_timestamp is None:
            return dict(enabled=enabled,status='waiting',reason='waiting_for_observations',state=None,candidates=[])
        if enabled:
            decision=self.tracker.decide(max(now,self.last_timestamp),self.camera)
        else:
            decision=dict(allow_fire=False,selected_plate=None,candidates=[],reason='prediction_disabled')
        if not self.metric_valid and decision.get('allow_fire'):
            decision.update(allow_fire=False,reason='camera_calibration_required')
        result=dict(schema='astra.rotation-preview.v1',enabled=enabled,status='tracking' if self.tracker.state is not None else 'waiting',
            state_timestamp=self.last_timestamp,last_observation_timestamp=self.tracker.last_observation,
            state=copy.deepcopy(self.last_rotation['state']),covariance=copy.deepcopy(self.last_rotation['covariance']),
            samples=self.tracker.frames,measurement_span_s=0. if self.tracker.started is None else self.last_timestamp-self.tracker.started,
            input_pose_count=self.input_count,color=self.color,config={k:getattr(self.config,k) for k in
                ('plate_count','radius_m','angular_jerk_std','unknown_acceleration_bound_rad_s2','max_gap_s','min_observations','min_duration_s','max_angular_speed_rad_s','hit_width_m','max_face_angle_deg','max_prediction_horizon_s')},
            calibration=self.calibration,metric_valid=self.metric_valid,camera_mount='fixed_level_assumed',
            coordinate_frame='camera_level_assumed: X forward Y left Z up',selected_horizon_s=horizon,
            decision=decision,filter_kind='angular_CA_EKF',observation_age_s=max(0.,now-(self.tracker.last_observation or self.last_timestamp)))
        return result
