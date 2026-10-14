"""Research-only publication check; never changes PnP, translation or rotation.

Thresholds screen temporal inconsistency, not physical truth or a speed limit.
The caller retains raw_pose and filtered_pose; a waiting readout has no rotation.
"""
from dataclasses import dataclass
import copy,math
import cv2,numpy as np

@dataclass(frozen=True)
class Config:
    max_gap_s:float=.1
    large_step_deg:float=45.
    coherent_motion_residual_deg:float=25.

def rotation_angle(a,b):
    cosine=(np.trace(np.asarray(a,float)@np.asarray(b,float).T)-1.)/2.
    return float(np.degrees(np.arccos(np.clip(cosine,-1.,1.))))

def check_orientation(history,raw_pose,filtered_pose,timestamp,config=Config()):
    """Pure function returning new history and a separate orientation readout.

    A large step is withheld until a second step supports the same angular
    motion. Fast coherent motion is passed unchanged, never clamped or frozen.
    Alternative IPPE solutions are diagnostic only and are never substituted.
    """
    q=filtered_pose.get('rotation_matrix')
    if q is None or np.asarray(q).shape!=(3,3) or not np.isfinite(q).all():
        return [],dict(status='unavailable',reason=filtered_pose.get('status'),rotation_for_display=None,rvec_for_display=None,axes_for_display=None)
    current=dict(timestamp=float(timestamp),rotation_matrix=copy.deepcopy(q))
    previous=history[-1] if history else None
    dt=float(timestamp-previous['timestamp']) if previous else None
    checked=previous is not None and 0<dt<=config.max_gap_s+1e-9
    step=rotation_angle(q,previous['rotation_matrix']) if checked else None
    coherent=False;residual=None;previous_step=None
    if checked and len(history)>=2:
        earlier=history[-2];old_dt=previous['timestamp']-earlier['timestamp']
        if 0<old_dt<=config.max_gap_s+1e-9:
            increment=np.asarray(previous['rotation_matrix'])@np.asarray(earlier['rotation_matrix']).T
            rv=cv2.Rodrigues(increment)[0]
            predicted=cv2.Rodrigues(rv*(dt/old_dt))[0]@np.asarray(previous['rotation_matrix'])
            residual=rotation_angle(q,predicted)
            previous_step=rotation_angle(previous['rotation_matrix'],earlier['rotation_matrix'])
            coherent=(previous_step>config.large_step_deg and step>config.large_step_deg
                and residual<=config.coherent_motion_residual_deg)
    waiting=bool(checked and step>config.large_step_deg and not coherent)
    alternatives=[]
    if checked:
        for i,h in enumerate(raw_pose.get('hypotheses',[])):
            alternatives.append(dict(index=i,reprojection_px=h['reprojection_px'],angle_to_previous_deg=rotation_angle(h['rotation_matrix'],previous['rotation_matrix'])))
    out=dict(status='waiting' if waiting else ('consistent_fast_motion' if coherent else 'consistent' if checked else 'unchecked'),
        reason='rotation_step_needs_confirmation' if waiting else None,
        dt=dt,rotation_step_deg=step,previous_rotation_step_deg=previous_step,constant_rate_residual_deg=residual,
        rotation_for_display=None if waiting else copy.deepcopy(q),
        rvec_for_display=None if waiting else copy.deepcopy(filtered_pose.get('rvec_rad')),
        axes_for_display=None if waiting else copy.deepcopy(filtered_pose.get('axes_image_px')),
        hypotheses=alternatives,raw_status=raw_pose.get('status'),filtered_status=filtered_pose.get('status'),
        orientation_reliable=False if waiting else filtered_pose.get('orientation_reliable',False),
        input_orientation_reliable=filtered_pose.get('orientation_reliable',False),
        metric_valid=filtered_pose.get('metric_valid',False),numerical_pose_changed=False)
    return ([copy.deepcopy(previous),current] if checked else [current]),out
