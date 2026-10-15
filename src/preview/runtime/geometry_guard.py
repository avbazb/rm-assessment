"""Separate physical corner admission from unchanged 2D detections.

Use geometry_for_pose only on the copy passed to PnP. The detector observation
and the detector's original tracking_measurement must remain the 2D inputs.
"""
import copy
import numpy as np

POLICY = {
    'name': 'pose_central_axis_rejection_v3',
    'left_axis_fraction_max': .40,
    'right_axis_fraction_min': .60,
    'outer_box_gate': False,
}


def geometry_for_pose(observation):
    """Return a deep copy whose corners may be withheld from PnP.

    Fractions are relative to the unchanged model box. This screens ownership;
    it is not a calibrated probability or a physical-corner guarantee.
    """
    out = copy.deepcopy(observation)
    if not observation.get('geometry_valid') or observation.get('corners') is None:
        out['pose_geometry_guard'] = dict(policy=POLICY['name'], status='unavailable')
        return out
    status = 'accepted'
    reason = None
    fractions = None
    evidence=observation.get('fusion_evidence') or {}
    if evidence.get('status')=='rotation_roi_verified' and evidence.get('measured_lamp_extent_fraction',1.)<.25:
        out.update(corners=None,geometry_valid=False,geometry_score=None,geometry_reason='incomplete_rotation_lamp_extent')
        out['pose_geometry_guard']=dict(policy='rotation_measured_lamp_extent',status='rejected',reason='incomplete_rotation_lamp_extent')
        return out
    try:
        q = np.asarray(observation['corners'], dtype=float)
        box = np.asarray(observation['bbox'], dtype=float)
        if q.shape != (4, 2) or box.shape != (4,) or not np.isfinite(q).all() or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
            raise ValueError('invalid geometry')
        axes = np.array([q[[0, 3]].mean(axis=0), q[[1, 2]].mean(axis=0)])
        uv = (axes - box[:2]) / (box[2:] - box[:2])
        fractions = uv.tolist()
        # Model box edges can clip true lamps. Screen only inward confusion
        # (often a numeral stroke), not slight displacement past a model edge.
        if not (uv[0, 0] <= POLICY['left_axis_fraction_max'] and uv[1, 0] >= POLICY['right_axis_fraction_min']):
            status, reason = 'rejected', 'pose_central_axis_rejected'
    except (TypeError, ValueError, KeyError, OverflowError):
        status, reason = 'rejected', 'pose_invalid_geometry'
    out['pose_geometry_guard'] = dict(policy=POLICY['name'], status=status,
        reason=reason, axis_box_fractions=fractions, thresholds=copy.deepcopy(POLICY))
    if status == 'rejected':
        out['pose_geometry_original'] = {k: copy.deepcopy(observation.get(k)) for k in
            ('corners', 'geometry_valid', 'geometry_reason', 'geometry_score')}
        out.update(corners=None, geometry_valid=False, geometry_score=None, geometry_reason=reason)
    return out


from dataclasses import dataclass
import copy,math
import cv2,numpy as np

@dataclass(frozen=True)
class OrientationConfig:
    max_gap_s:float=.1
    large_step_deg:float=45.
    coherent_motion_residual_deg:float=25.

def rotation_angle(a,b):
    cosine=(np.trace(np.asarray(a,float)@np.asarray(b,float).T)-1.)/2.
    return float(np.degrees(np.arccos(np.clip(cosine,-1.,1.))))

def check_orientation(history,raw_pose,filtered_pose,timestamp,config=OrientationConfig()):
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
