"""Causal image-time pose readout. Raw lamp/PnP measurements remain untouched."""
import copy,math
import cv2
import numpy as np
from .geometry import PlateGeometry

class PoseContinuity:
    def __init__(self,time_constant=.12,max_gap=.35):
        self.time_constant=time_constant;self.max_gap=max_gap;self.states={};self.pending={}

    def update(self,tracks,observations,timestamp,calibration):
        # A tracker may temporarily hide a coasting track while retaining its
        # ID internally. Absence from this frame's output must not erase pose
        # history; otherwise the same ID reappears with an unfiltered jump.
        self.states={k:v for k,v in self.states.items() if timestamp-v['timestamp']<=self.max_gap}
        self.pending={k:v for k,v in self.pending.items() if timestamp-v['timestamp']<=self.max_gap}
        for track in tracks:
            if not track['observed']:continue
            index=track.get('observation_index')
            if index is None:continue
            measurement=observations[index]['pose'];pose=copy.deepcopy(measurement)
            track['pose']=pose
            if not pose.get('tvec_m'):continue
            tid=track['track_id'];old=self.states.get(tid);chosen=pose
            # Near-equal planar hypotheses are still ambiguous. Continuity
            # chooses the display branch only; it does not certify orientation.
            if old and pose.get('hypotheses'):
                candidates=[h for h in pose['hypotheses'] if h['reprojection_px']<=pose['reprojection_px']+.25]
                chosen=min(candidates,key=lambda h:np.linalg.norm(cv2.Rodrigues(np.asarray(h['rotation_matrix'])@old['rotation'].T)[0]))
            raw=np.asarray(chosen['tvec_m']);rotation=np.asarray(chosen['rotation_matrix']);dt=timestamp-old['timestamp'] if old else 0
            fresh=old is None or dt<=0 or dt>self.max_gap or old['profile']!=pose['reference_profile']
            if not fresh:
                # A large apparent size change can be a different plate or a
                # camera cut. Start a new segment rather than blend identities.
                discontinuity=abs(math.log(raw[2]/old['raw'][2]))>.25+dt
                if discontinuity:
                    pending=self.pending.get(tid)
                    consistent=(pending is not None and 0<timestamp-pending['timestamp']<=.2 and abs(math.log(raw[2]/pending['depth']))<.10)
                    if not consistent:
                        self.pending[tid]=dict(timestamp=timestamp,depth=float(raw[2]))
                        track['pose']=dict(status='pose_temporal_waiting',timestamp=timestamp,reason='depth_discontinuity',raw_depth_m=float(raw[2]))
                        continue
                    fresh=True
            self.pending.pop(tid,None)
            alpha=1. if fresh else -math.expm1(-dt/self.time_constant)
            width,height=np.asarray(pose['lamp_reference_mm'])/1000
            reference=PlateGeometry(lamp_spacing_m=width,lamp_length_m=height).reference_points
            axes=np.vstack((np.zeros(3),np.eye(3)*height));k=np.asarray(calibration['camera_matrix']);dist=np.asarray(calibration['distortion_coefficients'])
            q=np.asarray(observations[index]['corners'])
            raw_rotation=rotation.copy()
            # Keep the current image bearing. Smoothing all XYZ coordinates
            # makes the axes trail a panning camera by tens of pixels.
            # Reduce smoothing until the readout still explains current lamps.
            while True:
                depth=raw[2] if fresh else old['position'][2]+alpha*(raw[2]-old['position'][2])
                filtered=raw*(depth/raw[2]);rotation=raw_rotation
                if not fresh:
                    delta=cv2.Rodrigues(raw_rotation@old['rotation'].T)[0]
                    rotation=cv2.Rodrigues(alpha*delta)[0]@old['rotation']
                rv=cv2.Rodrigues(rotation)[0].reshape(3)
                projected=cv2.projectPoints(reference,rv,filtered,k,dist)[0].reshape(4,2)
                fit=float(np.sqrt(np.mean(np.sum((projected-q)**2,axis=1))))
                if alpha>=1 or fit<=max(3.,chosen['reprojection_px']+2.):break
                alpha=min(1.,alpha+.25)
            pixels=cv2.projectPoints(axes,rv,filtered,k,dist)[0].reshape(4,2)
            velocity=None if fresh else ((filtered-old['position'])/dt).tolist()
            sy=math.hypot(rotation[0,0],rotation[1,0]);singular=sy<1e-6
            angles=(math.atan2(-rotation[1,2],rotation[1,1]) if singular else math.atan2(rotation[2,1],rotation[2,2]), math.atan2(-rotation[2,0],sy),0 if singular else math.atan2(rotation[1,0],rotation[0,0]))
            pose.update(tvec_m=filtered.tolist(),rvec_rad=rv.tolist(),rotation_matrix=rotation.tolist(),range_m=float(np.linalg.norm(filtered)),depth_m=float(filtered[2]),
                axes_image_px=pixels.tolist(),axes_camera_m=((rotation@axes.T).T+filtered).tolist(),corners_camera_m=((rotation@reference.T).T+filtered).tolist(),
                outward_normal=(-rotation[:,2]).tolist(),euler_xyz_deg=np.degrees(angles).tolist(),euler_singular=singular,
                pose_kind='causal_filtered_image_time_estimate',velocity_m_s=velocity,filter=dict(time_constant_s=self.time_constant,alpha=alpha,reset=fresh,
                raw_reprojection_px=measurement['reprojection_px'],filtered_reprojection_px=fit,projection_budget_px=max(3.,chosen['reprojection_px']+2.)),reprojection_px=fit)
            self.states[tid]=dict(timestamp=timestamp,position=filtered.copy(),raw=raw.copy(),rotation=rotation.copy(),profile=pose['reference_profile'])
