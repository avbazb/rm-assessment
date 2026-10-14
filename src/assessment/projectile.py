"""Causal monocular ballistic-contact estimator. No file/GT API.

Scope: calibrated stationary level camera, known vertical static impact plane,
known sphere radius, gravity-only flight. A forecast alone never confirms contact.
"""
from dataclasses import dataclass
import math
import cv2
import numpy as np


@dataclass
class FlightFit:
    origin_time: float
    state: np.ndarray
    camera: object
    rms_px: float
    max_px: float
    condition: float

    def position(self, timestamp):
        dt = timestamp - self.origin_time
        return self.state[:3] + self.state[3:] * dt + np.array((0., 4.905*dt*dt, 0.))

    def pixel(self, timestamp):
        p = self.position(timestamp)
        if p[2] <= .1:
            return None
        return cv2.projectPoints(p.reshape(1,3), np.zeros(3), np.zeros(3), self.camera.k,
                                 np.asarray(self.camera.distortion))[0].reshape(2)

    def intersection(self, plane_point, normal, radius):
        g = np.array((0.,9.81,0.))
        coefficients = (.5*np.dot(normal,g), np.dot(normal,self.state[3:]),
                        np.dot(normal,self.state[:3]-plane_point)-radius)
        roots = np.roots(np.trim_zeros(np.asarray(coefficients),'f'))
        candidates=[]
        for root in roots:
            if abs(root.imag) > 1e-8 or root.real < 0:
                continue
            dt=float(root.real)
            if np.dot(normal,self.state[3:]+g*dt) >= -.1:
                continue
            t=self.origin_time+dt
            center=self.position(t);surface=center-radius*normal
            pixel=cv2.projectPoints(surface.reshape(1,3),np.zeros(3),np.zeros(3),self.camera.k,
                                    np.asarray(self.camera.distortion))[0].reshape(2)
            candidates.append((t,pixel,surface))
        return min(candidates,key=lambda item:item[0]) if candidates else None


def fit_flight(points, camera):
    if len(points)<8 or points[-1]['timestamp']-points[0]['timestamp']<.15:
        return None
    stamps=np.asarray([p['timestamp'] for p in points]); times=stamps-stamps[0]
    pixels=np.asarray([p['center'] for p in points],dtype=float)
    rays=cv2.undistortPoints(pixels.reshape(-1,1,2),camera.k,np.asarray(camera.distortion)).reshape(-1,2)
    matrix=[];target=[]
    for (u,v),t in zip(rays,times):
        matrix.extend(((1,0,-u,t,0,-u*t),(0,1,-v,0,t,-v*t)))
        target.extend((0,-4.905*t*t))
    matrix=np.asarray(matrix);target=np.asarray(target)
    state,_,rank,sv=np.linalg.lstsq(matrix,target,rcond=None)
    if rank<6 or sv[-1]<=0 or sv[0]/sv[-1]>1e7:
        return None
    fit=FlightFit(float(stamps[0]),state,camera,0.,0.,float(sv[0]/sv[-1]))
    positions=np.array([fit.position(t) for t in stamps])
    if np.any(positions[:,2]<=.1) or not .2<state[5]<30:
        return None
    projected=np.array([fit.pixel(t) for t in stamps]);errors=np.linalg.norm(projected-pixels,axis=1)
    fit.rms_px=float(np.sqrt(np.mean(errors**2)));fit.max_px=float(np.max(errors))
    if fit.rms_px>1.5 or fit.max_px>3:
        return None
    return fit


def refine_contact(fit, samples, predicted_time, plane_point, normal, radius):
    """Fit the observed post-contact segment; do not snap to the armor plane.

    Searches past times only. A nearby surround may be at a different depth.
    Pre-contact fit is frozen before the event to prevent bounce contamination.
    """
    camera=fit.camera
    candidates=[]
    for event_time in np.arange(predicted_time-.05,predicted_time+.0801,.001):
        post=[s for s in samples if s['timestamp']>event_time+.002]
        if len(post)<4 or post[-1]['timestamp']-post[0]['timestamp']<.08:
            continue
        contact=fit.position(event_time)
        distance=float(np.dot(normal,contact-plane_point)-radius)
        if abs(distance)>.1:
            continue
        uv=np.array([s['center'] for s in post]);ts=np.array([s['timestamp']-event_time for s in post])
        rays=cv2.undistortPoints(uv.reshape(-1,1,2).astype(float),camera.k,np.asarray(camera.distortion)).reshape(-1,2)
        matrix=[];target=[]
        for (u,v),dt in zip(rays,ts):
            matrix.extend(((dt,0,-u*dt),(0,dt,-v*dt)))
            target.extend((-contact[0]+u*contact[2],-contact[1]+v*contact[2]-4.905*dt*dt))
        velocity=np.linalg.lstsq(matrix,target,rcond=None)[0]
        # Measured normal slowing/rebound is required. Ballistic continuation
        # beyond the target plane is an explicit negative control.
        incoming=fit.state[3:]+np.array((0.,9.81*(event_time-fit.origin_time),0.))
        if np.dot(normal,velocity)-np.dot(normal,incoming)<1. or velocity[2]>.5:
            continue
        positions=contact+ts[:,None]*velocity+np.column_stack((np.zeros(len(ts)),4.905*ts*ts,np.zeros(len(ts))))
        if np.any(positions[:,2]<.1):
            continue
        pred=cv2.projectPoints(positions,np.zeros(3),np.zeros(3),camera.k,np.asarray(camera.distortion))[0].reshape(-1,2)
        errors=list(np.linalg.norm(pred-uv,axis=1))
        before=[s for s in samples if s['timestamp']<=event_time+.002]
        errors.extend(np.linalg.norm(np.asarray(s['center'])-fit.pixel(s['timestamp'])) for s in before)
        cost=float(np.sqrt(np.mean(np.square(errors))))
        surface=contact-radius*normal
        pixel=cv2.projectPoints(surface.reshape(1,3),np.zeros(3),np.zeros(3),camera.k,np.asarray(camera.distortion))[0].reshape(2)
        candidates.append((cost,float(event_time),pixel,velocity))
    if not candidates:
        return None
    candidates.sort(key=lambda r:r[0]);best=candidates[0]
    equivalent=[r for r in candidates if r[0]<=best[0]+.25]
    spread=max(float(np.linalg.norm(r[2]-best[2])) for r in equivalent)
    if best[0]>1.5 or spread>8.:
        return None
    return {'timestamp':best[1],'pixel':tuple(map(float,best[2])),
            'post_velocity_optical_mps':best[3].tolist(),'two_segment_rms_px':best[0],
            'ambiguity_spread_px':spread,'uncertainty_px':max(3.,spread)}


def forecast_spread(points,camera,plane_point,normal,radius,reference):
    """Sensitivity to removing each recent observation, not a probability."""
    if len(points)<10:
        return None
    estimates=[]
    for index in range(max(0,len(points)-8),len(points)):
        fit=fit_flight(points[:index]+points[index+1:],camera)
        predicted=fit.intersection(plane_point,normal,radius) if fit else None
        if predicted is None:
            return None
        estimates.append(predicted)
    distance=max(float(np.linalg.norm(row[1]-reference[1])) for row in estimates)
    age=max(abs(row[0]-reference[0]) for row in estimates)
    return {'leave_one_out_spread_px':distance,'leave_one_out_time_spread_s':age,
            'scope':'local measurement sensitivity; not a calibrated confidence interval'}


class FlightContactCandidate:
    def __init__(self,camera,pose,radius_m=.02):
        if not math.isfinite(radius_m) or not 0<radius_m<.1 or pose.ambiguity:
            raise ValueError('unambiguous calibrated plane and positive sphere radius required')
        self.camera=camera;self.radius=radius_m
        x,y,z=pose.position
        self.plane_point=np.array((-y,camera.height_m-z,x))
        nx,ny,nz=pose.outward_normal
        self.normal=np.array((-ny,-nz,nx))
        self.points=[];self.fit=None;self.prediction=None;self.forecasts=[];self.departures=[]
        self.frozen=False;self.event_samples=[];self.reliability=None;self.last_timestamp=None
        self.result={'status':'warming_up','contact_confirmed':False,'forecast':None,'evidence':[]}

    def predict_pixel(self,timestamp):
        return self.fit.pixel(timestamp) if self.fit is not None else None

    def update(self,point,timestamp):
        if not math.isfinite(timestamp) or (self.last_timestamp is not None and timestamp<=self.last_timestamp):
            raise ValueError('strictly increasing finite acquisition timestamps required')
        self.last_timestamp=float(timestamp)
        if self.result['contact_confirmed']:
            return self.result
        if self.prediction is not None and timestamp>self.prediction[0]+.45:
            self.result.update(status='unconfirmed',forecast=None,state=None,evidence=['no_observed_contact_within_event_window'])
            return self.result
        if self.fit is not None:
            valid=self.prediction is None or timestamp<=self.prediction[0]
            self.result['state']={'timestamp':float(timestamp),
                'position_optical_m':self.fit.position(timestamp).tolist(),
                'velocity_optical_mps':(self.fit.state[3:]+np.array((0.,9.81*(timestamp-self.fit.origin_time),0.))).tolist(),
                'reprojection_rms_px':self.fit.rms_px,'points':len(self.points),
                'kind':'precontact_prediction'} if valid else None
        if point is None:
            # Missing pixels are never a collision observation.
            self.departures=[]
            self.result['evidence']=['missing_observation_not_contact']
            return self.result
        if self.points and timestamp<=self.points[-1]['timestamp']:
            raise ValueError('strictly increasing acquisition timestamps required')
        point=dict(point,timestamp=float(timestamp))
        if self.frozen:
            self.event_samples.append(point)
        if self.fit is not None and self.prediction is not None:
            expected=self.fit.pixel(timestamp)
            error=float(np.linalg.norm(np.asarray(point['center'])-expected)) if expected is not None else math.inf
            event_time,pixel,_=self.prediction
            if error>max(3.,6*self.fit.rms_px):
                # Require a measured departure close to a previously stable,
                # still-causal plane intersection. No disappearance shortcut.
                stable=self.frozen
                observed_depth=float(np.mean(np.diag(self.camera.k)[:2]))*self.radius/point['radius_px']
                plane_depth=float(self.plane_point[2])
                near_plane=.75*plane_depth<=observed_depth<=1.25*plane_depth
                recent=(not self.departures or timestamp-self.departures[-1]['timestamp']<=.15)
                end_window=.45 if self.departures else .22
                if stable and near_plane and recent and event_time-.025<=timestamp<=event_time+end_window:
                    self.departures.append(dict(point,error_px=error))
                    self.result.update(status='contact_candidate',evidence=['stable_ballistic_plane_intersection','measured_trajectory_departure'])
                    refined=refine_contact(self.fit,self.event_samples,event_time,self.plane_point,self.normal,self.radius) if self.frozen and len(self.departures)>=3 else None
                    if refined is not None and self.departures[-1]['error_px']>5:
                        self.result.update(status='visual_contact',contact_confirmed=True,
                            impact=refined['pixel'],contact_timestamp=refined['timestamp'],
                            confirmation_timestamp=float(timestamp),uncertainty_px=refined['uncertainty_px'],
                            contact_fit=refined,
                            evidence=['calibrated_known_static_plane','stable_ballistic_intersection',
                                      'three_observed_departure_frames','sphere_size_consistent_with_plane','fitted_observed_postcontact_segment'],
                            assumption='stationary_level_camera; known_vertical_plane; gravity_only_precontact; known_sphere_radius')
                    return self.result
                self.departures=[]
                self.result.update(status='fit_rejected',evidence=['observation_inconsistent_with_ballistic_model'])
                return self.result
            if self.frozen:
                return self.result
        self.points.append(point)
        fitted=fit_flight(self.points,self.camera)
        if fitted is None:
            return self.result
        prediction=fitted.intersection(self.plane_point,self.normal,self.radius)
        if prediction is None:
            self.result.update(status='no_forward_plane_intersection')
            return self.result
        t,pixel,surface=prediction
        if not timestamp-.025<=t<=timestamp+1.:
            self.result.update(status='plane_crossing_unconfirmed',evidence=['forecast_is_not_contact'])
            return self.result
        self.fit=fitted;self.prediction=prediction;self.forecasts.append(prediction)
        if 0<t-timestamp<.1:
            sensitivity=forecast_spread(self.points,self.camera,self.plane_point,self.normal,self.radius,prediction)
            if sensitivity and sensitivity['leave_one_out_spread_px']<=6 and sensitivity['leave_one_out_time_spread_s']<=.035:
                self.frozen=True;self.reliability=sensitivity
                self.event_samples=[p for p in self.points if p['timestamp']>=timestamp-.1]
        p=fitted.position(timestamp);v=fitted.state[3:]+np.array((0.,9.81*(timestamp-fitted.origin_time),0.))
        self.result.update(status='flight_predicted',forecast={'timestamp':float(t),'pixel':tuple(map(float,pixel)),
            'surface_optical_m':surface.tolist(),'scope':'predicted_known_plane_intersection; not confirmed_contact'},
            state={'timestamp':float(timestamp),'position_optical_m':p.tolist(),'velocity_optical_mps':v.tolist(),
                   'reprojection_rms_px':fitted.rms_px,'points':len(self.points)},
            evidence=['past_and_current_RGB_only','gravity_constrained_multi_frame_fit'])
        self.result['forecast_sensitivity']=self.reliability
        return self.result
