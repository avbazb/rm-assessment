"""Capture-time pixel motion, isolated from detection and identity decisions.

CV/CA interacting Kalman models use raw *associated* positions. This is an
image-motion forecast, not calibrated world motion or a ballistic solution.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class ForecastConfig:
    measurement_std_px: float = 2.5
    acceleration_std_px_s2: float = 100.
    jerk_std_px_s3: float = 250.
    mode_time_s: float = .35
    min_samples: int = 5
    min_span_s: float = .10
    max_horizon_s: float = .50
    innovation_gate: float = 16.


def dynamics(dt,mode,scale=1.,config=None):
    cfg=config or ForecastConfig()
    f=np.eye(6);f[0,2]=f[1,3]=dt
    q=np.zeros((6,6))
    if mode=="ca":
        f[0,4]=f[1,5]=dt*dt/2;f[2,4]=f[3,5]=dt
        # One random, constant jerk over this interval; std has px/s^3 units.
        g=np.array([dt**3/6,dt**2/2,dt])
        one=np.outer(g,g)*(cfg.jerk_std_px_s3*scale)**2
        for indices in ([0,2,4],[1,3,5]):q[np.ix_(indices,indices)]=one
    elif mode=="cv":
        f[4:,:]=0
        # Match the legacy filter's per-interval acceleration convention.
        g=np.array([dt*dt/2,dt])
        one=np.outer(g,g)*(cfg.acceleration_std_px_s2*scale)**2
        for indices in ([0,2],[1,3]):q[np.ix_(indices,indices)]=one
        q[4,4]=q[5,5]=(cfg.acceleration_std_px_s2*scale)**2
    else:raise ValueError("invalid forecast motion model")
    return f,q


def _mixture(xs,ps,weights):
    mean=sum(w*x for x,w in zip(xs,weights))
    covariance=sum(w*(p+np.outer(x-mean,x-mean)) for x,p,w in zip(xs,ps,weights))
    return mean,(covariance+covariance.T)/2


class _Motion:
    def __init__(self,center,timestamp,scale,noise):
        self.x=[np.r_[center,0.,0.,0.,0.] for _ in range(2)]
        self.p=[np.diag([noise**2]*2+[(350*scale)**2]*2+[(500*scale)**2]*2) for _ in range(2)]
        self.weights=np.array([.7,.3]);self.timestamp=self.created=timestamp;self.samples=1

    def update(self,center,timestamp,scale,noise,cfg):
        dt=timestamp-self.timestamp
        if dt<=0:raise ValueError("forecast measurements must increase in time")
        stay=.5+.5*math.exp(-dt/cfg.mode_time_s)
        transition=np.array([[stay,1-stay],[1-stay,stay]])
        priors=self.weights@transition
        mixed=[_mixture(self.x,self.p,self.weights*transition[:,j]/priors[j]) for j in range(2)]
        xs=[];ps=[];log_likelihood=[];innovations=[];r=np.eye(2)*noise**2
        for j,mode in enumerate(("cv","ca")):
            f,q=dynamics(dt,mode,scale,cfg);x,p=mixed[j]
            x=f@x;p=f@p@f.T+q
            residual=np.asarray(center,float)-x[:2];s=p[:2,:2]+r
            innovations.append(float(residual@np.linalg.solve(s,residual)))
            k=np.linalg.solve(s,p[:2,:]).T
            x=x+k@residual;ikh=np.eye(6);ikh[:,:2]-=k
            p=ikh@p@ikh.T+k@r@k.T;p=(p+p.T)/2
            log_likelihood.append(-.5*(float(residual@np.linalg.solve(s,residual))+np.linalg.slogdet(s)[1]))
            xs.append(x);ps.append(p)
        logw=np.log(priors)+log_likelihood;logw-=max(logw)
        weights=np.exp(logw);weights/=weights.sum()
        self.x=xs;self.p=ps;self.weights=weights;self.timestamp=timestamp;self.samples+=1
        return min(innovations)


class MotionForecaster:
    """Session-local estimates; coasting never adds a measurement or a hit."""
    def __init__(self,config=None):
        self.config=config or ForecastConfig();self.states={};self.segment=None

    def update(self,tracks,observations,timestamp,image_size):
        cfg=self.config;scale=max(.5,image_size[0]/640)
        segments={t["segment"] for t in tracks}
        if len(segments)>1:raise ValueError("mixed forecast segments")
        if segments and self.segment!=next(iter(segments)):
            self.states={};self.segment=next(iter(segments))
        result=[];active=set()
        for t in tracks:
            key=t["track_id"];active.add(key);motion=self.states.get(key)
            index=t.get("observation_index")
            if t["observed"] and index is not None:
                o=observations[index]
                noise=cfg.measurement_std_px*scale*(1 if o.get("geometry_valid") else 2.5)
                if motion is None or timestamp<motion.timestamp or timestamp-motion.timestamp>cfg.max_horizon_s:
                    motion=_Motion(o["center"],timestamp,scale,noise);self.states[key]=motion
                elif timestamp>motion.timestamp:
                    innovation=motion.update(o["center"],timestamp,scale,noise,cfg)
                    if innovation>cfg.innovation_gate:
                        # Both motion models failed their next-observation check.
                        # Restart motion evidence, without changing the visual ID.
                        motion=_Motion(o["center"],timestamp,scale,noise);self.states[key]=motion
                motion.last_center=list(o["center"])
            if motion is None:continue
            x,p=_mixture(motion.x,motion.p,motion.weights)
            result.append(dict(schema="astra.pixel-motion.v1",track_id=key,segment=t["segment"],
                state_timestamp=motion.timestamp,last_observation_timestamp=motion.timestamp,
                center=x[:2].tolist(),velocity=x[2:4].tolist(),acceleration=x[4:].tolist(),
                state_covariance=p.tolist(),models=[dict(kind=kind,weight=float(weight),mean=a.tolist(),covariance=b.tolist())
                    for kind,weight,a,b in zip(("cv","ca"),motion.weights,motion.x,motion.p)],
                samples=motion.samples,measurement_span_s=motion.timestamp-motion.created,
                confirmed=t["confirmed"],size=[t["bbox"][2]-t["bbox"][0],t["bbox"][3]-t["bbox"][1]],
                last_observed_center=motion.last_center,
                max_extrapolation_displacement_px=2*t.get("max_coasting_displacement_px",
                    max(12*scale,.5*float(np.linalg.norm(np.asarray(t["bbox"])[2:]-np.asarray(t["bbox"])[:2])))),
                scale=scale,coordinate_frame="image_pixels",velocity_unit="px/s",acceleration_unit="px/s2",
                covariance_status="model_based_uncalibrated",max_horizon_s=cfg.max_horizon_s,
                min_samples=cfg.min_samples,min_span_s=cfg.min_span_s,
                process_noise_kind="per_interval_constant_acceleration_or_jerk",
                acceleration_std_px_s2=cfg.acceleration_std_px_s2,jerk_std_px_s3=cfg.jerk_std_px_s3))
        self.states={k:s for k,s in self.states.items() if k in active and timestamp-s.timestamp<=cfg.max_horizon_s}
        return result


def forecast(estimates,target_timestamp,image_size=None):
    if not math.isfinite(target_timestamp):raise ValueError("invalid forecast target time")
    rows=[]
    for e in estimates:
        dt=target_timestamp-e["state_timestamp"]
        reason=None
        if dt< -1e-6:reason="clock_mismatch"
        elif dt>e["max_horizon_s"]:reason="prediction_expired"
        elif not e["confirmed"]:reason="tentative"
        elif e["samples"]<e["min_samples"] or e["measurement_span_s"]+1e-9<e["min_span_s"]:reason="insufficient_measurements"
        base=dict(track_id=e["track_id"],segment=e["segment"],target_timestamp=target_timestamp,
                  observation_timestamp=e["last_observation_timestamp"],horizon_s=dt,
                  coordinate_frame="image_pixels",covariance_status=e["covariance_status"])
        if reason in ("clock_mismatch","prediction_expired"):
            rows.append(dict(**base,status="waiting",reason=reason,center=None,position_covariance=None));continue
        cfg=ForecastConfig(acceleration_std_px_s2=e["acceleration_std_px_s2"],jerk_std_px_s3=e["jerk_std_px_s3"])
        xs=[];ps=[];weights=[]
        for m in e["models"]:
            f,q=dynamics(max(0,dt),m["kind"],e["scale"],cfg)
            xs.append(f@np.asarray(m["mean"]));ps.append(f@np.asarray(m["covariance"])@f.T+q);weights.append(m["weight"])
        x,p=_mixture(xs,ps,weights)
        radius=math.sqrt(max(0,float(np.linalg.eigvalsh(p[:2,:2])[-1])))*2.44774683068
        anchor=e.get("last_observed_center",e["center"])
        displacement=float(np.linalg.norm(x[:2]-anchor))
        # Explicit conservative publication policy for a local pixel-motion
        # model, not a bound on physical error or proof of armor identity.
        limit=e.get("max_extrapolation_displacement_px",max(24*e["scale"],float(np.linalg.norm(e["size"]))))
        if reason is None and displacement>limit:reason="unsupported_displacement"
        if reason is None and radius>max(4.,min(e["size"])/2):reason="uncertain"
        if reason is None and image_size and not (0<=x[0]<image_size[0] and 0<=x[1]<image_size[1]):reason="outside_image"
        rows.append(dict(**base,status="waiting" if reason else "ready",reason=reason,
            center=None if reason else x[:2].tolist(),candidate_center=x[:2].tolist(),
            displacement_px=displacement,max_extrapolation_displacement_px=limit,
            velocity=x[2:4].tolist(),acceleration=x[4:].tolist(),position_covariance=p[:2,:2].tolist(),
            uncertainty_radius_px=radius,model_weights=[m["weight"] for m in e["models"]]))
    return dict(schema="astra.pixel-forecast.v1",target_timestamp=target_timestamp,tracks=rows,
                scope="future image position; association errors and unobserved maneuvers remain possible")
