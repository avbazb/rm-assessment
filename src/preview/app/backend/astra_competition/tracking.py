"""Causal image-space tracks. Raw observations and physical lamp corners stay intact.

Measured lamp pairs can start tracks. Small model boxes with bilateral color
support require longer confirmation. Unsupported model boxes only continue
confirmed tracks briefly. IDs are local to a
source session; a number printed on armor is weak evidence, not its identity.
"""
from collections import deque
from dataclasses import dataclass
import copy
import math
from numbers import Integral, Real

import cv2
import numpy as np

from astra_vision.tracking import hungarian_assignment


@dataclass(frozen=True)
class TrackingConfig:
    min_hits: int = 3
    min_confirmation_s: float = .08
    color_supported_hits: int = 3
    color_supported_confirmation_s: float = .10
    tentative_gap_s: float = .22
    tentative_lifetime_s: float = .8
    max_lost_s: float = .65
    max_prediction_s: float = .30
    max_evidence_age_s: float = .60
    reset_gap_s: float = 1.0
    measurement_std_px: float = 2.5
    acceleration_std_px_s2: float = 100.
    max_tracks: int = 64


def _validate_inputs(observations,image,image_size):
    """Validate the observation/image contract before any tracker mutation.

    Pipeline images are nonempty uint8 BGR. Coordinates can extend outside the
    image, but must fit signed 32-bit pixel indices used by OpenCV/ROI extraction.
    Unknown string sources/colors remain admissible to the existing reject/gates.
    """
    def number(value):
        try:
            return (isinstance(value,Real) and not isinstance(value,(bool,np.bool_)) and
                    not (isinstance(value,np.floating) and value.dtype.itemsize>8) and math.isfinite(value))
        except (TypeError,ValueError,OverflowError):
            return False

    def coordinates(value,shape):
        try:
            array=np.asarray(value)
            return (array.shape==shape and array.dtype.kind in "iuf" and array.dtype.itemsize<=8 and
                    np.isfinite(array).all() and np.all(np.abs(array.astype(float))<=2147483647))
        except (TypeError,ValueError,OverflowError):
            return False

    if not isinstance(observations,(list,tuple)):
        raise ValueError("invalid tracking observations")
    if image is not None and (not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or
            image.ndim!=3 or image.shape[2]!=3 or min(image.shape[:2])<=0):
        raise ValueError("invalid tracking image; expected nonempty uint8 BGR")
    explicit_size=None
    if image_size is not None:
        if (not isinstance(image_size,(list,tuple,np.ndarray)) or np.shape(image_size)!=(2,) or
                any(not isinstance(v,Integral) or isinstance(v,(bool,np.bool_)) or
                    not 0<v<=2147483647 for v in image_size)):
            raise ValueError("invalid tracking image_size")
        explicit_size=tuple(image_size)
    size=tuple(image.shape[1::-1]) if image is not None else explicit_size or (640,480)
    if image is not None and explicit_size is not None and explicit_size!=size:
        raise ValueError("tracking image_size does not match image")
    for o in observations:
        if not isinstance(o,dict):
            raise ValueError("invalid tracking observation")
        if (not coordinates(o.get("bbox"),(4,)) or not coordinates(o.get("center"),(2,)) or
                not number(o.get("score")) or not isinstance(o.get("color"),str) or
                not isinstance(o.get("source"),str)):
            raise ValueError("invalid tracking observation")
        box=np.asarray(o["bbox"],float)
        if np.any(box[2:]<=box[:2]):
            raise ValueError("invalid tracking observation bbox")
        if o.get("geometry_score") is not None and not number(o["geometry_score"]):
            raise ValueError("invalid tracking geometry_score")
        if o.get("geometry_valid") is not None and not isinstance(o["geometry_valid"],(bool,np.bool_)):
            raise ValueError("invalid tracking geometry_valid")
        if o.get("corners") is not None and not coordinates(o["corners"],(4,2)):
            raise ValueError("invalid tracking corners")
        fusion=o.get("fusion_evidence")
        if fusion is not None and (not isinstance(fusion,dict) or
                (fusion.get("status") is not None and not isinstance(fusion["status"],str))):
            raise ValueError("invalid tracking fusion_evidence")
        digit=o.get("digit")
        if digit is not None:
            if not isinstance(digit,dict):
                raise ValueError("invalid tracking digit")
            if digit.get("supported") is not None and not isinstance(digit["supported"],(bool,np.bool_)):
                raise ValueError("invalid tracking digit supported")
            label=digit.get("label")
            if label is not None and not isinstance(label,str) and not number(label):
                raise ValueError("invalid tracking digit label")
            if digit.get("score") is not None and not number(digit["score"]):
                raise ValueError("invalid tracking digit score")
    return size


def admission(observation,image=None):
    """Strong geometry or weaker bilateral color, not a calibrated probability."""
    source=observation.get("source")
    score=float(observation.get("score",0))
    geometry=float(observation.get("geometry_score") or 0)
    evidence=observation.get('fusion_evidence') or {}
    if (source=='classical' and evidence.get('status')=='rotation_roi_verified' and
            evidence.get('current_lamp_measurement') is True and geometry>=.52 and
            0<=float(evidence.get('model_age_s',math.inf))<=.30):
        return 'strong'
    if (source=='classical' and evidence.get('status')=='traditional_reference_verified' and
            evidence.get('model_assistance') is False and score>=.45 and
            float(evidence.get('joint_margin') or 0)>=-.03):
        # This admits an image-plane observation only; it supplies no physical
        # corners and does not change PnP admission.
        return 'supported'
    if observation.get("geometry_valid") and observation.get("corners") is not None and len(observation["corners"]):
        if source=="fused" and (observation.get("fusion_evidence") or {}).get("status")=="joint_confirmed" and geometry>=.55:
            return "strong"
        if source=="classical" and geometry>=.60:
            return "strong"
    # At a few pixels per lamp, contour endpoints are unreliable. Verify actual
    # same-color pixels on BOTH sides of the model ROI, with an aspect gate that
    # rejects narrow field trim. This supplies no corner or pose measurement.
    if (source in ("neural","fused") and score>=.65 and image is not None and
            observation.get("color") in ("red","blue")):
        x1,y1,x2,y2=np.rint(observation["bbox"]).astype(int)
        h,w=image.shape[:2]
        roi=image[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]
        rh,rw=roi.shape[:2]
        if rh>=4 and rw>=6 and .65<=rw/rh<=4.0:
            b,g,r=cv2.split(roi.astype(np.int16))
            mask=((r-b>25)&(r-g>15)&(r>70)) if observation["color"]=="red" else ((b-r>25)&(b-g>10)&(b>70))
            side=max(1,int(rw*.4))
            if min(float(mask[:,:side].mean()),float(mask[:,-side:].mean()))>=.06:
                return "supported"
    return "weak" if source in ("neural","fused") and score>=.40 else "reject"


def _feature(image,observation):
    if image is None:
        return None
    x1,y1,x2,y2=np.rint(observation["bbox"]).astype(int)
    h,w=image.shape[:2]
    roi=image[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]
    if min(roi.shape[:2],default=0)<4:
        return None
    # A coarse grayscale appearance descriptor is auxiliary to motion and color.
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    feature=cv2.resize(gray,(8,8)).astype(float).ravel()
    feature-=feature.mean()
    norm=np.linalg.norm(feature)
    return feature/norm if norm>30 else None


class _State:
    def __init__(self,identifier,observation,timestamp,feature,scale,quality):
        self.identifier=identifier
        self.x=np.array([*observation["center"],0.,0.])
        # A newly visible rotating plate may already move rapidly. A zero mean
        # velocity must not be confused with a known zero velocity.
        self.p=np.diag([6.25*scale**2]*2+[350**2*scale**2]*2)
        box=np.asarray(observation["bbox"],float)
        self.size=box[2:]-box[:2]
        self.offset=(box[2:]+box[:2])/2-self.x[:2]
        self.color=observation["color"]
        self.observation=copy.deepcopy(observation)
        self.feature=feature
        self.created=self.seen=self.strong_seen=timestamp
        self.hits=self.streak=1
        self.geometry_streak=1 if quality=="strong" else 0
        self.streak_started=timestamp
        self.geometry_streak_started=timestamp if quality=="strong" else None
        self.evidence_kind=quality
        self.confirmation_kind=None
        self.confirmed=False
        self.observed=True
        self.observation_index=None
        self.history=deque(maxlen=30)

    def predict(self,dt,config,scale):
        f=np.eye(4);f[0,2]=f[1,3]=dt
        g=np.array([[dt*dt/2,0],[0,dt*dt/2],[dt,0],[0,dt]])
        self.x=f@self.x
        self.p=f@self.p@f.T+(config.acceleration_std_px_s2*scale)**2*(g@g.T)
        self.observed=False;self.observation_index=None

    def correct(self,observation,timestamp,feature,quality,config,scale,index):
        noise=(config.measurement_std_px*scale*(1 if quality=="strong" else 2.5))**2
        s=self.p[:2,:2]+np.eye(2)*noise
        residual=np.asarray(observation["center"])-self.x[:2]
        innovation=float(residual@np.linalg.solve(s,residual))
        if innovation>16 and quality in ("strong","supported"):
            # Only associations independently admitted by the appearance and
            # displacement gates reach this maneuver update.
            self.p*=min(4.,innovation/12)
            s=self.p[:2,:2]+np.eye(2)*noise
        k=np.linalg.solve(s,self.p[:2,:]).T
        self.x+=k@(np.asarray(observation["center"])-self.x[:2])
        ikh=np.eye(4)-k@np.eye(2,4)
        self.p=ikh@self.p@ikh.T+k@k.T*noise
        self.p=(self.p+self.p.T)/2
        # Time-based smoothing keeps box size/offset stable at changing input FPS.
        alpha=1-math.exp(-(timestamp-self.seen)/.16)
        box=np.asarray(observation["bbox"],float)
        self.size+=alpha*((box[2:]-box[:2])-self.size)
        self.offset+=alpha*((box[:2]+box[2:])/2-observation["center"]-self.offset)
        self.seen=timestamp;self.hits+=1;self.observed=True;self.observation_index=index
        if quality in ("strong","supported"):
            self.strong_seen=timestamp
            if self.streak==0:
                self.streak_started=timestamp
            self.streak+=1
            if quality=="strong":
                if self.geometry_streak==0:
                    self.geometry_streak_started=timestamp
                self.geometry_streak+=1
            else:
                self.geometry_streak=0;self.geometry_streak_started=None
            self.evidence_kind=quality
            # Confirmation duration belongs to the current evidence run, not
            # track birth. A missing frame keeps its ID but restarts both runs.
            # Weak matches intentionally bridge without adding evidence hits.
            # The nanosecond tolerance only absorbs floating-point boundaries.
            geometry_ready=(self.geometry_streak>=config.min_hits and
                self.geometry_streak_started is not None and
                timestamp-self.geometry_streak_started+1e-9>=config.min_confirmation_s)
            admitted_ready=(self.streak>=config.color_supported_hits and
                timestamp-self.streak_started+1e-9>=config.color_supported_confirmation_s)
            if not self.confirmed and (geometry_ready or admitted_ready):
                self.confirmed=True
                self.confirmation_kind="strong" if geometry_ready else "supported"
            if feature is not None:
                self.feature=feature if self.feature is None else .85*self.feature+.15*feature
                self.feature/=max(np.linalg.norm(self.feature),1e-8)
        if observation["color"]!="unknown":
            self.color=observation["color"]
        self.observation=copy.deepcopy(observation)


class ArmorTracker:
    def __init__(self,config=None):
        self.config=config or TrackingConfig()
        self.states=[];self.timestamp=None;self.next_id=1;self.segment=0
        self.shape=None;self.thumbnail=None;self.scene_histogram=None;self.last_result=None

    def reset(self,reason="explicit"):
        self.states=[];self.timestamp=None;self.thumbnail=None;self.scene_histogram=None;self.last_result=None
        self.segment+=1
        # Do not reuse an ID after a rewind, scene cut or long loss in one session.
        return reason

    def _cost(self,track,observation,feature,scale,quality):
        if track.color!="unknown" and observation["color"]!="unknown" and track.color!=observation["color"]:
            return None
        box=np.asarray(observation["bbox"],float);size=box[2:]-box[:2]
        ratio=np.maximum(size/track.size,track.size/size)
        if max(ratio)>2.2:
            return None
        residual=np.asarray(observation["center"])-track.x[:2]
        gate=max(35*scale,.8*np.linalg.norm(track.size))
        if not track.confirmed:
            gate=max(gate,min(120*scale,3*math.sqrt(max(np.linalg.eigvalsh(track.p[:2,:2])))))
        if np.linalg.norm(residual)>gate:
            return None
        noise=(self.config.measurement_std_px*scale*(1 if quality=="strong" else 2.5))**2
        distance=float(residual@np.linalg.solve(track.p[:2,:2]+np.eye(2)*noise,residual))
        similarity=None if feature is None or track.feature is None else float(np.clip(feature@track.feature,-1,1))
        if not track.confirmed and quality=="weak" and (track.streak<2 or observation["score"]<.50 or similarity is None or similarity<.75):
            return None
        maneuver=(quality in ("strong","supported") and similarity is not None and similarity>=.70 and
                  max(ratio)<1.6 and distance<=64)
        if distance>16 and not maneuver:
            return None
        appearance=0 if similarity is None else 3*(1-similarity)
        a,b=track.observation.get("digit") or {},observation.get("digit") or {}
        classical_pair=track.observation.get("source")==observation.get("source")=="classical"
        digit_penalty=2 if classical_pair and a.get("supported") and b.get("supported") and a.get("label")!=b.get("label") else 0
        return distance+appearance+digit_penalty+2*float(np.log(ratio).sum())

    def update(self,observations,timestamp,image=None,image_size=None):
        try:
            timestamp=float(timestamp)
        except (TypeError,ValueError,OverflowError) as exc:
            raise ValueError("invalid tracking timestamp") from exc
        if not math.isfinite(timestamp) or timestamp<0:
            raise ValueError("invalid tracking timestamp")
        size=_validate_inputs(observations,image,image_size)
        # Admission/appearance extraction can reject input too; complete these
        # pure computations before reset, prediction, history or ID allocation.
        qualities=[admission(o,image) for o in observations]
        ranked=sorted(range(len(observations)),key=lambda i:({"strong":3,"supported":2,"weak":1,"reject":0}[qualities[i]],observations[i]["score"]),reverse=True)[:128]
        features={i:_feature(image,observations[i]) for i in ranked}
        reason=None
        if self.shape is not None and size!=self.shape:
            reason=self.reset("resolution_changed")
        if self.timestamp is not None:
            if timestamp<self.timestamp:
                reason=self.reset("timestamp_rewind")
            elif timestamp==self.timestamp:
                result=copy.deepcopy(self.last_result);result["tracking"]["duplicate_timestamp"]=True
                return result
            elif timestamp-self.timestamp>self.config.reset_gap_s:
                reason=self.reset("timestamp_gap")
        thumbnail=None
        histogram=None
        if image is not None:
            thumbnail=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(48,27)).astype(float)
            hsv=cv2.cvtColor(cv2.resize(image,(96,54)),cv2.COLOR_BGR2HSV)
            histogram=cv2.calcHist([hsv],[0,1],None,[12,8],[0,180,0,256])
            cv2.normalize(histogram,histogram,alpha=1,norm_type=cv2.NORM_L1)
            if self.thumbnail is not None:
                difference=np.abs(thumbnail-self.thumbnail)
                color_cut=self.scene_histogram is not None and cv2.compareHist(histogram,self.scene_histogram,cv2.HISTCMP_BHATTACHARYYA)>.35
                if ((difference.mean()>55 and (difference>48).mean()>.60) or
                        (color_cut and difference.mean()>40 and (difference>32).mean()>.45)):
                    reason=self.reset("scene_cut")
        self.thumbnail=thumbnail;self.scene_histogram=histogram;self.shape=size
        scale=max(.5,size[0]/640)
        dt=0 if self.timestamp is None else timestamp-self.timestamp
        cfg=self.config
        self.states=[s for s in self.states if timestamp-s.seen<=cfg.max_lost_s and timestamp-s.strong_seen<=cfg.max_evidence_age_s
                     and (s.confirmed or (timestamp-s.seen<=cfg.tentative_gap_s and timestamp-s.created<=cfg.tentative_lifetime_s))]
        for s in self.states:
            s.predict(dt,cfg,scale)
        matched=set();used=set()
        # Confirmed tracks get first choice, then tentative tracks. Weak boxes can
        # continue confirmed identities only, and cannot refresh lamp evidence age.
        for confirmed,quality in ((True,"strong"),(True,"supported"),(False,"strong"),(False,"supported"),(True,"weak"),(False,"weak")):
            track_indices=[i for i,s in enumerate(self.states) if s.confirmed==confirmed and i not in matched]
            detection_indices=[j for j in ranked if j not in used and qualities[j]==quality]
            n,m=len(track_indices),len(detection_indices)
            costs=np.full((n,m+n),100.)
            costs[:,:m]=10000.
            for ri,i in enumerate(track_indices):
                for cj,j in enumerate(detection_indices):
                    cost=self._cost(self.states[i],observations[j],features[j],scale,quality)
                    if cost is not None:
                        costs[ri,cj]=cost
            for ri,cj in hungarian_assignment(costs):
                if cj<m and costs[ri,cj]<100:
                    i,j=track_indices[ri],detection_indices[cj]
                    self.states[i].correct(observations[j],timestamp,features[j],quality,cfg,scale,j)
                    matched.add(i);used.add(j)
        # One missing observation does not destroy the tentative ID. It still
        # breaks confirmation: repeated flashes cannot accumulate confirmation.
        for i,s in enumerate(self.states):
            if not s.confirmed and i not in matched:
                s.streak=0;s.geometry_streak=0
                s.streak_started=None;s.geometry_streak_started=None
        for j in ranked:
            if qualities[j] in ("strong","supported") and j not in used and len(self.states)<cfg.max_tracks:
                state=_State(self.next_id,observations[j],timestamp,features[j],scale,qualities[j])
                state.observation_index=j
                self.states.append(state);self.next_id+=1
        tracks=[];tentative=[];hidden_coasting=0
        for s in self.states:
            s.history.append(dict(timestamp=timestamp,center=s.x[:2].tolist(),observed=s.observed))
            age=timestamp-s.seen
            if age>cfg.max_prediction_s or (not s.confirmed and not s.observed):
                continue
            if not s.observed and np.linalg.norm(s.x[:2]-s.observation["center"])>max(12*scale,.5*np.linalg.norm(s.size)):
                # Keep the ID for reacquisition, but do not draw an unsupported
                # fast linear extrapolation far from the last measured plate.
                hidden_coasting+=1;continue
            center=s.x[:2];box_center=center+s.offset
            target=tracks if s.confirmed else tentative
            confirmation_kind=s.confirmation_kind if s.confirmed else s.evidence_kind
            target.append(dict(track_id=s.identifier,segment=self.segment,timestamp=timestamp,
                state_timestamp=timestamp,state_covariance=s.p.tolist(),
                acceleration_noise_std_px_s2=cfg.acceleration_std_px_s2*scale,
                max_prediction_s=cfg.max_prediction_s,
                last_observed_center=list(s.observation["center"]),
                max_coasting_displacement_px=max(12*scale,.5*float(np.linalg.norm(s.size))),
                center=center.tolist(),velocity=s.x[2:].tolist(),bbox=np.r_[box_center-s.size/2,box_center+s.size/2].tolist(),
                color=s.color,status=("tracked" if s.observed else "coasting") if s.confirmed else "tentative",confirmed=s.confirmed,
                confirmation_hits=s.streak,confirmation_required=cfg.min_hits if confirmation_kind=="strong" else cfg.color_supported_hits,
                observed=s.observed,hits=s.hits,
                confirmation_evidence="lamp_geometry" if confirmation_kind=="strong" else "model_and_color",
                seconds_since_observation=age,evidence_age_s=timestamp-s.strong_seen,
                position_covariance=s.p[:2,:2].tolist(),velocity_covariance=s.p[2:,2:].tolist(),
                history=list(s.history),observation_index=s.observation_index,
                score=s.observation["score"],source=s.observation["source"],
                digit=s.observation.get("digit") if s.observation["source"]=="classical" else None,
                score_kind=s.observation.get("score_kind"),score_timestamp=s.seen,
                coordinate_frame="image_pixels",position_unit="px",velocity_unit="px/s"))
        self.timestamp=timestamp
        result=dict(tracks=tracks,tentative_tracks=tentative,tracking=dict(schema="astra.armor-tracks.v2",coordinate_frame="image_pixels",
            position_unit="px",velocity_unit="px/s",segment=self.segment,reset_reason=reason,duplicate_timestamp=False,
            confirmed=len(tracks),tentative=sum(not s.confirmed for s in self.states),
            raw_candidates=len(observations),strong_candidates=qualities.count("strong"),
            color_supported_candidates=qualities.count("supported"),
            hidden_coasting=hidden_coasting,
            weak_candidates=qualities.count("weak"),rejected_candidates=qualities.count("reject"),
            policy="lamp_geometry_or_model_color_birth; classical_only_digits; three_frame_confirmation; tentative_preview; appearance_gated_maneuver; bounded_weak_continuation",
            max_prediction_s=cfg.max_prediction_s,max_lost_s=cfg.max_lost_s))
        self.last_result=copy.deepcopy(result)
        return result
