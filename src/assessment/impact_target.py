"""Static base-board reference for impact replay, with explicit model support."""
from dataclasses import replace
import math,numpy as np


class BaseTargetReference:
    def __init__(self, geometry, min_frames=3, min_geometry=.85, max_gap_s=.25):
        self.aspect=geometry.lamp_spacing_m/geometry.lamp_length_m
        self.min_frames,self.min_geometry,self.max_gap_s=min_frames,min_geometry,max_gap_s
        self.previous=None;self.stamp=None;self.count=0

    def eligible(self,o):
        if not o.geometry_valid or o.corners is None or (o.geometry_score or 0)<self.min_geometry:return False
        q=np.asarray(o.corners,dtype=float)
        width=(np.linalg.norm(q[1]-q[0])+np.linalg.norm(q[2]-q[3]))/2
        height=(np.linalg.norm(q[3]-q[0])+np.linalg.norm(q[2]-q[1]))/2
        return height>=10 and .75*self.aspect<=width/height<=self.aspect/.75

    @staticmethod
    def same(a,b):
        scale=math.hypot(a.bbox[2]-a.bbox[0],a.bbox[3]-a.bbox[1])
        return a.color==b.color and np.max(np.linalg.norm(np.asarray(a.corners)-b.corners,axis=1))<=max(4.,.04*scale)

    def update(self,classical,model,timestamp):
        candidates=[o for o in classical if self.eligible(o)]
        supported=[(c,m) for c in candidates for m in model if self.eligible(m) and self.same(c,m)]
        if supported:
            identities={id(c) for c,m in supported}
            if len(identities)!=1:self.previous=None;self.count=0;return None
            chosen,match=max(supported,key=lambda pair:pair[1].score)
        elif len(candidates)==1:chosen,match=candidates[0],None
        else:self.previous=None;self.count=0;return None
        consistent=(self.previous is not None and self.stamp is not None and 0<timestamp-self.stamp<=self.max_gap_s and self.same(self.previous,chosen))
        self.count=self.count+1 if consistent else 1
        self.previous,self.stamp=chosen,float(timestamp)
        if match is None and self.count<self.min_frames:return None
        observation=match if match is not None else chosen
        evidence=dict(observation.fusion_evidence or {},base_target_policy='persistent_geometry_model_assisted_v1',model_support=match is not None,consistent_geometry_frames=self.count,reference_scope='known_static_base_front_view',score_kind='model_confidence' if match is not None else 'geometric_quality')
        return replace(observation,fusion_evidence=evidence)
