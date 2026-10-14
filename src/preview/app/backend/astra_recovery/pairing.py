"""v3 recovery math with per-call extraction dependency, no module mutation."""
import math
from dataclasses import replace
from types import FunctionType
import numpy as np
import astra_competition.guided_fusion as gf
from astra_competition.perception import ArmorObservation
from .core import find_extra_bars


def extractor(confidence,base_extract=gf.extract_roi):
    extras=None
    def extract(image,proposal,config,*,color_first=False):
        nonlocal extras
        bars,meta,roi=base_extract(image,proposal,config,color_first=color_first)
        if not color_first:return bars,meta,roi
        for i,j,pair,_ in gf.lamp_pair_candidates(image,bars,config):
            spatial=gf.association(proposal,pair)
            if spatial and pair.score>=.45 and (proposal.score>=confidence or (pair.score>=.55 and spatial['quality']>=.65)):
                return bars,meta,roi
        if extras is None:extras=find_extra_bars(image,config)
        xa,ya,xb,yb=roi;local=[]
        for bar in extras:
            if not (np.all(bar.corners.min(axis=0)>(xa,ya)) and np.all(bar.corners.max(axis=0)<(xb-1,yb-1))):continue
            axis=bar.axis_endpoints[1]-bar.axis_endpoints[0]
            moved=replace(bar,center=tuple(bar.axis_endpoints.mean(axis=0)),length=float(np.linalg.norm(axis)),tilt_degrees=math.degrees(math.atan2(axis[0],axis[1])))
            if any(gf.same_lamp(moved,old) for old in bars):continue
            local.append(moved)
        choices=[];combined=sorted(bars+local,key=lambda b:(b.center[0],b.center[1],b.color));new_ids={id(b) for b in local}
        for i,j,pair,_ in gf.lamp_pair_candidates(image,combined,config):
            a,b=combined[i],combined[j]
            if not ({id(a),id(b)}&new_ids) or not owned_pair(proposal,a,b):continue
            spatial=gf.association(proposal,pair)
            if spatial and pair.score>=.55 and spatial['quality']>=.65:
                choices.append((.4*spatial['quality']+.35*pair.score,a,b))
        if not choices:return bars,meta,roi
        _,a,b=max(choices,key=lambda v:v[0]);selected=[a,b]
        newmeta={**meta,'methods':{id(bar):('bilateral_neutral_core_v3' if id(bar) in new_ids else meta['methods'][id(bar)]) for bar in selected}}
        return selected,newmeta,roi
    return extract

def paired_observations(detector,frame,timestamp,frame_id):
    proposals=detector.neural.detect(frame)
    provenance={}
    regional,regional_diagnostics=gf.verify_classical_regions(frame,proposals,detector.classical,detector.neural,detector.digits,provenance=provenance)
    if detector.regional_policy!='off':
        regional,policy_diagnostics=gf.filter_cross_global_regions(proposals,regional,provenance,detector.confidence)
        regional_diagnostics=dict(regional_diagnostics,**policy_diagnostics)
    global_count=len(proposals)
    proposals=[*proposals,*regional];roi_cache={}
    def cached_extract(image,proposal,config,*,color_first=False):
        key=(id(proposal),color_first)
        if key not in roi_cache:roi_cache[key]=gf.extract_roi(image,proposal,config,color_first=color_first)
        return roi_cache[key]
    def run(extract):
        # Reuse exact pairing with a private globals dictionary, never mutate modules.
        original=gf.guided_fuse
        private=FunctionType(original.__code__,dict(original.__globals__,extract_roi=extract),
            original.__name__,original.__defaults__,original.__closure__)
        private.__kwdefaults__=dict(original.__kwdefaults__ or {})
        rows,diagnostics=private(frame,proposals,detector.classical.config,
            detector.digits,detector.confidence,color_first=detector.lamp_method=='color')
        for row in rows:row['evidence']['model_scope']='cv_proposed_region' if row['detection'].bbox in {p.bbox for p in regional} else 'full_frame'
        diagnostics.update(global_model_proposals=global_count,regional_verification=regional_diagnostics,
            lamp_method='model_guided_color_roi' if detector.lamp_method=='color' else 'model_guided_roi',
            digit_method=detector.digits.method if detector.digits else 'off')
        observations=[ArmorObservation(float(timestamp),frame_id,row['detection'].bbox,row['detection'].center,
            row['detection'].color,row['detection'].score,row['detection'].source,row['corners'],row['corners'] is not None,
            'model_guided_lamp_reference' if row['corners'] else 'neural_box_only',row['geometry_score'],
            digit=row['digit'],fusion_evidence=row['evidence']) for row in rows]
        return observations,diagnostics
    baseline,diagnostics=run(cached_extract)
    if all(o.geometry_valid for o in baseline):return baseline,baseline,diagnostics,diagnostics
    recovered,recovery_diagnostics=run(extractor(detector.confidence,cached_extract))
    return baseline,recovered,diagnostics,recovery_diagnostics

def owned_pair(proposal,a,b):
 x1,y1,x2,y2=proposal.bbox
 if x2<=x1:return False
 p=sorted((bar.center[0]-x1)/(x2-x1) for bar in (a,b))
 return 0<=p[0]<=.35 and .65<=p[1]<=1 and all(y1<=bar.center[1]<=y2 for bar in (a,b))
