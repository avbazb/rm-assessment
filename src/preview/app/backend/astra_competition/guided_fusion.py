"""Model proposals guide lamp extraction; measured lamp pairs verify proposals.

This module never unions independent full-frame detector results. Model confidence,
geometry and association quality keep separate meanings and units.
"""
from dataclasses import replace
import math

import cv2
import numpy as np

from astra_vision.detector import ArmorDetector
from astra_vision.fusion import box_iou
from astra_vision.types import Detection
from .classical import BrightCoreDetector,ColorCoreDetector,lamp_pair_candidates,lamp_quad,lamp_span_ratio
from .color_lamps import ColorFirstDetector
from .geometry import polygon_array,polygon_center

PROPOSAL_CONFIDENCE=.10
REGIONAL_CONFIRMATION_CONFIDENCE=.50
REGIONAL_POLICY_VERSION="cross_global_membership_v1"
DEFAULT_REGIONAL_POLICY=REGIONAL_POLICY_VERSION
POLICY="bidirectional_roi_fusion; model_guided_lamps; cv_regions_verified_by_model; no_unverified_cv_outputs"


def same_lamp(a,b):
    return (a.color==b.color and math.dist(a.center,b.center)<max(2.,.22*min(a.length,b.length))
            and max(a.length,b.length)/min(a.length,b.length)<1.6)


def extract_roi(frame,proposal,config,*,color_first=False):
    height,width=frame.shape[:2]
    x1,y1,x2,y2=proposal.bbox
    margin_x=max(4.,(x2-x1)*.15);margin_y=max(4.,(y2-y1)*.15)
    xa,ya=max(0,int(x1-margin_x)),max(0,int(y1-margin_y))
    xb,yb=min(width,int(math.ceil(x2+margin_x))),min(height,int(math.ceil(y2+margin_y)))
    if xb<=xa or yb<=ya:return [],{},(xa,ya,xb,yb)
    crop=frame[ya:yb,xa:xb]
    # Lower luminance gates only inside a learned armor proposal, retaining
    # measured red/blue support, elongation and same-color pairing checks.
    local_config=replace(config,min_contour_area=min(config.min_contour_area,2.),
                         min_bar_length=min(config.min_bar_length,4.))
    value=np.max(crop,axis=2)
    adaptive=int(np.clip(np.percentile(value,94)*.85,70,170))
    # With a model box available, incomplete cores may remain model-only. Pure
    # CV keeps its original shape-fill threshold because it has no such fallback.
    methods=[("bright",BrightCoreDetector(local_config,180,min_core_fill=.60)),
             ("adaptive",BrightCoreDetector(local_config,adaptive,min_core_fill=.60)),
             ("color",ColorCoreDetector(local_config,min_core_fill=.60))]
    if color_first:
        methods=[("adaptive",ColorFirstDetector(local_config,adaptive,min_core_fill=.60,model_guided=True))]
    bars=[];origins={}
    for name,detector in methods:
        for bar in detector.find_light_bars(crop):
            # A contour cut by the ROI boundary cannot give complete endpoints.
            if (np.min(bar.corners[:,0])<=0 or np.min(bar.corners[:,1])<=0
                    or np.max(bar.corners[:,0])>=crop.shape[1]-1 or np.max(bar.corners[:,1])>=crop.shape[0]-1):
                continue
            shift=np.array((xa,ya),np.float32)
            kwargs=dict(center=(bar.center[0]+xa,bar.center[1]+ya),corners=bar.corners+shift)
            if hasattr(bar,"axis_endpoints"):
                kwargs["axis_endpoints"]=bar.axis_endpoints+shift
                axis=bar.axis_endpoints[1]-bar.axis_endpoints[0]
                kwargs.update(center=tuple(bar.axis_endpoints.mean(axis=0)+shift),
                    length=float(np.linalg.norm(axis)),tilt_degrees=math.degrees(math.atan2(axis[0],axis[1])))
            shifted=replace(bar,**kwargs)
            if any(same_lamp(shifted,existing) for existing in bars):continue
            origins[id(shifted)]=name;bars.append(shifted)
    return sorted(bars,key=lambda b:(b.center[0],b.center[1],b.color)),dict(
        adaptive_threshold=adaptive,methods=origins), (xa,ya,xb,yb)


def association(proposal,pair):
    x1,y1,x2,y2=proposal.bbox
    width,height=x2-x1,y2-y1
    if width<=0 or height<=0:return None
    cx,cy=pair.center
    dx=abs(cx-proposal.center[0])/width;dy=abs(cy-proposal.center[1])/height
    span=(pair.bbox[2]-pair.bbox[0])/width
    length=(pair.bbox[3]-pair.bbox[1])/height
    overlap=box_iou(proposal.bbox,pair.bbox)
    if not (x1<=cx<=x2 and y1<=cy<=y2 and dx<=.28 and dy<=.35
            and .42<=span<=1.35 and .14<=length<=1.3 and overlap>=.08):return None
    return dict(quality=1/(1+2*dx+dy+.4*abs(math.log(span))),iou=overlap,
                center_dx_ratio=dx,center_dy_ratio=dy,span_ratio=span)


def verify_classical_regions(frame,proposals,classical,neural,digits,max_regions=4,*,provenance=None):
    """Traditional proposals get an actual network evaluation before admission.

    The bounded second pass is local; no unverified CV box becomes output. A
    network hit elsewhere in the crop cannot endorse the proposed lamp pair.

    Optional provenance records original frame-local objects during this same
    pass. It never changes admission, deduplication, or this public return value.
    The caller may filter only after the complete verified list is finalized.
    """
    if provenance is not None:
        provenance.clear()
        provenance.update(bars=[],pairs=[],selected=[],admitted=[],regions=[])
    verifier=getattr(neural,"detect_regions",None)
    if verifier is None:return [],dict(regions=[],verified=0,available=False)
    bars=classical.find_light_bars(frame)
    pairs=lamp_pair_candidates(frame,bars,classical.config)
    if provenance is not None:
        pairs=list(pairs)
        provenance.update(bars=bars,pairs=pairs)
    candidates=[]
    for i,j,pair,_ in pairs:
        if pair.score<.55 or any(association(model,pair) is not None for model in proposals):continue
        # Full-frame model-supported pairs were excluded above. Only unbacked
        # CV proposals face this extra broad physical-aspect plausibility gate;
        # ROI measurements under an actual model box retain existing checks.
        limit=getattr(classical,"max_unbacked_span_ratio",None)
        if limit is not None and lamp_span_ratio(bars[i],bars[j])>limit:continue
        digit=digits.pair(frame,lamp_quad(bars[i],bars[j])) if digits else None
        candidates.append((bool(digit and digit["supported"]),pair.score,i,j,pair))
    candidates.sort(key=lambda c:(-c[0],-c[1],c[2],c[3]))
    selected=[];used=set();regions=[]
    height,width=frame.shape[:2]
    for _,_,i,j,pair in candidates:
        if i in used or j in used:continue
        x1,y1,x2,y2=pair.bbox;w=x2-x1;h=y2-y1
        region=(max(0,int(x1-.4*w)),max(0,int(y1-h)),min(width,int(math.ceil(x2+.4*w))),min(height,int(math.ceil(y2+h))))
        if min(region[2]-region[0],region[3]-region[1])<8:continue
        regions.append(region);selected.append(pair);used.update((i,j))
        if provenance is not None:
            provenance["selected"].append(dict(i=i,j=j,pair=pair,region_index=len(regions)-1))
        if len(regions)>=max_regions:break
    if provenance is not None:provenance["regions"]=regions
    verified=[]
    for region_index,(pair,detections) in enumerate(zip(selected,verifier(frame,regions))):
        matches=[]
        for hit_index,detection in enumerate(detections):
            # Crops introduce additional search hypotheses. A second-pass hit
            # needs stronger model support than a geometry-assisted global hit.
            if detection.score<REGIONAL_CONFIRMATION_CONFIDENCE:continue
            spatial=association(detection,pair)
            if spatial is not None and spatial["quality"]>=.65:
                matches.append((spatial["quality"]*detection.score,detection,hit_index))
        if not matches:continue
        chosen=max(matches,key=lambda row:row[0])
        detection=chosen[1]
        if any(box_iou(detection.bbox,old.bbox)>=.5 for old in [*proposals,*verified]):continue
        verified.append(detection)
        if provenance is not None:
            provenance["admitted"].append(dict(**provenance["selected"][region_index],
                regional_index=len(verified)-1,hit_index=chosen[2]))
    return verified,dict(available=True,regions=regions,verified=len(verified),cv_candidates=len(candidates),budget=max_regions)


def _inside_global(bar,proposal):
    x,y=bar.center
    x1,y1,x2,y2=proposal.bbox
    return x1<=x<=x2 and y1<=y<=y2


def _disjoint_globals(a,b):
    ax1,ay1,ax2,ay2=a.bbox;bx1,by1,bx2,by2=b.bbox
    return min(ax2,bx2)<=max(ax1,bx1) or min(ay2,by2)<=max(ay1,by1)


def filter_cross_global_regions(proposals,regional,provenance,confidence):
    """Reject only the frozen cross-global membership contradiction.

    This pure post-admission stage uses the verifier's original full-frame lamp
    indices, complete pair list and original unexpanded global boxes. It does
    not run a detector, change a threshold, revisit hits or refill a crop slot.
    Even vetoed admissions must already have participated in original dedup.
    """
    bars=provenance["bars"];pairs=provenance["pairs"]
    admitted=provenance["admitted"]
    if [seed["regional_index"] for seed in admitted]!=list(range(len(regional))):
        raise ValueError("regional provenance does not match complete admission order")
    facts=[]
    for seed in admitted:
        endpoints=(seed["i"],seed["j"])
        owners=[[k for k,g in enumerate(proposals) if g.score>=confidence
            and g.color in ("red","blue") and g.color==bars[index].color
            and _inside_global(bars[index],g)] for index in endpoints]
        fact=dict(seed_ids=list(endpoints),owner_sets=owners,contradicted=False,alternatives=[],
            region_index=seed["region_index"],regional_index=seed["regional_index"],hit_index=seed["hit_index"],
            reason="non_unique_global_owner")
        facts.append(fact)
        if any(len(owner)!=1 for owner in owners):continue
        a,b=owners[0][0],owners[1][0]
        fact["owners"]=[a,b]
        if a==b:
            fact["reason"]="same_global_owner";continue
        if not _disjoint_globals(proposals[a],proposals[b]):
            fact["reason"]="overlapping_global_owners";continue
        fact["owners_disjoint"]=True
        for endpoint,owner in zip(endpoints,(a,b)):
            for pair_index,(i,j,pair,_) in enumerate(pairs):
                if set((i,j))==set(endpoints):continue
                if endpoint not in (i,j) or pair.score<.55:continue
                if not (_inside_global(bars[i],proposals[owner]) and _inside_global(bars[j],proposals[owner])):continue
                spatial=association(proposals[owner],pair)
                if spatial is None:continue
                fact["alternatives"].append(dict(pair_index=pair_index,seed_ids=[i,j],owner=owner,
                    shared_endpoint=endpoint,association=spatial))
        fact["contradicted"]=bool(fact["alternatives"])
        fact["reason"]="cross_global_membership_with_alternative" if fact["contradicted"] else "no_within_owner_alternative"
    removed={f["regional_index"] for f in facts if f["contradicted"]}
    keep=[i for i in range(len(regional)) if i not in removed]
    sources=[dict(model_index=i,scope="full_frame",global_index=i) for i in range(len(proposals))]
    for index in keep:
        seed=admitted[index]
        sources.append(dict(model_index=len(sources),scope="cv_proposed_region",region_index=seed["region_index"],
            hit_index=seed["hit_index"],regional_index=index))
    return [regional[i] for i in keep],dict(verified=len(keep),verified_before_veto=len(regional),
        vetoed=len(removed),policy_version=REGIONAL_POLICY_VERSION,
        provenance=dict(admissions=facts,retained_regional_indices=keep,combined_proposals=sources))


def validate_regional_verification(diagnostics,global_count=None):
    """Strict archive-reader contract; never discard unknown diagnostic fields.

    Legacy public-verifier records have no policy_version. Versioned records
    describe the detector's post-admission input to guided_fuse, not its final
    regional output count. Raw admissions are needed to reproduce a veto.
    """
    def require(condition,message):
        if not condition:raise ValueError("invalid regional verification: "+message)
    def count(value):return type(value) is int and value>=0
    def finite_number(value):
        if not isinstance(value,(int,float)) or isinstance(value,bool):return False
        try:return math.isfinite(value)
        except (OverflowError,ValueError):return False
    require(isinstance(diagnostics,dict),"expected dictionary")
    require(type(diagnostics.get("available")) is bool,"available must be boolean")
    legacy={"available","regions","verified"}
    if diagnostics["available"]:legacy.update(("cv_candidates","budget"))
    version=diagnostics.get("policy_version","off")
    extra={"verified_before_veto","vetoed","policy_version","provenance"}
    require(isinstance(version,str) and version in ("off",REGIONAL_POLICY_VERSION),"unknown policy_version")
    require(set(diagnostics)==(legacy if version=="off" else legacy|extra),"unexpected or missing fields")
    require(count(diagnostics["verified"]),"verified must be a nonnegative integer")
    require(isinstance(diagnostics["regions"],(list,tuple)),"regions must be a sequence")
    for region in diagnostics["regions"]:
        require(isinstance(region,(list,tuple)) and len(region)==4 and all(type(n) is int for n in region),"region coordinates")
    if diagnostics["available"]:
        require(count(diagnostics["cv_candidates"]) and count(diagnostics["budget"]),"candidate/budget counters")
        require(len(diagnostics["regions"])<=diagnostics["cv_candidates"],"selected regions exceed candidates")
    else:
        require(not diagnostics["regions"] and diagnostics["verified"]==0,"unavailable verifier has admissions")
    if version=="off":
        require(diagnostics["verified"]<=len(diagnostics["regions"]),"admissions exceed selected regions")
        return version
    before=diagnostics["verified_before_veto"];vetoed=diagnostics["vetoed"]
    require(count(before) and count(vetoed),"before/vetoed counters")
    require(before<=len(diagnostics["regions"]) and before==diagnostics["verified"]+vetoed,"counter equation")
    provenance=diagnostics["provenance"]
    require(isinstance(provenance,dict) and set(provenance)=={"admissions","retained_regional_indices","combined_proposals"},"provenance fields")
    admissions=provenance["admissions"];keep=provenance["retained_regional_indices"];combined=provenance["combined_proposals"]
    require(isinstance(admissions,list) and len(admissions)==before,"admission count")
    require(isinstance(keep,list) and isinstance(combined,list),"provenance sequences")
    base={"seed_ids","owner_sets","contradicted","alternatives","region_index","regional_index","hit_index","reason"}
    reasons={"non_unique_global_owner","same_global_owner","overlapping_global_owners","no_within_owner_alternative","cross_global_membership_with_alternative"}
    for index,fact in enumerate(admissions):
        require(isinstance(fact,dict) and base<=set(fact)<=base|{"owners","owners_disjoint"},"admission fields")
        require(count(fact["regional_index"]) and fact["regional_index"]==index and count(fact["hit_index"]) and count(fact["region_index"])
            and fact["region_index"]<len(diagnostics["regions"]),"admission identity/order")
        require(type(fact["contradicted"]) is bool and isinstance(fact["reason"],str)
            and fact["reason"] in reasons,"admission decision")
        require(isinstance(fact["seed_ids"],list) and len(fact["seed_ids"])==2 and all(count(i) for i in fact["seed_ids"])
            and len(set(fact["seed_ids"]))==2,"seed IDs")
        require(isinstance(fact["owner_sets"],list) and len(fact["owner_sets"])==2
            and all(isinstance(owners,list) and all(count(i) for i in owners) and owners==sorted(set(owners))
                for owners in fact["owner_sets"]),"owner sets")
        unique=all(len(owners)==1 for owners in fact["owner_sets"])
        require(("owners" in fact)==unique,"unique-owner fields")
        if unique:require(isinstance(fact["owners"],list) and all(count(i) for i in fact["owners"])
            and fact["owners"]==[owners[0] for owners in fact["owner_sets"]],"owner identity")
        disjoint=fact["reason"] in ("no_within_owner_alternative","cross_global_membership_with_alternative")
        require(("owners_disjoint" in fact)==disjoint and (not disjoint or fact["owners_disjoint"] is True),"disjoint field")
        require(isinstance(fact["alternatives"],list),"alternative sequence")
        expected_reason=("non_unique_global_owner" if not unique else
            "same_global_owner" if fact["owners"][0]==fact["owners"][1] else
            "overlapping_global_owners" if "owners_disjoint" not in fact else
            "cross_global_membership_with_alternative" if fact["alternatives"] else "no_within_owner_alternative")
        require(fact["reason"]==expected_reason,"reason contradicts ownership/support")
        require(fact["contradicted"]==bool(fact["alternatives"])==(fact["reason"]=="cross_global_membership_with_alternative"),"decision/support equation")
        for alternative in fact["alternatives"]:
            require(isinstance(alternative,dict) and set(alternative)=={"pair_index","seed_ids","owner","shared_endpoint","association"},"alternative fields")
            require(all(count(alternative[k]) for k in ("pair_index","owner","shared_endpoint")),"alternative identity")
            require(isinstance(alternative["seed_ids"],list) and len(alternative["seed_ids"])==2 and all(count(i) for i in alternative["seed_ids"])
                and len(set(alternative["seed_ids"]))==2 and set(alternative["seed_ids"])!=set(fact["seed_ids"]),"alternative seed IDs")
            require(alternative["owner"] in fact["owners"] and alternative["shared_endpoint"] in fact["seed_ids"]
                and alternative["shared_endpoint"] in alternative["seed_ids"]
                and alternative["owner"]==fact["owners"][fact["seed_ids"].index(alternative["shared_endpoint"])],"alternative owner/endpoint")
            spatial=alternative["association"]
            require(isinstance(spatial,dict) and set(spatial)=={"quality","iou","center_dx_ratio","center_dy_ratio","span_ratio"}
                and all(finite_number(n) for n in spatial.values()),"alternative association")
    region_ids=[fact["region_index"] for fact in admissions]
    require(region_ids==sorted(set(region_ids)),"admitted region order/identity")
    require(all(count(i) for i in keep),"retained index types")
    expected_keep=[i for i,fact in enumerate(admissions) if not fact["contradicted"]]
    require(keep==expected_keep and len(keep)==diagnostics["verified"],"retained identity/count")
    require(sum(fact["contradicted"] for fact in admissions)==vetoed,"veto count")
    if global_count is None:global_count=len(combined)-len(keep)
    require(count(global_count) and len(combined)==global_count+len(keep),"combined proposal count")
    expected=[dict(model_index=i,scope="full_frame",global_index=i) for i in range(global_count)]
    for index in keep:
        fact=admissions[index]
        expected.append(dict(model_index=len(expected),scope="cv_proposed_region",region_index=fact["region_index"],
            hit_index=fact["hit_index"],regional_index=index))
    require(all(isinstance(row,dict) and all(count(value) for key,value in row.items() if key!="scope") for row in combined),"combined index types")
    require(combined==expected,"combined proposal lineage")
    require(all(owner<global_count for fact in admissions for owners in fact["owner_sets"] for owner in owners),"owner out of range")
    return version


def validate_fusion_proposal_counts(diagnostics,global_count=None):
    """Validate the enclosing effective-input equation for strict replay tools."""
    if not isinstance(diagnostics,dict):raise ValueError("invalid fusion diagnostics: expected dictionary")
    recorded_global=diagnostics.get("global_model_proposals")
    combined=diagnostics.get("model_proposals")
    if type(recorded_global) is not int or recorded_global<0 or type(combined) is not int or combined<0:
        raise ValueError("invalid fusion proposal counters")
    if global_count is not None and (type(global_count) is not int or global_count!=recorded_global):
        raise ValueError("original global proposal count differs")
    version=validate_regional_verification(diagnostics.get("regional_verification"),recorded_global)
    if combined!=recorded_global+diagnostics["regional_verification"]["verified"]:
        raise ValueError("model_proposals must equal global_model_proposals + retained verified")
    return version


def guided_fuse(frame,proposals,config,digits,confidence=.25,*,color_first=False):
    candidates=[];roi_records=[];all_bars=[]
    for index,proposal in enumerate(proposals):
        bars,extraction,roi=(extract_roi(frame,proposal,config,color_first=True) if color_first else extract_roi(frame,proposal,config))
        all_bars.extend(b for b in bars if not any(same_lamp(b,old) for old in all_bars))
        admitted=0
        # Character classification is performed only after the model association
        # gate, so out-of-region pairs cannot consume the CNN budget or lamps.
        for i,j,pair,_ in lamp_pair_candidates(frame,bars,config):
            spatial=association(proposal,pair)
            if spatial is None or pair.score<.45:continue
            corners=lamp_quad(bars[i],bars[j])
            try:polygon_array(corners)
            except ValueError:continue
            digit=digits.pair(frame,corners) if digits else None
            supported=bool(digit and digit["supported"])
            quality=.40*spatial["quality"]+.35*pair.score+.20*proposal.score+.05*supported
            weak=proposal.score<confidence
            if weak and (proposal.score<PROPOSAL_CONFIDENCE or pair.score<.55 or spatial["quality"]<.65):continue
            candidates.append(dict(index=index,pair=pair,bars=(bars[i],bars[j]),corners=corners,digit=digit,
                quality=quality,spatial=spatial,roi=roi,adaptive_threshold=extraction["adaptive_threshold"],
                extraction=[extraction["methods"][id(bars[k])] for k in (i,j)]))
            admitted+=1
        roi_records.append(dict(model_index=index,roi=roi,lamps=len(bars),admissible_pairs=admitted))
    candidates.sort(key=lambda c:(-c["quality"],-proposals[c["index"]].score,c["index"]))
    chosen={};used=[];conflicts=[]
    for candidate in candidates:
        index=candidate["index"]
        if index in chosen:continue
        if any(same_lamp(a,b) for a in candidate["bars"] for b in used):
            conflicts.append(index);continue
        chosen[index]=candidate;used.extend(candidate["bars"])
    output=[];rejected=[]
    for index,proposal in enumerate(proposals):
        candidate=chosen.get(index)
        if candidate is None:
            if index in conflicts and any(box_iou(proposal.bbox,proposals[other].bbox)>=.5 for other in chosen):
                rejected.append(dict(model_index=index,score=proposal.score,reason="shared_lamp_duplicate"));continue
            if proposal.score<confidence:
                rejected.append(dict(model_index=index,score=proposal.score,reason="weak_without_geometry"));continue
            # Unlit/occluded candidates stay explicit: neither valid corners nor
            # an invented fused confidence can be supplied by a neural rectangle.
            output.append(dict(detection=replace(proposal,source="neural"),corners=None,geometry_score=None,
                digit=digits.box(frame,proposal.bbox) if digits else None,
                evidence=dict(status="model_only",model_confidence=proposal.score,geometry_quality=None,
                    association_quality=None,joint_quality=None,rescued=False,reason="no_unique_lamp_pair")))
            continue
        pair=candidate["pair"];corners=tuple(tuple(map(float,p)) for p in candidate["corners"])
        output.append(dict(detection=Detection(proposal.bbox,polygon_center(corners),pair.color,proposal.score,"fused"),
            corners=corners,geometry_score=pair.score,digit=candidate["digit"],evidence=dict(
                status="joint_confirmed",model_confidence=proposal.score,geometry_quality=pair.score,
                association_quality=candidate["spatial"]["quality"],joint_quality=candidate["quality"],
                quality_kind="heuristic_not_probability",rescued=proposal.score<confidence,
                roi=candidate["roi"],lamp_bbox=pair.bbox,extraction=candidate["extraction"],
                adaptive_threshold=candidate["adaptive_threshold"],
                model_color=proposal.color,color_corrected=proposal.color not in ("unknown",pair.color))))
        output[-1]["evidence"].update(lamp_axis_method="pixel_least_squares",
            lamp_core_fill=[b.core_fill for b in candidate["bars"]],
            lamp_axis_residual_px=[b.axis_residual_px for b in candidate["bars"]])
    diagnostics=dict(fusion_policy=POLICY,model_proposals=len(proposals),joint_confirmed=len(chosen),
        weak_rescued=sum(o["evidence"]["rescued"] for o in output),model_only=sum(o["corners"] is None for o in output),
        rejected_proposals=rejected,shared_lamp_conflicts=conflicts,rois=roi_records,
        red_bars=sum(b.color=="red" for b in all_bars),blue_bars=sum(b.color=="blue" for b in all_bars),
        reason="detected" if output else "no_jointly_supported_proposal")
    return output,diagnostics
