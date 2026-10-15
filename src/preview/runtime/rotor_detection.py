"""Fresh lamp measurements at camera rate; one asynchronous model job, no queue.

Old model rectangles define search regions and feature seeds. Current pixels
must support each returned box. Feature tracking never creates PnP endpoints.
"""
import copy
import math
import time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from . import fast_classical_v2 as lamps
from .fast_classical import _iou


class ModelSlot:
    def __init__(self, pipeline):
        self.pipeline=pipeline;self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='rotor-model')
        self.pending=None;self.owner=None;self.results={};self.jobs=0

    def _run(self, image, timestamp):
        neural=copy.copy(self.pipeline._model());neural.imgsz=min(832,self.pipeline._input_size(image))
        started=time.perf_counter()
        rows=[r.to_dict() for r in neural.detect(image)]
        return dict(timestamp=timestamp,models=rows,gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),
                    shape=image.shape[:2],processing_ms=(time.perf_counter()-started)*1000)

    def poll(self, owner, image, timestamp):
        if self.pending is not None and self.pending.done():
            completed=self.pending;completed_owner=self.owner;self.pending=None
            self.results[completed_owner]=completed.result()
            while len(self.results)>8:self.results.pop(next(iter(self.results)))
        if self.pending is None:
            self.pending=self.executor.submit(self._run,image.copy(),timestamp);self.owner=owner;self.jobs+=1
        result=self.results.get(owner)
        if result is None or result['shape']!=image.shape[:2] or not 0<=timestamp-result['timestamp']<=.30:
            return None
        return result

    def close(self):
        self.executor.shutdown(wait=False,cancel_futures=True);self.results.clear()


def detect(image,timestamp,owner,slot):
    started=time.perf_counter();model=slot.poll(owner,image,timestamp)
    if model is None:
        rows,diag=lamps.detect(image,timestamp)
        return rows,rows,[],dict(rotation_camera_rate=True,model_proposals_ready=False,model_queue_capacity=0,
            model_inferences=0,model_assistance=False,image_database_used=False,classical=diag)
    h,w=image.shape[:2];regions=[]
    for row in model['models']:
        if row['score']<.30:continue
        box=np.asarray(row['bbox'],float);size=box[2:]-box[:2]
        # Cover the adjacent face when the rotor exposes the next plate.
        pad=np.array([1.7*size[1],.55*size[1]])
        lo=np.maximum(0,box[:2]-pad);hi=np.minimum([w,h],box[2:]+pad)
        if np.any(hi-lo<6):continue
        region=dict(bbox=np.r_[lo,hi],color=row.get('color','unknown'),model_score=row['score'],plate_heights=[float(size[1])])
        overlaps=[i for i,old in enumerate(regions) if old['color']==region['color'] and _iou(old['bbox'],region['bbox'])>.05]
        for i in reversed(overlaps):
            old=regions.pop(i);region['bbox']=np.r_[np.minimum(region['bbox'][:2],old['bbox'][:2]),np.maximum(region['bbox'][2:],old['bbox'][2:])]
            region['model_score']=max(region['model_score'],old['model_score'])
            region['plate_heights'].extend(old['plate_heights'])
        regions.append(region)
    if not regions:
        rows,diag=lamps.detect(image,timestamp)
        return rows,rows,[],dict(rotation_camera_rate=True,model_proposals_ready=True,model_age_s=timestamp-model['timestamp'],
            model_queue_capacity=0,model_inferences=0,model_assistance=False,image_database_used=False,classical=diag)
    candidates=[];bars_count=0
    for group,region in enumerate(regions):
        x1,y1,x2,y2=np.rint(region['bbox']).astype(int);crop=image[y1:y2,x1:x2]
        if not crop.size:continue
        bars=lamps._bars(crop,chromatic_cores=True);pairs=lamps._pairs(bars,crop.shape);bars_count+=len(bars)
        for p in pairs:
            if p['score']<.52 or not .65<=p['span']<=3.2 or p['color_fraction']<.08 or p['length']<5:continue
            if region['color'] not in ('unknown',p['color']):continue
            offset=np.array([x1,y1]);p=dict(p,bbox=p['bbox']+np.tile(offset,2),corners=p['corners']+offset,center=p['center']+offset)
            o=lamps._output(p,timestamp)
            o['fusion_evidence'].update(status='rotation_roi_verified',model_assistance=True,
                model_scope='recent_robot_region_only',model_timestamp=model['timestamp'],model_age_s=timestamp-model['timestamp'],
                model_confidence=region['model_score'],robot_region=region['bbox'].tolist(),roi_group=group,
                current_lamp_measurement=True,max_small_plate_span=3.2,
                measured_lamp_extent_fraction=p['length']/max(1.,float(np.median(region['plate_heights']))))
            candidates.append(o)
    # Exclusive physical bars and overlapping pairs are resolved before tracking.
    from astra_recovery.attachment import same_axis
    accepted=[];owned=[]
    for o in sorted(candidates,key=lambda r:-r['geometry_score']):
        q=np.asarray(o['corners']);axes=(q[[0,3]],q[[1,2]])
        if any(same_axis(a,b) for a in axes for b in owned):continue
        if any(_iou(o['bbox'],old['bbox'])>.6 for old in accepted):continue
        accepted.append(o);owned.extend(axes)
    measured=list(accepted)
    # A plate remains detectable while its lamps blink off or only one is
    # exposed. Re-measure its model-supported image features in this frame;
    # never label a delayed model rectangle as a current measurement.
    tracked=model_image_tracks(image,model,timestamp,regions)
    for o in tracked:
        if any(_iou(o['bbox'],old['bbox'])>.18 or np.linalg.norm(np.asarray(o['center'])-old['center'])<.35*(o['bbox'][3]-o['bbox'][1]) for old in accepted):continue
        accepted.append(o)
    return accepted,measured,[],dict(rotation_camera_rate=True,model_proposals_ready=True,model_age_s=timestamp-model['timestamp'],
        model_processing_ms=model['processing_ms'],model_queue_capacity=0,model_inferences=0,model_assistance=True,
        image_database_used=False,classical=dict(light_bars=bars_count,candidate_pairs=len(candidates),measured_count=len(measured),image_tracked_count=len(accepted)-len(measured),regions=len(regions),
            processing_ms=(time.perf_counter()-started)*1000),classical_module=lamps.VERSION)


def model_image_tracks(image,model,timestamp,regions):
    if model.get('gray') is None or not 0<=timestamp-model['timestamp']<=.30:return []
    old=model['gray'];gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);height,width=gray.shape
    feature_groups=[];points=[];proposals=[]
    for m in sorted(model['models'],key=lambda r:-r['score']):
        if m['score']<.45 or any(_iou(m['bbox'],p['bbox'])>.5 for p in proposals):continue
        x1,y1,x2,y2=np.rint(m['bbox']).astype(int);x1=max(0,x1);y1=max(0,y1);x2=min(width,x2);y2=min(height,y2)
        if x2-x1<8 or y2-y1<8:continue
        p=cv2.goodFeaturesToTrack(old[y1:y2,x1:x2],maxCorners=30,qualityLevel=.02,minDistance=2,blockSize=3)
        if p is None or len(p)<6:continue
        p=p+np.float32([x1,y1]);feature_groups.append((len(points),len(points)+len(p)));points.extend(p);proposals.append(m)
    if not points:return []
    source=np.float32(points).reshape(-1,1,2)
    target,status,_=cv2.calcOpticalFlowPyrLK(old,gray,source,None,winSize=(15,15),maxLevel=2)
    back,reverse,_=cv2.calcOpticalFlowPyrLK(gray,old,target,None,winSize=(15,15),maxLevel=2)
    good=(status.ravel()!=0)&(reverse.ravel()!=0)&(np.linalg.norm(back-source,axis=(1,2))<1.)
    accepted=[]
    for (start,end),m in zip(feature_groups,proposals):
        valid=good[start:end]
        if sum(valid)<6 or np.mean(valid)<.55:continue
        a=source[start:end,0][valid];b=target[start:end,0][valid]
        matrix,inliers=cv2.estimateAffine2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=1.5,maxIters=100,confidence=.95)
        if matrix is None or inliers is None or inliers.sum()<6 or inliers.mean()<.65:continue
        scales=np.linalg.svd(matrix[:,:2],compute_uv=False);angle=abs(math.atan2(matrix[1,0],matrix[0,0]))
        if np.linalg.det(matrix[:,:2])<=0 or scales.min()<.65 or scales.max()>1.4 or angle>.30:continue
        box=np.asarray(m['bbox']);corners=np.array([[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]])
        q=corners@matrix[:,:2].T+matrix[:,2];lo=q.min(axis=0);hi=q.max(axis=0)
        if np.any(lo<0) or np.any(hi>np.array([width,height])-1):continue
        center=(lo+hi)/2;delta=np.linalg.norm(center-(box[:2]+box[2:])/2)
        if delta>max(12.,2.*(box[3]-box[1])):continue
        color=m.get('color','unknown')
        region=max(regions,key=lambda r:_iou(r['bbox'],np.r_[lo,hi]))
        if _iou(region['bbox'],np.r_[lo,hi])<=0:continue
        if color=='unknown':
            crop=image[max(0,int(lo[1])):int(hi[1])+1,max(0,int(lo[0])):int(hi[0])+1]
            blue,green,red=[p.astype(np.int16) for p in cv2.split(crop)]
            counts={'red':int(np.sum((red-blue>30)&(red-green>18)&(red>100))),
                    'blue':int(np.sum((blue-red>30)&(blue-green>-15)&(blue>100)))}
            best=max(counts,key=counts.get)
            if counts[best]>=5 and counts[best]>=2*min(counts.values()):color=best
        accepted.append(dict(timestamp=float(timestamp),frame_id='camera_optical',bbox=np.r_[lo,hi].tolist(),center=center.tolist(),
            color=color,score=float(m['score']),score_kind='source_model_confidence',source='image_tracked_model',
            corners=None,measured_lamp_ends=None,geometry_valid=False,geometry_score=None,geometry_reason='image_features_only',
            fusion_evidence=dict(status='current_image_features',model_assistance=True,model_confidence=m['score'],
                model_timestamp=model['timestamp'],model_age_s=timestamp-model['timestamp'],current_image_measurement=True,
                feature_inliers=int(inliers.sum()),feature_inlier_fraction=float(inliers.mean()),
                robot_region=np.asarray(region['bbox']).tolist(),model_bbox=box.tolist(),classical_module=lamps.VERSION)))
    return accepted
