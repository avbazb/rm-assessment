"""Measure red/blue-seeded lamp cores, including enclosed overexposure."""
import math
import cv2,numpy as np
from astra_competition.classical import BrightCoreDetector,fit_lamp_axis
from astra_vision.detector import LightBar


class _ColorComponents(BrightCoreDetector):
    def __init__(self,config=None,brightness=180,min_core_fill=None,*,model_guided=False,recover_open_core=False):
        super().__init__(config,brightness,min_core_fill);self.model_guided=model_guided
        self.recover_open_core=recover_open_core or model_guided

    def find_light_bars(self,image):
        if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or not image.size:
            raise ValueError('expected nonempty uint8 BGR image')
        cfg=self.config;b,g,r=cv2.split(image);value=cv2.max(cv2.max(b,g),r)
        achromatic=None
        if self.recover_open_core:
            minimum=cv2.min(cv2.min(b,g),r)
            achromatic=minimum.astype(np.uint16)*100>=value.astype(np.uint16)*35
        hue,saturation,_=cv2.split(cv2.cvtColor(image,cv2.COLOR_BGR2HSV))
        margin=cv2.LUT(value,np.maximum(12,(np.arange(256)+3)//4).astype(np.uint8))
        if self.model_guided:margin=np.full_like(value,12)
        chroma={'red':((hue<=12)|(hue>=168))&(cv2.subtract(r,b)>=25)&(cv2.subtract(r,g)>=margin)&(value>=70),
                'blue':(hue>=85)&(hue<=135)&(cv2.subtract(b,r)>=25)&(cv2.subtract(b,g)>=12)&(value>=70)}
        candidates=[]
        height,frame_width=image.shape[:2]
        for color in ('red','blue'):
            if cfg.color not in ('both',color):continue
            other='red' if color=='blue' else 'blue'
            seed=chroma[color].astype(np.uint8)*255
            # Select red/blue connected regions first. Enclosed achromatic
            # centers can belong to an overexposed lamp; unrelated white areas
            # never initiate a component. Closing is only region selection.
            closed=cv2.morphologyEx(seed,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
            envelopes=np.zeros_like(seed)
            regions,_=cv2.findContours(closed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(envelopes,regions,-1,255,-1)
            envelopes=cv2.bitwise_or(envelopes,cv2.dilate(seed,np.ones((3,3),np.uint8)))
            color_support=envelopes>0
            if self.recover_open_core:
                adjacent=cv2.dilate(seed,np.ones((3,3),np.uint8))>0
                # Reconstruct achromatic components only after a colored seed
                # reaches them. Reuse this support across luminance levels.
                neutral=(achromatic&(value>=min(self.brightness,180))&~chroma[other]).astype(np.uint8)
                count,labels=cv2.connectedComponents(neutral)
                admitted=np.zeros(count,bool);admitted[np.unique(labels[adjacent&(neutral>0)])]=True;admitted[0]=False
                color_support|=admitted[labels]
            for threshold in sorted(set([self.brightness,180,210,235,250])):
                bright=cv2.inRange(value,threshold,255)
                mask=((bright>0)&color_support&~chroma[other]).astype(np.uint8)*255
                contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area=cv2.contourArea(contour)
                    if area<cfg.min_contour_area:continue
                    rect=cv2.minAreaRect(contour);corners=cv2.boxPoints(rect)
                    edges=np.roll(corners,-1,axis=0)-corners;lengths=np.linalg.norm(edges,axis=1)
                    length,width=float(lengths.max()),float(lengths.min())
                    if width<.8 or length<cfg.min_bar_length:continue
                    edge=edges[int(np.argmax(lengths))]
                    tilt=(math.degrees(math.atan2(float(edge[0]),float(edge[1])))+90)%180-90
                    if not cfg.min_bar_aspect<=length/width<=cfg.max_bar_aspect or abs(tilt)>cfg.max_bar_tilt_degrees:continue
                    if area/(length*width)<cfg.min_rectangularity:continue
                    x,y,w,h=cv2.boundingRect(contour)
                    component=np.zeros((h,w),np.uint8);cv2.drawContours(component,[contour-np.array((x,y))],-1,255,-1)
                    component=cv2.bitwise_and(component,mask[y:y+h,x:x+w])
                    x0,y0=max(0,x-3),max(0,y-3);x1,y1=min(frame_width,x+w+3),min(height,y+h+3)
                    support=np.zeros((y1-y0,x1-x0),np.uint8)
                    support[y-y0:y-y0+h,x-x0:x-x0+w]=component
                    support=cv2.dilate(support,np.ones((7,7),np.uint8))>0
                    counts={name:int((support&seed[y0:y1,x0:x1]).sum()) for name,seed in chroma.items()}
                    if counts[color]<max(3,int(support.sum()*.05)) or counts[color]<2*counts[other]:continue
                    bar=LightBar(tuple(map(float,rect[0])),length,width,tilt,corners,color)
                    fitted=fit_lamp_axis(bar,component,np.array((x,y)),self.min_core_fill)
                    if fitted is not None:candidates.append((fitted,threshold))
        # Prefer complete valid measured components over their shorter bright
        # inner versions. Distinct axes are retained for downstream pairing.
        chosen=[];n=len(candidates)
        centers=np.empty((n,2));lengths=np.empty(n);widths=np.empty(n);tilts=np.empty(n);colors=np.empty(n,bool)
        for bar,threshold in sorted(candidates,key=lambda item:(-item[0].length,item[1])):
            axis=bar.axis_endpoints[1]-bar.axis_endpoints[0];axis=axis/np.linalg.norm(axis)
            count=len(chosen);delta=centers[:count]-bar.center
            normal=np.abs(axis[0]*delta[:,1]-axis[1]*delta[:,0])
            overlap=(bar.length+lengths[:count])/2-np.abs(delta@axis)
            duplicate=np.any((colors[:count]==(bar.color=='blue'))&
                (normal<=np.maximum(2.,.5*np.maximum(bar.width,widths[:count])))&
                (overlap>.5*np.minimum(bar.length,lengths[:count]))&(np.abs(bar.tilt_degrees-tilts[:count])<12))
            if not duplicate:
                centers[count]=bar.center;lengths[count]=bar.length;widths[count]=bar.width;tilts[count]=bar.tilt_degrees;colors[count]=bar.color=='blue'
                chosen.append(bar)
        return sorted(chosen,key=lambda bar:(bar.center[0],bar.center[1],bar.color))


class ColorFirstDetector(_ColorComponents):
    min_pair_quality=.35
    exclusive_overlap=True
    # Tighten only unbacked full-frame pairing. Model-backed ROI admission
    # retains its independently validated envelope below.
    max_classical_span_ratio=4.7
    # Broad admission envelope, covering small and large lamp references.
    # The largest existing reference is 230/55; allow 20% measurement margin.
    # No lower aspect gate: yaw can make a genuine plate appear much narrower.
    max_unbacked_span_ratio=230/55*1.20
    def find_light_bars(self,image):
        bars=super().find_light_bars(image);accepted=[];self.last_color_evidence=[]
        if self.model_guided:return bars
        for bar in bars:
            x,y,w,h=cv2.boundingRect(bar.corners.astype(np.float32))
            margin=8 if self.recover_open_core else 4
            x0,y0=max(0,x-margin),max(0,y-margin);x1,y1=min(image.shape[1],x+w+margin),min(image.shape[0],y+h+margin)
            crop=image[y0:y1,x0:x1].astype(float);maximum=crop.max(axis=2);minimum=crop.min(axis=2)
            saturation=(maximum-minimum)/np.maximum(maximum,1)
            footprint=np.zeros(maximum.shape,np.uint8);cv2.fillConvexPoly(footprint,np.rint(bar.corners-[x0,y0]).astype(np.int32),255)
            inside=footprint>0
            peak=np.percentile(maximum[inside],90)
            core=inside&(maximum>=.9*peak)
            b,g,r=np.moveaxis(crop,-1,0)
            colored=((b-r>=25)&(b-g>=12)) if bar.color=='blue' else ((r-b>=25)&(r-g>=12))
            around=(cv2.dilate(footprint,np.ones((7,7),np.uint8))>0)&~inside&colored&(maximum>=70)
            core_sat=float(np.median(saturation[core])) if core.any() else 0.
            halo_sat=float(np.median(saturation[around])) if around.any() else 0.
            accept=core_sat>=.65 or (halo_sat>=.45 and halo_sat-core_sat>=.15)
            coverage=None
            if self.recover_open_core:
                # A locally verified crop must originate in a measured colored
                # lamp, not a white glyph surrounded by dim colored pavement.
                # Cyan bloom is valid blue evidence even when G approaches B.
                color_support=((b-r>=25)&(b-g>=-40)) if bar.color=='blue' else ((r-b>=25)&(r-g>=12))
                nearby=cv2.dilate(footprint,np.ones((7,7),np.uint8))>0
                rim=(cv2.dilate(footprint,np.ones((15,15),np.uint8))>0)&~nearby
                channel=b if bar.color=='blue' else r
                background=float(np.median(channel[rim])) if rim.any() else peak
                # Saturated halos can be dimmer than an overexposed white core.
                # Admit their contrast against the surrounding scene; a flat
                # blue floor around a white subtitle has no such color contrast.
                contrast=(channel>=background+30)&(maximum>=70)
                luminous=nearby&color_support&(saturation>=.35)&((maximum>=.85*peak)|contrast)
                yy,xx=np.nonzero(luminous);axis=bar.axis_endpoints[1]-bar.axis_endpoints[0];length=float(np.linalg.norm(axis));axis/=length
                along=(np.column_stack((xx+x0,yy+y0))-bar.axis_endpoints[0])@axis
                along=along[(along>=0)&(along<=length)]
                bins=np.bincount(np.clip((along/length*5).astype(int),0,4),minlength=5)
                coverage=int(np.count_nonzero(bins))
                accept=coverage>=3 and (accept or (bar.color=='blue' and core_sat>=.35))
            self.last_color_evidence.append(dict(center=bar.center,color=bar.color,core_saturation=core_sat,halo_saturation=halo_sat,color_axis_bins=coverage,accepted=accept))
            if accept:accepted.append(bar)
        return accepted
