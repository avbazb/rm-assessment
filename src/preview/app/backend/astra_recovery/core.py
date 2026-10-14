"""Research candidate: neutral core with bilateral chromatic emission support.

Returns additional measured bars only. Pair proportions, original extraction,
model ownership and production defaults remain untouched.
"""
import math
import cv2
import numpy as np
cv2.setNumThreads(1)
from astra_vision.detector import LightBar
from astra_competition.classical import BrightCoreDetector,fit_lamp_axis

def core_candidates(image, config=None):
    cfg=config or BrightCoreDetector().config
    minimum=image.min(axis=2)
    mask=(minimum>=180).astype(np.uint8)*255
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
        placeholder=LightBar(tuple(map(float,rect[0])),length,width,tilt,corners,'blue')
        fitted=fit_lamp_axis(placeholder,component,np.array((x,y)),.60)
        if fitted is not None:yield fitted

def band_evidence(image,bar):
    # Use normal-distance bands around the measured axis, not image axes.
    a,b=bar.axis_endpoints;axis=b-a;length=np.linalg.norm(axis);axis=axis/length
    normal=np.array([axis[1],-axis[0]])
    outer=max(12.,6.*bar.width)
    mn=np.floor(np.minimum(a,b)-outer-2).astype(int);mx=np.ceil(np.maximum(a,b)+outer+2).astype(int)
    mn=np.maximum(mn,0);mx=np.minimum(mx,[image.shape[1],image.shape[0]])
    x0,y0=mn;x1,y1=mx
    yy,xx=np.mgrid[y0:y1,x0:x1];delta=np.stack([xx-a[0],yy-a[1]],-1)
    along=delta@axis;side=delta@normal
    crop=image[y0:y1,x0:x1].astype(np.int16);bb,gg,rr=np.moveaxis(crop,-1,0)
    mxv=crop.max(axis=2);mnv=crop.min(axis=2);sat=(mxv-mnv)/np.maximum(mxv,1)
    interior=(along>=.1*length)&(along<=.9*length)
    inner=max(1.,bar.width*.35);near=max(4.,bar.width*1.15)
    left=interior&(side>=inner)&(side<=near);right=interior&(side<=-inner)&(side>=-near)
    far=interior&(np.abs(side)>=max(near+3,bar.width*4.5))&(np.abs(side)<=outer)
    result={}
    for color,ch,op in [('blue',bb,rr),('red',rr,bb)]:
        # Cyan is a valid blue halo; green dominates neither RGB chroma end.
        green_margin=-25 if color=='blue' else 12
        colored=(ch-op>=30)&(ch-gg>=green_margin)&(sat>=.4)&(ch>=100)
        background=float(np.median(ch[far])) if far.any() else 255.
        contrast=ch>=background+35
        halves=[]
        for band in (left,right):
            supported=band&colored&contrast
            bins=np.floor(along[supported]/length*5).astype(int)
            counts=np.bincount(np.clip(bins,0,4),minlength=5)
            halves.append(dict(pixels=int(supported.sum()),fraction=float(supported.sum()/max(1,band.sum())),bins=int((counts>=2).sum()),median_ch=float(np.median(ch[band])) if band.any() else 0.,median_sat=float(np.median(sat[band])) if band.any() else 0.))
        result[color]=dict(background=background,left=halves[0],right=halves[1])
    return result

def find_extra_bars(image,config=None,diagnostics=None):
    cfg=config or BrightCoreDetector().config
    out=[]
    for bar in core_candidates(image,cfg):
        evidence=band_evidence(image,bar);accepted=[]
        for color,ev in evidence.items():
            if cfg.color not in ('both',color):continue
            if all(ev[k]['bins']>=4 and ev[k]['fraction']>=.25 for k in ('left','right')):accepted.append(color)
        if diagnostics is not None:diagnostics.append(dict(center=bar.center,length=bar.length,width=bar.width,evidence=evidence,accepted=accepted))
        if len(accepted)==1:
            from dataclasses import replace
            out.append(replace(bar,color=accepted[0]))
    return out
