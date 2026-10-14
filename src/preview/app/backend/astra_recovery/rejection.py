import cv2
import numpy as np
def features(im,o):
    h,w=im.shape[:2];x0,y0,x1,y1=o['bbox'];x0,y0=np.floor([x0,y0]).astype(int);x1,y1=np.ceil([x1,y1]).astype(int)
    x0=max(0,x0);y0=max(0,y0);x1=min(w,x1);y1=min(h,y1);crop=im[y0:y1,x0:x1]
    if crop.size==0:return {'invalid':True}
    b,g,r=np.moveaxis(crop.astype('int16'),-1,0);v=crop.max(2);gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    blue=(b-r>=25)&(b-g>=-25)&(b>=80);red=(r-b>=25)&(r-g>=12)&(r>=80)
    mask=(blue|red).astype('uint8');n,lab,stats,_=cv2.connectedComponentsWithStats(mask,8)
    spans=[int(max(s[cv2.CC_STAT_WIDTH],s[cv2.CC_STAT_HEIGHT])) for s in stats[1:] if s[cv2.CC_STAT_AREA]>=3]
    ch,cw=crop.shape[:2];colored=mask>0
    return dict(area_fraction=(x1-x0)*(y1-y0)/(w*h),width_fraction=(x1-x0)/w,height_fraction=(y1-y0)/h,
                color_pixels=int(mask.sum()),color_fraction=float(mask.mean()),max_colored_span=max(spans,default=0),
                max_colored_span_to_box_height=max(spans,default=0)/ch,gray_p90=float(np.percentile(gray,90)),gray_p99=float(np.percentile(gray,99)),
                left_color_pixels=int(colored[:,:max(1,int(cw*.35))].sum()),right_color_pixels=int(colored[:,int(cw*.65):].sum()),
                hue_color=o['color'],geometry_valid=o['geometry_valid'])
"""Read-only research predicate for an unsupported dark model-only large box.

Parameters frozen from the developmental failure mechanism before the 149-box
video scan; not a model threshold change and not a production admission rule.
"""
POLICY=dict(min_area_fraction=.10,max_gray_p99=32.,max_color_fraction=.005,
            max_colored_span_to_box_height=.15,require_no_geometry=True)
def reject(observation,features):
    return (not observation['geometry_valid'] and features['area_fraction']>=.10
            and features['gray_p99']<=32. and features['color_fraction']<=.005
            and features['max_colored_span_to_box_height']<.15)
