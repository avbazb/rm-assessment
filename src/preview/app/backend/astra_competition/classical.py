"""Bright lamp cores with independently measured red/blue support."""
import math
from dataclasses import dataclass
import cv2
import numpy as np
from astra_vision.detector import ArmorDetector, DetectorConfig, LightBar
from astra_vision.types import Detection


@dataclass(frozen=True)
class FittedLightBar(LightBar):
    # Keep the measured footprint separate from the fitted PnP reference axis.
    axis_endpoints: np.ndarray
    core_fill: float
    axis_residual_px: float


def lamp_endpoints(bar):
    fitted=getattr(bar,"axis_endpoints",None)
    if fitted is not None:
        return fitted[0],fitted[1]
    return footprint_endpoints(bar)


def lamp_axis_angle(bar):
    fitted=getattr(bar,"axis_endpoints",None)
    if fitted is None:
        return math.radians(bar.tilt_degrees)
    delta=fitted[1]-fitted[0]
    return math.atan2(float(delta[0]),float(delta[1]))


def footprint_endpoints(bar):
    points=np.asarray(bar.corners,dtype=float)
    index=int(np.argmin(np.linalg.norm(np.roll(points,-1,axis=0)-points,axis=1)))
    a=(points[index]+points[(index+1)%4])/2
    b=(points[(index+2)%4]+points[(index+3)%4])/2
    return (a,b) if a[1]<b[1] else (b,a)


def fit_lamp_axis(bar,mask,origin=(0,0),min_fill=.60):
    """Fit actual component pixels, using the rotated rectangle's geometric area.

    Inspired by FaterYU/rm_auto_aim findLights (MIT; see reference manifest).
    Orthogonal projection avoids slope division and the upstream horizontal-line
    special case. Empty interiors are not filled in as if they were luminous.
    """
    footprint=np.zeros(mask.shape,np.uint8)
    cv2.fillConvexPoly(footprint,np.rint(bar.corners-origin).astype(np.int32),255)
    occupied=(mask>0)&(footprint>0)
    # Rasterizing a tiny rotated box adds a border of pixels to its area. Using
    # that enlarged denominator rejects real thin lamps near the fill threshold.
    # Follow findLights' continuous minAreaRect area and cap pixel-area rounding.
    fill=min(1.,float(occupied.sum()/max(1.,bar.length*bar.width)))
    if fill<min_fill or occupied.sum()<5:
        return None
    yy,xx=np.nonzero(occupied)
    points=np.column_stack((xx,yy)).astype(np.float32)
    line=cv2.fitLine(points,cv2.DIST_L2,0,.01,.01).ravel().astype(float)
    axis=line[:2];axis/=np.linalg.norm(axis)
    if axis[1]<0:axis=-axis
    along=(points-line[2:])@axis
    top=line[2:]+axis*along.min()+origin
    bottom=line[2:]+axis*along.max()+origin
    length=float(along.max()-along.min())
    if length<1 or abs(axis[1])<.1:
        return None
    residual=(points-line[2:])@np.array((-axis[1],axis[0]))
    # Keep the footprint geometry used by pairing stable. The fitted centerline
    # is the PnP/observation reference, not a replacement bounding rectangle.
    return FittedLightBar(bar.center,bar.length,bar.width,
        bar.tilt_degrees,bar.corners,bar.color,
        np.array((top,bottom)),fill,float(np.sqrt(np.mean(residual**2))))


def lamp_quad(left, right, fitted=True):
    endpoints=lamp_endpoints if fitted else footprint_endpoints
    lt,lb=endpoints(left);rt,rb=endpoints(right)
    return np.array((lt,rt,rb,lb))


def lamp_span_ratio(left, right):
    """Spacing perpendicular to the measured lamp axes / mean lamp length.

    Do not use an image-aligned bounding box: camera roll changes its aspect.
    This is an image-space plausibility check, not a physical pose guarantee.
    """
    lt,lb=lamp_endpoints(left);rt,rb=lamp_endpoints(right)
    a=np.asarray(lb)-lt;b=np.asarray(rb)-rt
    lengths=np.array((np.linalg.norm(a),np.linalg.norm(b)))
    if not np.isfinite(lengths).all() or lengths.min()<=0:return float('inf')
    axis=a/lengths[0]+b/lengths[1]
    norm=float(np.linalg.norm(axis))
    if norm<=0:return float('inf')
    axis/=norm
    delta=(np.asarray(rt)+rb-np.asarray(lt)-lb)/2
    return float(abs(axis[0]*delta[1]-axis[1]*delta[0])/lengths.mean())


def lamp_pair_candidates(image, bars, config, evidence=None, *, axis_relative=False):
    """Return all admissible pairs before assigning shared lamps."""
    candidates=[]
    height,width=image.shape[:2]
    for i,left in enumerate(bars):
        for j in range(i+1,len(bars)):
            right=bars[j]
            if left.color!=right.color:
                continue
            length=(left.length+right.length)/2
            ratio=max(left.length,right.length)/min(left.length,right.length)
            tilt=abs(left.tilt_degrees-right.tilt_degrees)
            dy=abs(left.center[1]-right.center[1])/length
            dx=(right.center[0]-left.center[0])/length
            image_aligned=dy<=config.max_vertical_offset_ratio
            if ratio>config.max_length_ratio or tilt>config.max_tilt_difference_degrees:
                continue
            if not config.min_pair_separation_ratio<=dx<=config.max_pair_separation_ratio:
                continue
            if axis_relative:
                # A rolled plate's lamps align perpendicular to their own axes.
                # Perspective can make valid lamps nonparallel. Test alignment
                # over the interval between both measured axes, rather than
                # assuming their average is the only possible plate direction.
                # Fusion retains its old image-vertical gate and score.
                angles=[lamp_axis_angle(bar) for bar in (left,right)]
                angle=sum(angles)/2
                delta=np.array(right.center)-left.center
                dy=abs(float(delta@np.array((math.sin(angle),math.cos(angle)))))/length
                dx=float(delta@np.array((math.cos(angle),-math.sin(angle))))/length
                offsets=[float(delta@np.array((math.sin(a),math.cos(a))))/length for a in angles]
                alignment_error=0. if offsets[0]*offsets[1]<=0 else min(map(abs,offsets))
                axis_aligned=(alignment_error<=config.max_vertical_offset_ratio and
                    config.min_pair_separation_ratio<=dx<=config.max_pair_separation_ratio)
                if not axis_aligned:
                    continue
            elif not image_aligned:
                continue
            if any(b.color==left.color and left.center[0]<b.center[0]<right.center[0]
                   and abs(b.center[1]-(left.center[1]+right.center[1])/2)<length/2
                   and .65<b.length/length<1.5 for b in bars):
                continue
            quality=1/(1+2*(ratio-1)+dy+tilt/30+.1*min(abs(dx-2.4),abs(dx-4.1)))
            points=np.vstack((left.corners,right.corners))
            lo=np.maximum(points.min(axis=0),(0,0));hi=np.minimum(points.max(axis=0),(width-1,height-1))
            if np.any(hi<=lo):
                continue
            center=tuple((np.array(left.center)+right.center)/2)
            detection=Detection(tuple(map(float,(*lo,*hi))),center,left.color,float(quality),"classical")
            # The existing templates were normalized using footprint midpoints.
            # Preserve that crop convention; fitted axes refine final PnP points.
            character=evidence.pair(image,lamp_quad(left,right,fitted=False)) if evidence is not None else {"supported":False}
            candidates.append((i,j,detection,character))
    return candidates


def pair_with_characters(image, bars, config, evidence, *, axis_relative=False, min_quality=0., max_unbacked_span_ratio=None, exclusive_overlap=False, overlap_fraction=.8):
    """Score every admissible pair before consuming lamps; use glyphs to break conflicts."""
    candidates=lamp_pair_candidates(image,bars,config,evidence if min_quality<=0 else None,axis_relative=axis_relative)
    proportion_rejections=[]
    if max_unbacked_span_ratio is not None:
        retained=[]
        for candidate in candidates:
            i,j,detection,_=candidate;span=lamp_span_ratio(bars[i],bars[j])
            if span>max_unbacked_span_ratio:
                proportion_rejections.append(dict(bbox=detection.bbox,span_ratio=span,reason='overwide_without_model_support'))
            else:retained.append(candidate)
        candidates=retained
    if min_quality>0:
        candidates=[(i,j,detection,evidence.pair(image,lamp_quad(bars[i],bars[j],fitted=False))
            if evidence is not None else character) for i,j,detection,character in candidates if detection.score>=min_quality]
    # Weak character support is used only with an already valid lamp pair.
    # Unknown digits remain eligible; no arbitrary OCR label can create a pair.
    ranked=sorted(candidates,key=lambda c:(-int(c[3]["supported"]),-c[2].score,c[0],c[1]))
    # Several luminance levels can measure the same luminous footprint with
    # different axes. Treat near-contained footprints as mutually exclusive,
    # without merging their measured endpoints or changing fusion's ROI path.
    aliases={i:{i} for i in range(len(bars))}
    if exclusive_overlap:
        relevant=sorted({i for c in candidates for i in c[:2]})
        for offset,i in enumerate(relevant):
            a=np.asarray(bars[i].corners,np.float32)
            for j in relevant[offset+1:]:
                if bars[i].color!=bars[j].color:continue
                b=np.asarray(bars[j].corners,np.float32)
                if np.any(a.max(axis=0)<=b.min(axis=0)) or np.any(b.max(axis=0)<=a.min(axis=0)):continue
                area=min(cv2.contourArea(a),cv2.contourArea(b))
                if area>0 and cv2.intersectConvexConvex(a,b)[0]>=overlap_fraction*area:
                    aliases[i].add(j);aliases[j].add(i)
    used=set();selected=[];characters={};suppressed=[]
    for i,j,detection,character in ranked:
        if i in used or j in used or j in aliases[i]:
            suppressed.append(dict(bbox=detection.bbox,reason="shared_lamp",digit=character))
            continue
        used.update(aliases[i]|aliases[j]);selected.append(detection)
        if evidence is not None:
            characters[detection.bbox]=character
    selected.sort(key=lambda d:d.center)
    return selected,characters,dict(candidate_pairs=len(candidates),suppressed_pairs=suppressed,
        digit_supported_pairs=sum(c[3]["supported"] for c in candidates),
        geometry_reference="lamp_axis" if axis_relative else "image_vertical",
        exclusive_overlap=exclusive_overlap,
        overlap_fraction=overlap_fraction,
        proportion_guard=dict(max_unbacked_span_ratio=max_unbacked_span_ratio,rejected=proportion_rejections))


class BrightCoreDetector(ArmorDetector):
    def __init__(self, config=None, brightness=180,min_core_fill=None):
        super().__init__(config or DetectorConfig(min_contour_area=3.,min_bar_length=5.,
            max_bar_aspect=35.,max_bar_tilt_degrees=50.,max_length_ratio=1.8,
            max_vertical_offset_ratio=.7,min_pair_separation_ratio=.5))
        self.brightness=brightness
        self.min_core_fill=self.config.min_rectangularity if min_core_fill is None else min_core_fill

    def find_light_bars(self, image):
        if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or not image.size:
            raise ValueError("expected nonempty uint8 BGR image")
        cfg=self.config
        b,g,r=cv2.split(image)
        value=cv2.max(cv2.max(b,g),r)
        # Extract the luminous core before assigning color. This separates the
        # lamp from dim saturated halos and admits overexposed white centers.
        mask=(value>=self.brightness).astype(np.uint8)*255
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        bars=[]
        height,width=image.shape[:2]
        for contour in contours:
            area=cv2.contourArea(contour)
            if area<cfg.min_contour_area:
                continue
            rect=cv2.minAreaRect(contour)
            corners=cv2.boxPoints(rect)
            edges=np.roll(corners,-1,axis=0)-corners
            lengths=np.linalg.norm(edges,axis=1)
            length,bar_width=float(lengths.max()),float(lengths.min())
            if bar_width<.8 or length<cfg.min_bar_length:
                continue
            edge=edges[int(np.argmax(lengths))]
            tilt=(math.degrees(math.atan2(float(edge[0]),float(edge[1])))+90)%180-90
            if not cfg.min_bar_aspect<=length/bar_width<=cfg.max_bar_aspect or abs(tilt)>cfg.max_bar_tilt_degrees:
                continue
            if area/(length*bar_width)<cfg.min_rectangularity:
                continue
            x,y,w,h=cv2.boundingRect(contour)
            x0,y0=max(0,x-3),max(0,y-3)
            x1,y1=min(width,x+w+3),min(height,y+h+3)
            local=np.zeros((y1-y0,x1-x0),np.uint8)
            shifted=contour-np.array((x0,y0))
            cv2.drawContours(local,[shifted],-1,255,-1)
            local=cv2.dilate(local,np.ones((7,7),np.uint8))>0
            bb,gg,rr=(channel[y0:y1,x0:x1].astype(np.int16) for channel in (b,g,r))
            vv=value[y0:y1,x0:x1]
            blue=local&(bb-rr>=25)&(bb-gg>=12)&(vv>=85)
            red=local&(rr-bb>=25)&(rr-gg>=12)&(vv>=85)
            red_count,blue_count=int(red.sum()),int(blue.sum())
            count=max(red_count,blue_count)
            if count<max(3,int(local.sum()*.05)) or count<2*min(red_count,blue_count):
                continue
            color="red" if red_count>blue_count else "blue"
            if cfg.color not in ("both",color):
                continue
            bar=LightBar(tuple(map(float,rect[0])),length,bar_width,tilt,corners,color)
            component=np.zeros((h,w),np.uint8)
            cv2.drawContours(component,[contour-np.array((x,y))],-1,255,-1)
            component=cv2.bitwise_and(component,mask[y:y+h,x:x+w])
            fitted=fit_lamp_axis(bar,component,np.array((x,y)),self.min_core_fill)
            if fitted is not None and abs(fitted.tilt_degrees)<=cfg.max_bar_tilt_degrees:
                bars.append(fitted)
        return sorted(bars,key=lambda bar:(bar.center[0],bar.center[1],bar.color))


class ColorCoreDetector(ArmorDetector):
    """Retain the color fallback, but measure axes from its actual color pixels."""
    def __init__(self,config=None,min_core_fill=None):
        super().__init__(config)
        self.min_core_fill=self.config.min_rectangularity if min_core_fill is None else min_core_fill

    def find_light_bars(self,image):
        bars=super().find_light_bars(image)
        result=[];cfg=self.config
        for bar in bars:
            x,y,w,h=cv2.boundingRect(bar.corners.astype(np.float32))
            x0,y0=max(0,x-2),max(0,y-2)
            x1,y1=min(image.shape[1],x+w+2),min(image.shape[0],y+h+2)
            crop=image[y0:y1,x0:x1]
            hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
            hue,sat,value=cv2.split(hsv)
            b,g,r=cv2.split(crop.astype(np.int16))
            mask=((hue<=12)|(hue>=168))&(r-b>=cfg.min_channel_difference)&(r-g>=cfg.min_channel_difference) if bar.color=="red" else (
                (hue>=90)&(hue<=135)&(b-r>=cfg.min_channel_difference)&(b-g>=cfg.min_channel_difference))
            mask=(mask&(sat>=cfg.min_saturation)&(value>=cfg.min_value)).astype(np.uint8)*255
            fitted=fit_lamp_axis(bar,mask,np.array((x0,y0)),self.min_core_fill)
            if fitted is not None and abs(fitted.tilt_degrees)<=cfg.max_bar_tilt_degrees:
                result.append(fitted)
        return result
