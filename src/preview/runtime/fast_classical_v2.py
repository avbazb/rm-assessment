"""Bounded model-ROI lamp measurement and independent vectorized pure CV.

Model-owned measurements and independent detections are deliberately separate.
Model rectangles select pixels; every PnP endpoint still comes from those pixels.
"""
from __future__ import annotations
import math
import time
import cv2
import numpy as np
from .fast_classical import _Bar, _image, _iou, _observation, _owner_quality

VERSION = 'fast_color_geometry_roi_v2'
MODEL_ROI_MAX_SIDE = 256
_K3 = np.ones((3, 3), np.uint8)
_K7 = np.ones((7, 7), np.uint8)


def _deduplicate(candidates):
    ordered = sorted(candidates, key=lambda a: (not a.white_core, -a.length))
    n = len(ordered)
    centers = np.empty((n, 2)); axes = np.empty((n, 2))
    lengths = np.empty(n); widths = np.empty(n); colors = np.empty(n, bool)
    selected = []
    for bar in ordered:
        count = len(selected)
        axis = (bar.endpoints[1]-bar.endpoints[0])/bar.length
        if count:
            delta = centers[:count]-bar.center
            along = np.abs(delta@axis)
            normal = np.abs(delta@np.array((-axis[1], axis[0])))
            duplicate = ((colors[:count] == (bar.color == 'blue'))
                         & (np.abs(axes[:count]@axis) >= .94)
                         & (normal <= np.maximum(2., .65*np.maximum(bar.width, widths[:count])))
                         & (along < .4*(lengths[:count]+bar.length)))
            if np.any(duplicate):
                continue
        centers[count] = bar.center; axes[count] = axis
        lengths[count] = bar.length; widths[count] = bar.width; colors[count] = bar.color == 'blue'
        selected.append(bar)
    return sorted(selected, key=lambda a: (a.center[0], a.center[1]))


def _bars(image, chromatic_cores=False):
    height, width = image.shape[:2]
    b, g, r = cv2.split(image)
    value = cv2.max(cv2.max(b, g), r)
    minimum = cv2.min(cv2.min(b, g), r)
    # Cyan edges of an overexposed blue lamp remain useful. They cannot start
    # an unrelated white component without sufficient red/blue contrast.
    seeds = {
        'blue': (cv2.subtract(b, r) >= 30) & (b.astype(np.int16)-g.astype(np.int16) >= -15) & (value >= 85),
        'red': (cv2.subtract(r, b) >= 30) & (cv2.subtract(r, g) >= 18) & (value >= 85),
    }
    seed_masks = {c: m.astype(np.uint8)*255 for c, m in seeds.items()}
    # In a white core, the weakest channel is bright too. max(B,G,R) would
    # instead include the entire saturated blue halo and destroy elongation.
    masks = [('white', cv2.inRange(minimum, 125, 255)),
             ('white', cv2.inRange(minimum, 155, 255)),
             ('white', cv2.inRange(minimum, 215, 255))]
    for color in ('red', 'blue'):
        mask = (seeds[color] & (value >= 150)).astype(np.uint8)*255
        masks.append((color, cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _K3)))
        if chromatic_cores:
            # Saturated lamps can have yellow/cyan cores with a dark opposite
            # channel. min(B,G,R) misses them while a full color blob includes
            # the bloom. The middle channel isolates their measured bright core.
            for threshold in (100, 160):
                masks.append((color, (seeds[color] & (g >= threshold) & (value >= 170)).astype(np.uint8)*255))
    candidates = []
    for origin, mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 2:
                continue
            rect = cv2.minAreaRect(contour)
            rw, rh = rect[1]
            length, thickness = max(rw, rh), min(rw, rh)
            if length < 4 or thickness < .65 or not 1.8 <= length/thickness <= 40:
                continue
            angle = math.radians(rect[2])
            axis_y = math.sin(angle) if rw >= rh else math.cos(angle)
            # Rolled cameras/plates are allowed; nearly horizontal strips are
            # rejected in the image-only first pass.
            if abs(axis_y) < .18 or area/max(length*thickness, 1.) < .30:
                continue
            box = cv2.boxPoints(rect)
            x, y, w, h = cv2.boundingRect(contour)
            component = np.zeros((h, w), np.uint8)
            cv2.drawContours(component, [contour-np.array((x, y))], -1, 255, -1)
            component = cv2.bitwise_and(component, mask[y:y+h, x:x+w])
            yy, xx = np.nonzero(component)
            if len(xx) < 5:
                continue
            points = np.column_stack((xx+x, yy+y)).astype(np.float32)
            line = cv2.fitLine(points, cv2.DIST_L2, 0, .01, .01).ravel()
            axis = line[:2].astype(float)
            axis /= np.linalg.norm(axis)
            if axis[1] < 0:
                axis = -axis
            projected = (points-line[2:])@axis
            top = line[2:]+axis*float(projected.min())
            bottom = line[2:]+axis*float(projected.max())
            actual_length = float(projected.max()-projected.min())
            if actual_length < 3 or abs(float(axis[1])) < .18:
                continue
            # Measure colored pixels around this actual component, not around
            # an arbitrary model rectangle or a synthetic line.
            pad = max(4, min(12, int(thickness*.8)))
            xa, ya, xb, yb = max(0, x-pad), max(0, y-pad), min(width, x+w+pad), min(height, y+h+pad)
            footprint = np.zeros((yb-ya, xb-xa), np.uint8)
            footprint[y-ya:y-ya+h, x-xa:x-xa+w] = component
            near = cv2.dilate(footprint, _K7) > 0
            counts = {c: int(np.count_nonzero(near & m[ya:yb, xa:xb])) for c, m in seeds.items()}
            color = max(counts, key=counts.get)
            other = 'red' if color == 'blue' else 'blue'
            if (origin != 'white' and color != origin) or counts[color] < 3 or counts[color] < 2*counts[other]:
                continue
            cy, cx = np.nonzero(near & seeds[color][ya:yb, xa:xb])
            along = (np.column_stack((cx+xa, cy+ya))-top)@axis
            along = along[(along >= -.1*actual_length) & (along <= 1.1*actual_length)]
            bins = np.bincount(np.clip((along/actual_length*5).astype(int), 0, 4), minlength=5)
            covered = int(np.count_nonzero(bins))
            if covered < 3:
                continue
            fraction = counts[color]/max(int(near.sum()), 1)
            if origin == 'white' and fraction < .045:
                continue
            fill = min(1., len(xx)/max(length*thickness, 1.))
            residual = (points-line[2:])@np.array((-axis[1], axis[0]))
            candidates.append(_Bar(color, (top+bottom)/2, np.array((top, bottom)), box,
                                   actual_length, thickness, fill,
                                   float(np.sqrt(np.mean(residual**2))), fraction, origin == 'white'))
    return _deduplicate(candidates)



def _pairs(bars, shape):
    n = len(bars)
    if n < 2:
        return []
    height, width = shape[:2]
    centers = np.array([b.center for b in bars])
    lengths = np.array([b.length for b in bars])
    axes = np.array([(b.endpoints[1]-b.endpoints[0])/b.length for b in bars])
    colors = np.array([b.color == 'blue' for b in bars])
    ii, jj = np.triu_indices(n, 1)
    ratio = np.maximum(lengths[ii], lengths[jj])/np.minimum(lengths[ii], lengths[jj])
    dot = np.einsum('ij,ij->i', axes[ii], axes[jj])
    pre = (colors[ii] == colors[jj]) & (ratio <= 1.85) & (dot >= math.cos(math.radians(28)))
    ii, jj, ratio, dot = ii[pre], jj[pre], ratio[pre], dot[pre]
    if not len(ii):
        return []
    average = axes[ii]+axes[jj]
    average /= np.linalg.norm(average, axis=1)[:, None]
    normal = np.column_stack((average[:, 1], -average[:, 0]))
    delta = centers[jj]-centers[ii]
    mean_length = (lengths[ii]+lengths[jj])/2
    separation = np.einsum('ij,ij->i', delta, normal)/mean_length
    offset = np.abs(np.einsum('ij,ij->i', delta, average))/mean_length
    tilt = np.degrees(np.arccos(np.clip(dot, -1., 1.)))
    quality = 1/(1+1.7*(ratio-1)+offset+tilt/40+.08*np.minimum(abs(separation-2.4), abs(separation-4.1)))
    pre = (separation >= .4) & (separation <= 5.05) & (offset <= .7) & (quality >= .4)
    ii, jj = ii[pre], jj[pre]
    average, normal = average[pre], normal[pre]
    mean_length, separation, quality = mean_length[pre], separation[pre], quality[pre]
    keep = np.ones(len(ii), bool)
    # Only shortlisted pairs pay the interior-lamp check. Work in fixed blocks
    # so a textured white background cannot create an N x N x N allocation.
    for start in range(0, len(ii), 128):
        end = min(len(ii), start+128)
        d = centers[None, :, :]-centers[ii[start:end], None, :]
        along = np.einsum('bni,bi->bn', d, normal[start:end])/mean_length[start:end, None]
        off = np.abs(np.einsum('bni,bi->bn', d, average[start:end]))/mean_length[start:end, None]
        relative = lengths[None, :]/mean_length[start:end, None]
        interior = ((colors[None, :] == colors[ii[start:end], None]) & (relative >= .6) & (relative <= 1.55)
                    & (along > .15) & (along < separation[start:end, None]-.15) & (off < .55))
        local = np.arange(end-start)
        interior[local, ii[start:end]] = False; interior[local, jj[start:end]] = False
        keep[start:end] = ~np.any(interior, axis=1)
    pairs = []
    for k in np.nonzero(keep)[0]:
        i, j = int(ii[k]), int(jj[k]); left, right = bars[i], bars[j]
        q = np.array((left.endpoints[0], right.endpoints[0], right.endpoints[1], left.endpoints[1]))
        if not cv2.isContourConvex(q.astype(np.float32)) or abs(cv2.contourArea(q.astype(np.float32))) < 3:
            continue
        footprint = np.vstack((left.corners, right.corners))
        pad = np.abs(average[k])*mean_length[k]*.42
        lo = np.maximum(footprint.min(axis=0)-pad, (0., 0.))
        hi = np.minimum(footprint.max(axis=0)+pad, (width-1., height-1.))
        pairs.append(dict(i=i, j=j, color=left.color, bbox=np.r_[lo, hi], corners=q,
                          center=q.mean(axis=0), score=float(quality[k]), span=float(separation[k]),
                          residual=max(left.residual, right.residual), length=float(mean_length[k]),
                          color_fraction=min(left.color_fraction, right.color_fraction)))
    return pairs


def _output(pair, timestamp, scale=1., model_index=None):
    obs = _observation(pair, timestamp, scale, model_index)
    obs['fusion_evidence']['classical_module'] = VERSION
    return obs


def detect(image, timestamp=0., models=None):
    start = time.perf_counter(); image = _image(image)
    height, width = image.shape[:2]
    if models is not None:
        observations = []; details = []; total_bars = total_pairs = 0
        for index, model in enumerate(models):
            box = np.asarray(model.get('bbox'), float)
            if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
                continue
            margin = np.maximum(3., (box[2:]-box[:2])*.15)
            lo = np.maximum(0, np.floor(box[:2]-margin)).astype(int)
            hi = np.minimum((width, height), np.ceil(box[2:]+margin)).astype(int)
            if np.any(hi <= lo):
                continue
            crop = image[lo[1]:hi[1], lo[0]:hi[0]]
            scale = min(1., MODEL_ROI_MAX_SIDE/max(crop.shape[:2]))
            if scale < 1:
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            bars = _bars(crop); pairs = _pairs(bars, crop.shape)
            total_bars += len(bars); total_pairs += len(pairs)
            chosen = None; best = -1.
            for local in pairs:
                pair = dict(local, bbox=local['bbox']/scale+np.tile(lo, 2),
                            corners=local['corners']/scale+lo, center=local['center']/scale+lo,
                            length=local['length']/scale, residual=local['residual']/scale)
                quality = _owner_quality(pair, model, 1.)
                if quality is not None and quality > best:
                    chosen, best = pair, quality
            if chosen is not None:
                observations.append(_output(chosen, timestamp, model_index=index))
            details.append(dict(model_index=index, roi=[*lo.tolist(), *hi.tolist()], scale=scale,
                                light_bars=len(bars), candidate_pairs=len(pairs), measured=chosen is not None))
        return observations, dict(version=VERSION, image_database_used=False, model_inference_used=False,
                                  mode='model_roi', pure_observations=[], pure_not_computed=True,
                                  light_bars=total_bars, candidate_pairs=total_pairs,
                                  measured_count=len(observations), rois=details,
                                  processing_ms=(time.perf_counter()-start)*1000)
    scale = min(1., 960/max(image.shape[:2]))
    working = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
    bars = _bars(working); pairs = _pairs(bars, working.shape)
    observations = []; used = set()
    for pair in sorted(pairs, key=lambda p: -p['score']):
        if pair['i'] in used or pair['j'] in used or pair['color_fraction'] < .45 or pair['length'] < 5:
            continue
        obs = _output(pair, timestamp, scale)
        if any(_iou(obs['bbox'], old['bbox']) >= .65 for old in observations):
            continue
        observations.append(obs); used.update((pair['i'], pair['j']))
    observations.sort(key=lambda o: o['center'][0])
    return observations, dict(version=VERSION, image_database_used=False, model_inference_used=False,
                              mode='independent_full_frame', pure_observations=observations,
                              pure_not_computed=False, light_bars=len(bars), candidate_pairs=len(pairs),
                              measured_count=len(observations), working_scale=scale,
                              processing_ms=(time.perf_counter()-start)*1000)
