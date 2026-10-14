"""Image-only red/blue lamp measurement, without a template/image database.

White luminous cores and saturated colored components are measured separately:
a blue bloom must not turn the white lamp core into a square contour. A white
component is eligible only with distributed red or blue evidence beside it.
Scores describe image geometry, not calibrated detection probabilities.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

VERSION = 'fast_color_geometry_v1'
_K3 = np.ones((3, 3), np.uint8)
_K7 = np.ones((7, 7), np.uint8)


@dataclass
class _Bar:
    color: str
    center: np.ndarray
    endpoints: np.ndarray
    corners: np.ndarray
    length: float
    width: float
    fill: float
    residual: float
    color_fraction: float
    white_core: bool


def _image(image):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or not image.size):
        raise ValueError('Expected a nonempty uint8 BGR image')
    return image


def _bars(image):
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
    # Prefer the independently measured white core over its broad blue/red
    # bloom, while retaining distinct physical axes.
    selected = []
    for bar in sorted(candidates, key=lambda a: (not a.white_core, -a.length)):
        axis = (bar.endpoints[1]-bar.endpoints[0])/bar.length
        duplicate = False
        for old in selected:
            if old.color != bar.color:
                continue
            delta = old.center-bar.center
            along = abs(float(delta@axis))
            normal = abs(float(delta@np.array((-axis[1], axis[0]))))
            old_axis = (old.endpoints[1]-old.endpoints[0])/old.length
            if (abs(float(axis@old_axis)) >= .94 and normal <= max(2., .65*max(bar.width, old.width))
                    and along < .4*(old.length+bar.length)):
                duplicate = True
                break
        if not duplicate:
            selected.append(bar)
    selected.sort(key=lambda a: (a.center[0], a.center[1]))
    return selected


def _iou(a, b):
    w, h = max(0., min(a[2], b[2])-max(a[0], b[0])), max(0., min(a[3], b[3])-max(a[1], b[1]))
    overlap = w*h
    return overlap/max(1e-9, (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap)


def _pair_candidates(bars, shape):
    height, width = shape[:2]
    candidates = []
    for i, left in enumerate(bars):
        for j in range(i+1, len(bars)):
            right = bars[j]
            if left.color != right.color:
                continue
            length = (left.length+right.length)/2
            ratio = max(left.length, right.length)/min(left.length, right.length)
            a = (left.endpoints[1]-left.endpoints[0])/left.length
            b = (right.endpoints[1]-right.endpoints[0])/right.length
            tilt = math.degrees(math.acos(float(np.clip(a@b, -1., 1.))))
            if ratio > 1.85 or tilt > 28:
                continue
            axis = a+b
            axis /= np.linalg.norm(axis)
            delta = right.center-left.center
            separation = float(delta@np.array((axis[1], -axis[0])))/length
            offset = abs(float(delta@axis))/length
            if not .4 <= separation <= 5.05 or offset > .7:
                continue
            # A comparable interior lamp means the pair could span two plates.
            interior = False
            for k, middle in enumerate(bars):
                if k in (i, j) or middle.color != left.color or not .6 <= middle.length/length <= 1.55:
                    continue
                d = middle.center-left.center
                t = float(d@np.array((axis[1], -axis[0])))/length
                if .15 < t < separation-.15 and abs(float(d@axis))/length < .55:
                    interior = True
                    break
            if interior:
                continue
            quality = 1/(1+1.7*(ratio-1)+offset+tilt/40+.08*min(abs(separation-2.4), abs(separation-4.1)))
            if quality < .40:
                continue
            q = np.array((left.endpoints[0], right.endpoints[0], right.endpoints[1], left.endpoints[1]))
            if not cv2.isContourConvex(q.astype(np.float32)) or abs(cv2.contourArea(q.astype(np.float32))) < 3:
                continue
            footprint = np.vstack((left.corners, right.corners))
            # The dark panel extends beyond the lamp endpoints. Keep this 2D
            # box distinct from the four measured center-line PnP references.
            lo = footprint.min(axis=0)-np.abs(axis)*length*.42
            hi = footprint.max(axis=0)+np.abs(axis)*length*.42
            lo = np.maximum(lo, (0., 0.)); hi = np.minimum(hi, (width-1., height-1.))
            candidates.append(dict(i=i, j=j, color=left.color, bbox=np.r_[lo, hi], corners=q,
                                   center=q.mean(axis=0), score=quality, span=separation,
                                   residual=max(left.residual, right.residual), length=length,
                                   color_fraction=min(left.color_fraction, right.color_fraction)))
    return candidates


def _observation(pair, timestamp, scale, model_index=None):
    q = pair['corners']/scale
    box = (pair['bbox']/scale).tolist()
    quality = float(pair['score'])
    evidence = dict(status='measured_red_blue_pair', classical_module=VERSION,
                    model_assistance=model_index is not None, span_ratio=float(pair['span']),
                    measured_axis_residual_px=float(pair['residual']/scale),
                    color_fraction=float(pair['color_fraction']),
                    traditional_detection=dict(bbox=box, score=quality, origin='pure_color_geometry'))
    if model_index is not None:
        evidence['model_index'] = int(model_index)
    return dict(timestamp=float(timestamp), frame_id='camera_optical', bbox=box,
                center=(pair['center']/scale).tolist(), color=pair['color'], score=quality,
                decision_score=quality, source='classical', score_kind='color_geometry_quality',
                corners=q.tolist(), measured_lamp_ends=[q[[0, 3]].tolist(), q[[1, 2]].tolist()],
                geometry_valid=True, geometry_reason='measured_red_blue_lamp_axes',
                geometry_score=quality, fusion_evidence=evidence)


def _owner_quality(pair, model, scale):
    if model.get('color') not in (pair['color'], 'unknown', None):
        return None
    box = np.asarray(model['bbox'], float)*scale
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        return None
    size = box[2:]-box[:2]
    center = (pair['center']-box[:2])/size
    axes = np.array([pair['corners'][[0, 3]].mean(axis=0), pair['corners'][[1, 2]].mean(axis=0)])
    uv = (axes-box[:2])/size
    # Both measured lamps must belong to the same model and sit on its sides.
    # The model rectangle never supplies an endpoint or a four-point result.
    if not (-.15 <= uv[0, 0] <= .43 and .57 <= uv[1, 0] <= 1.15
            and np.all(uv[:, 1] >= -.1) and np.all(uv[:, 1] <= 1.1)
            and .18 <= center[0] <= .82 and .1 <= center[1] <= .9):
        return None
    return pair['score']/(1+float(np.linalg.norm(center-.5)))


def detect(image, timestamp=0., models=None):
    """Return measured observations and diagnostics in original image pixels.

    models=None returns independently selected pure CV pairs. When models are
    provided, returned pairs have model_index ownership. pure_observations in
    diagnostics comes from exactly the same full-frame extraction, without
    using model boxes to alter that independent result.
    """
    start = time.perf_counter()
    image = _image(image)
    scale = min(1., 960/max(image.shape[:2]))
    working = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
    bars = _bars(working)
    pairs = _pair_candidates(bars, working.shape)
    used = set(); pure = []
    for pair in sorted(pairs, key=lambda p: -p['score']):
        if pair['i'] in used or pair['j'] in used:
            continue
        # White numeral strokes can sit on a dim colored panel. Pure CV needs
        # stronger distributed lamp color than a model-owned measurement.
        if pair['color_fraction'] < .45 or pair['length'] < 5:
            continue
        obs = _observation(pair, timestamp, scale)
        if any(_iou(obs['bbox'], old['bbox']) >= .65 for old in pure):
            continue
        pure.append(obs); used.update((pair['i'], pair['j']))
    pure.sort(key=lambda o: o['center'][0])
    observations = pure
    if models is not None:
        observations = []; used = set(); owners = set(); ranked = []
        for pair in pairs:
            for index, model in enumerate(models):
                quality = _owner_quality(pair, model, scale)
                if quality is not None:
                    ranked.append((quality, index, pair))
        for _, index, pair in sorted(ranked, key=lambda p: (-p[0], p[1])):
            if index in owners or pair['i'] in used or pair['j'] in used:
                continue
            owners.add(index); used.update((pair['i'], pair['j']))
            observations.append(_observation(pair, timestamp, scale, index))
    diagnostics = dict(version=VERSION, math_backend='cpu', image_database_used=False,
                       model_inference_used=False, working_scale=scale, light_bars=len(bars),
                       candidate_pairs=len(pairs), pure_count=len(pure),
                       measured_count=len(observations), pure_observations=pure,
                       processing_ms=(time.perf_counter()-start)*1000)
    return observations, diagnostics
