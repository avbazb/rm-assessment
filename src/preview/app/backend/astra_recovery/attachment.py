"""Research candidate: attach exclusive lamp measurements to frozen detections.

The original detector owns admission, score, color, order and image-track anchor.
The recovery pass may only attach measured corners to a matching model-only box.
One image-local network cache shares actual inference between the two CV passes.
"""
import copy
import threading
from dataclasses import replace
import numpy as np
from astra_competition.perception import CompetitionDetector
def uses_recovered_core(obs):
    return 'bilateral_neutral_core_v3' in (obs.get('fusion_evidence') or {}).get('extraction', [])
from .pairing import paired_observations


def axes(observation):
    if not observation.get('geometry_valid') or observation.get('corners') is None:
        return []
    q = np.asarray(observation['corners'], dtype=float)
    return [q[[0, 3]], q[[1, 2]]]


def same_axis(a, b):
    """Reserve overlapping physical axes even if exposure changes their length."""
    u = a[1] - a[0]
    v = b[1] - b[0]
    la, lb = np.linalg.norm(u), np.linalg.norm(v)
    if min(la, lb) <= 0:
        return False
    u, v = u / la, v / lb
    if abs(float(u @ v)) < .94:
        return False
    normal = np.array([-u[1], u[0]])
    if abs(float((b.mean(axis=0) - a.mean(axis=0)) @ normal)) > max(2., .08 * min(la, lb)):
        return False
    lo, hi = sorted((b - a[0]) @ u)
    return max(0., min(la, hi) - max(0., lo)) >= .5 * min(la, lb)


def attach_geometry(baseline, recovered):
    """Dictionary API shared by actual inference and exact-pixel cache replay."""
    result = copy.deepcopy(baseline)
    used = [(o['color'], axis) for o in baseline for axis in axes(o)]
    choices = []
    for i, old in enumerate(baseline):
        if old['geometry_valid']:
            continue
        for new in recovered:
            if (tuple(new['bbox']) != tuple(old['bbox']) or new['score'] != old['score']
                    or new['color'] != old['color'] or not new['geometry_valid']
                    or not uses_recovered_core(new)):
                continue
            quality = (new.get('fusion_evidence') or {}).get('joint_quality', 0.)
            choices.append((quality, old['score'], i, new))
    choices.sort(key=lambda x: (-x[0], -x[1], x[2]))
    chosen = set()
    for _, _, i, new in choices:
        if i in chosen:
            continue
        measured = axes(new)
        if any(same_axis(a, b)
               for color, a in used for b in measured):
            continue
        old = baseline[i]
        evidence = copy.deepcopy(new['fusion_evidence'])
        evidence.update(
            lamp_center_px=copy.deepcopy(new['center']),
            tracking_center_source='frozen_detector_center',
            geometry_policy='exclusive_measurement_attachment_v7',
            tracking_measurement_original={k: copy.deepcopy(old[k]) for k in
                ('source', 'corners', 'geometry_valid', 'geometry_score', 'geometry_reason', 'fusion_evidence')})
        result[i].update(source='fused', corners=copy.deepcopy(new['corners']),
            geometry_valid=True, geometry_score=new['geometry_score'],
            geometry_reason='model_owned_bilateral_core', fusion_evidence=evidence)
        chosen.add(i)
        used.extend((new['color'], a) for a in measured)
    return result


def tracking_measurement(observation):
    evidence = observation.get('fusion_evidence') or {}
    original = evidence.get('tracking_measurement_original')
    if evidence.get('geometry_policy') != 'exclusive_measurement_attachment_v7' or original is None:
        return observation
    return dict(observation, **copy.deepcopy(original))


class ImageNetworkCache:
    def __init__(self, network):
        self.network = network
        self.full = None
        self.regions = {}

    def detect(self, frame):
        if self.full is None:
            self.full = self.network.detect(frame)
        return list(self.full)

    def detect_regions(self, frame, regions):
        key = tuple(tuple(r) for r in regions)
        if key not in self.regions:
            verifier = getattr(self.network, 'detect_regions', None)
            self.regions[key] = verifier(frame, regions) if verifier else [[] for _ in regions]
        return [list(items) for items in self.regions[key]]


class FrozenDetectionFusion(CompetitionDetector):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._call_lock=threading.RLock()

    def detect(self, frame, timestamp=0., frame_id='camera_optical'):
        with self._call_lock:
            return self._detect_locked(frame,timestamp,frame_id)

    def _detect_locked(self, frame, timestamp, frame_id):
        if self.mode != 'fusion':
            raise ValueError('v7 is a fusion measurement candidate')
        original,recovered,baseline_diagnostics,recovery_diagnostics=paired_observations(self,frame,timestamp,frame_id)
        rows = attach_geometry([o.to_dict() for o in original], [o.to_dict() for o in recovered])
        self.last_diagnostics = dict(baseline_diagnostics, measurement_recovery=dict(
            policy='exclusive_measurement_attachment_v7',
            added=sum(o['geometry_valid'] and not b.geometry_valid for o, b in zip(rows, original)),
            recovery_pass=recovery_diagnostics))
        keys = ('source', 'corners', 'geometry_valid', 'geometry_score', 'geometry_reason', 'fusion_evidence')
        return [replace(old, **{k: row[k] for k in keys}) for old, row in zip(original, rows)]
