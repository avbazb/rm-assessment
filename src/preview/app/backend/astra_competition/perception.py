from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
import math
import cv2
import numpy as np

from astra_vision.detector import ArmorDetector, DetectorConfig
from .geometry import finite, polygon_array, polygon_center
from .classical import BrightCoreDetector, ColorCoreDetector, pair_with_characters, lamp_endpoints
from .color_lamps import ColorFirstDetector
from .guided_fusion import DEFAULT_REGIONAL_POLICY, REGIONAL_POLICY_VERSION


class SeededLampDetector(ArmorDetector):
    """Recover a saturated lamp's white core from nearby colored edge pixels.

    Geometry/pairing still uses the existing detector. Uncolored white regions
    without a red/blue seed are never admitted. Shared white growth is rejected.
    """
    def find_light_bars(self, frame_bgr):
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("expected uint8 BGR image")
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        b, g, r = [c.astype(np.int16) for c in cv2.split(frame_bgr)]
        blue = (hue >= 90) & (hue <= 135) & (saturation >= 50) & (value >= 90) & (b-r >= 24) & (b-g >= 12)
        red = ((hue <= 12) | (hue >= 168)) & (saturation >= 50) & (value >= 90) & (r-b >= 24) & (r-g >= 12)
        white = (saturation < 100) & (value >= 180)
        masks = []
        for seed in (red, blue):
            nearby = cv2.dilate(seed.astype(np.uint8), np.ones((7,7),np.uint8)).astype(bool)
            masks.append(seed | (nearby & white))
        conflict = masks[0] & masks[1]
        classified = np.zeros_like(frame_bgr)
        classified[masks[0] & ~conflict] = (0,0,255)
        classified[masks[1] & ~conflict] = (255,0,0)
        return super().find_light_bars(classified)


@dataclass(frozen=True)
class ArmorObservation:
    timestamp: float
    frame_id: str
    bbox: tuple
    center: tuple
    color: str
    score: float
    source: str
    corners: tuple | None
    geometry_valid: bool
    geometry_reason: str
    geometry_score: float | None
    polygon_scope: str = "lamp_axis_reference; physical_hit_surface_requires_dimensions"
    digit: dict | None = None
    fusion_evidence: dict | None = None

    def to_dict(self):
        return dict(asdict(self),score_kind="geometric_quality" if self.source=="classical" else "model_confidence")


def corners_for_detection(detection, bars, frame_shape):
    options = []
    height, width = frame_shape[:2]
    scale = max(math.hypot(detection.bbox[2] - detection.bbox[0], detection.bbox[3] - detection.bbox[1]), 1)
    # Necessary condition for the existing .05 bbox/center error bound: each
    # lamp's footprint must lie within the candidate bbox plus that tolerance.
    # Prune impossible combinations before enumerating all same-color lamps.
    bounds=np.asarray(detection.bbox);tolerance=.05*scale
    matching=[b for b in bars if b.color==detection.color
        and np.all(b.corners.min(axis=0)>=bounds[:2]-tolerance)
        and np.all(b.corners.max(axis=0)<=bounds[2:]+tolerance)]
    for left, right in combinations(matching, 2):
        if left.center[0] > right.center[0]:
            left, right = right, left
        points = np.vstack((left.corners, right.corners))
        lo = np.maximum(points.min(axis=0), (0, 0))
        hi = np.minimum(points.max(axis=0), (width - 1, height - 1))
        bbox = np.concatenate((lo, hi))
        center = (np.asarray(left.center) + right.center) / 2
        error = np.linalg.norm(bbox - detection.bbox) / scale + np.linalg.norm(center - detection.center) / scale
        if error > .05:
            continue
        lt, lb = lamp_endpoints(left)
        rt, rb = lamp_endpoints(right)
        q = np.array((lt, rt, rb, lb))
        try:
            polygon_array(q)
        except ValueError:
            continue
        if np.any(q[:, 0] < 0) or np.any(q[:, 0] >= width) or np.any(q[:, 1] < 0) or np.any(q[:, 1] >= height):
            continue
        options.append((float(error), q))
    options.sort(key=lambda item: item[0])
    if not options:
        return None, "no_complete_lamp_pair"
    if len(options) > 1 and options[1][0] - options[0][0] < .005:
        return None, "ambiguous_lamp_pair"
    return tuple(tuple(map(float, p)) for p in options[0][1]), "valid_lamp_reference"


class CompetitionDetector:
    def __init__(self, mode="classical", weights=None, device="cpu", imgsz=960, confidence=.25,
                 config=None, neural_detector=None, lamp_method="color", digit_method="auto", digit_threshold=.25,
                 *, regional_policy=DEFAULT_REGIONAL_POLICY):
        if mode not in ("classical", "fusion", "neural"):
            raise ValueError("invalid detector mode")
        if regional_policy not in ("off",REGIONAL_POLICY_VERSION):
            raise ValueError("invalid regional fusion policy")
        self._regional_policy=regional_policy if mode=="fusion" else "off"
        self.mode = mode
        self.confidence = confidence
        if lamp_method not in ("legacy", "seeded", "bright", "color"):
            raise ValueError("invalid lamp extraction method")
        self.lamp_method = lamp_method
        detector_class={"legacy":ArmorDetector,"seeded":SeededLampDetector,"bright":BrightCoreDetector,"color":ColorFirstDetector}[lamp_method]
        self.classical = (detector_class(config,recover_open_core=mode=="fusion")
            if lamp_method=="color" else detector_class(config))
        self.legacy_fallback = ColorCoreDetector() if lamp_method=="bright" and config is None else None
        self.last_diagnostics = {}
        if digit_method=="auto":
            digit_method="template" if mode=="classical" else "off"
        if digit_method not in ("off","template","model") or (mode=="classical" and digit_method=="model"):
            raise ValueError("pure classical mode cannot use a digit model")
        # Model-backed detection uses model/color/geometry evidence only. Even
        # legacy callers requesting a digit backend cannot re-enable its gates.
        if mode!="classical":
            digit_method="off"
        self.digits=None
        if digit_method!="off":
            from .digits import DigitEvidence
            self.digits=DigitEvidence(digit_method,digit_threshold)
        self.neural = None
        if mode != "classical":
            if neural_detector is None:
                if weights is None:
                    raise ValueError("model weights required")
                # Use the same shipped CPU backend as the preview; no Torch dependency.
                from runtime.onnx_detector import GuidedOnnxDetector
                from .guided_fusion import PROPOSAL_CONFIDENCE
                proposal_confidence=min(confidence,PROPOSAL_CONFIDENCE) if mode=="fusion" else confidence
                onnx_weights = Path(weights)
                if onnx_weights.suffix.lower() == '.pt':
                    onnx_weights = onnx_weights.with_suffix('.onnx')
                if onnx_weights.suffix.lower() != '.onnx' or not onnx_weights.is_file():
                    raise ValueError('需要 ONNX 模型；PT 权重需提供同名 ONNX 文件')
                neural_detector = GuidedOnnxDetector(onnx_weights, device, proposal_confidence, "both", imgsz)
            self.neural = neural_detector

    @property
    def regional_policy(self):
        """Construction-only policy; rebuild detector/session to change it."""
        return self._regional_policy

    def detect(self, frame, timestamp=0.0, frame_id="camera_optical"):
        finite(timestamp, "image timestamp")
        if self.mode=="fusion":
            from .guided_fusion import guided_fuse,verify_classical_regions,filter_cross_global_regions
            proposals=self.neural.detect(frame)
            if self.regional_policy=="off":
                regional,regional_diagnostics=verify_classical_regions(frame,proposals,self.classical,self.neural,self.digits)
            else:
                provenance={}
                regional,original_diagnostics=verify_classical_regions(
                    frame,proposals,self.classical,self.neural,self.digits,provenance=provenance)
                regional,policy_diagnostics=filter_cross_global_regions(proposals,regional,provenance,self.confidence)
                regional_diagnostics=dict(original_diagnostics,**policy_diagnostics)
            global_count=len(proposals)
            proposals=[*proposals,*regional]
            fusion_options={"color_first":True} if self.lamp_method=="color" else {}
            records,self.last_diagnostics=guided_fuse(frame,proposals,self.classical.config,self.digits,self.confidence,**fusion_options)
            for row in records:
                row["evidence"]["model_scope"]="cv_proposed_region" if row["detection"].bbox in {p.bbox for p in regional} else "full_frame"
            self.last_diagnostics.update(global_model_proposals=global_count,regional_verification=regional_diagnostics)
            self.last_diagnostics.update(lamp_method="model_guided_color_roi" if self.lamp_method=="color" else "model_guided_roi",digit_method=self.digits.method if self.digits else "off")
            return [ArmorObservation(float(timestamp),frame_id,row["detection"].bbox,row["detection"].center,
                row["detection"].color,row["detection"].score,row["detection"].source,row["corners"],row["corners"] is not None,
                "model_guided_lamp_reference" if row["corners"] else "neural_box_only",row["geometry_score"],
                digit=row["digit"],fusion_evidence=row["evidence"]) for row in records]
        bars = self.classical.find_light_bars(frame)
        digits = self.digits.for_frame(frame) if self.digits is not None else None
        characters={}
        pairing={}
        pair_options={}
        if getattr(self.classical,"min_pair_quality",0)>0:
            pair_options["min_quality"]=self.classical.min_pair_quality
        if self.mode=="classical" and hasattr(self.classical,"max_unbacked_span_ratio"):
            pair_options["max_unbacked_span_ratio"]=getattr(self.classical,"max_classical_span_ratio",self.classical.max_unbacked_span_ratio)
            pair_options["exclusive_overlap"]=getattr(self.classical,"exclusive_overlap",False)
        classical,characters,pairing=pair_with_characters(frame,bars,self.classical.config,digits,
            axis_relative=self.mode=="classical",**pair_options)
        fallback_used=False
        if not classical and self.legacy_fallback is not None:
            fallback_bars=self.legacy_fallback.find_light_bars(frame)
            classical,characters,pairing=pair_with_characters(frame,fallback_bars,self.legacy_fallback.config,digits,
                axis_relative=self.mode=="classical")
            if classical:
                bars=fallback_bars
                fallback_used=True
        if self.mode == "classical":
            detections = classical
            references = {i: d for i, d in enumerate(classical)}
        else:
            neural = self.neural.detect(frame)
            detections, references = neural, {}
        output = []
        for index, detection in enumerate(detections):
            reference = references.get(index)
            corners, reason = corners_for_detection(reference, bars, frame.shape) if reference else (None, "neural_box_only")
            center = polygon_center(corners) if corners else detection.center
            digit=characters.get(reference.bbox) if reference is not None else None
            if digit is None and digits is not None:
                digit=digits.box(frame,detection.bbox)
            output.append(ArmorObservation(float(timestamp), frame_id, detection.bbox, center, detection.color,
                                           detection.score, detection.source, corners, corners is not None, reason,
                                           reference.score if reference else None,digit=digit))
        self.last_diagnostics=dict(lamp_method=self.lamp_method,legacy_fallback_used=fallback_used,red_bars=sum(b.color=="red" for b in bars),
            fitted_axis_bars=sum(hasattr(b,"axis_endpoints") for b in bars),
            blue_bars=sum(b.color=="blue" for b in bars),reason="detected" if output else
            ("no_valid_same_color_pair" if bars else "no_colored_lamp_cores"),
            classical_fallback_policy=None,
            digit_method=self.digits.method if self.digits else "off",pairing=pairing)
        return output


def verified_weights(root):
    return verified_weights_identity(root)[0]


def verified_weights_identity(root):
    """Keep the validated bytes' identity for callers that must report it."""
    import hashlib
    import json
    root = Path(root).resolve()
    manifest = json.loads((root / 'models/manifest.json').read_text(encoding='utf-8'))
    contract = next(row for row in manifest['files'] if row['path'] == 'models/recommended.pt')
    weights = (root / contract['path']).resolve()
    if not weights.is_relative_to(root):
        raise ValueError('模型清单路径无效')
    with weights.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    identity = dict(bytes=weights.stat().st_size, sha256=checksum)
    if identity["bytes"] != contract["bytes"] or identity["sha256"] != contract["sha256"]:
        raise ValueError("recommended model failed release verification")
    return weights, dict(status="verified", name=weights.name, **identity)
