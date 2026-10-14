"""Pixel-space fusion; classical scores are evidence, not calibrated probabilities."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Protocol

import numpy as np

from .detector import ArmorDetector
from .tracking import hungarian_assignment
from .types import Detection


class Detector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]: ...


@dataclass(frozen=True)
class FusionConfig:
    min_iou: float = 0.25
    max_center_distance_ratio: float = 0.35
    classical_center_weight: float = 0.5
    allow_classical_fallback: bool = False

    def __post_init__(self):
        for name in ("min_iou", "max_center_distance_ratio", "classical_center_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0,1]")
        if self.min_iou == 0 or self.max_center_distance_ratio == 0:
            raise ValueError("fusion gates must be positive")
        if not isinstance(self.allow_classical_fallback, bool):
            raise ValueError("allow_classical_fallback must be bool")


@dataclass(frozen=True)
class FusionResult:
    detections: tuple[Detection, ...]
    neural: tuple[Detection, ...]
    classical: tuple[Detection, ...]
    matches: tuple[dict, ...]
    discarded_classical_indices: tuple[int, ...]

    def to_dict(self):
        return asdict(self)


def box_iou(a, b) -> float:
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - overlap
    return float(overlap / union)


class FusionArmorDetector:
    """One neural measurement per target, optionally refined by a light-bar pair.

    Unlit neural candidates survive without a classical pair. The physical-box
    extent and score remain neural; a score average would mix incompatible scales.
    """

    def __init__(self, neural: Detector, classical: Detector | None = None,
                 config: FusionConfig | None = None):
        self.neural = neural
        self.classical = classical or ArmorDetector()
        self.config = config or FusionConfig()

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        return list(self.detect_with_evidence(frame_bgr).detections)

    def detect_with_evidence(self, frame_bgr: np.ndarray) -> FusionResult:
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or not frame_bgr.size:
            raise ValueError("frame_bgr must be a nonempty uint8 HxWx3 BGR image")
        return self.fuse(self.neural.detect(frame_bgr), self.classical.detect(frame_bgr))

    def fuse(self, neural: list[Detection], classical: list[Detection]) -> FusionResult:
        if any(not isinstance(item, Detection) for item in [*neural, *classical]):
            raise TypeError("detectors must return Detection instances")
        cfg = self.config
        n, m = len(neural), len(classical)
        unmatched = (n + 1) * 3.0
        costs = np.full((n, m + n), unmatched)
        costs[:, :m] = (n + m + 2) * unmatched
        geometry = {}
        for i, nd in enumerate(neural):
            for j, cd in enumerate(classical):
                if nd.color != "unknown" and cd.color != "unknown" and nd.color != cd.color:
                    continue
                x1, y1, x2, y2 = nd.bbox
                if not (x1 <= cd.center[0] <= x2 and y1 <= cd.center[1] <= y2):
                    continue
                iou = box_iou(nd.bbox, cd.bbox)
                distance_ratio = math.dist(nd.center, cd.center) / math.hypot(x2-x1, y2-y1)
                if iou >= cfg.min_iou and distance_ratio <= cfg.max_center_distance_ratio:
                    costs[i, j] = 1 - iou + distance_ratio
                    geometry[i, j] = (iou, distance_ratio)
        output = [replace(d, source="neural") for d in neural]
        matched_classical = set()
        matches = []
        for i, j in hungarian_assignment(costs):
            if j >= m or costs[i, j] >= unmatched:
                continue
            nd, cd = neural[i], classical[j]
            weight = cfg.classical_center_weight
            center = tuple((1-weight)*a + weight*b for a, b in zip(nd.center, cd.center))
            output[i] = Detection(nd.bbox, center, cd.color if cd.color != "unknown" else nd.color, nd.score, "fused")
            matched_classical.add(j)
            iou, ratio = geometry[i, j]
            matches.append(dict(neural_index=i, classical_index=j, iou=iou,
                                center_distance_ratio=ratio, neural_score=nd.score,
                                classical_geometry_score=cd.score))
        discarded = []
        for j, cd in enumerate(classical):
            if j in matched_classical:
                continue
            # Do not create a second ID for a conflicting/duplicate measurement.
            overlaps_neural = any(
                box_iou(cd.bbox, nd.bbox) >= cfg.min_iou
                or (nd.bbox[0] <= cd.center[0] <= nd.bbox[2] and nd.bbox[1] <= cd.center[1] <= nd.bbox[3])
                for nd in neural)
            if cfg.allow_classical_fallback and not overlaps_neural:
                output.append(replace(cd, source="classical"))
            else:
                discarded.append(j)
        return FusionResult(tuple(output), tuple(neural), tuple(classical), tuple(matches), tuple(discarded))
