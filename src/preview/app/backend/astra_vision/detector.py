from __future__ import annotations

from dataclasses import dataclass
import math
import cv2
import numpy as np

from .types import Detection


@dataclass(frozen=True)
class DetectorConfig:
    color: str = "both"
    min_saturation: int = 90
    min_value: int = 90
    min_channel_difference: int = 30
    min_contour_area: float = 10.0
    min_bar_length: float = 8.0
    min_bar_aspect: float = 1.8
    max_bar_aspect: float = 25.0
    max_bar_tilt_degrees: float = 40.0
    min_rectangularity: float = 0.4
    max_length_ratio: float = 1.6
    max_tilt_difference_degrees: float = 20.0
    max_vertical_offset_ratio: float = 0.5
    min_pair_separation_ratio: float = 0.7
    max_pair_separation_ratio: float = 5.2

    def __post_init__(self) -> None:
        if self.color not in ("both", "red", "blue"):
            raise ValueError("color must be both, red or blue")
        for name in ("min_saturation", "min_value", "min_channel_difference"):
            if not 0 <= getattr(self, name) <= 255:
                raise ValueError(f"{name} must be in [0, 255]")
        for name in ("min_contour_area", "min_bar_length", "min_bar_aspect", "max_bar_aspect",
                     "max_bar_tilt_degrees", "min_rectangularity", "max_length_ratio",
                     "max_tilt_difference_degrees", "max_vertical_offset_ratio",
                     "min_pair_separation_ratio", "max_pair_separation_ratio"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.min_bar_aspect > self.max_bar_aspect or self.min_pair_separation_ratio > self.max_pair_separation_ratio:
            raise ValueError("minimum limits must not exceed maximum limits")


@dataclass(frozen=True)
class LightBar:
    center: tuple[float, float]
    length: float
    width: float
    tilt_degrees: float
    corners: np.ndarray
    color: str


class ArmorDetector:
    """Find same-color, near-parallel luminous bars. Scores are geometric heuristics."""

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()

    def find_light_bars(self, frame_bgr: np.ndarray) -> list[LightBar]:
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or not frame_bgr.size:
            raise ValueError("frame_bgr must be a nonempty uint8 HxWx3 BGR image")
        cfg = self.config
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        colors = ("red", "blue") if cfg.color == "both" else (cfg.color,)
        bars: list[LightBar] = []
        b, g, r = [channel.astype(np.int16) for channel in cv2.split(frame_bgr)]
        for color in colors:
            if color == "red":
                mask = cv2.inRange(hsv, (0, cfg.min_saturation, cfg.min_value), (12, 255, 255))
                mask |= cv2.inRange(hsv, (168, cfg.min_saturation, cfg.min_value), (179, 255, 255))
                dominant = (r - b >= cfg.min_channel_difference) & (r - g >= cfg.min_channel_difference)
            else:
                mask = cv2.inRange(hsv, (90, cfg.min_saturation, cfg.min_value), (135, 255, 255))
                dominant = (b - r >= cfg.min_channel_difference) & (b - g >= cfg.min_channel_difference)
            mask[~dominant] = 0
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < cfg.min_contour_area:
                    continue
                rect = cv2.minAreaRect(contour)
                corners = cv2.boxPoints(rect)
                edges = np.roll(corners, -1, axis=0) - corners
                lengths = np.linalg.norm(edges, axis=1)
                long_index = int(np.argmax(lengths))
                length, width = float(lengths.max()), float(lengths.min())
                if width < 1 or length < cfg.min_bar_length:
                    continue
                aspect = length / width
                edge = edges[long_index]
                tilt = (math.degrees(math.atan2(float(edge[0]), float(edge[1]))) + 90) % 180 - 90
                if not cfg.min_bar_aspect <= aspect <= cfg.max_bar_aspect or abs(tilt) > cfg.max_bar_tilt_degrees:
                    continue
                if area / (length * width) < cfg.min_rectangularity:
                    continue
                bars.append(LightBar(tuple(map(float, rect[0])), length, width, tilt, corners, color))
        return sorted(bars, key=lambda bar: (bar.center[0], bar.center[1], bar.color))

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        bars, cfg = self.find_light_bars(frame_bgr), self.config
        candidates: list[tuple[float, int, int, Detection]] = []
        height, width = frame_bgr.shape[:2]
        for i, left in enumerate(bars):
            for j in range(i + 1, len(bars)):
                right = bars[j]
                if left.color != right.color:
                    continue
                length = (left.length + right.length) / 2
                length_ratio = max(left.length, right.length) / min(left.length, right.length)
                tilt_delta = abs(left.tilt_degrees - right.tilt_degrees)
                dy = abs(left.center[1] - right.center[1]) / length
                dx = (right.center[0] - left.center[0]) / length
                if length_ratio > cfg.max_length_ratio or tilt_delta > cfg.max_tilt_difference_degrees or dy > cfg.max_vertical_offset_ratio:
                    continue
                if not cfg.min_pair_separation_ratio <= dx <= cfg.max_pair_separation_ratio:
                    continue
                # A matching intervening lamp indicates that this pair spans another target.
                if any(bar.color == left.color and left.center[0] < bar.center[0] < right.center[0]
                       and abs(bar.center[1] - (left.center[1] + right.center[1]) / 2) < length / 2
                       and 0.65 < bar.length / length < 1.5 for bar in bars):
                    continue
                quality = 1 / (1 + 2 * (length_ratio - 1) + dy + tilt_delta / 30
                               + 0.1 * min(abs(dx - 2.4), abs(dx - 4.1)))
                points = np.vstack((left.corners, right.corners))
                x1, y1 = np.maximum(points.min(axis=0), (0, 0))
                x2, y2 = np.minimum(points.max(axis=0), (width - 1, height - 1))
                if x2 <= x1 or y2 <= y1:
                    continue
                center = ((left.center[0] + right.center[0]) / 2, (left.center[1] + right.center[1]) / 2)
                detection = Detection(tuple(map(float, (x1, y1, x2, y2))), center, left.color, float(quality), "classical")
                candidates.append((quality, i, j, detection))
        used: set[int] = set()
        detections: list[Detection] = []
        for _, i, j, detection in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
            if i not in used and j not in used:
                used.update((i, j))
                detections.append(detection)
        return sorted(detections, key=lambda item: item.center)
