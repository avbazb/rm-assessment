from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    color: str = "unknown"
    score: float = 1.0
    source: str = "unknown"

    def __post_init__(self) -> None:
        if len(self.bbox) != 4 or len(self.center) != 2:
            raise ValueError("bbox must be xyxy and center must be xy")
        if not all(math.isfinite(v) for v in (*self.bbox, *self.center, self.score)):
            raise ValueError("detection values must be finite")
        if self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError("bbox must have positive width and height")
        if self.color not in ("red", "blue", "unknown"):
            raise ValueError("color must be red, blue or unknown")
        if not 0 <= self.score <= 1:
            raise ValueError("score must be between zero and one")
        if self.source not in ("unknown", "classical", "neural", "fused"):
            raise ValueError("invalid detection source")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class HistoryPoint:
    timestamp: float
    center: tuple[float, float]
    observed: bool


@dataclass(frozen=True)
class TrackSnapshot:
    track_id: int
    timestamp: float
    center: tuple[float, float]
    velocity: tuple[float, float]
    predicted_center: tuple[float, float]
    prediction_seconds: float
    bbox: tuple[float, float, float, float]
    color: str
    observed: bool
    hits: int
    misses: int
    status: str
    history: tuple[HistoryPoint, ...]
    prediction_timestamp: float = 0.0
    prediction_ready: bool = False
    seconds_since_observation: float = 0.0
    position_covariance: tuple = ()
    predicted_position_covariance: tuple = ()
    last_detection_source: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)
