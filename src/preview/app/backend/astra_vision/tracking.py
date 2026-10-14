from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import numpy as np

from .types import Detection, HistoryPoint, TrackSnapshot


@dataclass(frozen=True)
class TrackerConfig:
    prediction_seconds: float = 0.2
    measurement_std: float = 3.0
    acceleration_std: float = 35.0
    initial_velocity_std: float = 100.0
    mahalanobis_gate: float = 9.21
    max_distance_pixels: float = 120.0
    max_size_ratio: float = 3.0
    max_lost_seconds: float = 0.6
    max_missed_frames: int = 20
    min_hits: int = 2
    history_size: int = 120

    def __post_init__(self) -> None:
        for name in ("measurement_std", "acceleration_std", "initial_velocity_std", "mahalanobis_gate", "max_distance_pixels", "max_size_ratio", "max_lost_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.prediction_seconds) or self.prediction_seconds < 0:
            raise ValueError("prediction_seconds must be finite and nonnegative")
        if self.max_size_ratio < 1:
            raise ValueError("max_size_ratio must be at least 1")
        for name in ("max_missed_frames", "min_hits", "history_size"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) < (0 if name == "max_missed_frames" else 1):
                raise ValueError(f"invalid {name}")


def hungarian_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost rectangular assignment, no SciPy dependency, O(rows^2*cols)."""
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or not np.isfinite(cost).all():
        raise ValueError("cost must be a finite matrix")
    rows, cols = cost.shape
    if not rows or not cols:
        return []
    if rows > cols:
        return [(j, i) for i, j in hungarian_assignment(cost.T)]
    u, v = np.zeros(rows + 1), np.zeros(cols + 1)
    p, way = np.zeros(cols + 1, dtype=int), np.zeros(cols + 1, dtype=int)
    for i in range(1, rows + 1):
        p[0] = i
        minv, used = np.full(cols + 1, np.inf), np.zeros(cols + 1, dtype=bool)
        j0 = 0
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], np.inf, 0
            for j in range(1, cols + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return [(int(p[j] - 1), j - 1) for j in range(1, cols + 1) if p[j]]


class _Track:
    def __init__(self, track_id: int, detection: Detection, timestamp: float, config: TrackerConfig):
        self.track_id, self.color = track_id, detection.color
        self.state = np.array([*detection.center, 0.0, 0.0], dtype=float)
        self.covariance = np.diag([config.measurement_std**2] * 2 + [config.initial_velocity_std**2] * 2)
        self.size = np.array([detection.bbox[2] - detection.bbox[0], detection.bbox[3] - detection.bbox[1]])
        self.bbox_offset = (np.asarray(detection.bbox[:2]) + np.asarray(detection.bbox[2:])) / 2 - detection.center
        self.last_detection_source = detection.source
        self.last_seen, self.hits, self.misses, self.observed = timestamp, 1, 0, True
        self.history: deque[HistoryPoint] = deque(maxlen=config.history_size)

    def predict(self, dt: float, cfg: TrackerConfig) -> None:
        f = np.eye(4)
        f[0, 2] = f[1, 3] = dt
        g = np.array([[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt]])
        self.state = f @ self.state
        self.covariance = f @ self.covariance @ f.T + cfg.acceleration_std**2 * (g @ g.T)
        self.observed = False

    def correct(self, detection: Detection, timestamp: float, cfg: TrackerConfig) -> None:
        h = np.eye(2, 4)
        r = np.eye(2) * cfg.measurement_std**2
        innovation = np.asarray(detection.center) - self.state[:2]
        s = self.covariance[:2, :2] + r
        k = np.linalg.solve(s, self.covariance[:2, :]).T
        self.state += k @ innovation
        ikh = np.eye(4) - k @ h
        self.covariance = ikh @ self.covariance @ ikh.T + k @ r @ k.T
        self.covariance = (self.covariance + self.covariance.T) / 2
        self.size = np.array([detection.bbox[2] - detection.bbox[0], detection.bbox[3] - detection.bbox[1]])
        self.bbox_offset = (np.asarray(detection.bbox[:2]) + np.asarray(detection.bbox[2:])) / 2 - detection.center
        self.last_detection_source = detection.source
        if detection.color != "unknown":
            self.color = detection.color
        self.last_seen, self.misses, self.observed = timestamp, 0, True
        self.hits += 1


class MultiObjectTracker:
    """Constant-velocity 2D Kalman tracks with gated global data association."""

    def __init__(self, config: TrackerConfig | None = None):
        self.config = config or TrackerConfig()
        self._tracks: list[_Track] = []
        self._timestamp: float | None = None
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._timestamp, self._next_id = None, 1

    def update(self, detections: list[Detection], timestamp: float) -> list[TrackSnapshot]:
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be finite and nonnegative")
        if self._timestamp is not None and timestamp <= self._timestamp:
            raise ValueError("timestamps must be strictly increasing")
        if any(not isinstance(detection, Detection) for detection in detections):
            raise TypeError("detections must contain Detection instances")
        cfg = self.config
        dt = timestamp - self._timestamp if self._timestamp is not None else 0
        # Expire before association: a late reappearance must obtain a new ID.
        self._tracks = [track for track in self._tracks if timestamp - track.last_seen <= cfg.max_lost_seconds]
        for track in self._tracks:
            track.predict(dt, cfg)
        self._timestamp = timestamp
        n, m = len(self._tracks), len(detections)
        # Dummy columns explicitly represent unmatched tracks. A dominating
        # penalty prioritizes maximum valid match count before total distance.
        unmatched_cost = (n + 1) * (cfg.mahalanobis_gate + 1)
        invalid_cost = (n + m + 2) * unmatched_cost
        costs = np.full((n, m + n), unmatched_cost)
        if m:
            costs[:, :m] = invalid_cost
        for i, track in enumerate(self._tracks):
            s = track.covariance[:2, :2] + np.eye(2) * cfg.measurement_std**2
            for j, detection in enumerate(detections):
                if track.color != "unknown" and detection.color != "unknown" and track.color != detection.color:
                    continue
                size = np.array([detection.bbox[2] - detection.bbox[0], detection.bbox[3] - detection.bbox[1]])
                if np.max(np.maximum(size / track.size, track.size / size)) > cfg.max_size_ratio:
                    continue
                residual = np.asarray(detection.center) - track.state[:2]
                if np.linalg.norm(residual) > cfg.max_distance_pixels:
                    continue
                distance = float(residual @ np.linalg.solve(s, residual))
                if distance <= cfg.mahalanobis_gate:
                    costs[i, j] = distance
        matched_detections: set[int] = set()
        matched_tracks: set[int] = set()
        for i, j in hungarian_assignment(costs):
            if j < m and costs[i, j] < unmatched_cost:
                self._tracks[i].correct(detections[j], timestamp, cfg)
                matched_detections.add(j)
                matched_tracks.add(i)
        for i, track in enumerate(self._tracks):
            if i not in matched_tracks:
                track.misses += 1
        self._tracks = [track for track in self._tracks if track.misses <= cfg.max_missed_frames]
        for j, detection in enumerate(detections):
            if j not in matched_detections:
                self._tracks.append(_Track(self._next_id, detection, timestamp, cfg))
                self._next_id += 1
        snapshots = []
        for track in self._tracks:
            center = tuple(map(float, track.state[:2]))
            track.history.append(HistoryPoint(timestamp, center, track.observed))
            half_size = track.size / 2
            bbox_center = track.state[:2] + track.bbox_offset
            bbox = tuple(map(float, np.r_[bbox_center - half_size, bbox_center + half_size]))
            horizon = cfg.prediction_seconds
            project = np.array([[1, 0, horizon, 0], [0, 1, 0, horizon]], dtype=float)
            future_covariance = project @ track.covariance @ project.T + np.eye(2) * cfg.acceleration_std**2 * horizon**4 / 4
            covariance_tuple = lambda matrix: tuple(tuple(map(float, row)) for row in matrix)
            snapshots.append(TrackSnapshot(
                track.track_id, timestamp, center, tuple(map(float, track.state[2:])),
                tuple(map(float, track.state[:2] + track.state[2:] * cfg.prediction_seconds)),
                cfg.prediction_seconds, bbox, track.color, track.observed, track.hits, track.misses,
                "confirmed" if track.hits >= cfg.min_hits else "tentative", tuple(track.history),
                timestamp + horizon, track.hits >= max(2, cfg.min_hits), timestamp - track.last_seen,
                covariance_tuple(track.covariance[:2, :2]), covariance_tuple(future_covariance), track.last_detection_source))
        return snapshots
