from __future__ import annotations

from dataclasses import dataclass
import math
import cv2
import numpy as np

from .geometry import finite, impact_offset, polygon_array


@dataclass(frozen=True)
class ColorBlobConfig:
    hue_low: int
    hue_high: int
    min_saturation: int = 100
    min_value: int = 80
    min_area: float = 3.0
    max_area: float = 1600.0

    def __post_init__(self):
        if not 0 <= self.hue_low <= self.hue_high <= 179 or not 0 <= self.min_saturation <= 255 or not 0 <= self.min_value <= 255:
            raise ValueError("invalid HSV configuration")
        if not 0 < self.min_area <= self.max_area or not math.isfinite(self.max_area):
            raise ValueError("invalid blob area")


BALL_COLOR = ColorBlobConfig(10, 35)
MARKER_COLOR = ColorBlobConfig(140, 165)


def blob_mask(frame, config):
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("expected BGR image")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (config.hue_low, config.min_saturation, config.min_value),
                       (config.hue_high, 255, 255))


def blobs(frame, config=BALL_COLOR, subtract_mask=None):
    mask = blob_mask(frame, config)
    if subtract_mask is not None:
        if subtract_mask.shape != mask.shape:
            raise ValueError("baseline image dimensions changed")
        mask[subtract_mask > 0] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not config.min_area <= area <= config.max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if not .4 <= w / h <= 2.5:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
        found.append(dict(center=center, radius_px=float(max(w, h) / 2), area=area))
    return found


class ImpactSession:
    """A single image-only shot. Collision messages are not part of this API."""

    def __init__(self, shot_id, polygon, start_timestamp, baseline, mode="marker", timeout=4.0, contact_estimator=None):
        if mode not in ("marker", "unmarked") or not shot_id:
            raise ValueError("invalid shot configuration")
        finite(start_timestamp)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("invalid timeout")
        self.polygon = polygon_array(polygon)
        self.shot_id, self.start, self.mode, self.timeout = str(shot_id), float(start_timestamp), mode, timeout
        self.shape = baseline.shape
        self.baseline_mask = cv2.dilate(blob_mask(baseline, MARKER_COLOR), np.ones((3, 3), np.uint8))
        self.last_timestamp = self.start
        self.marker = None
        self.marker_hits = 0
        self.trajectory = []
        if contact_estimator is not None and mode != "unmarked":
            raise ValueError("ballistic contact estimator requires unmarked mode")
        self.contact_estimator = contact_estimator
        self.result = dict(shot_id=self.shot_id, mode=mode, status="waiting", impact=None,
                           correction=None, evidence=[], automatic=True)

    def update(self, frame, timestamp):
        finite(timestamp)
        if frame.shape != self.shape:
            raise ValueError("image dimensions changed within shot")
        if timestamp <= self.last_timestamp:
            raise ValueError("shot timestamps must strictly increase")
        self.last_timestamp = timestamp
        if self.result["status"] in ("confirmed", "unconfirmed"):
            return dict(self.result)
        if timestamp - self.start > self.timeout:
            self.result.update(status="unconfirmed", evidence=["timeout_without_confirmed_visual_contact"])
            return dict(self.result)
        ball_candidates = blobs(frame)
        if self.trajectory:
            last = self.trajectory[-1]
            dt = timestamp - last["timestamp"]
            gate = min(160., 20. + 1500 * dt)
            ball_candidates = [b for b in ball_candidates if math.dist(b["center"], last["center"]) <= gate]
            if self.contact_estimator is not None and len(ball_candidates)>1:
                ball_candidates.sort(key=lambda b:math.dist(b["center"],last["center"]))
                if math.dist(ball_candidates[1]["center"],last["center"])-math.dist(ball_candidates[0]["center"],last["center"])>8.:
                    ball_candidates=ball_candidates[:1]
        measured = None
        if len(ball_candidates) == 1:
            point = dict(ball_candidates[0], timestamp=float(timestamp))
            measured = point
            self.trajectory.append(point)
            if self.result["status"] != "candidate":
                self.result["status"] = "tracking"
        if self.mode == "marker":
            candidates = blobs(frame, MARKER_COLOR, self.baseline_mask)
            if len(candidates) == 1:
                current = candidates[0]
                self.marker_hits = self.marker_hits + 1 if self.marker and math.dist(current["center"], self.marker["center"]) <= 3 else 1
                self.marker = current
                if self.marker_hits >= 2:
                    sigma = max(1., current["radius_px"] * .25)
                    self.result.update(status="confirmed", impact=current["center"],
                        correction=impact_offset(self.polygon, current["center"], sigma),
                        evidence=["new_visible_marker", "persistent_two_frames"],
                        assumption="simulator_renders_contact_marker; detected_from_RGB_only")
            else:
                self.marker_hits = 0
        elif self.contact_estimator is not None:
            flight=self.contact_estimator.update(measured,float(timestamp))
            self.result["flight"] = dict(flight)
            self.result["forecast"] = flight.get("forecast")
            if flight.get("contact_confirmed"):
                self.result.update(status="confirmed",impact=flight["impact"],
                    contact_kind="observed_two_segment_trajectory",contact_timestamp=flight["contact_timestamp"],
                    confirmation_timestamp=flight["confirmation_timestamp"],
                    correction=impact_offset(self.polygon,flight["impact"],flight["uncertainty_px"]),
                    evidence=flight["evidence"],assumption=flight["assumption"])
            elif flight["status"]=="contact_candidate":
                self.result["status"]="candidate"
            elif flight["status"]=="unconfirmed":
                self.result.update(status="unconfirmed",evidence=flight["evidence"])
        elif len(self.trajectory) >= 3 and self.trajectory[-1]["timestamp"] == timestamp:
            a, b, c = self.trajectory[-3:]
            # A projected reversal is useful evidence but cannot establish 3-D contact.
            v1 = (np.array(b["center"]) - a["center"]) / (b["timestamp"] - a["timestamp"])
            v2 = (np.array(c["center"]) - b["center"]) / (c["timestamp"] - b["timestamp"])
            if np.linalg.norm(v1) > 20 and np.linalg.norm(v2) > 20 and np.dot(v1, v2) < -.25 * np.linalg.norm(v1) * np.linalg.norm(v2):
                sigma = b["radius_px"] + math.dist(a["center"], b["center"]) / 2
                self.result.update(status="candidate", impact=b["center"],
                    correction=impact_offset(self.polygon, b["center"], sigma),
                    evidence=["image_trajectory_reversal"],
                    assumption="candidate_only; monocular_motion_does_not_prove_contact")
        self.result["timestamp"] = float(timestamp)
        self.result["trajectory"] = list(self.trajectory)
        return dict(self.result)


def manual_impact(polygon, point, shot_id="manual", uncertainty_px=2.0):
    return dict(shot_id=shot_id, status="manual", automatic=False, input_method="human_selected_image_point",
                impact=point, correction=impact_offset(polygon, point, uncertainty_px),
                evidence=["manual_pixel_selection"])
