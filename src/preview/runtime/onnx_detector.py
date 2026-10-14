"""YOLO armor inference without Torch, CUDA, Ultralytics, or network access.

Preprocessing and postprocessing follow the deployed Ultralytics 8.4.166
predictor: stride-32 rectangular full frames, square regional crops, class 0,
confidence > threshold, NMS IoU 0.7, max_det 300, original-pixel coordinates.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from astra_vision.types import Detection


def letterbox(image: np.ndarray, size: int, rect: bool) -> np.ndarray:
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    new_width, new_height = round(width * ratio), round(height * ratio)
    dw, dh = size - new_width, size - new_height
    if rect:
        dw, dh = dw % 32, dh % 32
    left, right = round(dw / 2 - .1), round(dw / 2 + .1)
    top, bottom = round(dh / 2 - .1), round(dh / 2 + .1)
    if (width, height) != (new_width, new_height):
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = .7, max_det: int = 300) -> list[int]:
    order = np.argsort(-scores, kind="stable")[:30000]
    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    keep = []
    while order.size and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)
        remaining = order[1:]
        intersection = np.maximum(np.minimum(boxes[i, 2:], boxes[remaining, 2:]) - np.maximum(boxes[i, :2], boxes[remaining, :2]), 0)
        intersection = intersection[:, 0] * intersection[:, 1]
        union = areas[i] + areas[remaining] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = remaining[iou <= threshold]
    return keep


class GuidedOnnxDetector:
    def __init__(self, weights, device="cpu", confidence=.10, color="both", imgsz=960, *, threads=2):
        if not Path(weights).is_file():
            raise FileNotFoundError(weights)
        if not 0 < confidence < 1 or color not in ("red", "blue", "both"):
            raise ValueError("Invalid confidence or color filter")
        if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz < 32 or imgsz % 32:
            raise ValueError("imgsz must be a positive multiple of 32")
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(weights), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.device = "cpu"
        self.confidence, self.color, self.imgsz = confidence, color, imgsz

    @staticmethod
    def lamp_color(crop):
        if not crop.size:
            return "unknown"
        hue, saturation, value = cv2.split(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))
        bright = (saturation > 90) & (value > 110)
        red = np.count_nonzero(bright & ((hue < 12) | (hue > 168)))
        blue = np.count_nonzero(bright & (hue > 95) & (hue < 135))
        if max(red, blue) < 4 or max(red, blue) < 1.5 * min(red, blue):
            return "unknown"
        return "red" if red > blue else "blue"

    @staticmethod
    def _validate(frame):
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or not frame.size:
            raise ValueError("Expected uint8 BGR image")

    def _predict(self, frame, size, rect):
        self._validate(frame)
        padded = letterbox(frame, size, rect)
        tensor = np.ascontiguousarray(padded[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255)
        prediction = self.session.run([self.output_name], {self.input_name: tensor})[0][0].T
        if prediction.ndim != 2 or prediction.shape[1] < 5:
            raise RuntimeError("Unsupported armor model output")
        classes = prediction[:, 4:].argmax(axis=1)
        scores = prediction[:, 4:].max(axis=1)
        valid = (classes == 0) & (scores > self.confidence) & np.isfinite(prediction).all(axis=1)
        xywh, scores = prediction[valid, :4], scores[valid]
        boxes = np.empty_like(xywh)
        boxes[:, :2] = xywh[:, :2] - xywh[:, 2:] / np.float32(2)
        boxes[:, 2:] = xywh[:, :2] + xywh[:, 2:] / np.float32(2)
        chosen = nms(boxes, scores)
        boxes, scores = boxes[chosen], scores[chosen]
        height, width = frame.shape[:2]
        gain = min(padded.shape[0] / height, padded.shape[1] / width)
        padx = round((padded.shape[1] - round(width * gain)) / 2 - .1)
        pady = round((padded.shape[0] - round(height * gain)) / 2 - .1)
        boxes[:, [0, 2]] -= padx
        boxes[:, [1, 3]] -= pady
        boxes /= gain
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
        return boxes, scores

    def _detections(self, frame, size, rect, offset=(0, 0), filter_color=True):
        boxes, scores = self._predict(frame, size, rect)
        detections = []
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(float, box)
            if x2 <= x1 or y2 <= y1 or not 0 <= score <= 1:
                continue
            color = self.lamp_color(frame[int(y1):math.ceil(y2), int(x1):math.ceil(x2)])
            if filter_color and self.color != "both" and color != self.color:
                continue
            bounds = (x1 + offset[0], y1 + offset[1], x2 + offset[0], y2 + offset[1])
            detections.append(Detection(bounds, ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2), color, float(score), "neural"))
        return detections

    def detect(self, frame_bgr):
        return self._detections(frame_bgr, self.imgsz, True)

    def detect_regions(self, frame, regions):
        self._validate(frame)
        # Serial bounded crops avoid large RAM peaks on a CPU-only computer.
        # The original regional predictor uses rect=False, imgsz=640.
        output = []
        for x1, y1, x2, y2 in regions:
            if not 0 <= x1 < x2 <= frame.shape[1] or not 0 <= y1 < y2 <= frame.shape[0]:
                raise ValueError("Invalid regional crop")
            output.append(self._detections(frame[y1:y2, x1:x2], 640, False, (x1, y1), False))
        return output
