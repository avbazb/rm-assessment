"""One model pass plus measured red/blue lamps, without image-bank searches."""
import copy
import os
import time
import numpy as np
from . import fast_classical_v2 as fast_classical


class DetectionPipeline:
    VERSION = fast_classical.VERSION
    MAX_NETWORK_PIXELS = 640000

    def __init__(self, model_path, *, imgsz=0, threads=None):
        self.model_path = model_path
        self.imgsz = imgsz
        self.threads = threads or min(8, os.cpu_count() or 1)
        self.neural = None
        self.last_diagnostics = {}
        self.last_pure_observations = []

    def _model(self):
        if self.neural is None:
            from .onnx_detector import GuidedOnnxDetector
            self.neural = GuidedOnnxDetector(self.model_path, confidence=.10,
                                            imgsz=self.imgsz or 960, threads=self.threads)
        return self.neural

    def _input_size(self, image):
        if self.imgsz:
            return self.imgsz
        # Rectangular input preserves 960 on wide video, while limiting work
        # on tall/square frames. This rule uses shape alone, never detections/GT.
        height, width = image.shape[:2]
        for size in range(960, 607, -32):
            gain = size/max(height,width)
            nh, nw = round(height*gain), round(width*gain)
            pixels = (nh+(size-nh)%32)*(nw+(size-nw)%32)
            if pixels <= self.MAX_NETWORK_PIXELS:
                return size
        return 608

    def warmup(self, mode='fusion'):
        if mode not in ('fusion', 'classical'):
            raise ValueError('检测方式无效')
        started = time.perf_counter()
        if mode == 'fusion':
            # Session loading and kernel preparation happen before "ready".
            neural = self._model()
            for height in (480,360):
                image = np.zeros((height,640,3), dtype=np.uint8)
                neural.imgsz = self._input_size(image)
                neural.detect(image)
        return dict(classical_version=self.VERSION, math_backend='cpu',
                    image_database_used=False, model_ready=self.neural is not None,
                    preparation_ms=(time.perf_counter()-started)*1000)

    def detect(self, image, timestamp, mode):
        if mode == 'classical':
            traditional, diagnostics = fast_classical.detect(image, timestamp)
            self.last_pure_observations = copy.deepcopy(traditional)
            diagnostics.pop('pure_observations', None)
            self.last_diagnostics = dict(classical=diagnostics, model_assistance=False,
                                        image_database_used=False, model_inferences=0)
            return traditional, traditional, []
        started = time.perf_counter()
        neural = self._model()
        neural.imgsz = self._input_size(image)
        models = [dict(row.to_dict(), score_kind='model_confidence')
                  for row in neural.detect(image)]
        model_ms = (time.perf_counter()-started)*1000
        measured, diagnostics = fast_classical.detect(image, timestamp, models=models)
        self.last_pure_observations = diagnostics.pop('pure_observations')
        # Two model proposals may not claim the same measured physical lamp.
        from astra_recovery.attachment import same_axis
        reserved = []
        exclusive = []
        for pair in sorted(measured, key=lambda p: -models[p['fusion_evidence']['model_index']]['score']):
            q = np.asarray(pair['corners'], dtype=float)
            axes = (q[[0,3]], q[[1,2]])
            if any(same_axis(axis, old) for axis in axes for old in reserved):
                continue
            exclusive.append(pair)
            reserved.extend(axes)
        diagnostics['shared_axis_rejected'] = len(measured)-len(exclusive)
        measured = exclusive
        owned = {row['fusion_evidence']['model_index']: row for row in measured}
        observations = []
        for index, model in enumerate(models):
            pair = owned.get(index)
            if pair is None and model['score'] < .30:
                continue
            box = list(model['bbox'])
            evidence = dict(status='joint_confirmed' if pair else 'model_only',
                classical_module=self.VERSION, model_confidence=model['score'],
                model_bbox=box, model_scope='full_frame', model_index=index,
                image_database_used=False, model_assistance=pair is not None,
                classical_geometry_is_not_inferred_from_model=True)
            if pair is not None:
                evidence.update(copy.deepcopy(pair['fusion_evidence']))
                evidence.update(status='joint_confirmed', model_confidence=model['score'],
                                model_bbox=box, model_scope='full_frame')
            observations.append(dict(timestamp=float(timestamp), frame_id='camera_optical',
                bbox=box, center=[(box[0]+box[2])/2, (box[1]+box[3])/2],
                color=pair['color'] if pair else model['color'], score=model['score'],
                score_kind='model_confidence', source='fused' if pair else 'neural',
                corners=copy.deepcopy(pair['corners']) if pair else None,
                measured_lamp_ends=copy.deepcopy(pair['measured_lamp_ends']) if pair else None,
                geometry_valid=bool(pair), geometry_reason='model_owned_measured_lamps' if pair else 'neural_box_only',
                geometry_score=pair['geometry_score'] if pair else None,
                fusion_evidence=evidence))
        self.last_diagnostics = dict(classical=diagnostics, model_ms=model_ms,
            model_input_size=neural.imgsz, model_pixel_budget=self.MAX_NETWORK_PIXELS if not self.imgsz else None,
            model_threads=self.threads, model_inferences=1,
            image_database_used=False, classical_module=self.VERSION,
            fusion_supported=sum(row['source']=='fused' for row in observations),
            model_only=sum(row['source']=='neural' for row in observations))
        return observations, measured, models
