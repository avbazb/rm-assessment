"""CPU-only, image-time processing for the standalone preview."""
import copy
import hashlib
import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

import cv2
import numpy as np
from astra_recovery import CandidateDetector, tracking_measurement
from astra_competition.camera_pose import default_calibration, validate_calibration, observation_pose
from astra_competition.tracking import ArmorTracker
from astra_competition.forecasting import MotionForecaster, forecast
from astra_competition.pose_filter import PoseContinuity
from .detection_pipeline import DetectionPipeline
from .orientation_guard import check_orientation


class Busy(Exception):
    pass


class StaleFrame(ValueError):
    pass


class InferenceTimeout(TimeoutError):
    pass


class Engine:
    def __init__(self, model_path, request_timeout=8., warmup_timeout=30.):
        self.model_path = model_path
        self.detector = None
        self.sessions = {}
        self.lock = threading.Lock()
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='astra-inference')
        self.closed = False
        self.request_timeout = request_timeout
        self.warmup_timeout = warmup_timeout
        self.pending = None
        self.task_started = None
        self.last_error = None
        cv2.setNumThreads(1)

    def health(self):
        pending = self.pending is not None and not self.pending.done()
        age = time.monotonic()-self.task_started if pending and self.task_started is not None else 0.
        return {'status': 'busy' if pending else 'ready', 'inference_busy': pending,
                'inference_age_s': age, 'last_error': self.last_error, 'closed': self.closed,
                'model_ready': self.detector is not None and self.detector.neural is not None}

    def _wait_task(self, function, *args, timeout):
        if self.pending is not None and not self.pending.done():
            raise Busy('上一帧仍在处理，请稍后重新检测；长时间无响应时重启程序')
        future = self.worker.submit(function, *args)
        self.pending = future
        self.task_started = time.monotonic()
        try:
            result = future.result(timeout=timeout)
            self.last_error = None
            return result
        except FutureTimeout as error:
            self.last_error = 'inference_timeout'
            raise InferenceTimeout('检测超时，请重新检测；持续超时请重启程序') from error
        finally:
            if future.done():
                self.pending = None

    def _infer(self, image, timestamp, mode):
        if self.detector is None:
            self.detector = DetectionPipeline(self.model_path)
        return self.detector.detect(image, timestamp, mode)

    def warmup(self, mode='fusion'):
        if not self.lock.acquire(False):
            raise Busy('正在处理画面，请稍后重试')
        try:
            if self.closed:
                raise ValueError('程序已关闭')
            if self.detector is None:
                self.detector = DetectionPipeline(self.model_path)
            details=self._wait_task(self.detector.warmup, mode, timeout=self.warmup_timeout)
            return {'status': 'ready', 'device': 'CPU', **details}
        finally:
            self.lock.release()

    def analyze(self, payload, session, epoch, sequence, timestamp, horizon=.15, calibration=None, mode='fusion',
                frame_format='encoded', frame_width=None, frame_height=None):
        if not self.lock.acquire(False):
            raise Busy('正在处理画面，请稍后重试')
        started = time.perf_counter()
        try:
            if self.closed:
                raise ValueError('程序已关闭')
            if mode not in ('fusion','classical'):
                raise ValueError('检测方式无效')
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', session):
                raise ValueError('会话编号无效')
            if not 0 <= epoch <= 1_000_000_000 or sequence < 0:
                raise ValueError('帧编号无效')
            if not math.isfinite(timestamp) or timestamp < 0 or not math.isfinite(horizon) or not 0 <= horizon <= .5:
                raise ValueError('画面时间或预测时长无效')
            if not payload or len(payload) > 32*1024*1024:
                raise ValueError('图像帧须在 32 MB 以内')
            if frame_format == 'rgba8':
                if any(isinstance(n,bool) or not isinstance(n,int) or n<8 for n in (frame_width,frame_height)):
                    raise ValueError('原始画面尺寸无效')
                if frame_width*frame_height>9_000_000 or len(payload)!=frame_width*frame_height*4:
                    raise ValueError('原始画面尺寸与像素数量不符')
                rgba=np.frombuffer(payload,np.uint8).reshape(frame_height,frame_width,4)
                image=cv2.cvtColor(rgba,cv2.COLOR_RGBA2BGR)
                frame_encoding='rgba8'
            elif frame_format == 'encoded':
                image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                frame_encoding='png' if payload.startswith(b'\x89PNG') else 'other'
            else:
                raise ValueError('画面格式无效')
            if image is None or min(image.shape[:2]) < 8 or image.shape[0]*image.shape[1] > 9_000_000:
                raise ValueError('图像无法解码或超过 900 万像素')
            h, w = image.shape[:2]
            calibration = validate_calibration(calibration) if calibration else default_calibration(w, h)
            # Imported calibration must match the uploaded source resolution.
            # An assumed field of view is regenerated at the actual input size.
            signature = json.dumps([image.shape[:2], calibration, mode], sort_keys=True)
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            now = time.monotonic()
            self.sessions = {k: v for k, v in self.sessions.items() if now-v['touched'] < 600}
            state = self.sessions.get(session)
            if state and (epoch < state['epoch'] or (epoch == state['epoch'] and sequence < state['sequence'])):
                raise StaleFrame('已跳过过期画面')
            if state and epoch == state['epoch'] and signature == state['signature']:
                if timestamp < state['timestamp']:
                    raise StaleFrame('进度已改变，请重置跟踪段')
                if timestamp == state['timestamp']:
                    if digest != state['digest']:
                        raise ValueError('同一帧时间对应不同图像')
                    out = copy.deepcopy(state['response'])
                    out.update(sequence=sequence, processing_ms=0.)
                    out['future_prediction'] = forecast(out['motion_estimates'], timestamp+horizon, (w, h))
                    out['cache_hit'] = True
                    state.update(sequence=sequence, touched=now)
                    return out
            else:
                state = dict(epoch=epoch, signature=signature, tracker=ArmorTracker(),
                             forecaster=MotionForecaster(), poses=PoseContinuity(), orientations={})
            stage_started = time.perf_counter()
            stages = {'decode_and_prepare_ms': (stage_started-started)*1000}
            observations,traditional,models = self._wait_task(self._infer, image, timestamp, mode, timeout=self.request_timeout)
            stages['detection_ms'] = (time.perf_counter()-stage_started)*1000
            stage_started = time.perf_counter()
            measured = [tracking_measurement(o) for o in observations]
            # This guard affects PnP only. Detection admission, score, original
            # two-dimensional measurement and association remain unchanged.
            from .geometry_guard import geometry_for_pose
            pose_observations = [geometry_for_pose(o) for o in observations]
            for obs, safe in zip(observations, pose_observations):
                safe['pose'] = observation_pose(safe, calibration, (w, h))
                safe['pose']['timestamp'] = timestamp
                obs['pose'] = copy.deepcopy(safe['pose'])
                if 'pose_geometry_guard' in safe:
                    obs['pose_geometry_guard'] = copy.deepcopy(safe['pose_geometry_guard'])
                if 'pose_geometry_original' in safe:
                    obs['pose_geometry_original'] = copy.deepcopy(safe['pose_geometry_original'])
            stages['pose_measurement_ms'] = (time.perf_counter()-stage_started)*1000
            stage_started = time.perf_counter()
            tracked = state['tracker'].update(measured, timestamp, image)
            stages['tracking_ms'] = (time.perf_counter()-stage_started)*1000
            stage_started = time.perf_counter()
            tracks = [*tracked['tracks'], *tracked['tentative_tracks']]
            estimates = state['forecaster'].update(tracks, measured, timestamp, (w, h))
            state['poses'].update(tracks, pose_observations, timestamp, calibration)
            orientations = {key: history for key, history in state['orientations'].items()
                            if history and 0 <= timestamp-history[-1]['timestamp'] <= .1+1e-9}
            for track in tracks:
                index = track.get('observation_index')
                if track.get('observed') and index is not None:
                    obs = observations[index]
                    obs.update(track_id=track['track_id'], track_status=track['status'])
                    hist, readout = check_orientation(state['orientations'].get(track['track_id'], []),
                        obs['pose'], track.get('pose', {}), timestamp)
                    orientations[track['track_id']] = hist
                    if 'pose' in track:
                        track['pose']['orientation_readout'] = readout
                        track['pose']['orientation_status'] = readout['status']
                    obs['pose']['orientation_readout'] = readout
                    obs['pose']['orientation_status'] = readout['status']
            state['orientations'] = orientations
            stages['state_and_pose_filter_ms'] = (time.perf_counter()-stage_started)*1000
            out = dict(timestamp=timestamp, sequence=sequence, epoch=epoch, width=w, height=h,
                       device='CPU', observations=observations, **tracked,
                       detection_mode=mode, classical_module=self.detector.VERSION,traditional_observations=traditional,
                       model_detections=models,detection_diagnostics=copy.deepcopy(self.detector.last_diagnostics),
                       motion_estimates=estimates, future_prediction=forecast(estimates, timestamp+horizon, (w, h)),
                       camera_calibration=calibration, processing_ms=(time.perf_counter()-started)*1000,
                       stage_timings=stages,
                       input_frame=dict(pixel_sha256=digest, payload_sha256=hashlib.sha256(payload).hexdigest(),
                                        encoding=frame_encoding),
                       queue_capacity=0, cache_hit=False,
                       neural_model_used=mode=='fusion',math_backend='cpu')
            state.update(sequence=sequence, timestamp=timestamp, digest=digest, touched=now, response=copy.deepcopy(out))
            self.sessions[session] = state
            return out
        finally:
            self.lock.release()

    def prediction(self, session, epoch, timestamp=None, horizon=.15):
        state = self.sessions.get(session)
        if not state or state['epoch'] != epoch:
            raise StaleFrame('没有当前跟踪段的预测状态')
        response = state['response']
        timestamp = response['timestamp'] if timestamp is None else float(timestamp)
        if (not math.isfinite(timestamp) or not math.isfinite(horizon) or not 0 <= horizon <= .5
                or not -1e-6 <= timestamp-response['timestamp'] <= .25+1e-6):
            raise ValueError('预测时间无效或观测已过期')
        result = forecast(response['motion_estimates'], timestamp+horizon, (response['width'], response['height']))
        result.update(session=session, epoch=epoch, sequence=response['sequence'],
                      state_timestamp=response['timestamp'], presentation_timestamp=timestamp,
                      selected_horizon_s=horizon, coordinate_frame='image_pixels',position_unit='px')
        return result

    def close(self):
        with self.lock:
            self.closed = True
            self.worker.shutdown(wait=False, cancel_futures=True)
            self.sessions.clear()
            self.detector = None


def impact_result(observation, point):
    from .geometry_guard import geometry_for_pose
    safe = geometry_for_pose(observation)
    corners = safe.get('corners')
    if not safe.get('geometry_valid') or corners is None:
        raise ValueError('等待完整灯条')
    q = np.asarray(corners, np.float32)
    p = np.asarray(point, float)
    if q.shape != (4, 2) or p.shape != (2,) or not np.isfinite(q).all() or not np.isfinite(p).all():
        raise ValueError('落点或灯条坐标无效')
    if not cv2.isContourConvex(q) or abs(cv2.contourArea(q)) < 1:
        raise ValueError('灯条区域无效')
    target = q.mean(axis=0)
    distance = float(cv2.pointPolygonTest(q, tuple(p.tolist()), True))
    return dict(hit=distance >= 0, state='edge' if abs(distance) <= 2 else 'inside' if distance > 0 else 'outside',
                point=p.tolist(), target=target.tolist(), dx=float(target[0]-p[0]), dy=float(target[1]-p[1]),
                distance_to_edge_px=distance, automatic=False, input_method='human_selected_image_point')
