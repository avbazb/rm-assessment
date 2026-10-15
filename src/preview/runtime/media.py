"""Local media library. Uploaded originals and playable copies remain local."""
import hashlib
import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import uuid
from fractions import Fraction

import cv2
import numpy as np


class ImportBusy(RuntimeError):
    pass


class MediaLibrary:
    def __init__(self, assets, data):
        self.assets = Path(assets)
        self.data = Path(data)
        self.uploads = self.data/'uploads'
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.jobs_lock = threading.Lock()
        self.jobs = set()
        self.import_lock = threading.Lock()
        self.import_jobs = {}
        self.import_threads = set()
        self.closed = False
        self.entries = {}
        self.files = {}
        self.thumbs = {}
        for row in json.loads((self.assets/'samples/catalog.json').read_text('utf-8')):
            self._register(row, self.assets/'samples')
        for manifest in self.uploads.glob('*/item.json'):
            try:
                self._register(json.loads(manifest.read_text('utf-8')), manifest.parent)
            except (ValueError, KeyError, OSError):
                continue

    def _register(self, row, base):
        item = dict(row)
        media = (base/item.pop('file')).resolve()
        thumb = (base/item.pop('thumb')).resolve()
        if base.resolve() not in media.parents or base.resolve() not in thumb.parents:
            raise ValueError('素材路径无效')
        if not media.is_file() or not thumb.is_file():
            raise ValueError('素材文件缺失')
        key = item['id']
        item.update(url=f'/media/{key}', thumbnail=f'/thumb/{key}.jpg')
        self.entries[key] = item
        self.files[key] = media
        self.thumbs[key] = thumb

    def listing(self):
        with self.lock:
            return {'items': list(self.entries.values())}

    def executable(self, name):
        if name not in ('ffmpeg', 'ffprobe'):
            raise ValueError('视频处理组件无效')
        bundled = self.assets/'bin'/f'{name}.exe'
        if bundled.is_file():
            return str(bundled)
        system = shutil.which(name)
        if system:
            return system
        raise ValueError('未找到 FFmpeg/FFprobe，请安装并加入 PATH')

    def run(self, args, timeout=180, progress=None):
        with self.jobs_lock:
            if self.closed:
                raise ValueError('程序已关闭')
            process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding='utf-8', errors='replace', creationflags=0x08000000 if os.name=='nt' else 0)
            self.jobs.add(process)
        stdout_parts, stderr_parts = [], []
        def read_output():
            for line in process.stdout:
                stdout_parts.append(line)
                if progress and '=' in line:
                    key, value = line.strip().split('=', 1)
                    if key == 'out_time_us':
                        try:
                            progress(float(value)/1_000_000)
                        except (ValueError, OverflowError):
                            pass
        def read_errors():
            for line in process.stderr:
                # Diagnostic tails are bounded even for malformed long videos.
                stderr_parts.append(line)
                if len(stderr_parts) > 100:
                    del stderr_parts[:50]
        readers = [threading.Thread(target=read_output, daemon=True),
                   threading.Thread(target=read_errors, daemon=True)]
        for reader in readers:
            reader.start()
        try:
            process.wait(timeout=timeout)
            for reader in readers:
                reader.join(2)
            if process.returncode:
                detail = ''.join(stderr_parts).strip()[-600:]
                raise ValueError('素材无法解码：'+detail if detail else '素材无法解码，请检查文件是否完整')
            return ''.join(stdout_parts)
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.wait()
            for reader in readers:
                reader.join(2)
            raise ValueError('视频处理超时') from error
        finally:
            for stream in (process.stdout, process.stderr):
                stream.close()
            with self.jobs_lock:
                self.jobs.discard(process)

    def close(self):
        with self.jobs_lock:
            self.closed = True
            processes = list(self.jobs)
            imports = list(self.import_threads)
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for thread in imports:
            if thread is not threading.current_thread():
                thread.join(3)

    def _prepare_file(self, temporary, filename):
        suffix = Path(filename).suffix.lower()
        images = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
        videos = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.wmv', '.mpeg', '.mpg'}
        if suffix not in images|videos:
            raise ValueError('支持 JPG、PNG、BMP、WebP、TIFF 和常见视频格式')
        key = 'upload-'+uuid.uuid4().hex
        folder = self.uploads/key
        folder.mkdir()
        original = folder/('original'+suffix)
        shutil.move(str(temporary), str(original))
        return key, folder, original, suffix, images

    def _job_update(self, job_id, **values):
        if job_id is not None:
            with self.jobs_lock:
                self.import_jobs[job_id].update(values, updated_at=time.time())

    @staticmethod
    def _source_hash(temporary, source_sha256):
        if source_sha256 is not None:
            return source_sha256
        with Path(temporary).open('rb') as source:
            return hashlib.file_digest(source, 'sha256').hexdigest()

    def _reuse_original(self, temporary, digest):
        """Reuse verified content, never a filename or modification-time guess."""
        size = Path(temporary).stat().st_size
        with self.lock:
            candidates = [(key, dict(item), self.files[key]) for key, item in self.entries.items()
                          if item.get('origin') == 'upload']
        for key, item, media in candidates:
            if not media.is_file() or not self.thumbs[key].is_file():
                continue
            known = item.get('original_sha256')
            if known is None:
                # Upgrade old libraries lazily; do not scan every video at startup.
                originals = list(media.parent.glob('original.*'))
                if len(originals) != 1 or originals[0].stat().st_size != size:
                    continue
                with originals[0].open('rb') as source:
                    known = hashlib.file_digest(source, 'sha256').hexdigest()
                manifest = media.parent/'item.json'
                row = json.loads(manifest.read_text('utf-8'))
                row.update(original_sha256=known, original_bytes=size)
                pending = media.parent/'item.json.tmp'
                pending.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding='utf-8')
                pending.replace(manifest)
                with self.lock:
                    self.entries[key].update(original_sha256=known, original_bytes=size)
                    item = dict(self.entries[key])
            if known == digest and item.get('original_bytes') == size:
                return {**item, 'import_reused': True}
        return None

    def begin_import(self, temporary, filename, source_sha256=None):
        if not self.import_lock.acquire(False):
            raise ImportBusy('正在导入素材，请稍后重试')
        prepared = None
        try:
            with self.jobs_lock:
                if self.closed:
                    raise ValueError('程序已关闭')
            digest = self._source_hash(temporary, source_sha256)
            reused = self._reuse_original(temporary, digest)
            if reused is not None:
                job = dict(id=uuid.uuid4().hex, status='ready', progress=100., stage='complete',
                           filename=Path(filename).name[:180], created_at=time.time(),
                           updated_at=time.time(), item=reused)
                with self.jobs_lock:
                    self.import_jobs[job['id']] = job
                self.import_lock.release()
                return copy.deepcopy(job)
            # Own the uploaded original before the HTTP request returns.
            prepared = self._prepare_file(temporary, filename)
            job_id = uuid.uuid4().hex
            job = dict(id=job_id, status='processing', progress=0., stage='probing',
                       filename=Path(filename).name[:180], created_at=time.time(), updated_at=time.time())
            def work():
                try:
                    item = self._process_file(prepared, filename, job_id, digest)
                    self._job_update(job_id, status='ready', progress=100., stage='complete', item=item)
                except Exception as exc:
                    self._job_update(job_id, status='failed', stage='failed', error=str(exc))
                finally:
                    self.import_lock.release()
                    with self.jobs_lock:
                        self.import_threads.discard(threading.current_thread())
            thread = threading.Thread(target=work, name='astra-media-import', daemon=True)
            with self.jobs_lock:
                if self.closed:
                    raise ValueError('程序已关闭')
                self.import_jobs[job_id] = job
                self.import_threads.add(thread)
                snapshot = copy.deepcopy(job)
                thread.start()
            return snapshot
        except BaseException as exc:
            if prepared is not None:
                self._failure(prepared[1], filename, exc)
            self.import_lock.release()
            raise

    def import_status(self, job_id):
        with self.jobs_lock:
            job = self.import_jobs.get(job_id)
            return copy.deepcopy(job) if job is not None else None

    def add_file(self, temporary, filename, source_sha256=None):
        if not self.import_lock.acquire(False):
            raise ImportBusy('正在导入素材，请稍后重试')
        try:
            with self.jobs_lock:
                if self.closed:
                    raise ValueError('程序已关闭')
            digest = self._source_hash(temporary, source_sha256)
            reused = self._reuse_original(temporary, digest)
            if reused is not None:
                return reused
            return self._process_file(self._prepare_file(temporary, filename), filename, source_sha256=digest)
        finally:
            self.import_lock.release()

    @staticmethod
    def _failure(folder, filename, error):
        (folder/'failed.json').write_text(json.dumps(dict(filename=filename,
            status='processing_failed', error=str(error)), ensure_ascii=False, indent=2), encoding='utf-8')

    def _probe(self, file):
        probe = json.loads(self.run([self.executable('ffprobe'), '-v', 'error', '-show_format',
            '-show_streams', '-of', 'json', str(file)]))
        stream = next((s for s in probe.get('streams', []) if s.get('codec_type') == 'video'), None)
        if not stream:
            raise ValueError('文件中没有视频画面')
        width, height = stream.get('width', 0), stream.get('height', 0)
        if min(width, height) < 8 or width*height > 9_000_000:
            raise ValueError('视频画面尺寸无效或超过 900 万像素')
        duration = float(probe.get('format', {}).get('duration', stream.get('duration', 0)))
        if not math.isfinite(duration) or not 0 < duration <= 7200:
            raise ValueError('视频时长须在 2 小时以内')
        return probe, stream, duration

    @staticmethod
    def _square_pixel_copy(stream):
        sar = stream.get('sample_aspect_ratio', '1:1')
        if sar in ('1:1', 'N/A', '0:1'):
            return True
        try:
            ratio = Fraction(sar.replace(':', '/'))
            width, height = int(stream['width']), int(stream['height'])
            # The existing normalization filter would leave these pixels intact.
            # Change the H.264 aspect metadata instead of re-encoding every frame.
            return ratio > 0 and width % 2 == height % 2 == 0 and int(width * ratio / 2) * 2 == width
        except (ValueError, TypeError, KeyError, ZeroDivisionError, OverflowError):
            return False

    @staticmethod
    def _copy_compatible(suffix, stream):
        rotations = [stream.get('tags', {}).get('rotate', 0)]
        rotations.extend(side.get('rotation', 0) for side in stream.get('side_data_list', []))
        try:
            rotation_ok = all(math.isfinite(float(value)) and abs(float(value) % 360) < 1e-6
                              for value in rotations)
        except (TypeError, ValueError, OverflowError):
            rotation_ok = False
        return (stream.get('codec_name') == 'h264'
                and stream.get('pix_fmt') == 'yuv420p' and rotation_ok
                and MediaLibrary._square_pixel_copy(stream))

    def _process_file(self, prepared, filename, job_id=None, source_sha256=None):
        key, folder, original, suffix, images = prepared
        try:
            if suffix in images:
                image = cv2.imdecode(np.fromfile(original, np.uint8), cv2.IMREAD_COLOR)
                if image is None or min(image.shape[:2]) < 8 or image.shape[0]*image.shape[1] > 9_000_000:
                    raise ValueError('图片无法解码或超过 900 万像素')
                file = folder/'image.png'
                cv2.imencode('.png', image)[1].tofile(file)
                height, width = image.shape[:2]
                meta = dict(kind='image', width=width, height=height, duration=0, fps=0,
                            import_method='image_decode', timestamp_preserved=True)
            else:
                _, stream, duration = self._probe(original)
                file = folder/'preview.mp4'
                copy_stream = self._copy_compatible(suffix, stream)
                method = 'lossless_remux' if copy_stream else 'timestamp_preserving_transcode'
                self._job_update(job_id, stage='remuxing' if copy_stream else 'transcoding', progress=10.)
                command = [self.executable('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-y',
                    '-copyts', '-i', str(original), '-map', '0:v:0', '-an']
                if copy_stream:
                    command += ['-c:v', 'copy', '-copytb', '1']
                    if stream.get('sample_aspect_ratio', '1:1') not in ('1:1', 'N/A', '0:1'):
                        command += ['-bsf:v', 'h264_metadata=sample_aspect_ratio=1/1']
                        method = 'lossless_aspect_remux'
                else:
                    # No -r or CFR conversion: each decoded frame keeps its PTS.
                    command += ['-vf', 'scale=trunc(iw*sar/2)*2:ih,setsar=1,pad=ceil(iw/2)*2:ceil(ih/2)*2',
                        '-fps_mode', 'passthrough', '-enc_time_base', '-1', '-c:v', 'libx264', '-preset', 'fast', '-crf', '19',
                        '-pix_fmt', 'yuv420p', '-metadata:s:v:0', 'rotate=0', '-threads', '2']
                command += ['-avoid_negative_ts', 'disabled', '-movflags', '+faststart',
                            '-progress', 'pipe:1', '-nostats', str(file)]
                self.run(command, timeout=7200, progress=lambda seconds: self._job_update(job_id,
                    progress=10.+min(78., max(0., seconds/duration*78.))))
                self._job_update(job_id, stage='thumbnail', progress=90.)
                _, preview_stream, preview_duration = self._probe(file)
                cap = cv2.VideoCapture(str(file))
                ok, image = cap.read()
                height, width = (image.shape[:2] if ok else (0, 0))
                cap.release()
                if not ok:
                    raise ValueError('视频预览生成失败')
                rate = preview_stream.get('avg_frame_rate', '0/1').split('/')
                fps = float(rate[0])/max(1., float(rate[-1]))
                meta = dict(kind='video', width=width, height=height, duration=preview_duration, fps=fps,
                            original_duration=duration, original_width=stream.get('width'),
                            original_height=stream.get('height'), preview_normalized=not copy_stream,
                            import_method=method, timestamp_preserved=True,
                            start_time=float(preview_stream.get('start_time', 0)),
                            average_frame_rate=preview_stream.get('avg_frame_rate'),
                            frame_timing='native_pts')
            thumbnail = cv2.resize(image, (256, max(1, round(image.shape[0]*256/image.shape[1]))))
            cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(folder/'thumb.jpg')
            self._job_update(job_id, stage='registering', progress=95.)
            with file.open('rb') as source:
                digest = hashlib.file_digest(source, 'sha256').hexdigest()
            row = dict(id=key, title=Path(filename).name[:180], origin='upload', file=file.name,
                       thumb='thumb.jpg', sha256=digest,
                       original_sha256=source_sha256 or self._source_hash(original, None),
                       original_bytes=original.stat().st_size, **meta)
            (folder/'item.json').write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding='utf-8')
            with self.lock:
                if self.closed:
                    raise ValueError('程序已关闭')
                self._register(row, folder)
                return dict(self.entries[key])
        except BaseException as error:
            # Keep the uploaded original if processing fails. The exception
            # and a failed manifest allow retry without destroying user data.
            self._failure(folder, filename, error)
            raise
