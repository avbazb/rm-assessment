"""Local media library. Uploaded originals and playable copies remain local."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import uuid

import cv2
import numpy as np


class MediaLibrary:
    def __init__(self, assets, data):
        self.assets = Path(assets)
        self.data = Path(data)
        self.uploads = self.data/'uploads'
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.jobs_lock = threading.Lock()
        self.jobs = set()
        self.closed = False
        self.entries = {}
        self.files = {}
        self.thumbs = {}
        catalog = self.assets/'samples/catalog.json'
        samples = json.loads(catalog.read_text('utf-8')) if catalog.is_file() else []
        for row in samples:
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

    def run(self, args, timeout=180):
        with self.jobs_lock:
            if self.closed:
                raise ValueError('程序已关闭')
            process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding='utf-8', errors='replace', creationflags=0x08000000 if os.name=='nt' else 0)
            self.jobs.add(process)
        try:
            stdout, _ = process.communicate(timeout=timeout)
            if process.returncode:
                raise ValueError('素材无法解码，请检查文件是否完整')
            return stdout
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.communicate()
            raise ValueError('视频处理超时') from error
        finally:
            with self.jobs_lock:
                self.jobs.discard(process)

    def close(self):
        with self.jobs_lock:
            self.closed = True
            processes = list(self.jobs)
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    def add_file(self, temporary, filename):
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
        try:
            if suffix in images:
                image = cv2.imdecode(np.fromfile(original, np.uint8), cv2.IMREAD_COLOR)
                if image is None or min(image.shape[:2]) < 8 or image.shape[0]*image.shape[1] > 9_000_000:
                    raise ValueError('图片无法解码或超过 900 万像素')
                file = folder/'image.png'
                cv2.imencode('.png', image)[1].tofile(file)
                height, width = image.shape[:2]
                meta = dict(kind='image', width=width, height=height, duration=0, fps=0)
            else:
                # The normalized preview preserves duration and has explicit
                # frame rate; originals stay beside it for reproducibility.
                probe = json.loads(self.run([self.executable('ffprobe'), '-v', 'error', '-show_format',
                    '-show_streams', '-of', 'json', str(original)]))
                stream = next((s for s in probe['streams'] if s['codec_type']=='video'), None)
                if not stream:
                    raise ValueError('文件中没有视频画面')
                if stream.get('width', 0)*stream.get('height', 0) > 9_000_000:
                    raise ValueError('视频画面超过 900 万像素')
                duration = float(probe.get('format', {}).get('duration', 0))
                if duration <= 0 or duration > 7200:
                    raise ValueError('视频时长须在 2 小时以内')
                rate = stream.get('avg_frame_rate', '30/1').split('/')
                fps = min(30., max(1., float(rate[0])/max(1., float(rate[-1]))))
                file = folder/'preview.mp4'
                self.run([self.executable('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-y', '-i', str(original),
                    '-map', '0:v:0', '-an', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                    '-r', str(fps), '-c:v', 'libx264', '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', '-threads', '2', str(file)], timeout=7200)
                cap = cv2.VideoCapture(str(file))
                ok, image = cap.read()
                height, width = (image.shape[:2] if ok else (0, 0))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()
                if not ok:
                    raise ValueError('视频预览生成失败')
                meta = dict(kind='video', width=width, height=height, duration=frame_count/fps, fps=fps,
                            original_duration=duration, original_width=stream.get('width'),
                            original_height=stream.get('height'), preview_normalized=True)
            thumbnail = cv2.resize(image, (256, max(1, round(image.shape[0]*256/image.shape[1]))))
            cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(folder/'thumb.jpg')
            row = dict(id=key, title=Path(filename).name[:180], origin='upload', file=file.name,
                       thumb='thumb.jpg', sha256=hashlib.sha256(file.read_bytes()).hexdigest(), **meta)
            (folder/'item.json').write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding='utf-8')
            with self.lock:
                self._register(row, folder)
                return dict(self.entries[key])
        except BaseException:
            # Keep the uploaded original if processing fails. The exception
            # and a failed manifest allow retry without destroying user data.
            (folder/'failed.json').write_text(json.dumps({'filename': filename, 'status': 'processing_failed'}), encoding='utf-8')
            raise
