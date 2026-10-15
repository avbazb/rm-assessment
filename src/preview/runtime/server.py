"""Loopback-only HTTP API and range-aware media serving."""
import json
import mimetypes
import os
from pathlib import Path
import re
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs, unquote, quote

from .engine import Engine, Busy, StaleFrame, InferenceTimeout, impact_result
from .media import MediaLibrary
from .version import VERSION, BUILD_ID


class Application(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, root, data, port=0):
        self.root, self.data = Path(root), Path(data)
        self.media = MediaLibrary(self.root/'app/assets', self.data)
        self.exports = self.data/'exports'
        self.exports.mkdir(exist_ok=True)
        self.engine = Engine(self.root/'app/assets/models/recommended.onnx')
        self.upload_lock = threading.Lock()
        super().__init__(('127.0.0.1', port), Handler)
        self.url = f'http://127.0.0.1:{self.server_port}'

    def server_close(self):
        self.media.close()
        super().server_close()


class Handler(BaseHTTPRequestHandler):
    server_version = 'AstraVision/' + VERSION
    protocol_version = 'HTTP/1.1'
    disable_nagle_algorithm = True

    def setup(self):
        super().setup()
        self.connection.settimeout(20)

    def log_message(self, *_):
        pass

    def json_response(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def local_request(self):
        host = self.headers.get('Host', '')
        if host not in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'):
            raise ValueError('请求来源无效')
        origin = self.headers.get('Origin')
        if origin and origin not in (self.server.url, f'http://localhost:{self.server.server_port}'):
            raise ValueError('请求来源无效')

    def body(self, limit):
        length = int(self.headers.get('Content-Length', '0'))
        if not 0 <= length <= limit:
            raise ValueError('文件大小超过限制')
        data = self.rfile.read(length)
        if len(data) != length:
            raise ValueError('文件上传不完整')
        return data

    def do_GET(self):
        try:
            self.local_request()
            path = unquote(urlsplit(self.path).path)
            if path == '/api/health':
                self.json_response(dict(app='astra-portable-vision', device='CPU', version=VERSION, build_id=BUILD_ID,
                                        classical_module='fast_color_geometry_roi_v2',detection_modes=['fusion','classical'],
                                        cpu_scheduling=getattr(self.server, 'cpu_scheduling', {}),
                                        **self.server.engine.health()))
            elif path == '/api/media':
                self.json_response(self.server.media.listing())
            elif path == '/api/prediction':
                query = {k: v[-1] for k, v in parse_qs(urlsplit(self.path).query).items()}
                timestamp = float(query['timestamp']) if 'timestamp' in query else None
                self.json_response(self.server.engine.prediction(query.get('session','preview'),
                    int(query.get('epoch',0)), timestamp, float(query.get('horizon',.15))))
            elif path.startswith('/media/'):
                self.file_response(self.server.media.files.get(path.removeprefix('/media/')))
            elif path.startswith('/thumb/'):
                self.file_response(self.server.media.thumbs.get(path.removeprefix('/thumb/').removesuffix('.jpg')))
            elif re.fullmatch(r'/exports/[a-f0-9]{32}\.png', path):
                self.file_response(self.server.exports/path.split('/')[-1], attachment='ASTRA.png')
            elif path.startswith('/api/imports/'):
                job = self.server.media.import_status(path.removeprefix('/api/imports/'))
                self.json_response({'job': job} if job else {'error': '导入任务不存在'}, 200 if job else 404)
            elif path in ('/', '/index.html', '/impact.html', '/motion.html', '/pose.html', '/styles.css', '/motion.js', '/app.js'):
                self.file_response(self.server.root/'app/web'/('index.html' if path=='/' else path[1:]))
            else:
                self.json_response({'error': '内容不存在'}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except StaleFrame as exc:
            self.json_response({'error': str(exc)}, 409)
        except ValueError as exc:
            self.json_response({'error': str(exc)}, 400)

    def do_POST(self):
        try:
            self.local_request()
            url = urlsplit(self.path)
            query = {k: v[-1] for k, v in parse_qs(url.query).items()}
            if url.path == '/api/warmup':
                self.body(1024)
                self.json_response(self.server.engine.warmup(query.get('mode','fusion')))
            elif url.path == '/api/analyze':
                prediction_switch = query.get('prediction', '1')
                if prediction_switch not in ('0','1'):
                    raise ValueError('预测开关无效')
                body_started = time.perf_counter()
                payload = self.body(32*1024*1024)
                body_ms = (time.perf_counter()-body_started)*1000
                calibration = self.headers.get('X-Calibration')
                result = self.server.engine.analyze(payload, query.get('session', 'preview'),
                    int(query.get('epoch', 0)), int(query.get('sequence', 0)), float(query.get('timestamp', 0)),
                    float(query.get('horizon', .15)), json.loads(unquote(calibration)) if calibration else None,
                    query.get('mode','fusion'),
                    frame_format='rgba8' if self.headers.get('X-Frame-Encoding')=='rgba8' else 'encoded',
                    frame_width=int(self.headers.get('X-Frame-Width','0')),
                    frame_height=int(self.headers.get('X-Frame-Height','0')),
                    prediction_enabled=prediction_switch == '1')
                result['request_body_ms'] = body_ms
                self.json_response(result)
            elif url.path == '/api/impact':
                data = json.loads(self.body(128*1024))
                self.json_response(impact_result(data['observation'], data['point']))
            elif url.path == '/api/media':
                self.upload(query.get('filename', ''))
            elif url.path == '/api/export':
                payload = self.body(32*1024*1024)
                import cv2
                import numpy as np
                if not payload.startswith(b'\x89PNG\r\n\x1a\n') or cv2.imdecode(np.frombuffer(payload, np.uint8), 1) is None:
                    raise ValueError('保存画面须为 PNG')
                name = uuid.uuid4().hex+'.png'
                (self.server.exports/name).write_bytes(payload)
                self.json_response(dict(url='/exports/'+name, filename=Path(query.get('filename', 'ASTRA.png')).name[:180]))
            elif url.path == '/api/shutdown':
                self.body(1024)
                self.json_response({'status': 'closing'})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self.json_response({'error': '接口不存在'}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Busy as exc:
            self.json_response({'error': str(exc)}, 429)
        except StaleFrame as exc:
            self.json_response({'error': str(exc)}, 409)
        except InferenceTimeout as exc:
            self.json_response({'error': str(exc), 'code': 'inference_timeout'}, 504)
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            self.json_response({'error': str(exc)}, 400)
        except Exception as exc:
            import traceback
            with (self.server.data/'errors.log').open('a', encoding='utf-8') as log:
                traceback.print_exc(file=log)
            incident = uuid.uuid4().hex[:8]
            with (self.server.data/'errors.log').open('a', encoding='utf-8') as log:
                log.write(f'error_id={incident} path={self.path}\n')
            self.json_response({'error': f'检测处理失败（记录 {incident}），可以重新检测',
                                'error_id': incident, 'code': type(exc).__name__}, 500)

    def upload(self, filename):
        if not self.server.upload_lock.acquire(False):
            raise Busy('正在导入素材，请稍后重试')
        temporary = None
        try:
            length = int(self.headers.get('Content-Length', '-1'))
            if not 0 < length <= 1024*1024*1024:
                raise ValueError('素材须在 1 GB 以内')
            fd, temporary = tempfile.mkstemp(prefix='upload-', dir=self.server.data)
            with os.fdopen(fd, 'wb') as target:
                left = length
                while left:
                    chunk = self.rfile.read(min(left, 1024*1024))
                    if not chunk:
                        raise ValueError('文件上传不完整')
                    target.write(chunk)
                    left -= len(chunk)
            if parse_qs(urlsplit(self.path).query).get('async') == ['1']:
                from .media import ImportBusy
                try:
                    job = self.server.media.begin_import(temporary, filename)
                except ImportBusy as exc:
                    raise Busy(str(exc)) from exc
                self.json_response({'job': job}, 202)
            else:
                item = self.server.media.add_file(temporary, filename)
                self.json_response({'item': item})
        finally:
            if temporary and Path(temporary).exists():
                Path(temporary).unlink()
            self.server.upload_lock.release()

    def file_response(self, path, attachment=None):
        if path is None or not Path(path).is_file():
            self.json_response({'error': '素材不存在'}, 404)
            return
        path = Path(path)
        size = path.stat().st_size
        start, end, partial = 0, size-1, False
        value = self.headers.get('Range')
        if value:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', value)
            if not match or not any(match.groups()):
                self.send_error(416)
                return
            a, b = match.groups()
            start = int(a) if a else max(0, size-int(b))
            end = min(size-1, int(b)) if a and b else size-1
            if start > end or start >= size:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            partial = True
        mime = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        self.send_response(206 if partial else 200)
        self.send_header('Content-Type', mime+('; charset=utf-8' if mime.startswith('text/') or mime=='application/javascript' else ''))
        self.send_header('Content-Length', str(end-start+1))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-cache')
        if attachment:
            self.send_header('Content-Disposition', 'attachment; filename="'+attachment+'"')
        if partial:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        with path.open('rb') as source:
            source.seek(start)
            remaining = end-start+1
            while remaining:
                data = source.read(min(remaining, 256*1024))
                if not data:
                    break
                self.wfile.write(data)
                remaining -= len(data)
