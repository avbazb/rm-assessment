"""Single-file Windows application entry point."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import webbrowser

ROOT = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT/'app/backend'))
from runtime.version import VERSION, BUILD_ID
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')


def main():
    parser = argparse.ArgumentParser(description='robotmaster')
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--health-file', type=Path)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--self-test-output', type=Path)
    parser.add_argument('--self-test-image', type=Path)
    parser.add_argument('--self-test-mode', choices=['fusion','classical'], default='fusion')
    args = parser.parse_args()
    from runtime.windows_runtime import configure_cpu_scheduling
    cpu_scheduling = configure_cpu_scheduling()
    data = args.data_dir or Path(os.environ.get('LOCALAPPDATA', str(Path.home())))/'AstraVision'/VERSION
    data.mkdir(parents=True, exist_ok=True)
    # Windowless builds have no stdio; libraries still receive writable handles.
    if sys.stdout is None:
        sys.stdout = (data/'runtime.log').open('a', encoding='utf-8')
    if sys.stderr is None:
        sys.stderr = sys.stdout
    if args.self_test:
        from runtime.engine import Engine
        engine = Engine(ROOT/'app/assets/models/recommended.onnx')
        try:
            sample = next((row for row in json.loads((ROOT/'app/assets/samples/catalog.json').read_text('utf-8'))
                           if row['kind']=='image'), None)
            if args.self_test_image:
                payload = args.self_test_image.read_bytes()
                test_input = 'external_image'
            elif sample:
                payload = (ROOT/'app/assets/samples'/sample['file']).read_bytes()
                test_input = 'bundled_image'
            else:
                import cv2
                import numpy as np
                ok, encoded = cv2.imencode('.png', np.zeros((480, 640, 3), np.uint8))
                if not ok:
                    raise RuntimeError('自检图像生成失败')
                payload = encoded.tobytes()
                test_input = 'synthetic_blank_runtime_only'
            out = engine.analyze(payload, 'self-test', 0, 0, 0.,mode=args.self_test_mode)
            if test_input == 'bundled_image' and not out['observations']:
                raise RuntimeError('内置蓝板自检未得到装甲板')
            report = dict(status='passed', device='CPU', frozen=bool(getattr(sys, 'frozen', False)),
                          torch_loaded='torch' in sys.modules, observations=len(out['observations']),
                          test_input=test_input,
                          geometry=sum(o['geometry_valid'] for o in out['observations']),
                          processing_ms=out['processing_ms'], input_frame=out['input_frame'],
                          inference_providers=engine.detector.neural.session.get_providers() if engine.detector.neural else [],
                          classical_module=out['classical_module'],detection_mode=args.self_test_mode,
                          neural_model_used=out['neural_model_used'],math_backend=out['math_backend'])
            text = json.dumps(report, ensure_ascii=False, indent=2)
            if args.self_test_output:
                args.self_test_output.parent.mkdir(parents=True, exist_ok=True)
                args.self_test_output.write_text(text, encoding='utf-8')
            print(text)
        finally:
            engine.close()
        return
    if not args.no_browser and args.port == 0:
        # A second double-click reopens this user's running preview.
        # Verification is loopback-only and cannot redirect to a saved URL.
        try:
            from urllib.parse import urlsplit
            from urllib.request import urlopen
            previous = json.loads((data/'last-session.json').read_text('utf-8'))
            parsed = urlsplit(previous['url'])
            if parsed.scheme == 'http' and parsed.hostname == '127.0.0.1' and parsed.port and parsed.path in ('', '/'):
                with urlopen(previous['url']+'/api/health', timeout=.75) as response:
                    running = json.load(response)
                if (running.get('app') == 'astra-portable-vision' and running.get('version') == VERSION
                        and running.get('build_id') == BUILD_ID and not running.get('last_error')
                        and not running.get('closed') and running.get('inference_age_s', 0) < 8
                        and running.get('classical_module') == 'fast_color_geometry_roi_v2'):
                    webbrowser.open(previous['url'])
                    return
        except (OSError, ValueError, KeyError):
            pass
    from runtime.server import Application
    server = Application(ROOT, data, args.port)
    server.cpu_scheduling = cpu_scheduling
    if args.health_file:
        args.health_file.parent.mkdir(parents=True, exist_ok=True)
        args.health_file.write_text(json.dumps({'url': server.url, 'pid': os.getpid(), 'status': 'serving'}), encoding='utf-8')
    (data/'last-session.json').write_text(json.dumps({'url': server.url, 'pid': os.getpid()}), encoding='utf-8')
    if not args.no_browser:
        webbrowser.open(server.url)
    try:
        server.serve_forever(poll_interval=.2)
    finally:
        server.engine.close()
        server.server_close()
        if getattr(sys, 'frozen', False):
            sys.stdout.flush()
            # A timed-out native worker must not hold a windowless EXE open.
            os._exit(0)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        error = traceback.format_exc()
        root = Path(os.environ.get('LOCALAPPDATA', tempfile.gettempdir()))/'AstraVision'
        root.mkdir(parents=True, exist_ok=True)
        (root/'startup-error.log').write_text(error, encoding='utf-8')
        if getattr(sys, 'frozen', False) and '--no-browser' not in sys.argv and '--self-test' not in sys.argv:
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, '程序启动失败，错误记录已保存到 '+str(root/'startup-error.log'), 'robotmaster', 0x10)
        raise
