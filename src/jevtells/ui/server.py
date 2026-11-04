"""Loopback-only Flask application. All media is rooted in samples/ or work/."""
from __future__ import annotations

import io
import json
import logging
import os
import secrets
import subprocess
import sys
import threading
import uuid
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import cv2
from dotenv import dotenv_values
from flask import Flask, abort, jsonify, request, send_file
from werkzeug.exceptions import HTTPException
from werkzeug.serving import make_server
from werkzeug.utils import secure_filename

from ..config import load_config
from ..doctor import checks
from ..resources import asset_path
from ..stages import people
from .jobs import Jobs, redact
from .params import cli_args, normalize, number, output_path, safe_path, source_rect, intervals_from_drag

VIDEOS = {'.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v'}
PUBLIC = {'.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v', '.png', '.jpg'}
RENDER_FIELDS = {'title', 'blur', 'subtitles', 'layout', 'reframe'}


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else default


def media_info(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30
        width, height = int(capture.get(3)), int(capture.get(4))
        duration = capture.get(cv2.CAP_PROP_FRAME_COUNT)/fps
        if not capture.isOpened() or not width or duration <= 0:
            raise ValueError('invalid_video')
        return {'width': width, 'height': height, 'duration': duration, 'fps': fps}
    finally:
        capture.release()


def create_app(home: Path | None = None) -> Flask:
    root = (home or Path(os.environ.get('JEVTELLS_HOME', Path.cwd()))).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for area in ('samples', 'work'):
        candidate = root / area
        if candidate.is_symlink():
            raise ValueError('unsafe_path')
        candidate.mkdir(exist_ok=True)
    static = Path(__file__).parent
    app = Flask(__name__, static_folder=None)
    maximum = int(os.environ.get('JEVTELLS_UI_MAX_UPLOAD_BYTES', 2*1024**3))
    app.config['MAX_CONTENT_LENGTH'] = maximum
    jobs = Jobs(root)
    app.extensions['jobs'] = jobs
    app.extensions['root'] = root
    token = secrets.token_urlsafe(32)
    settings_path = root/'work/.ui_settings.json'
    settings = {'lang': 'zh', 'output_dir': 'work', **(read_json(settings_path, {}) or {})}
    diagnosis: list[dict[str, Any]] = []

    def diagnose() -> None:
        nonlocal diagnosis
        # doctor only exposes key status; use the UI home even for test clients.
        diagnosis = checks(load_config())
        configured = bool(os.environ.get('OPENROUTER_API_KEY') or dotenv_values(root/'.env').get('OPENROUTER_API_KEY'))
        for row in diagnosis:
            if row['name'] == 'OPENROUTER_API_KEY':
                row.update(ok=configured, detail='已设置' if configured else '未设置')

    diagnose()

    @app.before_request
    def protect() -> None:
        if request.host.split(':')[0] != '127.0.0.1':
            abort(403)
        if request.remote_addr not in (None, '127.0.0.1'):
            abort(403)
        origin = request.headers.get('Origin')
        if origin and urlsplit(origin).netloc != request.host:
            abort(403)
        if request.method != 'GET' and request.headers.get('X-JevTells-Token') != token:
            abort(403)

    @app.after_request
    def headers(response: Any) -> Any:
        response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' blob:; media-src 'self' blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'"
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.errorhandler(Exception)
    def error(exc: Exception) -> Any:
        # Never echo arbitrary request values, tracebacks or secrets.
        if isinstance(exc, HTTPException):
            return jsonify(error=f'http_{exc.code}'), exc.code
        if isinstance(exc, RuntimeError) and str(exc) == 'busy':
            return jsonify(error='busy'), 409
        if isinstance(exc, (ValueError, TypeError, KeyError, FileNotFoundError)):
            return jsonify(error='invalid_request'), 400
        return jsonify(error='operation_failed'), 500

    @app.get('/')
    def index() -> Any:
        return send_file(static/'index.html')

    @app.get('/static/<path:name>')
    def assets(name: str) -> Any:
        path = (static/name).resolve()
        if not path.is_relative_to(static.resolve()) or path.suffix not in {'.css', '.js', '.json'}:
            abort(404)
        return send_file(path)

    @app.get('/font')
    def font() -> Any:
        path = asset_path('assets/fonts/NotoSansCJKsc-Regular.otf')
        return send_file(path) if path.exists() else ('', 404)

    @app.get('/api/bootstrap')
    def bootstrap() -> Any:
        return jsonify(token=token, settings=settings, checks=diagnosis, stages=__import__('jevtells.ui.jobs', fromlist=['STAGES']).STAGES)

    @app.get('/api/samples')
    def samples() -> Any:
        return jsonify([{'path': str(path.relative_to(root)), 'name': path.name} for path in sorted((root/'samples').iterdir())
                        if path.suffix.lower() in VIDEOS and path.is_file() and path.resolve().is_relative_to(root/'samples')])

    @app.post('/api/upload')
    def upload() -> Any:
        with jobs.lock:
            if jobs.busy():
                raise RuntimeError('busy')
            suffix = Path(request.headers.get('X-Filename', '')).suffix.lower()
            if suffix not in VIDEOS:
                abort(415)
            if request.content_length is not None and request.content_length > maximum:
                abort(413)
            # Stream a raw body. The extension and Content-Length are checked before
            # writing; the byte counter also protects chunked/unknown-length uploads.
            stem = secure_filename(Path(request.headers.get('X-Filename', '')).stem)[:64] or 'video'
            destination = safe_path(root, f'samples/{stem}_{uuid.uuid4().hex[:10]}{suffix}', 'samples')
            try:
                with destination.open('xb') as handle:
                    total = 0
                    while chunk := request.stream.read(1024*1024):
                        total += len(chunk)
                        if total > maximum:
                            abort(413)
                        handle.write(chunk)
                info = media_info(destination)
            except Exception:
                destination.unlink(missing_ok=True)
                raise
            return jsonify(path=str(destination.relative_to(root)), **info)

    @app.get('/api/inspect')
    def inspect_video() -> Any:
        source = safe_path(root, request.args['input'], 'samples')
        info = media_info(source)
        # Reuse M5's generic subtitle detection, never a sample-specific rule.
        cache = safe_path(root, f'work/.ui_probe/{source.stem}.json', 'work')
        signature = [source.stat().st_size, source.stat().st_mtime_ns]
        detection = read_json(cache, {})
        if detection.get('signature') != signature:
            with jobs.lock:
                if jobs.busy():
                    raise RuntimeError('busy')
                from ..render.reframe import detect_subtitles
                detection = {'signature': signature, 'detection': detect_subtitles(source, load_config()['render']['reframe'])}
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(detection))
        return jsonify(**info, subtitles=detection['detection'])

    @app.get('/media/<path:name>')
    def media(name: str) -> Any:
        path = safe_path(root, name)
        if path.suffix.lower() not in PUBLIC or not path.is_file():
            abort(404)
        return send_file(path, conditional=True, as_attachment=request.args.get('download') == '1')

    @app.get('/api/frame')
    def frame() -> Any:
        path = safe_path(root, request.args['input'])
        at = number(request.args.get('at', 0))
        capture = cv2.VideoCapture(str(path))
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, at*1000)
            ok, image = capture.read()
            if not ok:
                abort(404)
            ok, data = cv2.imencode('.jpg', image)
            if not ok:
                abort(404)
            return send_file(io.BytesIO(data.tobytes()), mimetype='image/jpeg')
        finally:
            capture.release()

    @app.post('/api/command')
    def command() -> Any:
        import shlex
        p = normalize(request.json['params'])
        args = cli_args(p, root)
        return jsonify(args=args, command=shlex.join(['jevtells', *args]), output=str(output_path(p, root).relative_to(root)))

    @app.post('/api/jobs')
    def start() -> Any:
        p = normalize(request.json['params'])
        source = safe_path(root, p['input'], 'samples')
        if not source.is_file() or source.suffix.lower() not in VIDEOS:
            raise ValueError('invalid_video')
        mode = request.json.get('mode', 'run')
        output = output_path(p, root) if mode != 'people' else root/'work'/f'{source.stem}_s0_dauto'
        with jobs.lock:
            if jobs.busy():
                raise RuntimeError('busy')
            previous = read_json(output/'ui_params.json', {})
            # Only a successful full run is eligible for a render-only repeat.
            prior = read_json(output/'ui_complete_params.json', {})
            render_only = bool(mode == 'run' and prior and all(p.get(k) == prior.get(k) for k in set(p)|set(prior) if k not in RENDER_FIELDS)
                               and (output/'narration.json').is_file())
            args = cli_args(p, root, mode, render_only=render_only)
            output.mkdir(parents=True, exist_ok=True)
            if mode != 'people':
                (output/'ui_params.json').write_text(json.dumps(p, ensure_ascii=False, indent=2))
            result = jobs.start(args, output, mode)
            result['render_only'] = render_only
            if mode == 'run':
                worker = jobs.worker
                def save_complete() -> None:
                    assert worker is not None
                    worker.join()
                    with jobs.lock:
                        if jobs.job.get('id') == result['id'] and jobs.job['status'] == 'complete':
                            (output/'ui_complete_params.json').write_text(json.dumps(p, ensure_ascii=False, indent=2))
                threading.Thread(target=save_complete, daemon=True).start()
        return jsonify(result)

    @app.get('/api/jobs')
    def job() -> Any:
        return jsonify(jobs.snapshot())

    @app.post('/api/cancel')
    def cancel() -> Any:
        jobs.cancel()
        return jsonify(ok=True)

    @app.get('/api/people')
    def person_boxes() -> Any:
        source = safe_path(root, request.args['input'], 'samples')
        out = safe_path(root, f'work/{source.stem}_s0_dauto', 'work')
        if not (out/'detections.npz').exists():
            return jsonify(error='detect_first'), 409
        with jobs.lock:
            if jobs.busy():
                raise RuntimeError('busy')
            rows = people.run(out/'clip.mp4', out, number(request.args.get('at', 0)), out/'detections.npz', config=load_config())
        return jsonify(rows)

    @app.get('/api/timeline')
    def timeline() -> Any:
        output = safe_path(root, request.args['output'], 'work')
        meta = read_json(output/'run_meta.json', {})
        rows = read_json(output/'windows.json', [])
        info = media_info(output/'clip.mp4') if (output/'clip.mp4').exists() else {'duration': 0}
        return jsonify(windows=rows, shots=read_json(output/'shots.json', []), duration=info['duration'],
                       actual_start=meta.get('parameters', {}).get('start', 0),
                       media=str((output/'clip.mp4').relative_to(root)), meta={'api': meta.get('api', {})})

    @app.post('/api/coordinates')
    def coordinates() -> Any:
        data = request.json
        if data.get('kind') == 'timeline':
            return jsonify(intervals=intervals_from_drag(data['a'], data['b'], data['width'], data['duration']))
        return jsonify(rect=source_rect(data['rect'], data['display'], data['source']))

    @app.route('/api/settings', methods=['GET', 'POST'])
    def preferences() -> Any:
        if request.method == 'POST':
            with jobs.lock:
                if jobs.busy():
                    raise RuntimeError('busy')
                data = request.json
                lang = data.get('lang', settings['lang'])
                if lang not in {'zh', 'en'}:
                    raise ValueError('invalid_language')
                output = data.get('output_dir', settings['output_dir']).rstrip('/')
                safe_path(root, output+'/.ui', 'work')
                key = data.get('key')
                if key is not None:
                    if not isinstance(key, str) or not key or len(key) > 512 or not all(c.isalnum() or c in '-_.' for c in key):
                        raise ValueError('invalid_key')
                    from dotenv import set_key
                    env_path = root/'.env'
                    if env_path.is_symlink():
                        raise ValueError('unsafe_path')
                    env_path.touch(mode=0o600, exist_ok=True)
                    env_path.chmod(0o600)
                    set_key(str(env_path), 'OPENROUTER_API_KEY', key)
                    # A previously loaded environment key must not override the new file.
                    os.environ.pop('OPENROUTER_API_KEY', None)
                settings.update(lang=lang, output_dir=output)
                settings_path.write_text(json.dumps(settings))
                diagnose()
        return jsonify(settings=settings, checks=diagnosis)

    def history_params(path: Path) -> dict[str, Any] | None:
        p = read_json(path/'ui_params.json')
        if p:
            return normalize(p)
        meta = read_json(path/'run_meta.json', {}).get('parameters', {})
        source = Path(meta.get('input', ''))
        if not source.is_absolute():
            source = root/source
        if not source.resolve().is_relative_to(root/'samples') or not source.is_file():
            return None
        targets = []
        for value in meta.get('target', []):
            n, _, at = value.partition('@')
            targets.append([int(n), float(at or 0)])
        return normalize({k: v for k,v in {**meta, 'input': str(source.relative_to(root)), 'target': targets or [[1,0]],
            'others_speaking': [[float(x) for x in v.split('-')] for v in meta.get('others_speaking', [])],
            'blur': [[int(x) for x in v.split(',')] for v in meta.get('blur', [])], 'face_identity': meta.get('face_identity', False),
            'output_dir': str(path.parent.relative_to(root))}.items() if k in {'input','speaker','start','duration','target','others_speaking','blur','face_identity','lang','title','subtitles','subtitle_source','layout','reframe','output_dir'}})

    @app.get('/api/history')
    def history() -> Any:
        rows = []
        for path in sorted((root/'work').rglob('output_h.mp4'), key=lambda p: p.stat().st_mtime, reverse=True):
            if not path.resolve().is_relative_to(root/'work') or path.is_symlink():
                continue
            parent = path.parent
            try:
                p = history_params(parent)
            except (ValueError, KeyError, TypeError):
                p = None
            rows.append({'output': str(parent.relative_to(root)), 'params': p,
                         'h': str(path.relative_to(root)), 'v': str((parent/'output_v.mp4').relative_to(root)) if (parent/'output_v.mp4').is_file() else None})
        return jsonify(rows)

    @app.post('/api/reveal')
    def reveal() -> Any:
        path = safe_path(root, request.json['path'], 'work')
        if not path.exists():
            abort(404)
        command = ['open', '-R', str(path)] if sys.platform == 'darwin' else ['xdg-open', str(path if path.is_dir() else path.parent)]
        subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return jsonify(ok=True)

    return app


def serve(port: int = 7860) -> None:
    if not 1 <= port <= 65535:
        raise ValueError('Port must be between 1 and 65535')
    app = create_app()
    # Binding before opening the browser avoids a startup race. No host option.
    server = make_server('127.0.0.1', port, app, threaded=True)
    logging.getLogger('werkzeug').disabled = True
    url = f'http://127.0.0.1:{port}'
    print(url, flush=True)
    threading.Timer(.2, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.extensions['jobs'].cancel()
        server.server_close()
