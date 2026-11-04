"""Interactive, source-time person anchors, shared with the run command."""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import cv2


def source_identity(source: Path) -> dict:
    stat = source.stat()
    return {'path': str(source.resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def save(path: Path, anchors: list[tuple[int, float]], source: Path) -> None:
    if not anchors:
        raise ValueError('No person selected; target.json was not changed')
    clean = sorted({float(t): int(n) for n, t in anchors}.items())
    if any(n < 1 or not math.isfinite(t) or t < 0 for t, n in clean):
        raise ValueError('Invalid target anchors')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'version': 1, 'source': source_identity(source),
        'time_basis': 'source', 'anchors': [[n, t] for t, n in clean]}, indent=2))
    temporary.replace(path)


def load(path: Path, source: Path) -> list[tuple[int, float]]:
    value = json.loads(path.read_text())
    if value.get('version') != 1 or value.get('source') != source_identity(source) or value.get('time_basis') != 'source':
        raise ValueError('target.json belongs to a different/changed video; run pick again or pass --target')
    anchors = [(int(n), float(t)) for n, t in value['anchors']]
    if not anchors or any(n < 1 or not math.isfinite(t) or t < 0 for n, t in anchors):
        raise ValueError('Invalid target.json anchors; run pick again or pass --target')
    return anchors


def resolve(explicit: list[str], output: Path, canonical: Path, source: Path):
    from ..cli import _parse_targets
    if explicit:
        return _parse_targets(explicit), None
    for directory in dict.fromkeys((output, canonical)):
        path = directory / 'target.json'
        if path.exists():
            return load(path, source), path
    return _parse_targets([]), None


def require_gui() -> None:
    fallback = 'No graphical display available. Use jevtells people VIDEO --at SECONDS, then run with --target N@SECONDS.'
    if os.environ.get('SSH_CONNECTION') or (sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))):
        raise RuntimeError(fallback)
    if 'GUI:                           NONE' in cv2.getBuildInformation():
        raise RuntimeError(fallback)
    if sys.platform == 'darwin':
        import subprocess
        # Reject background daemons before calling HighGUI (which can abort).
        console = subprocess.run(['stat', '-f', '%Su', '/dev/console'], capture_output=True, text=True).stdout.strip()
        if console in ('', 'root', 'loginwindow'):
            raise RuntimeError(fallback)


def run(clip, out, at, detections, source, config):
    from .people import run as numbered
    require_gui()
    fps = float(detections['fps']); total = len(detections['poses_all'])
    if not math.isfinite(at) or not 0 <= at < total / fps:
        raise ValueError('--at must be within the video')
    name = 'JevTells pick | click or 1-9 | Enter save | Esc cancel'
    anchors = []; records = []; current = round(at * fps); dirty = True
    def seek(frame):
        nonlocal current, dirty
        current = int(frame); dirty = True
    def choose(number):
        nonlocal anchors, dirty
        if any(r['number'] == number for r in records):
            anchors = [(n,t) for n,t in anchors if round(t*fps) != current]
            anchors.append((number, current/fps)); dirty = True
            print(f'Selected {number}@{current/fps:.6f}s ({len(anchors)} anchors)')
    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            hits = [r for r in records if r['box'][0] <= x <= r['box'][2] and r['box'][1] <= y <= r['box'][3]]
            if len(hits) == 1:
                choose(hits[0]['number'])
    try:
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar('Frame', name, current, total-1, seek)
        cv2.setMouseCallback(name, click)
        while True:
            if dirty:
                records = numbered(clip, out, at=current/fps, detections=detections, force=True, config=config)
                frame = cv2.imread(str(out / f'people_{current/fps:04.1f}s.png'))
                if frame is None:
                    raise RuntimeError('Could not read numbered frame')
                cv2.putText(frame, f'{current/fps:.2f}s | anchors: {len(anchors)}', (15,25), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
                cv2.imshow(name, frame); dirty = False
            key = cv2.waitKey(30) & 0xff
            if ord('1') <= key <= ord('9'):
                choose(key-ord('0'))
            elif key in (10,13):
                save(out/'target.json', anchors, source)
                print(f'Saved {out / "target.json"}'); break
            elif key == 27 or cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                print('Selection cancelled'); break
    except cv2.error as error:
        raise RuntimeError('Cannot open selection window; use people plus --target N@SECONDS') from error
    finally:
        cv2.destroyAllWindows()
