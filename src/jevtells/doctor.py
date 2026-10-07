"""Read-only installation checks with key-safe results and repair advice."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from .config import load_config
from .resources import asset_path


def checks(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    config=config or load_config()
    result=[]
    def add(name: str, ok: bool, detail: str, repair: str) -> None:
        result.append({'name':name,'ok':bool(ok),'detail':detail,'repair':repair if not ok else ''})
    add('Python',sys.version_info>=(3,11),'.'.join(str(i) for i in sys.version_info[:3]),'Install Python 3.11 or newer and recreate the virtual environment.')
    from importlib.metadata import PackageNotFoundError, version
    try:
        ocr_version = version('rapidocr-onnxruntime')
        import onnxruntime
        cpu = 'CPUExecutionProvider' in onnxruntime.get_available_providers()
    except (PackageNotFoundError, ImportError):
        ocr_version, cpu = 'unavailable', False
    add('RapidOCR CPU', cpu and ocr_version == '1.4.4', ocr_version, 'Install rapidocr-onnxruntime==1.4.4.')
    executable=shutil.which('ffmpeg')
    available=[]
    if executable:
        try:
            output=subprocess.run([executable,'-hide_banner','-encoders'],capture_output=True,text=True,check=True).stdout
            for name in ('libx264','h264_videotoolbox'):
                if name not in output:
                    continue
                trial=subprocess.run([executable,'-hide_banner','-loglevel','error',
                    '-f','lavfi','-i','color=size=16x16:rate=1',
                    '-frames:v','1','-c:v',name,'-pix_fmt','yuv420p','-f','null','-'],
                    capture_output=True,timeout=15)
                if trial.returncode==0:
                    available.append(name)
        except (OSError,subprocess.CalledProcessError,subprocess.TimeoutExpired):
            pass
    add('ffmpeg',bool(executable and available),', '.join(available) if available else 'unavailable','Install ffmpeg with libx264 (macOS: brew install ffmpeg).')
    for name in ('pose','hands','face_detector','face_recognizer','face_landmarker'):
        path=asset_path(Path('models')/str(config['models'][name]))
        add('MediaPipe '+name,path.is_file() and path.stat().st_size>0,str(path),'Run jevtells download; set JEVTELLS_HOME if assets live elsewhere.')
    for role,relative in config['render']['fonts'].items():
        path=asset_path(relative)
        add('font '+role,path.is_file() and path.stat().st_size>0,str(path),'Run jevtells download; set JEVTELLS_HOME if assets live elsewhere.')
    path=asset_path('.env')
    configured=bool(os.environ.get('OPENROUTER_API_KEY','').strip())
    if not configured and path.is_file():
        configured=bool((dotenv_values(path).get('OPENROUTER_API_KEY') or '').strip())
    add('OPENROUTER_API_KEY',configured,'已设置' if configured else '未设置','Set OPENROUTER_API_KEY in the environment or in .env under JEVTELLS_HOME/current directory/~/.jevtells.')
    return result


def run(config: dict[str, Any] | None = None) -> int:
    results=checks(config)
    for item in results:
        print(f"{'OK' if item['ok'] else 'FAIL'} {item['name']}: {item['detail']}")
        if item['repair']: print('  Fix: '+item['repair'])
    return 0 if all(item['ok'] for item in results) else 1
