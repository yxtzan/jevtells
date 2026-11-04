"""One subprocess at a time; bounded polling logs, cancellation and accounting."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

STAGES = ['prepare', 'pose', 'track', 'asr', 'voice', 'shots', 'segment', 'actions', 'scene', 'state', 'debug', 'judge', 'narrate', 'render']


def redact(text: str, root: Path) -> str:
    secrets = [os.environ.get('OPENROUTER_API_KEY', ''), dotenv_values(root / '.env').get('OPENROUTER_API_KEY') or '']
    for secret in secrets:
        if secret:
            text = text.replace(secret, '[redacted]')
    return re.sub(r'sk-(?:or-)?[A-Za-z0-9_-]+', '[redacted]', text)


class Jobs:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.lock = threading.RLock()
        self.job: dict[str, Any] = {'status': 'idle', 'logs': [], 'stage': '', 'progress': 0}
        self.process: subprocess.Popen[str] | None = None
        self.worker: threading.Thread | None = None

    def busy(self) -> bool:
        return self.job['status'] in {'running', 'cancelling'}

    def start(self, args: list[str], output: Path | None, mode: str) -> dict[str, Any]:
        with self.lock:
            if self.busy():
                raise RuntimeError('busy')
            self.job = {'id': uuid.uuid4().hex, 'status': 'running', 'stage': 'prepare', 'progress': 0, 'logs': [], 'mode': mode,
                        'output': str(output.relative_to(self.root)) if output else None, 'cost': 0.0, 'cached_cost': None}
            self.worker = threading.Thread(target=self._run, args=(args, output), daemon=True)
            self.worker.start()
            return dict(self.job)

    def _run(self, args: list[str], output: Path | None) -> None:
        ledger = self.root / 'work' / '.ui_api_ledger.jsonl'
        ledger.parent.mkdir(parents=True, exist_ok=True)
        before = len(ledger.read_text().splitlines()) if ledger.exists() else 0
        env = dict(os.environ, JEVTELLS_HOME=str(self.root), PYTHONUNBUFFERED='1',
                   JEVTELLS_API_LEDGER=str(ledger), JEVTELLS_API_LIMIT=os.environ.get('JEVTELLS_UI_API_LIMIT', '0.10'),
                   JEVTELLS_API_RESERVE='0.05')
        try:
            with self.lock:
                if self.job['status'] == 'cancelling':
                    self.job['status'] = 'cancelled'
                    return
                self.process = subprocess.Popen([sys.executable, '-u', '-m', 'jevtells.ui.runner', *args], cwd=self.root,
                    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
                process = self.process
            assert process.stdout is not None
            for line in process.stdout:
                line = redact(line.rstrip(), self.root)
                with self.lock:
                    if line.startswith('UI_STAGE '):
                        stage = line.split()[-1]
                        self.job['stage'] = stage
                        self.job['progress'] = round(STAGES.index(stage)/len(STAGES)*100) if stage in STAGES else 0
                    self.job['logs'] = [*self.job['logs'][-249:], line[:3000]]
            code = process.wait()
            with self.lock:
                self.job['status'] = 'cancelled' if self.job['status'] == 'cancelling' else 'complete' if code == 0 else 'failed'
                if code == 0:
                    self.job['progress'] = 100
                self.job['returncode'] = code
                if output and (output/'run_meta.json').exists():
                    meta = json.loads((output/'run_meta.json').read_text())
                    self.job['cached_cost'] = meta.get('api', {}).get('total_cost')
                rows = [json.loads(row) for row in ledger.read_text().splitlines()[before:]] if ledger.exists() else []
                self.job['cost'] = sum(float(row.get('cost') or 0) for row in rows)
                self.job['cost_unknown'] = any(row.get('cost') is None for row in rows)
        except Exception as exc:
            with self.lock:
                self.job['status'] = 'failed'
                self.job['logs'].append(redact(type(exc).__name__, self.root))
        finally:
            with self.lock:
                self.process = None

    def cancel(self) -> None:
        with self.lock:
            if not self.busy():
                return
            self.job['status'] = 'cancelling'
            process = self.process
            if process and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                threading.Thread(target=self._reap, args=(process,), daemon=True).start()

    def _reap(self, process: subprocess.Popen[str]) -> None:
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return json.loads(redact(json.dumps(self.job), self.root))
