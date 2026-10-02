"""Detect the target shot span for the single-speaker M1 clip."""

from __future__ import annotations

import json
from pathlib import Path

import cv2


def run(clip: Path, out: Path, force: bool = False) -> list[dict[str, object]]:
    """Write a target shot covering the actual video duration."""
    destination = out / "shots.json"
    if destination.exists() and not force:
        return json.loads(destination.read_text())
    capture = cv2.VideoCapture(str(clip))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()
    shots = [{"t0": 0.0, "t1": float(frames / fps), "label": "target"}]
    destination.write_text(json.dumps(shots, ensure_ascii=False, indent=2))
    return shots
