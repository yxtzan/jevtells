"""Detect the target shot span for the single-speaker M1 clip."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from typing import Any, Mapping


def run(clip: Path, out: Path, force: bool = False, points: Mapping[str, Any] | None = None, config: Mapping[str, Any] | None = None) -> list[dict[str, object]]:
    """Write shot spans labelled by whether the tracked target is present."""
    destination = out / "shots.json"
    if destination.exists() and not force:
        return json.loads(destination.read_text())
    capture = cv2.VideoCapture(str(clip))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()
    total = int(frames)
    present = None
    if points is not None and "pose_present" in points:
        present = np.asarray(points["pose_present"], dtype=bool)
        total = min(total, len(present))
    if present is None or total == 0:
        present = np.ones(total, dtype=bool)
    present = present[:total]
    shots: list[dict[str, object]] = []
    if total:
        start = 0
        current = bool(present[0])
        for index in range(1, total + 1):
            if index == total or bool(present[index]) != current:
                shots.append({"t0": float(start / fps), "t1": float(index / fps), "label": "target" if current else "other"})
                if index < total:
                    start, current = index, bool(present[index])
    destination.write_text(json.dumps(shots, ensure_ascii=False, indent=2))
    return shots
