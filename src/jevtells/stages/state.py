"""Build the five-field state sent to later Jev stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..schemas import State


def _voice_text(voice: dict[str, Any], window: dict[str, Any]) -> str:
    """Describe measured voice features for one window."""
    times = np.asarray(voice.get("t", []), dtype=float)
    values = np.asarray(voice.get("rms_db", []), dtype=float)
    selected = values[(times >= window["t0"]) & (times <= window["t1"])]
    baseline = voice.get("baseline_rms_db")
    if not len(selected) or baseline is None:
        return "no voice frames measured"
    return f"loudness {float(np.mean(selected) - baseline):+.1f} dB vs clip baseline"


def _actions(points: dict[str, Any], window: dict[str, Any]) -> list[str]:
    """Summarize measured wrist travel for both sides."""
    times = np.asarray(points["t"], dtype=float)
    pose = np.asarray(points["pose"], dtype=float)
    result: list[str] = []
    mask = (times >= window["t0"]) & (times <= window["t1"])
    for side, name, index in ((0, "left", 15), (1, "right", 16)):
        wrists = pose[mask, index, :2]
        wrists = wrists[np.isfinite(wrists).all(axis=1)]
        if len(wrists) < 1:
            result.append(f"{name} wrist: no detection in window")
            continue
        travel = float(np.linalg.norm(np.diff(wrists, axis=0), axis=1).sum()) if len(wrists) > 1 else 0.0
        result.append(f"{name} wrist: total travel {travel:.3f} normalized image units")
    return result


def run(windows: list[dict[str, Any]], voice: dict[str, Any], scene: str, speaker: str, out: Path, force: bool = False, points: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate and write exactly five state fields for every window."""
    destination = out / "states.json"
    if destination.exists() and not force:
        return json.loads(destination.read_text())
    states: dict[str, Any] = {}
    measured_points = points or {"t": [], "pose": np.empty((0, 33, 4))}
    for window in windows:
        state = State(scene=scene, speaker=speaker, subtitle={"current": window["subtitle"], "previous": window["prev_subtitle"]}, voice=_voice_text(voice, window), measured_actions=_actions(measured_points, window))
        states[window["id"]] = state.model_dump()
    destination.write_text(json.dumps(states, ensure_ascii=False, indent=2))
    return states
