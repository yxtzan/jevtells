"""Build the five-field state sent to later Jev stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..schemas import State
from .voice import format_voice_text, window_voice_metrics


def _voice_text(voice: dict[str, Any], window: dict[str, Any], transcript: dict[str, Any] | None = None, config: dict[str, Any] | None = None) -> str:
    """Describe all four measured voice dimensions for one window."""

    try:
        return format_voice_text(window_voice_metrics(voice, window, transcript, config))
    except (TypeError, ValueError, KeyError):
        return "loudness unknown; unknown pitch variation; speech rate 0.0 words/s; unknown pauses"


def action_text(event: dict[str, Any]) -> str:
    """Use one description for state generation and judgment ID matching."""
    midpoint = float(event.get("mid", event.get("tmid", (float(event.get("t0", 0.0)) + float(event.get("t1", 0.0))) / 2)))
    limb = event.get("limb", event.get("side", ""))
    action_type = event.get("type", "gesture")
    start = float(event.get("start", event.get("t0", midpoint)))
    end = float(event.get("end", event.get("t1", midpoint)))
    magnitude = event.get("amplitude", event.get("magnitude"))
    text = f"{limb}: {action_type}, {max(0.0, end - start):.1f}s"
    if magnitude is not None and action_type not in {"open_palm", "fist", "point", "palms_up"}:
        text += f", {magnitude}"
    return text


def _actions(
    points: dict[str, Any],
    window: dict[str, Any],
    actions: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> list[str]:
    """Read action events for this window, with an explicit empty result."""

    hands = np.asarray(points.get('hands', []), dtype=float)
    times = np.asarray(points.get('t', []), dtype=float)
    mostly_unseen = False
    if hands.ndim == 4 and len(hands) and len(times)==len(hands):
        mask = (times>=float(window['t0'])) & (times<float(window['t1']))
        visible = np.isfinite(hands[...,:2]).all(axis=-1) & (hands[...,0]>=0) & (hands[...,0]<=1) & (hands[...,1]>=0) & (hands[...,1]<=1)
        ratio = visible.mean(axis=-1)
        any_hand = (ratio >= float((config or {}).get('actions',{}).get('hand_detection_ratio',.8))).any(axis=1)
        mostly_unseen = bool(mask.any() and np.mean(~any_hand[mask]) > float((config or {}).get('actions',{}).get('hands_missing_ratio',.5)))
    if actions is not None:
        events = actions.get("events", actions) if isinstance(actions, dict) else actions
        if isinstance(events, list):
            selected: list[str] = []
            minimum_motion = "medium"
            if isinstance(config, dict):
                candidate = config.get("state_min_motion_magnitude")
                if candidate is None and isinstance(config.get("actions"), dict):
                    candidate = config["actions"].get("state_min_motion_magnitude")
                if isinstance(candidate, str):
                    minimum_motion = candidate
            if isinstance(actions, dict) and isinstance(actions.get("state_min_motion_magnitude"), str):
                minimum_motion = str(actions["state_min_motion_magnitude"])
            motion_types = {"raise", "press_down", "beat", "spread", "gather", "nod", "shake", "lean_in"}
            magnitude_rank = {"small": 0, "medium": 1, "large": 2}
            for event in events:
                if event.get("far"):
                    continue
                try:
                    midpoint = float(event.get("mid", event.get("tmid", (float(event.get("t0", 0.0)) + float(event.get("t1", 0.0))) / 2)))
                    if float(window["t0"]) <= midpoint <= float(window["t1"]):
                        action_type = event.get("type", "gesture")
                        magnitude = event.get("amplitude", event.get("magnitude"))
                        if action_type in motion_types and magnitude_rank.get(str(magnitude), 0) < magnitude_rank.get(minimum_motion, 1):
                            continue
                        selected.append(action_text(event))
                except (TypeError, ValueError, AttributeError):
                    continue
            return selected + (["hands mostly not visible"] if mostly_unseen else []) if selected else ["hands mostly not visible" if mostly_unseen else "no notable gestures"]

    times = np.asarray(points.get("t", []), dtype=float)
    pose = np.asarray(points.get("pose", []), dtype=float)
    result: list[str] = []
    mask = (times >= window["t0"]) & (times <= window["t1"])
    if pose.ndim != 3 or pose.shape[1] <= 16:
        return ["no notable gestures"]
    for side, name, index in ((0, "left", 15), (1, "right", 16)):
        wrists = pose[mask, index, :2]
        wrists = wrists[np.isfinite(wrists).all(axis=1)]
        if len(wrists) < 1:
            continue
        travel = float(np.linalg.norm(np.diff(wrists, axis=0), axis=1).sum()) if len(wrists) > 1 else 0.0
        if travel > 0:
            result.append(f"{name} wrist: total travel {travel:.3f} normalized image units")
    return result or ["hands mostly not visible" if mostly_unseen else "no notable gestures"]


def run(
    windows: list[dict[str, Any]],
    voice: dict[str, Any],
    scene: str,
    speaker: str,
    out: Path,
    force: bool = False,
    points: dict[str, Any] | None = None,
    actions: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    transcript: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and write exactly five state fields for every window."""

    destination = out / "states.json"
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    if transcript is None:
        transcript_path = out / "transcript.json"
        if transcript_path.exists():
            transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    if actions is None:
        actions_path = out / "actions.json"
        if actions_path.exists():
            loaded = json.loads(actions_path.read_text(encoding="utf-8"))
            actions = loaded if isinstance(loaded, dict) else {"events": loaded}
    states: dict[str, Any] = {}
    measured_points = points or {"t": [], "pose": np.empty((0, 33, 4))}
    for window in windows:
        state = State(scene=scene, speaker=speaker, subtitle={"current": window["subtitle"], "previous": window.get("prev_subtitle", "")}, voice=_voice_text(voice, window, transcript, config), measured_actions=_actions(measured_points, window, actions, config))
        states[window["id"]] = state.model_dump()
    destination.write_text(json.dumps(states, ensure_ascii=False, indent=2), encoding="utf-8")
    return states
