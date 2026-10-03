"""Conservative gesture event rules for M2."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_RULE_CONFIG: dict[str, float | int] = {
    "min_action_duration": 0.2,
    "posture_max_duration": 3.0,
    "movement_threshold": 0.12,
    "small_magnitude": 0.12,
    "large_magnitude": 0.35,
    "state_threshold": 0.65,
    "max_events_per_window": 4,
    "merge_gap": 0.15,
}


def _cfg(config: Mapping[str, Any] | None) -> dict[str, float | int]:
    values = dict(DEFAULT_RULE_CONFIG)
    values.update(config or {})
    return values


def _array(features: Mapping[str, Any], key: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    value = features.get(key)
    if value is None:
        if shape is None:
            return np.asarray([], dtype=float)
        return np.full(shape, np.nan, dtype=float)
    return np.asarray(value, dtype=float)


def _times(features: Mapping[str, Any], n: int) -> tuple[np.ndarray, float]:
    times = _array(features, "t")
    fps = float(features.get("fps", 30.0))
    if len(times) != n:
        times = np.arange(n, dtype=float) / fps
    return times, fps


def _magnitude(amplitude: float, settings: Mapping[str, Any]) -> str:
    small = float(settings["small_magnitude"])
    large = float(settings["large_magnitude"])
    if amplitude >= large:
        return "large"
    if amplitude >= small:
        return "medium"
    return "small"


def _runs(mask: np.ndarray, times: np.ndarray, min_duration: float) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    if not len(values):
        return []
    padded = np.r_[False, values, False]
    starts = np.flatnonzero(np.diff(padded.astype(int)) == 1)
    ends = np.flatnonzero(np.diff(padded.astype(int)) == -1) - 1
    output: list[tuple[int, int]] = []
    for start, end in zip(starts, ends):
        if float(times[end] - times[start]) >= float(min_duration):
            output.append((int(start), int(end)))
    return output


def _event(event_type: str, side: str, t0: float, t1: float, amplitude: float, params: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    return {"id": "", "t0": float(t0), "t1": float(t1), "tmid": float((t0 + t1) / 2), "type": event_type, "side": side, "magnitude": _magnitude(float(amplitude), settings), "params": dict(params)}


def _movement(features: Mapping[str, Any], times: np.ndarray, settings: Mapping[str, Any], direction: str) -> list[dict[str, Any]]:
    wrists = _array(features, "wrist_norm")
    if wrists.ndim != 3 or wrists.shape[1] < 2:
        wrists = _array(features, "wrist_px")
    if wrists.ndim != 3 or wrists.shape[1] < 2:
        return []
    threshold = float(settings["movement_threshold"])
    fps = float(features.get("fps", 30.0))
    velocity = _array(features, "wrist_velocity_px_s") if "wrist_velocity_px_s" in features and "wrist_norm" not in features else np.diff(wrists, axis=0, prepend=wrists[:1]) * fps
    if velocity.shape != wrists.shape:
        velocity = np.diff(wrists, axis=0, prepend=wrists[:1]) * fps
    events: list[dict[str, Any]] = []
    for side, name in enumerate(("left hand", "right hand")):
        vertical = velocity[:, side, 1]
        if direction == "up":
            mask = vertical < -threshold
        else:
            mask = vertical > threshold
        for start, end in _runs(np.isfinite(vertical) & mask, times, float(settings["min_action_duration"])):
            displacement = float(abs(wrists[end, side, 1] - wrists[start, side, 1]))
            if displacement < threshold:
                continue
            events.append(_event("raise" if direction == "up" else "press_down", name, times[start], times[end], displacement, {"dy": displacement if direction == "down" else -displacement}, settings))
    return events


def _state_events(features: Mapping[str, Any], times: np.ndarray, settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    openness = _array(features, "hand_open")
    if openness.ndim != 2:
        return []
    events: list[dict[str, Any]] = []
    for side, name in enumerate(("left hand", "right hand")):
        for state_name, mask in (("open_palm", openness[:, side] >= float(settings["state_threshold"])), ("fist", openness[:, side] <= 1.0 - float(settings["state_threshold"]))):
            for start, end in _runs(np.isfinite(openness[:, side]) & mask, times, float(settings["min_action_duration"])):
                duration = float(times[end] - times[start])
                # A long, unchanging state is a posture (e.g. holding a mic).
                if duration > float(settings["posture_max_duration"]):
                    continue
                events.append(_event(state_name, name, times[start], times[end], float(np.nanmean(openness[start : end + 1, side])), {"open_score": float(np.nanmean(openness[start : end + 1, side]))}, settings))
    straight = _array(features, "finger_straight")
    if straight.ndim == 3 and straight.shape[2] >= 5:
        for side, name in enumerate(("left hand", "right hand")):
            point = (straight[:, side, 0] >= float(settings["state_threshold"])) & (np.nanmean(straight[:, side, 1:], axis=1) <= 1.0 - float(settings["state_threshold"]))
            for start, end in _runs(np.nan_to_num(point, nan=False), times, float(settings["min_action_duration"])):
                if times[end] - times[start] <= float(settings["posture_max_duration"]):
                    events.append(_event("point", name, times[start], times[end], 1.0, {"index_extended": True}, settings))
    palm = _array(features, "palm_up_score")
    if palm.ndim == 2:
        for side, name in enumerate(("left hand", "right hand")):
            mask = palm[:, side] > 0
            for start, end in _runs(np.isfinite(palm[:, side]) & mask, times, float(settings["min_action_duration"])):
                if times[end] - times[start] <= float(settings["posture_max_duration"]):
                    events.append(_event("palms_up", name, times[start], times[end], 1.0, {"palm_orientation": "up"}, settings))
    return events


def _two_hand_events(features: Mapping[str, Any], times: np.ndarray, settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    distance = _array(features, "two_hands_distance")
    if distance.ndim != 1 or len(distance) < 2:
        return []
    delta = np.diff(distance, prepend=distance[:1])
    threshold = float(settings["movement_threshold"]) / 2.0
    events: list[dict[str, Any]] = []
    for event_type, mask in (("spread", delta > threshold), ("gather", delta < -threshold)):
        for start, end in _runs(np.isfinite(delta) & mask, times, float(settings["min_action_duration"])):
            amount = float(abs(distance[end] - distance[start]))
            if amount < threshold:
                continue
            events.append(_event(event_type, "both hands", times[start], times[end], amount, {"distance": amount}, settings))
    return events


def _oscillation(values: np.ndarray, times: np.ndarray, event_type: str, side: str, settings: Mapping[str, Any], min_reversals: int = 2) -> list[dict[str, Any]]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 4:
        return []
    delta = np.diff(values)
    signs = np.sign(delta)
    signs[~np.isfinite(delta)] = 0
    changes = np.flatnonzero(signs[1:] * signs[:-1] < 0) + 1
    if len(changes) < min_reversals:
        return []
    start, end = int(max(0, changes[0] - 1)), int(min(len(values) - 1, changes[min_reversals - 1] + 1))
    amplitude = float(np.nanmax(values[start : end + 1]) - np.nanmin(values[start : end + 1]))
    return [_event(event_type, side, times[start], times[end], amplitude, {"reversals": int(min_reversals)}, settings)] if times[end] - times[start] >= float(settings["min_action_duration"]) else []


def _head_events(features: Mapping[str, Any], times: np.ndarray, settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    relative = _array(features, "nose_relative")
    if relative.ndim != 2 or relative.shape[1] < 2:
        return []
    return _oscillation(relative[:, 1], times, "nod", "head", settings) + _oscillation(relative[:, 0], times, "shake", "head", settings)


def _beat_events(features: Mapping[str, Any], times: np.ndarray, settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Detect repeated small wrist beats as a single emphasis event."""
    wrists = _array(features, "wrist_norm")
    if wrists.ndim != 3 or wrists.shape[1] < 2:
        return []
    events: list[dict[str, Any]] = []
    for side, name in enumerate(("left hand", "right hand")):
        events.extend(_oscillation(wrists[:, side, 1], times, "beat", name, settings, min_reversals=3))
    return events


def detect_actions(features: Mapping[str, Any], windows: Sequence[Mapping[str, Any]] | None = None, config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Infer action events and optionally cap events independently per window."""
    settings = _cfg(config)
    n = len(_array(features, "t")) or len(_array(features, "wrist_norm"))
    times, _ = _times(features, n)
    events = _movement(features, times, settings, "up") + _movement(features, times, settings, "down") + _state_events(features, times, settings) + _two_hand_events(features, times, settings) + _head_events(features, times, settings) + _beat_events(features, times, settings)
    shoulder = _array(features, "shoulder_width_relative")
    if shoulder.ndim == 1 and len(shoulder):
        delta = shoulder - np.nanmedian(shoulder)
        for start, end in _runs(delta > float(settings["movement_threshold"]), times, float(settings["min_action_duration"])):
            events.append(_event("lean_in", "upper body", times[start], times[end], float(np.nanmax(delta[start : end + 1])), {"shoulder_width_change": float(np.nanmax(delta[start : end + 1]))}, settings))
    # Merge overlapping/nearby events sharing a type and limb.
    events.sort(key=lambda item: (item["tmid"], item["type"], item["side"]))
    merged: list[dict[str, Any]] = []
    for item in events:
        if merged and item["type"] == merged[-1]["type"] and item["side"] == merged[-1]["side"] and item["t0"] <= merged[-1]["t1"] + float(settings["merge_gap"]):
            merged[-1]["t1"] = max(merged[-1]["t1"], item["t1"])
            merged[-1]["tmid"] = (merged[-1]["t0"] + merged[-1]["t1"]) / 2
            continue
        merged.append(item)
    if windows:
        limited: list[dict[str, Any]] = []
        for window in windows:
            inside = [item for item in merged if float(window["t0"]) <= item["tmid"] <= float(window["t1"])]
            inside.sort(key=lambda item: {"large": 2, "medium": 1, "small": 0}.get(item["magnitude"], 0), reverse=True)
            limited.extend(inside[: int(settings["max_events_per_window"])])
        merged = sorted(limited, key=lambda item: item["tmid"])
    for index, item in enumerate(merged, start=1):
        item["id"] = f"A{index:03d}"
    return merged


__all__ = ["detect_actions", "DEFAULT_RULE_CONFIG"]
