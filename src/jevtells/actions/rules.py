"""Conservative, configuration-driven gesture event rules for M2."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


SHAPE_TYPES = {"open_palm", "fist", "point", "palms_up"}
MOTION_TYPES = {"raise", "press_down", "beat", "spread", "gather"}
SHAPE_PRIORITY = {"fist": 0, "open_palm": 1, "palms_up": 2, "point": 3}

DEFAULT_RULE_CONFIG: dict[str, Any] = {
    "pose_visibility_threshold": 0.5,
    "motion_visibility_threshold": 0.75,
    "edge_margin": 0.04,
    "hand_detection_ratio": 0.80,
    "occupied_window_seconds": 10.0,
    "occupied_face_radius": 1.0,
    "occupied_ratio": 0.60,
    "occupied_event_ratio": 0.50,
    "posture_window": 5.0,
    "posture_ratio": 0.70,
    "min_action_duration": 0.2,
    "movement_threshold": 0.12,
    "state_threshold": 0.65,
    "fist_max_openness": 0.20,
    "fist_max_tip_palm_ratio": 1.00,
    "point_frame_ratio": 0.80,
    "max_per_window": 4,
    "merge_gap": 0.20,
    "small_threshold": 0.15,
    "large_threshold": 0.35,
    "palms_up_min_score": 0.50,
    "palms_up_max_vertical_cos": 0.70710678,
    "palms_up_min_nose_offset": 0.30,
    "hand_point_frame_ratio": 1.0,
    "lean_in": {"enabled": False},
}


def _cfg(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Resolve the same keys used by ``config/default.yaml``."""
    values = dict(DEFAULT_RULE_CONFIG)
    if isinstance(config, Mapping):
        actions = config.get("actions")
        if isinstance(actions, Mapping):
            values.update(actions)
        values.update({key: value for key, value in config.items() if key != "actions"})
        if isinstance(config.get("lean_in"), Mapping):
            values["lean_in"] = dict(config["lean_in"])
    return values


def _array(features: Mapping[str, Any], key: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    value = features.get(key)
    if value is None:
        return np.full(shape, np.nan, dtype=float) if shape is not None else np.asarray([], dtype=float)
    return np.asarray(value, dtype=float)


def _times(features: Mapping[str, Any], n: int) -> tuple[np.ndarray, float]:
    times = _array(features, "t")
    fps = float(features.get("fps", 30.0))
    if len(times) != n:
        times = np.arange(n, dtype=float) / fps
    return times, fps


def _span(times: np.ndarray, t0: float, t1: float) -> np.ndarray:
    return np.flatnonzero((times >= float(t0)) & (times <= float(t1)))


def _magnitude(amplitude: float, settings: Mapping[str, Any]) -> str:
    if amplitude >= float(settings["large_threshold"]):
        return "large"
    if amplitude >= float(settings["small_threshold"]):
        return "medium"
    return "small"


def _runs(mask: np.ndarray, times: np.ndarray, min_duration: float) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    if not len(values):
        return []
    padded = np.r_[False, values, False]
    starts = np.flatnonzero(np.diff(padded.astype(int)) == 1)
    ends = np.flatnonzero(np.diff(padded.astype(int)) == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends) if float(times[end] - times[start]) >= float(min_duration)]


def _tolerant_runs(mask: np.ndarray, times: np.ndarray, min_duration: float, merge_gap: float) -> list[tuple[int, int]]:
    """Group state runs separated by a short detector flicker."""
    runs = _runs(mask, times, min_duration)
    if not runs:
        return []
    merged: list[list[int]] = [[runs[0][0], runs[0][1]]]
    for start, end in runs[1:]:
        if float(times[start] - times[merged[-1][1]]) <= float(merge_gap):
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _event(event_type: str, side: str, t0: float, t1: float, amplitude: float | None, params: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    magnitude = None if event_type in SHAPE_TYPES else _magnitude(float(amplitude or 0.0), settings)
    return {"id": "", "t0": float(t0), "t1": float(t1), "tmid": float((t0 + t1) / 2), "type": event_type, "side": side, "magnitude": magnitude, "params": dict(params)}


def _sides(event: Mapping[str, Any]) -> tuple[int, ...]:
    return {"left hand": (0,), "right hand": (1,), "both hands": (0, 1)}.get(str(event.get("side", "")), ())


def _motion_valid(features: Mapping[str, Any], times: np.ndarray, event: Mapping[str, Any], settings: Mapping[str, Any]) -> bool:
    indices = _span(times, float(event["t0"]), float(event["t1"]))
    if not len(indices):
        return False
    visibility = _array(features, "wrist_visibility")
    edge_ok = np.asarray(features.get("wrist_edge_ok", []), dtype=bool)
    if visibility.ndim != 2 or edge_ok.ndim != 2:
        return True
    for side in _sides(event):
        if side >= visibility.shape[1] or side >= edge_ok.shape[1]:
            return False
        if not np.all(np.isfinite(visibility[indices, side]) & (visibility[indices, side] >= float(settings["motion_visibility_threshold"]))):
            return False
        if not np.all(edge_ok[indices, side]):
            return False
    return True


def _hand_shape_valid(features: Mapping[str, Any], times: np.ndarray, event: Mapping[str, Any], settings: Mapping[str, Any]) -> bool:
    ratios = _array(features, "hand_point_ratio")
    indices = _span(times, float(event["t0"]), float(event["t1"]))
    if ratios.ndim != 2 or not len(indices):
        return True
    required = float(settings["hand_detection_ratio"])
    frame_threshold = float(settings["hand_point_frame_ratio"])
    return all(side < ratios.shape[1] and float(np.mean(ratios[indices, side] >= frame_threshold)) >= required for side in _sides(event))


def _fist_valid(features: Mapping[str, Any], times: np.ndarray, event: Mapping[str, Any], settings: Mapping[str, Any]) -> bool:
    """Require a closed hand and all four fingertips within the palm."""
    indices = _span(times, float(event["t0"]), float(event["t1"]))
    if not len(indices):
        return False
    openness = _array(features, "hand_open")
    palm_ratio = _array(features, "fingertip_palm_ratio")
    # Older synthetic callers only provided ``hand_open``.  Keep that input
    # usable while real extracted features always include the geometry check.
    if palm_ratio.ndim != 3 or palm_ratio.shape[2] < 4:
        return all(
            side < openness.shape[1]
            and float(np.nanmean(openness[indices, side])) <= float(settings["fist_max_openness"])
            for side in _sides(event)
        )
    for side in _sides(event):
        if side >= palm_ratio.shape[1] or side >= openness.shape[1]:
            return False
        closed = np.isfinite(openness[indices, side]) & (openness[indices, side] <= float(settings["fist_max_openness"]))
        near_palm = np.isfinite(palm_ratio[indices, side, :4]).all(axis=1) & np.all(
            palm_ratio[indices, side, :4] <= float(settings["fist_max_tip_palm_ratio"]), axis=1
        )
        if not np.all(closed & near_palm):
            return False
    return True


def _point_condition(features: Mapping[str, Any], settings: Mapping[str, Any]) -> np.ndarray:
    """Return per-frame, per-hand index-point geometry validity."""
    straight = _array(features, "finger_straight")
    if straight.ndim != 3 or straight.shape[2] < 4:
        return np.asarray([], dtype=bool)
    index_extended = straight[:, :, 0] >= float(settings["state_threshold"])
    tip_dist = _array(features, "finger_tip_wrist_distance")
    root_dist = _array(features, "finger_root_wrist_distance")
    if tip_dist.ndim == 3 and root_dist.ndim == 3 and tip_dist.shape[2] >= 4 and root_dist.shape[2] >= 4:
        curled = np.isfinite(tip_dist[:, :, 1:4]) & np.isfinite(root_dist[:, :, 1:4])
        curled &= tip_dist[:, :, 1:4] < root_dist[:, :, 1:4]
        return index_extended & np.all(curled, axis=2)
    # Compatibility for hand-crafted feature dictionaries from M2 tests.
    other = straight[:, :, 1:]
    finite = np.sum(np.isfinite(other), axis=2)
    other_mean = np.divide(np.nansum(other, axis=2), finite, out=np.full(index_extended.shape, np.nan), where=finite > 0)
    return index_extended & (other_mean <= 1.0 - float(settings["state_threshold"]))


def _point_valid(features: Mapping[str, Any], times: np.ndarray, event: Mapping[str, Any], settings: Mapping[str, Any]) -> bool:
    condition = _point_condition(features, settings)
    indices = _span(times, float(event["t0"]), float(event["t1"]))
    if condition.ndim != 2 or not len(indices):
        return False
    for side in _sides(event):
        if side >= condition.shape[1] or float(np.mean(condition[indices, side])) < float(settings["point_frame_ratio"]):
            return False
    return True


def _occupied(features: Mapping[str, Any], times: np.ndarray, event: Mapping[str, Any], settings: Mapping[str, Any]) -> bool:
    values = np.asarray(features.get("occupied_mask", []), dtype=bool)
    indices = _span(times, float(event["t0"]), float(event["t1"]))
    if values.ndim != 2 or not len(indices):
        return False
    required = float(settings["occupied_event_ratio"])
    return any(side < values.shape[1] and float(np.mean(values[indices, side])) >= required for side in _sides(event))


def _posture_mask(mask: np.ndarray, times: np.ndarray, settings: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    result = np.zeros(len(values), dtype=bool)
    half = float(settings["posture_window"]) / 2.0
    min_state_duration = float(settings["posture_window"]) * float(settings["posture_ratio"])
    for index, timestamp in enumerate(times):
        local = np.flatnonzero(np.abs(times - timestamp) <= half)
        if len(local) and float(np.mean(values[local])) >= float(settings["posture_ratio"]):
            active = local[values[local]]
            active_duration = float(times[active[-1]] - times[active[0]]) if len(active) > 1 else 0.0
            if active_duration >= min_state_duration:
                result[index] = True
    return result


def _movement(features: Mapping[str, Any], times: np.ndarray, settings: Mapping[str, Any], direction: str) -> list[dict[str, Any]]:
    wrists = _array(features, "wrist_norm")
    if wrists.ndim != 3 or wrists.shape[1] < 2:
        return []
    threshold = float(settings["movement_threshold"])
    fps = float(features.get("fps", 30.0))
    velocity = np.diff(wrists, axis=0, prepend=wrists[:1]) * fps
    events: list[dict[str, Any]] = []
    for side, name in enumerate(("left hand", "right hand")):
        vertical = velocity[:, side, 1]
        mask = vertical < -threshold if direction == "up" else vertical > threshold
        for start, end in _runs(np.isfinite(vertical) & mask, times, float(settings["min_action_duration"])):
            displacement = float(abs(wrists[end, side, 1] - wrists[start, side, 1]))
            if displacement < threshold:
                continue
            event = _event("raise" if direction == "up" else "press_down", name, times[start], times[end], displacement, {"dy": displacement if direction == "down" else -displacement}, settings)
            if _motion_valid(features, times, event, settings):
                events.append(event)
    return events


def _state_events(features: Mapping[str, Any], times: np.ndarray, settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    openness = _array(features, "hand_open")
    if openness.ndim != 2:
        openness = np.full((len(times), 2), np.nan, dtype=float)
    events: list[dict[str, Any]] = []
    for side, name in enumerate(("left hand", "right hand")):
        valid_open = np.isfinite(openness[:, side])
        for state_name, state_mask in (("open_palm", openness[:, side] >= float(settings["state_threshold"])), ("fist", openness[:, side] <= float(settings["fist_max_openness"]))):
            posture = _posture_mask(valid_open & state_mask, times, settings)
            runs = _tolerant_runs(valid_open & state_mask, times, float(settings["min_action_duration"]), float(settings["merge_gap"])) if state_name == "fist" else _runs(valid_open & state_mask, times, float(settings["min_action_duration"]))
            for start, end in runs:
                if np.any(posture[max(0, start - int(float(settings["posture_window"]) * float(features.get("fps", 30.0)) / 2)) : min(len(posture), end + 1 + int(float(settings["posture_window"]) * float(features.get("fps", 30.0)) / 2))]):
                    continue
                event = _event(state_name, name, times[start], times[end], None, {"open_score": float(np.nanmean(openness[start : end + 1, side]))}, settings)
                if state_name != "fist" or (_fist_valid(features, times, event, settings) and _hand_shape_valid(features, times, event, settings)):
                    events.append(event)
    straight = _array(features, "finger_straight")
    if straight.ndim == 3 and straight.shape[2] >= 4:
        for side, name in enumerate(("left hand", "right hand")):
            point = _point_condition(features, settings)[:, side]
            point = np.nan_to_num(point, nan=False)
            posture = _posture_mask(point, times, settings)
            for start, end in _tolerant_runs(point, times, float(settings["min_action_duration"]), float(settings["merge_gap"])):
                if np.any(posture[max(0, start - int(float(settings["posture_window"]) * float(features.get("fps", 30.0)) / 2)) : min(len(posture), end + 1 + int(float(settings["posture_window"]) * float(features.get("fps", 30.0)) / 2))]):
                    continue
                event = _event("point", name, times[start], times[end], None, {"index_extended": True}, settings)
                if _point_valid(features, times, event, settings) and _hand_shape_valid(features, times, event, settings):
                    events.append(event)
    palm = _array(features, "palm_up_score")
    vertical_cos = _array(features, "finger_vertical_cos")
    below_nose = _array(features, "wrist_below_nose")
    if palm.ndim == 2:
        for side, name in enumerate(("left hand", "right hand")):
            if side >= palm.shape[1]:
                continue
            mask = palm[:, side] >= float(settings["palms_up_min_score"])
            if vertical_cos.ndim == 2 and side < vertical_cos.shape[1]:
                mask &= vertical_cos[:, side] < float(settings["palms_up_max_vertical_cos"])
            if below_nose.ndim == 2 and side < below_nose.shape[1]:
                mask &= below_nose[:, side] >= float(settings["palms_up_min_nose_offset"])
            posture = _posture_mask(mask, times, settings)
            for start, end in _runs(np.isfinite(palm[:, side]) & mask, times, float(settings["min_action_duration"])):
                if np.any(posture[max(0, start - int(float(settings["posture_window"]) * float(features.get("fps", 30.0)) / 2)) : min(len(posture), end + 1 + int(float(settings["posture_window"]) * float(features.get("fps", 30.0)) / 2))]):
                    continue
                event = _event("palms_up", name, times[start], times[end], None, {"palm_orientation": "up"}, settings)
                if _hand_shape_valid(features, times, event, settings):
                    events.append(event)
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
            event = _event(event_type, "both hands", times[start], times[end], amount, {"distance": amount}, settings)
            if _motion_valid(features, times, event, settings):
                events.append(event)
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
    if times[end] - times[start] < float(settings["min_action_duration"]):
        return []
    return [_event(event_type, side, times[start], times[end], amplitude, {"reversals": int(min_reversals)}, settings)]


def _head_events(features: Mapping[str, Any], times: np.ndarray, settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    relative = _array(features, "nose_relative")
    if relative.ndim != 2 or relative.shape[1] < 2:
        return []
    return _oscillation(relative[:, 1], times, "nod", "head", settings) + _oscillation(relative[:, 0], times, "shake", "head", settings)


def _beat_events(features: Mapping[str, Any], times: np.ndarray, settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    wrists = _array(features, "wrist_norm")
    if wrists.ndim != 3 or wrists.shape[1] < 2:
        return []
    events: list[dict[str, Any]] = []
    for side, name in enumerate(("left hand", "right hand")):
        for event in _oscillation(wrists[:, side, 1], times, "beat", name, settings, min_reversals=3):
            if _motion_valid(features, times, event, settings):
                events.append(event)
    return events


def _resolve_shape_conflicts(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    shapes = [event for event in events if event["type"] in SHAPE_TYPES]
    others = [event for event in events if event["type"] not in SHAPE_TYPES]
    kept: list[dict[str, Any]] = []
    for event in sorted(shapes, key=lambda item: (-SHAPE_PRIORITY[item["type"]], item["t0"], item["t1"])):
        if any(existing["side"] == event["side"] and event["t0"] <= existing["t1"] and existing["t0"] <= event["t1"] for existing in kept):
            continue
        kept.append(event)
    return others + kept


def _merge(events: Sequence[dict[str, Any]], settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    ordered = sorted(events, key=lambda item: (item["tmid"], item["type"], item["side"]))
    merged: list[dict[str, Any]] = []
    for item in ordered:
        if merged and item["type"] == merged[-1]["type"] and item["side"] == merged[-1]["side"] and item["t0"] <= merged[-1]["t1"] + float(settings["merge_gap"]):
            merged[-1]["t1"] = max(merged[-1]["t1"], item["t1"])
            merged[-1]["tmid"] = (merged[-1]["t0"] + merged[-1]["t1"]) / 2
            continue
        merged.append(dict(item))
    return merged


def _filter_shape_quality(events: Sequence[dict[str, Any]], features: Mapping[str, Any], times: np.ndarray, settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Recheck hand-point coverage over the final merged event span."""
    kept: list[dict[str, Any]] = []
    for event in events:
        event_type = event["type"]
        if event_type == "fist":
            valid = _fist_valid(features, times, event, settings)
        elif event_type == "point":
            valid = _point_valid(features, times, event, settings) and _hand_shape_valid(features, times, event, settings)
        else:
            valid = event_type not in SHAPE_TYPES or _hand_shape_valid(features, times, event, settings)
        if valid:
            kept.append(event)
    return kept


def _filter_occupied(events: Sequence[dict[str, Any]], features: Mapping[str, Any], times: np.ndarray, settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for event in events:
        if not _occupied(features, times, event, settings):
            filtered.append(event)
            continue
        if event["type"] in SHAPE_TYPES:
            continue
        if event["type"] in MOTION_TYPES and event.get("magnitude") != "large":
            continue
        filtered.append(event)
    return filtered


def detect_actions(features: Mapping[str, Any], windows: Sequence[Mapping[str, Any]] | None = None, config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Infer events, filter unreliable/occupied hands, and cap each window."""
    settings = _cfg(config)
    n = len(_array(features, "t")) or len(_array(features, "wrist_norm"))
    times, _ = _times(features, n)
    events = _movement(features, times, settings, "up") + _movement(features, times, settings, "down")
    events += _state_events(features, times, settings) + _two_hand_events(features, times, settings)
    events += _head_events(features, times, settings) + _beat_events(features, times, settings)
    lean_config = settings.get("lean_in", {})
    if isinstance(lean_config, Mapping) and bool(lean_config.get("enabled", False)):
        shoulder = _array(features, "shoulder_width_relative")
        if shoulder.ndim == 1 and len(shoulder):
            delta = shoulder - np.nanmedian(shoulder)
            for start, end in _runs(delta > float(settings["movement_threshold"]), times, float(settings["min_action_duration"])):
                events.append(_event("lean_in", "upper body", times[start], times[end], float(np.nanmax(delta[start : end + 1])), {"shoulder_width_change": float(np.nanmax(delta[start : end + 1]))}, settings))
    events = _filter_occupied(events, features, times, settings)
    events = _resolve_shape_conflicts(events)
    events = _merge(events, settings)
    events = _filter_shape_quality(events, features, times, settings)
    if windows:
        limited: list[dict[str, Any]] = []
        for window in windows:
            inside = [item for item in events if float(window["t0"]) <= item["tmid"] <= float(window["t1"])]
            # Keep confirmed large motions first, while hand-shape events
            # compete on equal footing with medium motions instead of being
            # unconditionally displaced by them. Earlier events win ties.
            weights = {"large": 2, "medium": 1, None: 1, "small": 0}
            inside.sort(key=lambda item: (-weights.get(item.get("magnitude"), 0), item["tmid"], item["t0"]))
            limited.extend(inside[: int(settings["max_per_window"])])
        events = sorted(limited, key=lambda item: item["tmid"])
    for index, item in enumerate(events, start=1):
        item["id"] = f"A{index:03d}"
    return events


__all__ = ["detect_actions", "DEFAULT_RULE_CONFIG", "SHAPE_TYPES", "MOTION_TYPES"]
