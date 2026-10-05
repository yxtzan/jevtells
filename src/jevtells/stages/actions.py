"""M2 action stage.

The action detector itself lives in :mod:`jevtells.actions.features` and
:mod:`jevtells.actions.rules`.  This module is deliberately a small stage
adapter: it normalises the detector's internal event names to the public JSON
schema, assigns events to windows, and writes the human-review artifacts.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from ..actions.features import extract_features
from ..actions.rules import SHAPE_TYPES, detect_actions


_LIMB_NAMES = {
    "left hand": ("left_hand", "left_wrist"),
    "right hand": ("right_hand", "right_wrist"),
    "both hands": ("both_hands", "both_wrists"),
    "head": ("head", "nose"),
    "upper body": ("torso", "shoulders"),
    "torso": ("torso", "shoulders"),
    "left_hand": ("left_hand", "left_wrist"),
    "right_hand": ("right_hand", "right_wrist"),
    "both_hands": ("both_hands", "both_wrists"),
}


def mark_far(events: Sequence[Mapping[str, Any]], shots: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for event in events:
        value = dict(event)
        mid = (float(value["t0"]) + float(value["t1"])) / 2
        shot = next((shot for shot in shots if float(shot["t0"]) <= mid < float(shot["t1"])), None)
        value["far"] = bool(shot and shot.get("far"))
        value["shot_index"] = shot.get("index") if shot else None
        result.append(value)
    return result


def _far_events(events: Sequence[Mapping[str, Any]], out: Path) -> list[dict[str, Any]]:
    path = out / "shots.json"
    return mark_far(events, json.loads(path.read_text())) if path.exists() else list(events)


def _settings(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Flatten the config keys consumed by feature extraction and rules."""

    root = dict(config or {})
    result: dict[str, Any] = {}
    for section in ("actions", "smoothing"):
        value = root.get(section, {})
        if isinstance(value, Mapping):
            result.update(value)
    lean_in = root.get("lean_in")
    if isinstance(lean_in, Mapping):
        result["lean_in"] = dict(lean_in)
    result.setdefault("interpolation_max_gap", result.get("max_gap_frames", 5))
    result.setdefault("smoothing_window", result.get("window_length", 7))
    result.setdefault("smoothing_polynomial", result.get("polyorder", 2))
    return result


def _json_value(value: Any) -> Any:
    """Turn numpy scalars/arrays into values accepted by ``json.dumps``."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _normalise_event(event: Mapping[str, Any], windows: Sequence[Mapping[str, Any]], fallback_id: int) -> dict[str, Any]:
    """Map rule-internal names to the exact public ``actions.json`` schema."""

    t0 = float(event.get("t0", event.get("start", 0.0)))
    t1 = float(event.get("t1", event.get("end", t0)))
    if t1 < t0:
        t0, t1 = t1, t0
    midpoint = float(event.get("tmid", (t0 + t1) / 2.0))
    raw_limb = str(event.get("limb", event.get("side", ""))).strip()
    limb, anchor = _LIMB_NAMES.get(raw_limb, (raw_limb.replace(" ", "_"), raw_limb.replace(" ", "_")))
    window_id: str | None = None
    for window in windows:
        try:
            if float(window["t0"]) <= midpoint <= float(window["t1"]):
                window_id = str(window["id"])
                break
        except (KeyError, TypeError, ValueError):
            continue
    params = _json_value(event.get("params", {}))
    event_type = str(event.get("type", "gesture"))
    magnitude = None if event_type in SHAPE_TYPES else event.get("magnitude", event.get("amplitude", "small"))
    return {
        "id": str(event.get("id") or f"A{fallback_id:03d}"),
        "t0": t0,
        "t1": t1,
        "limb": limb,
        "type": event_type,
        "magnitude": magnitude,
        "params": params if isinstance(params, Mapping) else {},
        "anchor": str(event.get("anchor", anchor)),
        "window": window_id,
    }


_POSE_EDGES = (
    (0, 11), (0, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
)
_HIGHLIGHT_EDGES = {
    "left_hand": ((11, 13), (13, 15)),
    "right_hand": ((12, 14), (14, 16)),
    "both_hands": ((11, 13), (13, 15), (12, 14), (14, 16)),
    "head": ((0, 11), (0, 12), (11, 12)),
    "torso": ((11, 12), (11, 23), (12, 24), (23, 24)),
}


def _nearest_indices(times: np.ndarray, requested: Sequence[float]) -> list[int]:
    if len(times) == 0:
        return [0 for _ in requested]
    return [int(np.nanargmin(np.abs(times - float(value)))) for value in requested]


def _read_video_frames(clip: Path | None, indices: Sequence[int]) -> dict[int, np.ndarray]:
    """Read only the requested frames from the source clip."""

    if clip is None or not Path(clip).exists() or not indices:
        return {}
    capture = cv2.VideoCapture(str(clip))
    frames: dict[int, np.ndarray] = {}
    for index in sorted(set(int(i) for i in indices if int(i) >= 0)):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if ok and frame is not None:
            frames[index] = frame
    capture.release()
    return frames


def _review_image(
    features: Mapping[str, Any],
    points: Mapping[str, Any],
    event: Mapping[str, Any],
    output: Path,
    frames: Mapping[int, np.ndarray] | None = None,
) -> None:
    """Render a clear three-panel event review image."""

    pose = np.asarray(features.get("pose_px", []), dtype=float)
    times = np.asarray(features.get("t", []), dtype=float)
    if pose.ndim != 3 or len(pose) == 0:
        return
    t0, t1 = float(event["t0"]), float(event["t1"])
    tm = (t0 + t1) / 2.0
    indices = _nearest_indices(times, (t0, tm, t1))
    source_frames = frames or {}
    hands = np.asarray(points.get("hands", []), dtype=float)
    width = float(features.get("width", points.get("width", 1.0)) or 1.0)
    height = float(features.get("height", points.get("height", 1.0)) or 1.0)
    limb = str(event.get("limb", ""))
    highlight = set(_HIGHLIGHT_EDGES.get(limb, ()))

    panels: list[np.ndarray] = []
    panel_width, panel_height = 480, 360
    for panel_index, frame_index in enumerate(indices):
        frame = source_frames.get(frame_index)
        if frame is None:
            canvas = np.full((panel_height, panel_width, 3), 246, dtype=np.uint8)
            image_width, image_height = width, height
            sx = (panel_width - 30) / max(image_width, 1.0)
            sy = (panel_height - 65) / max(image_height, 1.0)
            ox, oy = 15.0, 45.0
        else:
            canvas = frame.copy()
            image_height, image_width = canvas.shape[:2]
            scale = min(panel_width / max(image_width, 1), panel_height / max(image_height, 1))
            resized = cv2.resize(canvas, (max(1, int(image_width * scale)), max(1, int(image_height * scale))))
            canvas = np.full((panel_height, panel_width, 3), 246, dtype=np.uint8)
            ox, oy = (panel_width - resized.shape[1]) / 2.0, 38.0 + (panel_height - 38 - resized.shape[0]) / 2.0
            canvas[int(oy) : int(oy) + resized.shape[0], int(ox) : int(ox) + resized.shape[1]] = resized
            sx = resized.shape[1] / max(image_width, 1)
            sy = resized.shape[0] / max(image_height, 1)

        measured = pose[min(max(frame_index, 0), len(pose) - 1)]

        def xy(index: int) -> tuple[int, int] | None:
            if index >= len(measured) or not np.isfinite(measured[index, :2]).all():
                return None
            return (int(ox + measured[index, 0] * sx), int(oy + measured[index, 1] * sy))

        for first, second in _POSE_EDGES:
            first_xy, second_xy = xy(first), xy(second)
            if first_xy and second_xy:
                color, thickness = ((40, 220, 245), 5) if (first, second) in highlight else ((150, 150, 150), 2)
                cv2.line(canvas, first_xy, second_xy, color, thickness, cv2.LINE_AA)
        for index in range(len(measured)):
            location = xy(index)
            if location:
                color = (30, 60, 230) if index in {0, 11, 12, 13, 14, 15, 16} else (90, 90, 90)
                cv2.circle(canvas, location, 4, color, -1, cv2.LINE_AA)

        if hands.ndim == 4 and frame_index < len(hands):
            for side in range(min(2, hands.shape[1])):
                hand = hands[frame_index, side]
                color = (50, 210, 255) if ((side == 0 and limb == "left_hand") or (side == 1 and limb == "right_hand") or limb == "both_hands") else (170, 170, 110)
                for point in hand:
                    if np.isfinite(point[:2]).all():
                        # Hand points are normalised coordinates; use the same
                        # conversion as pose pixels.
                        location = (int(ox + point[0] * width * sx), int(oy + point[1] * height * sy))
                        cv2.circle(canvas, location, 2, color, -1)
        cv2.rectangle(canvas, (0, 0), (panel_width - 1, panel_height - 1), (150, 150, 150), 1)
        cv2.putText(canvas, ("start", "middle", "end")[panel_index], (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"t={times[frame_index]:.2f}s", (125, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 2, cv2.LINE_AA)
        panels.append(canvas)

    canvas = np.hstack(panels)
    title = f"{event['id']}  {event['type']}  {event['limb']}  {event['magnitude']}  {t0:.2f}-{t1:.2f}s"
    cv2.rectangle(canvas, (0, panel_height - 38), (canvas.shape[1], panel_height), (245, 245, 245), -1)
    cv2.putText(canvas, title, (14, panel_height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (25, 70, 190), 2, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def _summary(events: Sequence[Mapping[str, Any]], occupied_hands: Mapping[str, Any] | None = None) -> str:
    by_type: Counter[str] = Counter(str(event.get("type", "unknown")) for event in events)
    by_limb: Counter[str] = Counter(str(event.get("limb", "unknown")) for event in events)
    by_pair: Counter[tuple[str, str]] = Counter((str(event.get("type", "unknown")), str(event.get("limb", "unknown"))) for event in events)
    lines = [f"total events: {len(events)}", "", "by type:"]
    lines.extend(f"{name}: {count}" for name, count in sorted(by_type.items()))
    lines += ["", "by limb:"]
    lines.extend(f"{name}: {count}" for name, count in sorted(by_limb.items()))
    lines += ["", "by type and limb:"]
    lines.extend(f"{event_type} ({limb}): {count}" for (event_type, limb), count in sorted(by_pair.items()))
    lines += ["", "occupied hands:"]
    if occupied_hands:
        lines.extend(f"{name}: {value}" for name, value in sorted(occupied_hands.items()))
    else:
        lines.append("none")
    return "\n".join(lines) + "\n"


def run(
    points: Mapping[str, Any],
    windows: Sequence[Mapping[str, Any]],
    out: Path,
    force: bool = False,
    config: Mapping[str, Any] | None = None,
    clip: Path | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Detect and persist action events and their review artifacts."""

    out = Path(out)
    destination = out / "actions.json"
    if destination.exists() and not force:
        loaded = json.loads(destination.read_text(encoding="utf-8"))
        payload = loaded if isinstance(loaded, dict) else {"events": loaded}
        updated = _far_events(payload["events"], out)
        if updated != payload["events"]:
            payload["events"] = updated
            destination.write_text(json.dumps(updated, ensure_ascii=False, indent=2))
        return payload

    settings = _settings(config)
    shots_path = out / 'shots.json'
    shot_list = json.loads(shots_path.read_text()) if shots_path.exists() else []
    fps = float(points.get('fps',30))
    cuts = sorted({round(float(s['t0'])*fps) for s in shot_list if s.get('cut_at_start')})
    features = extract_features({**points, 'cut_frames':cuts}, settings)
    if cuts:
        detected = []
        total = len(features['t'])
        for a,b in zip([0,*cuts],[*cuts,total]):
            piece = {k:(v[a:b] if isinstance(v,np.ndarray) and v.ndim and len(v)==total else v) for k,v in features.items()}
            t0,t1 = a/fps,b/fps
            piece_windows = [{**w,'t0':max(t0,float(w['t0'])),'t1':min(t1,float(w['t1']))} for w in windows if float(w['t0'])<t1 and float(w['t1'])>t0]
            detected.extend(detect_actions(piece,piece_windows,settings))
    else:
        detected = detect_actions(features, windows, settings)
    events = [_normalise_event(item, windows, index) for index, item in enumerate(detected, 1)]
    events.sort(key=lambda item: (float(item["t0"]), str(item["id"])))
    for index, event in enumerate(events, 1):
        event["id"] = f"A{index:03d}"
    events = _far_events(events, out)

    out.mkdir(parents=True, exist_ok=True)
    # The persisted boundary file follows SPEC §6.6 (a list).  The return
    # value keeps the CLI's stage adapter compatibility with its M1 mapping
    # convention.
    occupied_mask = np.asarray(features.get("occupied_mask", []), dtype=bool)
    occupied_hands: dict[str, Any] = {}
    if occupied_mask.ndim == 2:
        for side, name in enumerate(("left hand", "right hand")):
            if side < occupied_mask.shape[1] and occupied_mask[:, side].any():
                occupied_hands[name] = {
                    "frames": int(np.sum(occupied_mask[:, side])),
                    "start": float(features["t"][np.flatnonzero(occupied_mask[:, side])[0]]),
                    "end": float(features["t"][np.flatnonzero(occupied_mask[:, side])[-1]]),
                }
    payload = {"events": events, "occupied_hands": occupied_hands}
    destination.write_text(json.dumps(events, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    review = out / "actions_review"
    review.mkdir(parents=True, exist_ok=True)
    # Remove review images from a previous target/configuration so the folder
    # is an exact audit of this run rather than a mixture of stale events.
    for previous in review.glob("*.png"):
        previous.unlink()
    times = np.asarray(features.get("t", []), dtype=float)
    requested_indices: list[int] = []
    limit = settings.get("review_limit")
    review_events = events if limit is None else events[:max(0,int(limit))]
    for event in review_events:
        requested_indices.extend(_nearest_indices(times, (event["t0"], (event["t0"] + event["t1"]) / 2, event["t1"])))
    frames = _read_video_frames(Path(clip) if clip is not None else None, requested_indices)
    for event in review_events:
        _review_image(features, points, event, review / f"{event['id']}_{event['type']}.png", frames)
    (out / "actions_summary.txt").write_text(_summary(events, occupied_hands), encoding="utf-8")
    return payload


__all__ = ["run"]
