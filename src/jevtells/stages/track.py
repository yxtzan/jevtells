"""Target-person selection and conservative frame-to-frame tracking.

The pose detector writes every person to ``detections.npz``.  This module is
the cheap, repeatable stage that turns those detections into the single-person
``keypoints.npz`` consumed by later stages.  It deliberately prefers a lost
target over jumping to another speaker.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..utils.geometry import match_hands_to_pose


@dataclass(frozen=True)
class TrackConfig:
    """Geometry thresholds for conservative target matching."""

    visibility_threshold: float = 0.5
    iou_threshold: float = 0.05
    max_center_distance_shoulder_width: float = 1.5
    area_ratio_min: float = 0.35
    area_ratio_max: float = 2.8


def _config(config: Mapping[str, Any] | None) -> TrackConfig:
    if config is None:
        return TrackConfig()
    source: Mapping[str, Any] = config
    nested = config.get("tracking") if isinstance(config, Mapping) else None
    if isinstance(nested, Mapping):
        source = nested
    values = {
        field: source[field]
        for field in ("iou_threshold", "max_center_distance_shoulder_width", "area_ratio_min", "area_ratio_max")
        if field in source
    }
    if isinstance(config.get("detection"), Mapping) and "pose_visibility_threshold" in config["detection"]:
        values["visibility_threshold"] = config["detection"]["pose_visibility_threshold"]
    return TrackConfig(**values)


def _xy_pixels(points: np.ndarray, width: float, height: float) -> np.ndarray:
    values = np.asarray(points, dtype=float).copy()
    values[..., 0] *= float(width)
    values[..., 1] *= float(height)
    return values


def person_box(pose: np.ndarray, width: float = 1.0, height: float = 1.0, visibility_threshold: float = 0.5) -> np.ndarray:
    """Return ``[x0,y0,x1,y1]`` in pixels, or NaNs when no person is visible."""
    values = np.asarray(pose, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        return np.full(4, np.nan, dtype=float)
    visible = np.isfinite(values[:, :2]).all(axis=1)
    if values.shape[1] > 3:
        visibility = values[:, 3]
        visible &= ~np.isfinite(visibility) | (visibility >= float(visibility_threshold))
    points = _xy_pixels(values[visible, :2], width, height)
    if len(points) == 0:
        return np.full(4, np.nan, dtype=float)
    return np.asarray([points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()], dtype=float)


def bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    """Compute intersection-over-union for two ``xyxy`` boxes."""
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if a.shape != (4,) or b.shape != (4,) or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return 0.0
    left, top = np.maximum(a[:2], b[:2])
    right, bottom = np.minimum(a[2:], b[2:])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def _box_center(box: np.ndarray) -> np.ndarray:
    return (np.asarray(box, dtype=float)[:2] + np.asarray(box, dtype=float)[2:]) / 2.0


def _shoulder_width(pose: np.ndarray, width: float, height: float) -> float:
    values = np.asarray(pose, dtype=float)
    if values.ndim != 2 or len(values) <= 12:
        return 1.0
    shoulders = _xy_pixels(values[[11, 12], :2], width, height)
    return float(np.linalg.norm(shoulders[0] - shoulders[1])) if np.isfinite(shoulders).all() else 1.0


def _candidate_match(previous: np.ndarray, candidates: np.ndarray, width: float, height: float, config: TrackConfig) -> int:
    """Match one previous pose to this frame's candidates, or return ``-1``."""
    previous_box = person_box(previous, width, height, config.visibility_threshold)
    if not np.isfinite(previous_box).all():
        return -1
    previous_center = _box_center(previous_box)
    shoulder = max(_shoulder_width(previous, width, height), 1.0)
    scored: list[tuple[float, int, float, float]] = []
    for index, candidate in enumerate(np.asarray(candidates)):
        box = person_box(candidate, width, height, config.visibility_threshold)
        if not np.isfinite(box).all():
            continue
        iou = bbox_iou(previous_box, box)
        center_distance = float(np.linalg.norm(_box_center(box) - previous_center))
        area_previous = max(1.0, float((previous_box[2] - previous_box[0]) * (previous_box[3] - previous_box[1])))
        area_candidate = max(1.0, float((box[2] - box[0]) * (box[3] - box[1])))
        area_ratio = area_candidate / area_previous
        within_geometry = center_distance <= config.max_center_distance_shoulder_width * shoulder and config.area_ratio_min <= area_ratio <= config.area_ratio_max
        if iou >= config.iou_threshold or within_geometry:
            # IoU is the primary signal; distance and area resolve ties.
            scored.append((iou, index, center_distance / shoulder, abs(np.log(area_ratio))))
    if not scored:
        return -1
    scored.sort(key=lambda item: (-item[0], item[2], item[3]))
    return scored[0][1]


def sorted_person_indices(poses: np.ndarray, frame_index: int, width: float = 1.0, height: float = 1.0, visibility_threshold: float = 0.5) -> list[int]:
    """Return visible person indices ordered left-to-right by box centre."""
    frame = np.asarray(poses)[int(frame_index)]
    entries: list[tuple[float, int]] = []
    for index, pose in enumerate(frame):
        box = person_box(pose, width, height, visibility_threshold)
        if np.isfinite(box).all():
            entries.append((float(_box_center(box)[0]), index))
    return [index for _, index in sorted(entries)]


def select_anchor(poses: np.ndarray, frame_index: int, person_number: int, width: float = 1.0, height: float = 1.0, visibility_threshold: float = 0.5) -> int:
    """Resolve a one-based left-to-right ``people`` number to detector index."""
    ordered = sorted_person_indices(poses, frame_index, width, height, visibility_threshold)
    ordinal = int(person_number) - 1
    return ordered[ordinal] if 0 <= ordinal < len(ordered) else -1


def _track_segment(poses: np.ndarray, start: int, end: int, initial: int, width: float, height: float, config: TrackConfig, step: int = 1) -> np.ndarray:
    """Track from an anchor in either direction, inclusive of the anchor."""
    indices = np.full(abs(end - start) + 1, -1, dtype=int)
    indices[0] = initial
    if initial < 0:
        return indices
    previous = poses[start, initial]
    frame = start
    output_index = 0
    while frame != end:
        frame += step
        output_index += 1
        selected = _candidate_match(previous, poses[frame], width, height, config)
        indices[output_index] = selected
        if selected < 0:
            # Do not use a later person's detection as an implicit re-lock.
            break
        previous = poses[frame, selected]
    return indices


def track_people(poses: np.ndarray, anchors: Sequence[tuple[int, float]] | None = None, fps: float = 30.0, width: float = 1.0, height: float = 1.0, config: Mapping[str, Any] | None = None) -> np.ndarray:
    """Return per-frame detector indices for the requested target anchors.

    ``anchors`` contains ``(one_based_person_number, seconds)`` pairs.  The
    first anchor tracks backwards and forwards; each later anchor starts a new
    segment at its requested frame.  Frames that cannot be matched are -1.
    """
    values = np.asarray(poses)
    if values.ndim != 4:
        raise ValueError("poses must have shape [T,K,landmark,channels]")
    n_frames = values.shape[0]
    if n_frames == 0:
        return np.empty(0, dtype=int)
    rules = _config(config)
    requested = sorted(anchors or [(1, 0.0)], key=lambda item: float(item[1]))
    anchor_frames: list[tuple[int, int]] = []
    for person_number, seconds in requested:
        frame = int(round(float(seconds) * float(fps)))
        frame = min(max(frame, 0), n_frames - 1)
        anchor_frames.append((frame, select_anchor(values, frame, int(person_number), width, height, rules.visibility_threshold)))
    selected = np.full(n_frames, -1, dtype=int)
    first_frame, first_index = anchor_frames[0]
    selected[first_frame] = first_index
    # Track backwards from the first anchor.
    if first_index >= 0:
        backward = _track_segment(values, first_frame, 0, first_index, width, height, rules, -1)
        for offset, index in enumerate(backward):
            selected[first_frame - offset] = index
    # Track each anchor forward until the next anchor.  A new anchor is an
    # explicit user decision and therefore may switch to a different person.
    for anchor_position, (frame, index) in enumerate(anchor_frames):
        next_frame = anchor_frames[anchor_position + 1][0] if anchor_position + 1 < len(anchor_frames) else n_frames - 1
        selected[frame] = index
        if frame > 0 and anchor_position == 0 and selected[frame - 1] < 0 and index >= 0:
            selected[frame - 1] = -1
        if index < 0:
            continue
        forward = _track_segment(values, frame, next_frame, index, width, height, rules)
        for offset, item in enumerate(forward):
            selected[frame + offset] = item
    return selected


def _array_from_npz(source: Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, (str, Path)):
        with np.load(source, allow_pickle=False) as loaded:
            return {key: loaded[key] for key in loaded.files}
    return dict(source)


def run(detections: Path | Mapping[str, Any], out: Path, anchors: Sequence[tuple[int, float]] | None = None, force: bool = False, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Write ``keypoints.npz`` from all-person detector output."""
    destination = out / "keypoints.npz"
    if destination.exists() and not force:
        requested = [[int(person), float(seconds)] for person, seconds in (anchors or [(1, 0.0)])]
        metadata_path = out / "track_meta.json"
        cached_anchors: list[list[float]] | None = None
        if metadata_path.exists():
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                cached_anchors = [[int(item[0]), float(item[1])] for item in payload.get("anchors", [])]
            except (OSError, ValueError, TypeError, KeyError, IndexError):
                cached_anchors = None
        if cached_anchors == requested:
            with np.load(destination, allow_pickle=False) as loaded:
                return {key: loaded[key] for key in loaded.files}
    data = _array_from_npz(detections)
    poses = np.asarray(data.get("poses_all", data.get("poses", data.get("pose"))), dtype=float)
    hands = np.asarray(data.get("hands_all", data.get("hands")), dtype=float)
    if poses.ndim != 4:
        raise ValueError("detections.npz is missing poses_all [T,K,33,4]")
    fps = float(np.asarray(data.get("fps", 30.0)).reshape(-1)[0])
    width = float(np.asarray(data.get("width", 1.0)).reshape(-1)[0])
    height = float(np.asarray(data.get("height", 1.0)).reshape(-1)[0])
    indices = track_people(poses, anchors, fps, width, height, config)
    target_pose = np.full((len(indices), poses.shape[2], poses.shape[3]), np.nan, dtype=np.float32)
    target_hands = np.full((len(indices), 2, hands.shape[2], hands.shape[3]), np.nan, dtype=np.float32) if hands.ndim == 4 else np.full((len(indices), 2, 21, 3), np.nan, dtype=np.float32)
    for frame, person_index in enumerate(indices):
        if person_index < 0 or person_index >= poses.shape[1]:
            continue
        target_pose[frame] = poses[frame, person_index]
        if hands.ndim == 4:
            # hands_all is global [T,M,21,3], independent of person index.
            # Reassign each detected hand to this target's pose wrists only
            # after the person has been selected.
            wrists = np.asarray([candidate[0] for candidate in hands[frame] if np.isfinite(candidate[0, :2]).all()])
            if len(wrists):
                shoulder = _shoulder_width(target_pose[frame], width, height)
                hand_limit = 0.9
                if isinstance(config, Mapping):
                    detection = config.get("detection", {})
                    if isinstance(detection, Mapping):
                        hand_limit = float(detection.get("hand_match_max_shoulder_width", hand_limit))
                assignments = match_hands_to_pose(hands[frame, :, 0, :], target_pose[frame], (width, height), max_distance=hand_limit * shoulder)
                for hand_index, side in enumerate(assignments):
                    if side >= 0 and hand_index < hands.shape[1]:
                        target_hands[frame, side] = hands[frame, hand_index]
    result: dict[str, Any] = {
        "fps": fps,
        "width": width,
        "height": height,
        "n_frames": len(indices),
        "t": np.arange(len(indices), dtype=float) / fps,
        "pose": target_pose,
        "hands": target_hands,
        "pose_present": np.isfinite(target_pose[:, :, :2]).all(axis=2).any(axis=1),
        "hand_present": np.isfinite(target_hands[:, :, :, :2]).all(axis=3).any(axis=2),
        "target_index": indices,
    }
    out.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **result)
    (out / "track_meta.json").write_text(json.dumps({"anchors": list(anchors or [(1, 0.0)]), "lost_frames": int(np.sum(indices < 0))}, ensure_ascii=False, indent=2))
    return result


__all__ = ["TrackConfig", "bbox_iou", "person_box", "sorted_person_indices", "select_anchor", "track_people", "run"]
