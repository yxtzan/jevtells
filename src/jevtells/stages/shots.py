"""HSV cuts and stable framing spans, with measured target activity bounds."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


def histogram(frame: np.ndarray, settings: Mapping[str, Any]) -> list[np.ndarray]:
    frame = cv2.resize(frame, (320, 180))
    frame = frame[:round(180 * (1 - float(settings.get('subtitle_exclusion_ratio', .2))))]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    result = []
    # Spatial tiles retain sensitivity to cuts between left/right framing.
    grid = int(settings.get('histogram_grid', 2))
    for row in np.array_split(hsv, grid, axis=0):
        for tile in np.array_split(row, grid, axis=1):
            hist = cv2.calcHist([tile], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256])
            cv2.normalize(hist, hist, alpha=1, norm_type=cv2.NORM_L1)
            result.append(hist)
    return result


def shoulder_ratios(points: Mapping[str, Any], width: int, height: int, visibility: float) -> np.ndarray:
    pose = np.asarray(points.get('pose', []), dtype=float)
    if pose.ndim != 3 or pose.shape[1] < 13:
        return np.array([])
    valid = np.isfinite(pose[:, [11, 12], :2]).all(axis=(1, 2))
    if pose.shape[2] >= 4:
        valid &= (pose[:, [11, 12], 3] >= visibility).all(axis=1)
    widths = np.linalg.norm((pose[:, 11, :2] - pose[:, 12, :2]) * [width, height], axis=1) / width
    return np.where(valid, widths, np.nan)


def target_bounds(points: Mapping[str, Any], start: int, end: int, width: int, height: int, visibility: float, shoulder: float, padding: float) -> list[float] | None:
    pose = np.asarray(points.get('pose', []), dtype=float)[start:end]
    if pose.ndim != 3:
        return None
    valid = np.isfinite(pose[:, :, :2]).all(axis=2)
    if pose.shape[2] >= 4:
        valid &= pose[:, :, 3] >= visibility
    xy = pose[:, :, :2][valid]
    hands = np.asarray(points.get('hands', []), dtype=float)[start:end]
    if hands.ndim == 4:
        hand_xy = hands[:, :, :, :2].reshape(-1, 2)
        xy = np.concatenate([xy, hand_xy[np.isfinite(hand_xy).all(axis=1)]])
    if not len(xy):
        return None
    low, high = np.percentile(xy * [width, height], [5, 95], axis=0)
    low = np.maximum([0, 0], low - padding * shoulder)
    high = np.minimum([width, height], high + padding * shoulder)
    return [float(low[0]), float(low[1]), float(high[0]), float(high[1])]


def _framing_boundaries(states: np.ndarray, minimum: int) -> list[int]:
    """A sustained scale/presence change is an analytical span, not an edit.

    A whole-zoom median would hide its unreliable distant part. Scale-only
    boundaries keep cut_at_start false so reports do not call a zoom a cut.
    """
    if not len(states):
        return []
    current = states[0]
    pending, candidate = 0, current
    result = []
    for index, state in enumerate(states):
        if state == current:
            pending, candidate = index + 1, current
        else:
            if state != candidate:
                pending, candidate = index, state
            if index - pending + 1 >= minimum:
                result.append(pending)
                current = state
                pending, candidate = index + 1, current
    return result


def run(clip: Path, out: Path, force: bool = False, points: Mapping[str, Any] | None = None, config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    destination, meta_path = out / 'shots.json', out / 'shots_meta.json'
    settings = dict((config or {}).get('shots', {}))
    visibility = float((config or {}).get('detection', {}).get('pose_visibility_threshold', .5))
    signature = {'version': 2, 'settings': settings, 'visibility': visibility}
    if destination.exists() and meta_path.exists() and not force:
        if json.loads(meta_path.read_text()).get('signature') == signature:
            return json.loads(destination.read_text())
    capture = cv2.VideoCapture(str(clip))
    if not capture.isOpened():
        raise RuntimeError(f'cannot decode shots: {clip}')
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cuts, differences, previous = set(), [], None
    try:
        for index in range(total):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError('shot decoding ended early')
            current = histogram(frame, settings)
            difference = max(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA) for a, b in zip(previous, current)) if previous is not None else 0.0
            differences.append(float(difference))
            if difference > float(settings.get('histogram_threshold', .45)):
                cuts.add(index)
            previous = current
    finally:
        capture.release()
    measured = points or {}
    ratios = shoulder_ratios(measured, width, height, visibility)
    present = np.asarray(measured.get('pose_present', np.isfinite(ratios)), dtype=bool)
    minimum = max(1, round(float(settings.get('min_seconds', .5)) * fps))
    far_threshold = float(settings.get('far_shoulder_ratio', .15))
    states = np.where(present, np.where(ratios < far_threshold, 1, 0), -1) if len(ratios) else np.array([])
    boundaries = sorted(cuts | set(_framing_boundaries(states, minimum)))
    kept = [0]
    for boundary in boundaries:
        if boundary - kept[-1] >= minimum:
            kept.append(boundary)
    if total - kept[-1] < minimum and len(kept) > 1:
        kept.pop()
    kept.append(total)
    shots = []
    for start, end in zip(kept, kept[1:]):
        finite = ratios[start:end]
        finite = finite[np.isfinite(finite)]
        shoulder = float(np.median(finite) * width) if len(finite) else None
        box = target_bounds(measured, start, end, width, height, visibility, shoulder or 0, float(settings.get('target_padding_shoulder', .15)))
        target = bool(box and len(present[start:end]) and present[start:end].mean() >= float(settings.get('target_presence_ratio', .5)) and (box[3] - box[1]) / height >= float(settings.get('min_target_height_ratio', .25)))
        shots.append({'index': len(shots) + 1, 't0': start / fps, 't1': end / fps, 'cut_at_start': start in cuts, 'label': 'target' if target else 'other', 'target_box': box if target else None, 'target_center_x': (box[0] + box[2]) / 2 if target else None, 'shoulder_px': shoulder if target else None, 'far': bool(shoulder / width < far_threshold) if target and shoulder is not None else None})
    destination.write_text(json.dumps(shots, ensure_ascii=False, indent=2, allow_nan=False))
    meta_path.write_text(json.dumps({'signature': signature, 'histogram_cuts': sorted(i / fps for i in cuts), 'differences': differences, 'framing_boundaries': [i / fps for i in kept[1:-1]]}, indent=2))
    return shots
