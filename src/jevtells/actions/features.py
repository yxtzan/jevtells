"""Feature extraction for the M2 gesture rules.

The detector stores normalised MediaPipe coordinates.  This module converts
them to pixels before measuring motion and then provides shoulder-normalised
features for rules whose thresholds should be resolution independent.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.signal import savgol_filter


def interpolate_short_gaps(values: np.ndarray, max_gap: int = 5) -> np.ndarray:
    """Fill NaN runs no longer than ``max_gap`` frames, preserving long gaps."""
    result = np.asarray(values, dtype=float).copy()
    if result.ndim == 1:
        result = result[:, None]
    for column in range(result.shape[1]):
        valid = np.isfinite(result[:, column])
        indices = np.arange(len(result))
        if valid.sum() < 2:
            continue
        missing = np.flatnonzero(~valid)
        if not len(missing):
            continue
        starts = missing[np.r_[True, np.diff(missing) > 1]]
        ends = missing[np.r_[np.diff(missing) > 1, True]]
        for start, end in zip(starts, ends):
            if end - start + 1 <= int(max_gap):
                left = start - 1
                right = end + 1
                if left >= 0 and right < len(result) and valid[left] and valid[right]:
                    result[start : end + 1, column] = np.interp(indices[start : end + 1], [left, right], [result[left, column], result[right, column]])
    return result[:, 0] if np.asarray(values).ndim == 1 else result


def smooth_zero_phase(values: np.ndarray, window: int = 7, polynomial: int = 2) -> np.ndarray:
    """Apply Savitzky–Golay smoothing without introducing a temporal delay."""
    source = np.asarray(values, dtype=float)
    result = source.copy()
    if len(source) < 3:
        return result
    window = max(3, int(window) | 1)
    if window > len(source):
        window = len(source) if len(source) % 2 else len(source) - 1
    if window < 3:
        return result
    poly = min(int(polynomial), window - 1)
    if source.ndim == 1:
        valid = np.isfinite(source)
        if valid.sum() >= window:
            result[valid] = savgol_filter(source[valid], window, poly, mode="interp")
        return result
    for column in range(source.shape[1]):
        valid = np.isfinite(source[:, column])
        if valid.sum() >= window:
            result[valid, column] = savgol_filter(source[valid, column], window, poly, mode="interp")
    return result


def _pixels(points: np.ndarray, width: float, height: float) -> np.ndarray:
    result = np.asarray(points, dtype=float).copy()
    result[..., 0] *= float(width)
    result[..., 1] *= float(height)
    return result


def _safe_norm(vector: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.linalg.norm(np.asarray(vector, dtype=float), axis=axis)


def _finger_features(hand_px: np.ndarray, side: int = 0) -> tuple[float, np.ndarray, float, float, float]:
    """Return hand shape features from pixel-scaled 3-D landmarks.

    MediaPipe's image landmarks use x/y image coordinates and a z value in
    the same normalised image scale as x.  The caller scales all three axes
    before this function.  The cross product of the index and pinky MCP
    vectors gives an oriented palm normal; the right hand is flipped so the
    same physical palm direction has the same sign for both sides.
    """
    values = np.asarray(hand_px, dtype=float)
    if values.shape != (21, 3) or not np.isfinite(values[:, :2]).all():
        return np.nan, np.full(5, np.nan), np.nan, np.nan, np.nan
    xy = values[:, :2]
    wrist = xy[0]
    # MCP/PIP/DIP/TIP groups in the MediaPipe hand topology.
    groups = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20), (1, 2, 3, 4))
    straight: list[float] = []
    lengths: list[float] = []
    for mcp, pip, dip, tip in groups:
        if not np.isfinite(xy[[mcp, pip, dip, tip]]).all():
            straight.append(np.nan)
            continue
        tip_distance = float(np.linalg.norm(xy[tip] - wrist))
        pip_distance = float(np.linalg.norm(xy[pip] - wrist))
        lengths.append(tip_distance)
        straight.append(float(tip_distance > pip_distance * 1.15))
    openness = float(np.nanmean(straight)) if straight else np.nan
    spread = float(np.nanmean(lengths)) if lengths else np.nan
    if np.isfinite(values[[0, 5, 9, 17], :3]).all():
        index_vector = values[5, :3] - values[0, :3]
        pinky_vector = values[17, :3] - values[0, :3]
        normal = np.cross(index_vector, pinky_vector)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm > 0:
            # Landmark order produces opposite normals for anatomical left
            # and right hands, so orient the right-hand normal once here.
            if int(side) == 1:
                normal = -normal
            normal /= normal_norm
            screen_up = np.asarray([0.0, -1.0, 0.0])
            palm_up = float(np.dot(normal, screen_up))
        else:
            palm_up = np.nan
        finger = xy[9] - xy[0]
        finger_norm = float(np.linalg.norm(finger))
        finger_vertical_cos = float(abs(finger[1]) / finger_norm) if finger_norm > 0 else np.nan
    else:
        palm_up = np.nan
        finger_vertical_cos = np.nan
    return openness, np.asarray(straight, dtype=float), palm_up, spread, finger_vertical_cos


def _rolling_ratio(mask: np.ndarray, fps: float, window_seconds: float) -> np.ndarray:
    """Return the local true ratio in a centred time window."""
    values = np.asarray(mask, dtype=float)
    if values.ndim != 1 or not len(values):
        return np.asarray([], dtype=float)
    radius = max(0, int(round(float(window_seconds) * float(fps) / 2.0)))
    result = np.full(len(values), np.nan, dtype=float)
    for index in range(len(values)):
        start, end = max(0, index - radius), min(len(values), index + radius + 1)
        sample = values[start:end]
        finite = np.isfinite(sample)
        if finite.any():
            result[index] = float(np.mean(sample[finite]))
    return result


def extract_features(points: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Extract smoothed, shoulder-normalised frame features from keypoints."""
    pose = np.asarray(points.get("pose"), dtype=float)
    hands = np.asarray(points.get("hands"), dtype=float)
    times = np.asarray(points.get("t", np.arange(len(pose)) / float(points.get("fps", 30.0))), dtype=float)
    fps = float(points.get("fps", 1.0 / np.nanmedian(np.diff(times)) if len(times) > 1 else 30.0))
    width = float(points.get("width", 1.0))
    height = float(points.get("height", 1.0))
    if pose.ndim != 3 or pose.shape[1] < 33:
        raise ValueError("pose must have shape [T,33,4]")
    gap = int((config or {}).get("interpolation_max_gap", 5))
    smooth_window = int((config or {}).get("smoothing_window", 7))
    smooth_poly = int((config or {}).get("smoothing_polynomial", 2))
    visibility_threshold = float((config or {}).get("pose_visibility_threshold", 0.5))
    pose_xy = _pixels(pose[:, :, :2], width, height)
    if pose.shape[2] > 3:
        visibility = pose[:, :, 3]
        pose_xy[(np.isfinite(visibility)) & (visibility < visibility_threshold)] = np.nan
    if pose.shape[2] > 3:
        visible = np.isfinite(pose[:, :, 3]) & (pose[:, :, 3] >= visibility_threshold)
        pose_xy[~visible] = np.nan
    pose_xy = interpolate_short_gaps(pose_xy.reshape(len(pose), -1), gap).reshape(pose_xy.shape)
    # Smooth only the coordinates; visibility remains the detector's raw data.
    for landmark in range(pose_xy.shape[1]):
        pose_xy[:, landmark] = smooth_zero_phase(pose_xy[:, landmark], smooth_window, smooth_poly)
    shoulder_values = pose_xy[:, [11, 12]]
    shoulder_count = np.sum(np.isfinite(shoulder_values), axis=1)
    shoulder_center = np.divide(np.nansum(shoulder_values, axis=1), shoulder_count, out=np.full((len(pose), 2), np.nan), where=shoulder_count > 0)
    shoulder_width = _safe_norm(pose_xy[:, 11] - pose_xy[:, 12])
    valid_shoulder = np.isfinite(shoulder_width) & (shoulder_width > 0)
    baseline = float(np.nanmedian(shoulder_width[valid_shoulder])) if valid_shoulder.any() else 1.0
    baseline = max(baseline, np.finfo(float).eps)
    wrist_px = pose_xy[:, [15, 16]]
    wrist_velocity = np.full_like(wrist_px, np.nan)
    if len(wrist_px) > 1:
        wrist_velocity[1:] = np.diff(wrist_px, axis=0) * fps
    wrist_norm = (wrist_px - shoulder_center[:, None, :]) / shoulder_width[:, None, None]
    wrist_norm[~np.isfinite(wrist_norm)] = np.nan
    if hands.ndim == 4 and hands.shape[1] >= 2 and hands.shape[3] >= 3:
        hands_px = np.asarray(hands[:, :2, :, :3], dtype=float).copy()
        hands_px[..., 0] *= width
        hands_px[..., 1] *= height
        # MediaPipe image-landmark z uses the x scale.
        hands_px[..., 2] *= width
    else:
        hands_px = np.full((len(pose), 2, 21, 3), np.nan)
    open_score = np.full((len(pose), 2), np.nan)
    finger_straight = np.full((len(pose), 2, 5), np.nan)
    palm_orientation = np.full((len(pose), 2), np.nan)
    finger_vertical_cos = np.full((len(pose), 2), np.nan)
    hand_spread = np.full((len(pose), 2), np.nan)
    hand_point_ratio = np.zeros((len(pose), 2), dtype=float)
    for frame in range(len(pose)):
        for side in range(2):
            raw_hand = hands[frame, side] if hands.ndim == 4 and hands.shape[1] > side else np.full((21, 3), np.nan)
            present = np.isfinite(raw_hand[:, :2]).all(axis=1) if raw_hand.ndim == 2 and len(raw_hand) else np.zeros(21, dtype=bool)
            hand_point_ratio[frame, side] = float(np.mean(present)) if len(present) else 0.0
            hand_geometry = hands_px[frame, side] if frame < len(hands_px) else np.full((21, 3), np.nan)
            opened, straight, palm, spread, vertical_cos = _finger_features(hand_geometry, side=side)
            open_score[frame, side] = opened
            finger_straight[frame, side] = straight
            palm_orientation[frame, side] = palm
            finger_vertical_cos[frame, side] = vertical_cos
            hand_spread[frame, side] = spread
    two_hands_distance = _safe_norm(wrist_px[:, 0] - wrist_px[:, 1]) / shoulder_width
    nose_relative = (pose_xy[:, 0] - shoulder_center) / shoulder_width[:, None]
    nose_relative[~np.isfinite(nose_relative)] = np.nan
    nose_px = pose_xy[:, 0]
    wrist_visibility = np.ones((len(pose), 2), dtype=float)
    if pose.shape[2] > 3:
        wrist_visibility = np.asarray(pose[:, [15, 16], 3], dtype=float)
    wrist_edge_ok = np.zeros((len(pose), 2), dtype=bool)
    margin = float((config or {}).get("edge_margin", 0.04))
    for side in range(2):
        wrist_edge_ok[:, side] = (
            np.isfinite(wrist_px[:, side]).all(axis=1)
            & (wrist_px[:, side, 0] >= width * margin)
            & (wrist_px[:, side, 0] <= width * (1.0 - margin))
            & (wrist_px[:, side, 1] >= height * margin)
            & (wrist_px[:, side, 1] <= height * (1.0 - margin))
        )
    occupied_frame = np.full((len(pose), 2), np.nan, dtype=float)
    occupied_mask = np.zeros((len(pose), 2), dtype=bool)
    occupied_radius = float((config or {}).get("occupied_face_radius", 1.0))
    occupied_ratio = float((config or {}).get("occupied_ratio", 0.6))
    occupied_window = float((config or {}).get("occupied_window_seconds", 10.0))
    for side in range(2):
        distance = _safe_norm(wrist_px[:, side] - nose_px) / shoulder_width
        valid = np.isfinite(distance) & np.isfinite(shoulder_width) & (shoulder_width > 0)
        occupied_frame[valid, side] = (distance[valid] <= occupied_radius).astype(float)
        ratio = _rolling_ratio(occupied_frame[:, side], fps, occupied_window)
        occupied_mask[:, side] = np.isfinite(ratio) & (ratio >= occupied_ratio)
    features: dict[str, Any] = {
        "t": times,
        "fps": fps,
        "width": width,
        "height": height,
        "pose_px": pose_xy,
        "shoulder_center_px": shoulder_center,
        "shoulder_width_px": shoulder_width,
        "shoulder_width_relative": shoulder_width / baseline,
        "wrist_px": wrist_px,
        "wrist_velocity_px_s": wrist_velocity,
        "wrist_norm": wrist_norm,
        "hand_open": open_score,
        "finger_straight": finger_straight,
        "palm_orientation": palm_orientation,
        "palm_up_score": palm_orientation,
        "finger_vertical_cos": finger_vertical_cos,
        "hand_spread": hand_spread,
        "hand_point_ratio": hand_point_ratio,
        "wrist_visibility": wrist_visibility,
        "wrist_edge_ok": wrist_edge_ok,
        "nose_px": nose_px,
        "wrist_below_nose": (wrist_px[:, :, 1] - nose_px[:, None, 1]) / shoulder_width[:, None],
        "occupied_frame": occupied_frame,
        "occupied_mask": occupied_mask,
        "two_hands_distance": two_hands_distance,
        "nose_relative": nose_relative,
    }
    return features


__all__ = ["interpolate_short_gaps", "smooth_zero_phase", "extract_features"]
