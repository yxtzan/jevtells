"""Geometry helpers used by the pose stage."""

from __future__ import annotations

from itertools import combinations, permutations

import numpy as np


def _xy_pixels(points: np.ndarray, image_size: tuple[float, float] | None) -> np.ndarray:
    """Convert normalised x/y points to pixels before measuring distances."""
    values = np.asarray(points, dtype=float).copy()
    if image_size is not None:
        width, height = image_size
        values[..., 0] *= float(width)
        values[..., 1] *= float(height)
    return values


def match_hands_to_pose(
    hand_wrists: np.ndarray,
    pose: np.ndarray,
    image_size: tuple[float, float] | None = None,
    max_distance: float | None = None,
) -> np.ndarray:
    """Map candidate hands to pose left/right wrists using the best assignment.

    MediaPipe does not guarantee that the order of ``hand_landmarks`` matches
    the pose wrists.  For two candidates we explicitly evaluate both possible
    permutations and select the one with the lower total pixel distance.  A
    candidate farther than ``max_distance`` from its assigned wrist is dropped.
    ``image_size`` is supplied by the detector so all geometry is measured in
    pixels; omitting it keeps the helper useful for normalised synthetic tests.
    """
    wrists = np.asarray(hand_wrists, dtype=float)
    if wrists.ndim == 1:
        wrists = wrists.reshape(1, -1)
    anchors = np.asarray([np.asarray(pose)[15, :2], np.asarray(pose)[16, :2]], dtype=float)
    wrist_xy = _xy_pixels(wrists[:, :2], image_size)
    anchor_xy = _xy_pixels(anchors, image_size)
    result = np.full(len(wrists), -1, dtype=int)
    valid_hands = [idx for idx, value in enumerate(wrist_xy) if np.isfinite(value).all()]
    valid_sides = [idx for idx, value in enumerate(anchor_xy) if np.isfinite(value).all()]
    if not valid_hands or not valid_sides:
        return result

    # Evaluate every one-to-one assignment.  There are at most two sides in
    # the pose, but keeping the generic loop also makes this helper easy to
    # exercise with synthetic data and future multi-hand detectors.
    best: tuple[float, tuple[int, ...]] | None = None
    best_hands: tuple[int, ...] = ()
    for selected_hands in combinations(valid_hands, min(len(valid_hands), len(valid_sides))):
        for selected_sides in permutations(valid_sides, len(selected_hands)):
            distance = sum(float(np.linalg.norm(wrist_xy[h] - anchor_xy[s])) for h, s in zip(selected_hands, selected_sides))
            candidate = (distance, tuple(selected_sides))
            if best is None or candidate[0] < best[0]:
                best = candidate
                best_hands = selected_hands
    if best is None:
        return result
    for hand_index, side in zip(best_hands, best[1]):
        distance = float(np.linalg.norm(wrist_xy[hand_index] - anchor_xy[side]))
        if max_distance is None or distance <= float(max_distance):
            result[hand_index] = side
    return result
