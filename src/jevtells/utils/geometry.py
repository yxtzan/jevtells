"""Geometry helpers used by the pose stage."""

from __future__ import annotations

import numpy as np


def match_hands_to_pose(hand_wrists: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Map detected hands to pose left wrist (15) and right wrist (16)."""
    wrists = np.asarray(hand_wrists, dtype=float)
    anchors = np.asarray([pose[15, :2], pose[16, :2]], dtype=float)
    available = [0, 1]
    result = np.full(len(wrists), -1, dtype=int)
    for index, wrist in enumerate(wrists):
        if not np.isfinite(wrist[:2]).all() or not available:
            continue
        side = min(available, key=lambda candidate: float(np.linalg.norm(wrist[:2] - anchors[candidate])))
        result[index] = side
        available.remove(side)
    return result
