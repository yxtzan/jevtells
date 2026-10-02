from __future__ import annotations
import numpy as np

def match_hands_to_pose(hand_wrists: np.ndarray, pose: np.ndarray) -> np.ndarray:
    wrists = np.asarray(hand_wrists, float)
    p = np.asarray(pose, float)
    anchors = np.array([p[15, :2], p[16, :2]], float)
    available = [0, 1]
    result = np.full(len(wrists), -1, dtype=int)
    for i, wrist in enumerate(wrists):
        if not np.isfinite(wrist[:2]).all() or not available:
            continue
        side = min(available, key=lambda k: float(np.linalg.norm(wrist[:2] - anchors[k])))
        result[i] = side
        available.remove(side)
    return result
