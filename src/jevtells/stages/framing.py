"""All-person visibility and face geometry, independent of selected identity."""
from __future__ import annotations
from typing import Any, Mapping
import numpy as np



def multiple_people(poses: np.ndarray, settings: Mapping[str, Any]) -> dict[str, Any]:
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 4 or not len(poses):
        return {'multi_person': False, 'multi_person_ratio': 0.}
    valid = np.isfinite(poses[..., :2]).all(axis=-1)
    if poses.shape[-1] >= 4:
        valid &= poses[..., 3] >= float(settings.get('visibility', .5))
    y = poses[..., 1]
    heights = np.max(np.where(valid, np.clip(y,0,1), -np.inf), axis=-1) - np.min(np.where(valid, np.clip(y,0,1), np.inf), axis=-1)
    counts = np.sum(heights >= float(settings.get('multi_height_ratio', .35)), axis=1)
    ratio = float(np.mean(counts >= 2))
    return {'multi_person': ratio >= float(settings.get('multi_frame_ratio', .50)), 'multi_person_ratio': ratio}


def faces_at(poses: np.ndarray, size: tuple[int, int], visibility: float, *, raw: bool = False) -> list[list[float]]:
    from ..render.body_zones import body_zones
    result = []
    for pose in np.asarray(poses):
        face = body_zones(pose, size, visibility).get('face_raw' if raw else 'face')
        if face:
            result.append(list(face))
    return result
