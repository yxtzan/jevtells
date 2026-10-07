"""All-person visibility and face geometry, independent of selected identity."""
from __future__ import annotations
from typing import Any, Mapping
import numpy as np



def person_mask(poses: np.ndarray, settings: Mapping[str, Any]) -> np.ndarray:
    """Require an interior nose/eye and a sufficiently tall visible body."""
    poses = np.asarray(poses, dtype=float)
    if poses.ndim < 3 or poses.shape[-2] < 6:
        return np.zeros(poses.shape[:-2], dtype=bool)
    xy = poses[..., :2]
    valid = np.isfinite(xy).all(axis=-1)
    if poses.shape[-1] >= 4:
        valid &= poses[..., 3] >= float(settings.get('visibility', .5))
    margin = float(settings.get('person_edge_margin', .02))
    interior = valid & (xy >= margin).all(axis=-1) & (xy <= 1-margin).all(axis=-1)
    head = interior[..., 0] & (interior[..., 2] | interior[..., 5])
    y = np.clip(xy[..., 1], 0, 1)
    heights = np.max(np.where(valid, y, -np.inf), axis=-1) - np.min(np.where(valid, y, np.inf), axis=-1)
    return head & (heights >= float(settings.get('multi_height_ratio', .35)))


def people_counts(poses: np.ndarray, settings: Mapping[str, Any]) -> np.ndarray:
    poses = np.asarray(poses, dtype=float)
    return person_mask(poses, settings).sum(axis=1) if poses.ndim == 4 else np.array([], dtype=int)


def multiple_people(poses: np.ndarray, settings: Mapping[str, Any]) -> dict[str, Any]:
    counts = people_counts(poses, settings)
    ratio = float(np.mean(counts >= 2)) if len(counts) else 0.
    return {'multi_person': ratio >= float(settings.get('multi_frame_ratio', .50)), 'multi_person_ratio': ratio}


def faces_at(poses: np.ndarray, size: tuple[int, int], visibility: float, *, raw: bool = False, settings: Mapping[str, Any] | None = None) -> list[list[float]]:
    from ..render.body_zones import body_zones
    result = []
    poses = np.asarray(poses)
    for pose in poses[person_mask(poses, {**(settings or {}), "visibility": visibility})]:
        face = body_zones(pose, size, visibility).get('face_raw' if raw else 'face')
        if face:
            result.append(list(face))
    return result


def multi_timeline(poses: np.ndarray, fps: float, cuts: list[int], settings: Mapping[str, Any]) -> np.ndarray:
    """Smooth counts within edits; accept only sustained changes, at their onset."""
    from scipy.ndimage import median_filter
    from .shots import _framing_boundaries
    counts = people_counts(poses, settings)
    result = np.zeros(len(counts), dtype=bool)
    radius = max(1, round(float(settings.get('multi_smooth_seconds', .2))*fps))
    kernel = radius if radius % 2 else radius+1
    minimum = max(1, round(float(settings.get('multi_switch_seconds', 1.5))*fps))
    for a,b in zip([0,*cuts],[*cuts,len(counts)]):
        states = median_filter(counts[a:b], size=kernel, mode='nearest') >= 2
        if not len(states):
            continue
        boundaries = [0,*_framing_boundaries(states,minimum),len(states)]
        for lo,hi in zip(boundaries,boundaries[1:]):
            result[a+lo:a+hi] = states[lo]
    return result
