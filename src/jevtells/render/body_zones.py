"""Pose-derived face and torso cores; arm motion remains a soft obstacle."""
from __future__ import annotations
from typing import Any, Mapping
import numpy as np


def body_zones(pose: np.ndarray, source: tuple[int, int], visibility: float = .5) -> dict[str, list[float]]:
    points = np.asarray(pose, dtype=float).copy()
    if points.ndim == 2:
        points = points[None]
    if points.ndim != 3 or points.shape[1] < 25:
        return {}
    valid = np.isfinite(points[..., :2]).all(axis=-1)
    if points.shape[-1] >= 4:
        valid &= points[..., 3] >= visibility
    xy = points[..., :2] * np.array(source)
    xy[~valid] = np.nan
    widths = np.linalg.norm(xy[:, 11] - xy[:, 12], axis=-1)
    widths = widths[np.isfinite(widths)]
    shoulder = float(np.median(widths)) if len(widths) else source[0] * .1
    result = {}
    def bounds(name, values, padding=0):
        values = values.reshape(-1, 2)
        values = values[np.isfinite(values).all(axis=1)]
        if len(values):
            lo, hi = np.percentile(values, [5, 95], axis=0) if len(values) > 1 else (values[0], values[0])
            result[name] = [float(lo[0]-padding), float(lo[1]-padding), float(hi[0]+padding), float(hi[1]+padding)]
    bounds('face', xy[:, :11], .35 * shoulder)
    shoulders = xy[:, [11, 12]].reshape(-1, 2)
    shoulders = shoulders[np.isfinite(shoulders).all(axis=1)]
    if len(shoulders):
        hips = xy[:, [23, 24]].reshape(-1, 2)
        hips = hips[np.isfinite(hips).all(axis=1)]
        result['torso'] = [float(np.min(shoulders[:, 0])), float(np.min(shoulders[:, 1])), float(np.max(shoulders[:, 0])), float(np.max(hips[:, 1])) if len(hips) else float(source[1])]
    bounds('hands', xy[:, [15, 16, 17, 18, 19, 20, 21, 22]])
    return result
