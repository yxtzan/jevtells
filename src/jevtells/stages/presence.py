"""Window presence measured from actual frame coverage, including edit boundaries."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..schemas import Window


def presence_ratio(points: Mapping[str, Any], start: float, end: float) -> float:
    present = np.asarray(points.get('pose_present', []), dtype=bool)
    if not len(present) or end <= start:
        return 0.0
    fps = float(points.get('fps', 30))
    times = np.asarray(points.get('t', np.arange(len(present)) / fps))
    coverage = np.maximum(0, np.minimum(end, times + 1 / fps) - np.maximum(start, times))
    return min(1.0, float(coverage[present].sum() / (end - start)))


def mark_presence(windows: Sequence[Mapping[str, Any]], points: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for window in windows:
        ratio = presence_ratio(points, float(window['t0']), float(window['t1']))
        result.append({**window, 'target_presence_ratio': ratio, 'target_offscreen': ratio < .5})
    return result


def split_presence_changes(windows: Sequence[Mapping[str, Any]], shots: Sequence[Mapping[str, Any]], points: Mapping[str, Any], transcript: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Keep short offscreen edits from disappearing into a long visible sentence.

    Only changes of target presence split an existing window. Ordinary cuts and
    framing changes leave speech segmentation intact. Word midpoints allocate
    real transcript words once, without duplicating words at an edit.
    """
    boundaries = []
    previous = None
    for shot in shots:
        visible = presence_ratio(points, float(shot['t0']), float(shot['t1'])) >= .5
        if previous is not None and previous != visible:
            boundaries.append(float(shot['t0']))
        previous = visible
    if not boundaries:
        return mark_presence(windows, points)
    words = [word for segment in transcript.get('segments', []) for word in segment.get('words', [])]
    result = []
    for window in windows:
        start, end = float(window['t0']), float(window['t1'])
        edges = [start, *[t for t in boundaries if start < t < end], end]
        for a, b in zip(edges, edges[1:]):
            row = dict(window)
            row.update(t0=a, t1=b)
            if len(edges) > 2 and words:
                row['subtitle'] = ' '.join(str(word['w']) for word in words if a <= (float(word['t0']) + float(word['t1'])) / 2 < b)
            result.append(row)
    for index, row in enumerate(result):
        row.update(id=f'W{index:02d}', index=index, prev_subtitle=result[index-1]['subtitle'] if index else '')
        Window.model_validate(row)
    return mark_presence(result, points)


def skip_reasons(windows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {str(window['id']): 'speaker_other' if window.get('speaker_other') else 'target_offscreen' for window in windows if window.get('speaker_other') or window.get('target_offscreen')}
