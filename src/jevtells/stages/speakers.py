"""Mark windows that overlap supplied intervals of another speaker."""

from __future__ import annotations

import math
from typing import Any, Sequence


def parse_intervals(values: Sequence[str]) -> list[tuple[float, float]]:
    """Validate and merge intervals so overlap is never counted twice."""
    intervals: list[tuple[float, float]] = []
    for value in values:
        try:
            start, end = (float(part) for part in value.split("-"))
        except (ValueError, TypeError) as error:
            raise ValueError(f"--others-speaking requires START-END: {value!r}") from error
        if not all(math.isfinite(v) for v in (start, end)) or start < 0 or end <= start:
            raise ValueError(f"invalid other-speaker interval: {value!r}")
        intervals.append((start, end))
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def mark_windows(windows: Sequence[dict[str, Any]], intervals: Sequence[tuple[float, float]], threshold: float = 0.5) -> list[dict[str, Any]]:
    """Return annotated copies, leaving timing and the five-field states intact."""
    result: list[dict[str, Any]] = []
    for window in windows:
        start, end = float(window["t0"]), float(window["t1"])
        overlap = sum(max(0.0, min(end, b) - max(start, a)) for a, b in intervals)
        other = end > start and overlap / (end - start) >= threshold
        result.append({**window, "speaker_other": other})
    return result


def other_ids(windows: Sequence[dict[str, Any]]) -> set[str]:
    return {str(window["id"]) for window in windows if window.get("speaker_other")}


def split_speaker_changes(windows, intervals, transcript, config):
    """Preserve exact supplied speaker boundaries before minimum-duration merging."""
    edges = sorted({t for interval in intervals for t in interval})
    words = [word for segment in transcript.get('segments',[]) for word in segment.get('words',[])]
    tolerance = float(config.get('windows',{}).get('split_boundary_tolerance_frames',.5))/float(config.get('max_fps',30))
    result = []
    for window in windows:
        start,end = float(window['t0']),float(window['t1'])
        cuts = [start,*[t for t in edges if start+tolerance < t < end-tolerance],end]
        for a,b in zip(cuts,cuts[1:]):
            row = {**window,'t0':a,'t1':b}
            if len(cuts)>2:
                if words:
                    row['subtitle'] = ' '.join(str(word['w']) for word in words if a <= (float(word['t0'])+float(word['t1']))/2 < b)
                elif not a <= (start+end)/2 < b:
                    # A burned cue has no word timings: keep its literal text
                    # once in the piece containing its midpoint.
                    row['subtitle'] = ''
                    row['subtitle_translation'] = ''
            result.append(row)
    return result
