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
