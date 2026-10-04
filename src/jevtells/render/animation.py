"""Small, testable time and interpolation primitives."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def ease_out_cubic(value: float) -> float:
    return 1.0 - (1.0 - max(0.0, min(1.0, value))) ** 3


def progress(elapsed: float, duration: float) -> float:
    return ease_out_cubic(elapsed / duration) if duration > 0 else 1.0


def interpolate(old: float | None, new: float | None, amount: float) -> float | None:
    if new is None:
        return None
    return new if old is None else old + (new - old) * amount


def window_at(windows: Sequence[Mapping[str, Any]], seconds: float) -> tuple[int | None, bool]:
    index = next((i for i in range(len(windows) - 1, -1, -1) if float(windows[i]["t0"]) <= seconds), None)
    return index, bool(index is not None and seconds > float(windows[index]["t1"]))
