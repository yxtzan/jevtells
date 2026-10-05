"""Small, testable time and interpolation primitives."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def ease_out_cubic(value: float) -> float:
    return 1.0 - (1.0 - max(0.0, min(1.0, value))) ** 3


def ease_in_out_cubic(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return 4*value**3 if value < .5 else 1-(-2*value+2)**3/2


def progress(elapsed: float, duration: float) -> float:
    return ease_out_cubic(elapsed / duration) if duration > 0 else 1.0


def interpolate(old: float | None, new: float | None, amount: float) -> float | None:
    if new is None:
        return None
    return new if old is None else old + (new - old) * amount


def window_at(windows: Sequence[Mapping[str, Any]], seconds: float) -> tuple[int | None, bool]:
    index = next((i for i in range(len(windows) - 1, -1, -1) if float(windows[i]["t0"]) <= seconds), None)
    return index, bool(index is not None and seconds > float(windows[index]["t1"]))


def commentary_at(index: int, elapsed: float, animation: Mapping[str, Any]) -> tuple[int, float, float]:
    """Choose exactly one sentence and quote, fading out before fading in."""
    out = float(animation["commentary_out_seconds"]) if index else 0.0
    if elapsed < out - 1e-9:
        return index - 1, max(0.0, 1 - elapsed / out), 0.0
    amount = progress(elapsed - out, float(animation["commentary_in_seconds"]))
    return index, amount, float(animation["commentary_offset"]) * (1 - amount)


def panel_index(windows: Sequence[Mapping[str, Any]], index: int | None) -> int | None:
    while index is not None and index > 0 and windows[index].get('hold_previous_panel'):
        index -= 1
    return index


def panel_window_at(windows: Sequence[Mapping[str, Any]], seconds: float) -> tuple[int | None, bool]:
    index,gap = window_at(windows,seconds)
    return panel_index(windows,index),gap
