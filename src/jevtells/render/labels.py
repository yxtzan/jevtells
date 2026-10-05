"""Select one event per fixed label slot and animate its visibility."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .animation import progress


def visibility(event: Mapping[str, Any], seconds: float, animation: Mapping[str, Any]) -> tuple[float, float]:
    start, end = float(event["t0"]), float(event["t1"])
    fade_end = end + float(animation["label_hold_seconds"])
    if seconds < start or seconds >= fade_end + float(animation["label_out_seconds"]):
        return 0.0, 1.0
    amount = progress(seconds - start, float(animation["label_in_seconds"]))
    opacity = amount if seconds < fade_end else 1.0 - (seconds - fade_end) / float(animation["label_out_seconds"])
    initial = float(animation["label_start_scale"])
    return max(0.0, opacity), initial + (1 - initial) * amount


def schedule(events: Sequence[Mapping[str, Any]], seconds: float, settings: Mapping[str, Any], *, lost: bool = False, shot: str = "target") -> dict[str, tuple[Mapping[str, Any], float, float]]:
    if lost or shot == "other":
        return {}
    result: dict[str, tuple[Mapping[str, Any], float, float]] = {}
    for event in sorted(events, key=lambda value: (float(value["t0"]), str(value.get("id", "")))):
        if event.get("far"):
            continue
        if event.get("type") not in settings["label_types"] or event.get("magnitude", event.get("amplitude")) == "small":
            continue
        opacity, scale = visibility(event, seconds, settings["animation"])
        if opacity <= 0:
            continue
        limb = event.get("limb")
        slots = ["left", "right"] if limb == "both_hands" else ["left" if limb in {"right_hand", "head", "body"} else "right"]
        for slot in slots:
            result[slot] = (event, opacity, scale)
    return result
