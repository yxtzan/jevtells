"""Independent rectangle audit of the labels actually drawn on each frame."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def intersects(a: Sequence[float], b: Sequence[float]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def frame_violations(frame: Mapping[str, Any]) -> list[dict[str, Any]]:
    failures = []
    labels = frame.get("labels", [])
    for index, label in enumerate(labels):
        collisions = [zone["name"] for zone in frame.get("forbidden", []) if intersects(label["rect"], zone["rect"])]
        for other in labels[:index]:
            if intersects(label["rect"], other["rect"]):
                collisions.append("label:" + other["slot"])
        if collisions:
            failures.append({"slot": label["slot"], "event": label["event"], "zones": collisions, "fallback": bool(label.get("fallback"))})
    return failures
