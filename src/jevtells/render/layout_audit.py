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


def segment_intersects_rect(a: Sequence[float], b: Sequence[float], rect: Sequence[float]) -> bool:
    """Clip a segment to a closed rectangle, including endpoint crossings."""
    lo, hi = 0.0, 1.0
    for axis in (0, 1):
        delta = b[axis] - a[axis]
        if abs(delta) < 1e-12:
            if a[axis] < rect[axis] or a[axis] > rect[axis+2]:
                return False
        else:
            enter, leave = sorted(((rect[axis]-a[axis])/delta, (rect[axis+2]-a[axis])/delta))
            lo, hi = max(lo, enter), min(hi, leave)
            if lo > hi:
                return False
    return True


def leader_violations(frame: Mapping[str, Any]) -> list[dict[str, Any]]:
    failures = []
    for label in frame.get('labels', []):
        path = label.get('actual_hand_leader', label.get('leader', []))
        for zone in frame.get('leader_zones', []):
            if any(segment_intersects_rect(a, b, zone['rect']) for a, b in zip(path, path[1:])):
                failures.append({'slot': label['slot'], 'event': label['event'], 'zone': zone['name']})
    return failures
