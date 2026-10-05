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


def outside_endpoint(path: Sequence[Sequence[float]], radius: float) -> list[tuple]:
    """Keep only segment portions outside the hand's circular exemption."""
    import math
    if not path:
        return []
    hand = path[-1]
    pieces = []
    for a, b in zip(path, path[1:]):
        dx, dy = b[0]-a[0], b[1]-a[1]
        ax, ay = a[0]-hand[0], a[1]-hand[1]
        aa = dx*dx+dy*dy
        roots = [0., 1.]
        if aa:
            bb, cc = 2*(ax*dx+ay*dy), ax*ax+ay*ay-radius*radius
            disc = bb*bb-4*aa*cc
            if disc >= 0:
                roots += [t for t in ((-bb-math.sqrt(disc))/(2*aa),(-bb+math.sqrt(disc))/(2*aa)) if 0 < t < 1]
        roots.sort()
        for lo, hi in zip(roots, roots[1:]):
            mid = (lo+hi)/2
            if (ax+mid*dx)**2+(ay+mid*dy)**2 >= radius*radius-1e-8:
                pieces.append(((a[0]+lo*dx,a[1]+lo*dy),(a[0]+hi*dx,a[1]+hi*dy)))
    return pieces


def line_flags(path, face, midline, radius=60):
    pieces = outside_endpoint(path, radius)
    face_hit = bool(face and any(segment_intersects_rect(a,b,face) for a,b in pieces))
    mid_hit = midline is not None and any((a[0]-midline)*(b[0]-midline) < 0 for a,b in pieces)
    return face_hit, mid_hit


def leader_violations(frame: Mapping[str, Any]) -> list[dict[str, Any]]:
    failures = []
    face = next((z['rect'] for z in frame.get('leader_zones', []) if z['name']=='face'), None)
    for label in frame.get('labels', []):
        path = label.get('actual_hand_leader', label.get('leader', []))
        face_hit, mid_hit = line_flags(path, face, frame.get('midline'), frame.get('endpoint_exemption',0))
        if face_hit:
            failures.append({'slot':label['slot'],'event':label['event'],'zone':'face'})
        if mid_hit:
            failures.append({'slot':label['slot'],'event':label['event'],'zone':'midline'})
    return failures
