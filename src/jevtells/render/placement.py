"""Deterministic per-shot label positions, with explicit geometric fallbacks."""
from __future__ import annotations

import math
from itertools import product
from typing import Any, Mapping, Sequence

from .layout_audit import intersects, segment_intersects_rect


def card_side(target: Sequence[float] | None, width: float, card_width: float, margin: float) -> str:
    if target is None:
        return "right"
    left,right=target[0],width-target[2]
    if right >= card_width+margin:
        return "right"
    if left >= card_width+margin:
        return "left"
    return "right" if right >= left else "left"


def crosses(a: Sequence[float], b: Sequence[float], c: Sequence[float], d: Sequence[float]) -> bool:
    def orient(p: Sequence[float], q: Sequence[float], r: Sequence[float]) -> float:
        return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
    return orient(a,b,c)*orient(a,b,d) < 0 and orient(c,d,a)*orient(c,d,b) < 0


def leader(rect: Sequence[float], anchor: Sequence[float], fraction: float = .35) -> list[tuple[float, float]]:
    start = (rect[2] if anchor[0] >= (rect[0]+rect[2])/2 else rect[0], (rect[1]+rect[3])/2)
    return [start, (start[0]+(anchor[0]-start[0])*fraction,start[1]), tuple(anchor)]


def select_positions(video: Sequence[float], target: Sequence[float] | None, forbidden: Sequence[Sequence[float]], sizes: Mapping[str, Sequence[float]], anchors: Mapping[str, Sequence[float]], fallback: Mapping[str, Sequence[float]], *, margin: float = 24, gap: float = 12, step: float = 12, line_ratio: float = .45, soft: Sequence[Sequence[float]] = (), protected: Sequence[Sequence[float]] = ()) -> dict[str, Any]:
    left, top, right, bottom = video
    bounds = target or (left+(right-left)/2, top, left+(right-left)/2, bottom)
    choices: dict[str, list[dict[str, Any]]] = {}
    preferred: dict[str, tuple[float, float]] = {}
    for slot in ("left", "right"):
        width, height = sizes[slot]
        anchor = anchors[slot]
        other = anchors["right" if slot == "left" else "left"]
        side = "left" if anchor[0] <= other[0] else "right"
        px = bounds[0]-gap-width if side == "left" else bounds[2]+gap
        py = max(top+margin, anchor[1]-height-gap)
        preferred[slot] = (px,py)
        xs = {max(left+margin,min(right-margin-width,anchor[0]-width/2)), left+margin, right-margin-width, bounds[0]-gap-width, bounds[2]+gap}
        ys = {top+margin,bottom-margin-height,py}
        for zone in forbidden:
            xs.update((zone[0]-gap-width, zone[2]+gap))
            ys.update((zone[1]-gap-height,zone[3]+gap))
        # Scan vertically on both sides, adding exact obstacle boundaries.
        ys.update(top+margin+i*step for i in range(max(0, math.ceil((bottom-top-2*margin-height)/step))+1))
        candidates = []
        for x,y in product(sorted(xs), sorted(ys)):
            rect = [x,y,x+width,y+height]
            if x < left+margin or y < top+margin or rect[2] > right-margin or rect[3] > bottom-margin:
                continue
            if any(intersects(rect, zone) for zone in forbidden):
                continue
            path = leader(rect, anchor)
            if any(segment_intersects_rect(a,b,zone) for a,b in zip(path,path[1:]) for zone in protected):
                continue
            candidate_side = "left" if (rect[0]+rect[2])/2 <= (bounds[0]+bounds[2])/2 else "right"
            overlap = sum(max(0,min(rect[2],z[2])-max(rect[0],z[0]))*max(0,min(rect[3],z[3])-max(rect[1],z[1])) for z in soft)
            length = sum(math.dist(a,b) for a,b in zip(path,path[1:]))
            if length > (right-left)*line_ratio:
                continue
            candidates.append({"rect": rect, "position": [x,y], "score": math.dist((x,y),(px,py))+length+overlap/max(1,width*height)*(right-left)+(0 if candidate_side==side else 10*(right-left)), "leader": path})
        choices[slot] = sorted(candidates,key=lambda c:(c["score"],c["position"]))
    best = None
    # All candidates participate, so a low-ranked stacked position remains usable.
    for a,b in product(choices["left"],choices["right"]):
        ra,rb=a["rect"],b["rect"]
        padded=[ra[0]-gap,ra[1]-gap,ra[2]+gap,ra[3]+gap]
        if intersects(padded,rb):
            continue
        if any(crosses(x,y,z,w) for x,y in zip(a["leader"],a["leader"][1:]) for z,w in zip(b["leader"],b["leader"][1:])):
            continue
        score=a["score"]+b["score"]
        if best is None or score < best[0]:
            best=(score,a,b)
    result: dict[str, Any] = {"preferred": preferred, "candidates": {slot: values for slot,values in choices.items()}}
    if best:
        result.update({"positions": {"left": best[1]["position"], "right": best[2]["position"]}, "fallback": False, "reason": None})
    else:
        # SPEC explicitly keeps the legacy positions when no legal pair exists.
        fixed = {}
        for slot in choices:
            side = "left" if anchors[slot][0] <= anchors["right" if slot == "left" else "left"][0] else "right"
            fixed[slot] = list(fallback[side])
        result.update({"positions": fixed, "fallback": True, "reason": "no noncrossing legal pair", "fallback_slots": ["left","right"]})
    return result
