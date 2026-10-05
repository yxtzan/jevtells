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
    return "left" if left >= 1.5 * right else "right"


def crosses(a: Sequence[float], b: Sequence[float], c: Sequence[float], d: Sequence[float]) -> bool:
    def orient(p: Sequence[float], q: Sequence[float], r: Sequence[float]) -> float:
        return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
    return orient(a,b,c)*orient(a,b,d) < 0 and orient(c,d,a)*orient(c,d,b) < 0


def leader(rect: Sequence[float], anchor: Sequence[float], fraction: float = .35) -> list[tuple[float, float]]:
    start = (rect[2] if anchor[0] >= (rect[0]+rect[2])/2 else rect[0], (rect[1]+rect[3])/2)
    return [start, (start[0]+(anchor[0]-start[0])*fraction,start[1]), tuple(anchor)]


def select_positions(video: Sequence[float], target: Sequence[float] | None, forbidden: Sequence[Sequence[float]], sizes: Mapping[str, Sequence[float]], anchors: Mapping[str, Sequence[float]], fallback: Mapping[str, Sequence[float]] | None = None, *, margin: float = 24, gap: float = 12, step: float = 12, line_ratio: float = .45, soft: Sequence[Sequence[float]] = (), protected: Sequence[Sequence[float]] = (), samples: Mapping[str, Sequence[Mapping[str, Any]]] | None = None, midline: float | None = None, face: Sequence[float] | None = None, exemption: float = 60, fraction: float = .35, start_offsets: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Rank same-side legal, opposite legal, relaxed, then forced shot positions.

    Legacy fallback and length limits are deliberately unused. Every measured
    display frame participates in the 95% line check, including animation.
    """
    from .layout_audit import line_flags
    import numpy as np
    left, top, right, bottom = video
    midline = midline if midline is not None else ((target[0]+target[2])/2 if target else (left+right)/2)
    choices, preferred = {}, {}
    def area(a,b):
        return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    for slot, (width,height) in sizes.items():
        values = list((samples or {}).get(slot, []))
        anchor = anchors[slot]
        hand_left = (float(np.median([v['anchor'][0]-v.get('midline',midline) for v in values])) if values else anchor[0]-midline) <= 0
        px = anchor[0]-width-gap if hand_left else anchor[0]+gap
        py = max(top+margin, anchor[1]-height-gap)
        preferred[slot] = (px,py)
        xs = {left+margin,right-margin-width,anchor[0]-width/2,midline-width-gap,midline+gap,px}
        ys = {top+margin,bottom-margin-height,py}
        for zone in [*forbidden,*soft]:
            xs.update((zone[0]-gap-width,zone[2]+gap))
            ys.update((zone[1]-gap-height,zone[3]+gap))
        # Scan the full picture, so narrow gaps and above-hand positions count.
        xs.update(left+margin+i*step for i in range(max(0,math.ceil((right-left-2*margin-width)/step))+1))
        ys.update(top+margin+i*step for i in range(max(0,math.ceil((bottom-top-2*margin-height)/step))+1))
        candidates=[]
        for x,y in product(sorted(xs),sorted(ys)):
            x,y=round(x),round(y)
            rect=[x,y,x+width,y+height]
            if x<left+margin or y<top+margin or rect[2]>right-margin or rect[3]>bottom-margin:
                continue
            same = ((rect[0]+rect[2])/2 <= midline) == hand_left
            hits = [z for z in forbidden if intersects(rect,z)]
            # Forced positions are only considered on the hand's side.
            if hits and not same:
                continue
            overlap=sum(area(rect,z) for z in soft)
            distance=math.dist((x,y),(px,py))+math.dist((x+width/2,y+height/2),anchor)
            score=distance+overlap/max(1,width*height)*(right-left)
            path=leader(rect,anchor,fraction)
            offset=(start_offsets or {}).get(slot,height/2)
            def sample_path(v):
                w=v.get('size',[width,height])[0]*v.get('zoom',1)
                off=v.get('offset',offset)*v.get('zoom',1)
                a=v['anchor'];start=(x+w if a[0]>=x+w/2 else x,y+round(off))
                return [start,(round(start[0]+(a[0]-start[0])*fraction),start[1]),a]
            frames=values or [{'anchor':anchor,'face':face or (protected[0] if protected else None),'midline':midline}]
            # Evaluate only hard-safe candidates; forced ones need face overlap score.
            passing=0
            if not hits:
                for v in frames:
                    f,m=line_flags(sample_path(v),v.get('face'),v.get('midline',midline),exemption)
                    passing+=not (f or m)
            rate=passing/len(frames)
            rank=3 if hits else (0 if same else 1) if rate>=.95 else 2 if same else None
            if rank is None:
                continue
            face_overlap=area(rect,face) if face else 0
            candidates.append({'rect':rect,'position':[x,y],'score':score,'rank':rank,'leader':path,'line_pass_rate':rate,'same_side':same,'relaxed':rank==2,'forced':rank==3,'face_overlap':face_overlap,'hard_overlap':sum(area(rect,z) for z in hits)})
        # Sorting includes explicit exception classes before scoring.
        choices[slot]=sorted(candidates,key=lambda c:(c['rank'],c['face_overlap'] if c['forced'] else 0,c['score'],c['position']))
    if samples is not None and any(not samples.get(slot) for slot in sizes):
        if any(not values for values in choices.values()):
            raise ValueError('label cannot fit picture safety bounds')
        selected={slot:values[0] for slot,values in choices.items()}
        return {'preferred':preferred,'candidates':choices,'positions':{k:v['position'] for k,v in selected.items()},'selected':selected,'relaxed':any(v['relaxed'] for v in selected.values()),'forced':any(v['forced'] for v in selected.values()),'fallback':False,'reason':None}
    best=None
    # A candidate only needs pairing with the best non-overlapping alternative.
    for a in choices['left']:
        padded=[a['rect'][0]-gap,a['rect'][1]-gap,a['rect'][2]+gap,a['rect'][3]+gap]
        b=next((b for b in choices['right'] if not intersects(padded,b['rect'])),None)
        if b is None:
            continue
        key=(max(a['rank'],b['rank']),a['rank']+b['rank'],a['face_overlap']+b['face_overlap'],a['score']+b['score'])
        if best is None or key<best[0]:best=(key,a,b)
    if best is None:
        # Impossible hard geometry still draws both labels and reports forced.
        if any(not values for values in choices.values()):
            raise ValueError('label cannot fit picture safety bounds')
        a,b=choices['left'][0],choices['right'][0]
        a,b=dict(a,forced=True),dict(b,forced=True)
    else:
        _,a,b=best
    selected={'left':a,'right':b}
    return {'preferred':preferred,'candidates':choices,'positions':{k:v['position'] for k,v in selected.items()},'selected':selected,'relaxed':any(v['relaxed'] for v in selected.values()),'forced':any(v['forced'] for v in selected.values()),'fallback':False,'reason':None}
