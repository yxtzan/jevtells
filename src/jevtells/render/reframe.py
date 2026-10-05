"""Full-height 4:3 crops constrained by measured burned subtitle edges."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .animation import ease_in_out_cubic


def crop_left(width: float, height: float, center: float | None, subtitles: Sequence[float] | None, max_offset: float = .08) -> float:
    crop_width = min(width,height*4/3)
    center = width/2 if center is None else center
    if subtitles is not None:
        lo,hi=subtitles
        if hi-lo>crop_width:
            center=(lo+hi)/2
        else:
            return max(0.0,min(width-crop_width,max(hi-crop_width,min(lo,center-crop_width/2))))
    else:
        center=max(width*(.5-max_offset),min(width*(.5+max_offset),center))
    return max(0.0,min(width-crop_width,center-crop_width/2))


def detect_subtitles(clip: Path, settings: Mapping[str, Any]) -> dict[str, Any]:
    capture=cv2.VideoCapture(str(clip))
    width,height=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=capture.get(cv2.CAP_PROP_FPS) or 30
    total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    top=round(height*(1-max(.30,float(settings.get("subtitle_strip",.2)))))
    text_tops=[]
    columns=np.zeros(width,dtype=float)
    samples=[]
    try:
        for frame in range(0,total,max(1,round(fps*float(settings.get("sample_seconds",.5))))):
            capture.set(cv2.CAP_PROP_POS_FRAMES,frame)
            ok,image=capture.read()
            if not ok:
                raise RuntimeError("cannot decode burned-subtitle sample")
            hsv=cv2.cvtColor(image[top:],cv2.COLOR_BGR2HSV)
            # Bright neutral or yellow lettering; reject large background
            # components, then accumulate the actual text-edge columns.
            mask=(((hsv[:,:,2]>=settings.get("text_value",140)) & ((hsv[:,:,1]<=settings.get("text_saturation",85)) | ((hsv[:,:,0]>=15)&(hsv[:,:,0]<=45))))*255).astype("uint8")
            count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
            text=np.zeros_like(mask)
            accepted=[]
            for index in range(1,count):
                x,y,w,h,area=stats[index]
                surround=hsv[max(0,y-3):min(len(hsv),y+h+3),max(0,x-3):min(width,x+w+3),2]
                dark_fraction=float((surround<settings.get("background_value",90)).mean())
                if height*.006 <= h <= height*.065 and 2 <= w <= width*.08 and area>=8 and dark_fraction>=settings.get("background_ratio",.45):
                    accepted.append((index,x,y,w,h))
            # Text lines have several small glyph components at a common y.
            for index,x,y,w,h in accepted:
                neighbours=sum(abs((y+h/2)-(oy+oh/2))<height*.025 for _i,_x,oy,_w,oh in accepted)
                if neighbours>=int(settings.get("min_glyphs",6)):
                    text[labels==index]=255
            edges=cv2.Canny(text,settings.get("edge_low",80),settings.get("edge_high",180))
            observed=np.flatnonzero(edges.any(axis=0))
            if len(observed):
                columns+=edges.sum(axis=0)/255
                rows=np.flatnonzero(text.any(axis=1))
                text_top=top+int(rows[0])
                text_tops.append(text_top)
                samples.append({"seconds":frame/fps,"bounds":[max(0,int(observed[0])-2),min(width,int(observed[-1])+3)],"text_top":text_top})
    finally:
        capture.release()
    observed=np.flatnonzero(columns>0)
    return {"bounds":[max(0,int(observed[0])-2),min(width,int(observed[-1])+3)] if len(observed) else None,"samples":samples,"source":[width,height],"strip_y":subtitle_strip_y(height,text_tops),"sample_seconds":float(settings.get("sample_seconds",.5))}


def subtitle_strip_y(height: int, text_tops: Sequence[float]) -> int:
    return max(0, round(min([height*.8, *(top-6 for top in text_tops)])))


def subtitle_mode(detection: Mapping[str, Any], width: int, height: int, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Require consecutive oversized samples, never trigger on a lone edge."""
    interval=float(settings.get("sample_seconds",.5))
    limit=min(width,height*4/3)-32
    previous=None
    runs=[]
    for sample in detection.get("samples",[]):
        bounds=sample.get("bounds")
        oversized=bool(bounds and bounds[1]-bounds[0]>limit)
        if oversized and previous is not None and sample["seconds"]-previous["seconds"]<=interval*1.1:
            runs.append([previous["seconds"],sample["seconds"]])
        previous=sample if oversized else None
    policy=str(settings.get("oversized_subtitles","strip"))
    if policy not in {"strip","center","off"}:
        raise ValueError("oversized_subtitles must be strip, center or off")
    return {"mode":policy if runs else "center", "oversized":bool(runs),"oversized_pairs":runs,"strip_y":detection.get("strip_y",round(height*.8)),"width_limit":limit}


def plan_crops(width: int, height: int, shots: Sequence[Mapping[str, Any]], points: Mapping[str, Any], subtitles: Sequence[float] | None, settings: Mapping[str, Any]) -> dict[str, Any]:
    enabled=width/height>4/3
    if not enabled:
        return {"enabled":False,"shots":[]}
    fps=float(points.get("fps",30));pose=np.asarray(points.get("pose",[]),dtype=float)
    timeline=[]
    for shot in shots:
        if shot.get("multi_person"):
            timeline.append({"index":shot["index"],"t0":shot["t0"],"t1":shot["t1"],"left":0.,"width":width,"height":height,"pans":[],"multi_person":True})
            continue
        left=crop_left(width,height,shot.get("target_center_x"),subtitles,float(settings.get("max_offset",.08)))
        pans=[];pending=None;current=left
        start,end=round(float(shot["t0"])*fps),min(len(pose),round(float(shot["t1"])*fps))
        for index in range(start,end):
            xy=pose[index,:,:2]
            valid=np.isfinite(xy).all(axis=1)
            if pose.shape[-1]>=4:
                valid &= pose[index,:,3]>=float(settings.get("visibility",.5))
            x=xy[valid,0]*width
            if not len(x):
                pending=None;continue
            low,high=np.percentile(x,[5,95]);crop_width=height*4/3
            if low<current-crop_width*float(settings.get("pan_exit_ratio",.1)) or high>current+crop_width*(1+float(settings.get("pan_exit_ratio",.1))):
                pending=index if pending is None else pending
                if (index-pending)/fps>=float(settings.get("pan_wait_seconds",1)):
                    destination=crop_left(width,height,float((low+high)/2),subtitles,float(settings.get("max_offset",.08)))
                    if abs(destination-current)>1:
                        pans.append({"t0":index/fps,"t1":index/fps+float(settings.get("pan_seconds",.6)),"from":current,"to":destination})
                        current=destination
                    pending=None
            else:
                pending=None
        timeline.append({"index":shot["index"],"t0":shot["t0"],"t1":shot["t1"],"left":left,"width":height*4/3,"height":height,"pans":pans})
    for previous, current in zip(timeline, timeline[1:]):
        source_shot = next(s for s in shots if s['index'] == current['index'])
        if not source_shot.get('cut_at_start') and bool(previous.get('multi_person')) != bool(current.get('multi_person')):
            current['transition'] = {'t0':current['t0'], 'seconds':float(settings.get('zoom_seconds',.6)),
                                     'left':previous['left'], 'width':previous['width'], 'height':previous['height']}
    return {"enabled":True,"shots":timeline,"subtitle_bounds":subtitles,"subtitle_oversized":bool(subtitles is not None and subtitles[1]-subtitles[0]>height*4/3)}


def crop_at(plan: Mapping[str, Any], seconds: float) -> tuple[float,float,float,float] | None:
    if not plan.get("enabled") or not plan.get("shots"):
        return None
    shot=next((shot for shot in plan["shots"] if float(shot["t0"])<=seconds<float(shot["t1"])),plan["shots"][-1])
    left=float(shot["left"])
    for pan in shot["pans"]:
        if seconds>=pan["t0"]:
            amount=ease_in_out_cubic((seconds-pan["t0"])/(pan["t1"]-pan["t0"]))
            left=pan["from"]+(pan["to"]-pan["from"])*amount
    width, height = float(shot['width']), float(shot['height'])
    transition = shot.get('transition')
    if transition:
        amount = ease_in_out_cubic((seconds-transition['t0'])/transition['seconds'])
        left = transition['left']+(left-transition['left'])*amount
        width = transition['width']+(width-transition['width'])*amount
        height = transition['height']+(height-transition['height'])*amount
    return left,0,width,height
