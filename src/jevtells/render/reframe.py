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
    top=round(height*(1-float(settings.get("subtitle_strip",.2))))
    columns=np.zeros(width,dtype=float)
    samples=[]
    try:
        for frame in range(0,total,max(1,round(fps*float(settings.get("sample_seconds",1))))):
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
                samples.append({"seconds":frame/fps,"bounds":[max(0,int(observed[0])-2),min(width,int(observed[-1])+3)]})
    finally:
        capture.release()
    observed=np.flatnonzero(columns>0)
    return {"bounds":[max(0,int(observed[0])-2),min(width,int(observed[-1])+3)] if len(observed) else None,"samples":samples,"source":[width,height],"strip_y":top}


def plan_crops(width: int, height: int, shots: Sequence[Mapping[str, Any]], points: Mapping[str, Any], subtitles: Sequence[float] | None, settings: Mapping[str, Any]) -> dict[str, Any]:
    enabled=width/height>4/3
    if not enabled:
        return {"enabled":False,"shots":[]}
    fps=float(points.get("fps",30));pose=np.asarray(points.get("pose",[]),dtype=float)
    timeline=[]
    for shot in shots:
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
    return left,0,float(shot["width"]),float(shot["height"])
