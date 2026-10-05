"""Pillow frame composition and checked ffmpeg pipe encoding with source audio."""

from __future__ import annotations

import hashlib
import copy
import fnmatch
import json
import re
import subprocess
import time
from pathlib import Path
from dataclasses import replace
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..actions.features import smooth_zero_phase
from ..i18n import load_translations
from ..stages.prepare import _ffmpeg
from .animation import commentary_at, ease_in_out_cubic, progress, window_at
from .geometry import Layout
from .labels import schedule
from .panels import Panels, opacity
from .text import Fonts, draw_fitted
from .layout_audit import intersects, segment_intersects_rect
from .body_zones import body_zones
from .placement import card_side, leader, select_positions
from .reframe import crop_at, detect_subtitles, plan_crops, subtitle_mode


def blur_frame(frame: Image.Image, rectangles: Sequence[Sequence[float]], original: tuple[int, int], radius: float) -> Image.Image:
    """Blur source-space ROIs before resizing or drawing any overlay."""
    if not rectangles:
        return frame
    image = frame.copy()
    sx, sy = image.width / original[0], image.height / original[1]
    for x, y, width, height in rectangles:
        if x + width > original[0] or y + height > original[1]:
            raise ValueError("blur rectangle extends beyond the original video")
        box = (round(x * sx), round(y * sy), round((x + width) * sx), round((y + height) * sy))
        image.paste(image.crop(box).filter(ImageFilter.GaussianBlur(radius * min(sx, sy))), box)
    return image


class Composer:
    def __init__(self, settings: Mapping[str, Any], layout: Layout, windows: Sequence[Mapping[str, Any]], points: Mapping[str, Any], actions: Sequence[Mapping[str, Any]], judgments: Mapping[str, Any], narration: Mapping[str, Any], transcript: Mapping[str, Any], *, title: str, sources: str, lang: str, blur: Sequence[Sequence[float]], subtitles: bool, config: Mapping[str, Any], shots: Sequence[Mapping[str, Any]] = (), reframe_plan: Mapping[str, Any] | None = None) -> None:
        self.settings, self.layout, self.windows, self.points = settings, layout, windows, points
        self.events, self.judgments, self.transcript = actions, judgments, transcript
        self.blur, self.subtitles = blur, subtitles
        self.shots = list(shots) or [{"index":1,"t0":0.,"t1":len(points.get("pose",[]))/float(points.get("fps",30)),"label":"target","far":False}]
        self.reframe_plan = reframe_plan or {}
        self.audit: dict[str, Any] = {}
        self.tr = load_translations(lang)
        self.fonts = Fonts(settings, layout.scale)
        self.panels = Panels(settings, layout, self.fonts, self.tr, windows, judgments, narration, title, sources)
        self.c, self.p = settings["colors"], settings[layout.kind]
        self.label_cache: dict[tuple[str, str, float], Image.Image] = {}
        self.label_limits: dict[str, float] = {}
        self.subtitle_cache: dict[str, Image.Image] = {}
        self.visibility = float(config.get("detection", {}).get("pose_visibility_threshold", 0.5))
        smoothing = config.get("smoothing", {})
        smooth_window, polynomial = int(smoothing.get("window_length", 7)), int(smoothing.get("polyorder", 2))
        pose = np.asarray(points.get("pose", []), dtype=float)
        hands = np.asarray(points.get("hands", []), dtype=float)
        self.anchors: dict[str, np.ndarray] = {}
        self.raw_anchors: dict[str, np.ndarray] = {}
        for slot, side in (("left", 1), ("right", 0)):
            wrist = int(settings["label"]["wrist_points"][side])
            xy = pose[:, wrist, :2].copy()
            if pose.shape[-1] > 3:
                xy[pose[:, wrist, 3] < self.visibility] = np.nan
            if hands.ndim == 4 and hands.shape[1] > side:
                palm = hands[:, side, settings["label"]["palm_points"], :2]
                valid = np.isfinite(palm).all(axis=(1, 2))
                xy[valid] = palm[valid].mean(axis=1)
            self.raw_anchors[slot] = xy.copy()
            self.anchors[slot] = smooth_zero_phase(xy, smooth_window, polynomial) if smoothing.get("enabled", True) else xy
        head = pose[:, int(settings["label"]["head_point"]), :2]
        self.raw_anchors["head"] = head.copy()
        self.anchors["head"] = smooth_zero_phase(head, smooth_window, polynomial)
        self.card_sides = {}
        for shot in self.shots:
            if layout.kind == "h":
                start,end=(round(float(shot[k])*float(points.get("fps",30))) for k in ("t0","t1"))
                core=body_zones(pose[start:end],layout.source,self.visibility).get("torso",shot.get("target_box"))
                self.card_sides[int(shot["index"])] = card_side(core,layout.source[0],self.p["card"][2]*layout.source[0]/layout.video[2],self.layout.px(settings.get("layout",{}).get("safe_margin",24))/layout.scale*layout.source[0]/layout.video[2])
        self.placements: dict[int, dict[str, Any]] = {}
        for shot in self.shots:
            if self.reframe_plan.get("enabled"):
                self.layout=replace(self.layout,crop=crop_at(self.reframe_plan,float(shot["t0"])))
            self.placements[int(shot["index"])] = self.place(shot)
        self.layout=layout

    def place(self, shot: Mapping[str, Any]) -> dict[str, Any]:
        start = round(float(shot["t0"])*float(self.points.get("fps",30)))
        end = round(float(shot["t1"])*float(self.points.get("fps",30)))
        cores = body_zones(np.asarray(self.points["pose"])[start:end], self.layout.source, self.visibility)
        if cores.get("torso") and not shot.get("far"):
            x0,y0,x1,y1=cores["torso"]
            box=self.layout.source_rect((x0,y0,x1-x0,y1-y0))
            vx,vy,vw,vh=self.layout.video
            margin=self.layout.px(self.settings["layout"]["safe_margin"]+self.settings["layout"]["stack_gap"])
            available=sorted({min(self.p["label_width"],space/self.layout.scale) for space in (box[0]-self.layout.px(vx)-margin,self.layout.px(vx+vw)-box[2]-margin) if space>0})
            for event in self.events:
                mid=(float(event["t0"])+float(event["t1"]))/2
                if not float(shot["t0"])<=mid<float(shot["t1"]) or event.get("far"):
                    continue
                for width in available:
                    self.label_limits[str(event["id"])]=width
                    try:
                        self.label_sprite(event,"left")
                        self.label_sprite(event,"right")
                        break
                    except ValueError:
                        self.label_limits.pop(str(event["id"]),None)
        sizes, anchors = {}, {}
        for slot in ("left", "right"):
            sprites = [self.label_sprite(event,slot) for event in self.events if not event.get("far") and event.get("type") in self.settings["label_types"] and float(event["t0"]) < float(shot["t1"]) and float(event["t1"]) >= float(shot["t0"]) and (event.get("limb") == "both_hands" or (slot=="left") == (event.get("limb") in {"right_hand","head","body"}))]
            sizes[slot] = [max((sprite.width for sprite in sprites),default=self.layout.px(self.p["label_width"])),max((sprite.height for sprite in sprites),default=self.layout.px(80))]
            xy = self.anchors[slot][start:end]
            xy = xy[np.isfinite(xy).all(axis=1)]
            median = np.median(xy,axis=0) if len(xy) else np.array([.5,.5])
            anchors[slot] = self.layout.source_point(median[0]*self.layout.source[0],median[1]*self.layout.source[1])
        saved_layout = self.layout
        samples = {slot: [] for slot in ("left", "right")}
        hard_by_name = {}
        frame_faces, frame_mids = [], []
        fps = float(self.points.get("fps", 30))
        for index in range(start, min(end, len(self.points["pose"]))):
            seconds = index / fps
            if self.reframe_plan.get("enabled"):
                self.layout = replace(saved_layout, crop=crop_at(self.reframe_plan, seconds))
            frame_core = body_zones(np.asarray(self.points["pose"])[index], self.layout.source, self.visibility)
            face = frame_core.get("face_raw")
            face = list(self.layout.source_rect((face[0],face[1],face[2]-face[0],face[3]-face[1]))) if face else None
            mid = self.layout.source_point(frame_core.get("midline", self.layout.source[0]/2), 0)[0]
            wi, _ = window_at(self.windows, seconds)
            lost = not np.isfinite(np.asarray(self.points["pose"])[index,:,:2]).any()
            if "target_index" in self.points: lost |= int(self.points["target_index"][index]) < 0
            shot_label = str(self.windows[wi].get("shot", "target")) if wi is not None else "target"
            has_label = False
            for slot, (event, alpha, zoom) in schedule(self.events,seconds,self.settings,lost=lost,shot=shot_label).items():
                if shot.get("far") or shot.get("label")=="other" or (event.get("shot_index") is not None and event["shot_index"]!=shot["index"]):
                    continue
                anchor_slot="head" if event.get("limb") in {"head","body"} else slot
                xy=self.raw_anchors[anchor_slot][index]
                if not np.isfinite(xy).all():xy=self.anchors[anchor_slot][index]
                if not np.isfinite(xy).all():continue
                sprite=self.label_sprite(event,slot)
                has_label = True
                samples[slot].append({"anchor":self.layout.source_point(xy[0]*self.layout.source[0],xy[1]*self.layout.source[1]),"face":face,"midline":mid,"size":sprite.size,"zoom":zoom,"offset":self.layout.px(self.p["label_size"]*self.settings["components"]["line_height"]+2*self.settings["label"]["padding"][1])/2})
            if has_label:
                if face: frame_faces.append(face)
                frame_mids.append(mid)
                zones_at = self.forbidden(seconds)
                for zone_index, zone in enumerate(zones_at):
                    name, rect = zone["name"], zone["rect"]
                    if name == "edge": name += ":" + str(zone_index)
                    if name not in hard_by_name:
                        hard_by_name[name] = list(rect)
                    else:
                        old = hard_by_name[name]
                        hard_by_name[name] = [min(old[0],rect[0]),min(old[1],rect[1]),max(old[2],rect[2]),max(old[3],rect[3])]
        self.layout = saved_layout
        zones=[{"name":name,"rect":rect} for name,rect in hard_by_name.items()]
        soft=[]
        for name in ("torso","hands"):
            if name in cores:
                x0,y0,x1,y1=cores[name]
                soft.append(list(self.layout.source_rect((x0,y0,x1-x0,y1-y0))))
        target = soft[0] if cores.get("torso") else None
        for slot, values in samples.items():
            if values:
                anchors[slot]=list(np.median([v['anchor'] for v in values],axis=0))
        midline=float(np.median(frame_mids)) if frame_mids else None
        face=[min(f[0] for f in frame_faces),min(f[1] for f in frame_faces),max(f[2] for f in frame_faces),max(f[3] for f in frame_faces)] if frame_faces else None
        vx,vy,vw,vh=self.layout.video
        cfg=self.settings.get("layout",{})
        result=select_positions(self.layout.rect((vx,vy,vw,vh)),target,[zone["rect"] for zone in zones],sizes,anchors,margin=self.layout.px(cfg.get("safe_margin",24)),gap=self.layout.px(cfg.get("stack_gap",12)),step=self.layout.px(cfg.get("candidate_step",12)),soft=soft,samples=samples,midline=midline,face=face,exemption=self.layout.px(60),fraction=self.settings["label"]["leader_fraction"])
        return {"shot_index":shot["index"],"t0":shot["t0"],"t1":shot["t1"],"sizes":sizes,"anchors":anchors,"sample_counts":{slot:len(v) for slot,v in samples.items()},"forbidden":zones,"card_side":self.card_sides.get(int(shot["index"])),**result}

    def card_x(self, seconds: float) -> float:
        default=self.p.get("card",[0])[0]
        if self.layout.kind != "h" or not self.shots:
            return default
        index=next((i for i,shot in enumerate(self.shots) if float(shot["t0"])<=seconds<float(shot["t1"])),len(self.shots)-1)
        def position(i: int) -> float:
            return default if self.card_sides[int(self.shots[i]["index"])]=="right" else self.settings["layout"]["card_left_x"]
        current=position(index)
        if not index:
            return current
        before=position(index-1)
        amount=ease_in_out_cubic((seconds-float(self.shots[index]["t0"]))/self.settings["animation"]["card_seconds"])
        return before+(current-before)*amount

    def label_sprite(self, event: Mapping[str, Any], slot: str) -> Image.Image:
        available=self.label_limits.get(str(event["id"]),self.p["label_width"])
        cache_key = (str(event["id"]), slot, available)
        if cache_key in self.label_cache:
            return self.label_cache[cache_key]
        label = self.settings["label"]
        g = self.settings["components"]
        limb = str(event.get("limb", ""))
        prefix = self.tr["facts"]["limbs"].get(limb, "")
        action = self.tr["actions"].get(str(event["type"]), str(event["type"]))
        if limb == "both_hands" and action.startswith(prefix):
            action = action[len(prefix):].strip()
        text = (prefix + " · " if prefix else "") + action
        fit = self.fonts.fit(text, "sans_black", self.p["label_size"], available - 2 * label["padding"][0], 1)
        px, py = (self.layout.px(v) for v in label["padding"])
        width = round(fit.font.getlength(text)) + 2 * px
        height = round(fit.size * g["line_height"]) + 2 * py
        magnitude = event.get("magnitude", event.get("amplitude"))
        details = [self.tr["ui"]["magnitude"].format(value=self.tr["facts"]["magnitudes"].get(magnitude, magnitude))] if magnitude else []
        judgment = self.judgments.get(str(event.get("window"))) or {}
        probability = judgment.get("actions", {}).get(str(event["id"]))
        details.append(self.tr["ui"]["expressive"].format(value=f"{probability:.2f}" if isinstance(probability, (int, float)) else self.tr["ui"]["missing"]))
        badge_text = " · ".join(details)
        badge_fit = self.fonts.fit(badge_text, "sans", self.p["badge_size"], available - 2 * label["badge_padding"][0] - label["badge_indent"], 1)
        bx, by = (self.layout.px(v) for v in label["badge_padding"])
        badge_width = round(badge_fit.font.getlength(badge_text)) + 2 * bx
        badge_height = round(badge_fit.size * g["line_height"]) + 2 * by
        indent, gap = self.layout.px(label["badge_indent"]), self.layout.px(label["gap"])
        sprite = Image.new("RGBA", (max(width, badge_width + indent), height + gap + badge_height))
        draw = ImageDraw.Draw(sprite)
        draw.rounded_rectangle((0, 0, width, height), radius=height / 2, fill=self.c["ink"])
        draw_fitted(sprite, (px, py), fit, self.c, line_height=g["line_height"], fill=self.c["lime"])
        draw.rounded_rectangle((indent, height + gap, indent + badge_width, height + gap + badge_height), radius=badge_height / 2, fill=self.c["fg"])
        draw_fitted(sprite, (indent + bx, height + gap + by), badge_fit, self.c, line_height=g["line_height"], fill=self.c["ink"])
        self.label_cache[cache_key] = sprite
        return sprite

    def _labels(self, image: Image.Image, seconds: float, point_index: int, lost: bool, shot: str) -> None:
        selected = schedule(self.events, seconds, self.settings, lost=lost, shot=shot)
        if not selected:
            return
        draw = ImageDraw.Draw(image)
        label = self.settings["label"]
        for slot, (event, alpha, zoom) in selected.items():
            current_shot = next((value for value in self.shots if float(value["t0"]) <= seconds < float(value["t1"])),None)
            if current_shot and (current_shot.get("far") or current_shot["label"]=="other"):
                continue
            if current_shot and event.get("shot_index") and event["shot_index"] != current_shot["index"]:
                continue
            xy = self.anchors["head" if event.get("limb") in {"head", "body"} else slot][point_index]
            if not np.isfinite(xy).all():
                continue
            anchor = self.layout.source_point(xy[0] * self.layout.source[0], xy[1] * self.layout.source[1])
            projected_anchor = anchor
            vx,vy,vw,vh=self.layout.video
            radius=self.layout.px(label["point_radius"])+self.layout.px(label["point_outline"])
            anchor=(max(self.layout.px(vx)+radius,min(self.layout.px(vx+vw)-radius,anchor[0])),max(self.layout.px(vy)+radius,min(self.layout.px(vy+vh)-radius,anchor[1])))
            # Pose/shot eligibility controls visibility. Geometry only selects
            # the shot position; it never suppresses a label or its connection.
            self.audit.setdefault("expected_labels", []).append({"slot":slot,"event":event["id"]})
            sprite = self.label_sprite(event, slot)
            if zoom < 1:
                sprite = sprite.resize((round(sprite.width * zoom), round(sprite.height * zoom)), Image.Resampling.LANCZOS)
            placement = self.placements.get(current_shot["index"]) if current_shot else None
            x, y = placement["positions"][slot] if placement else self.layout.point(*self.p["label_slots"][0 if slot == "left" else 1])
            x,y=round(x),round(y)
            # Keep the fixed slot but cap its right edge for long English text.
            x = min(x, image.width - sprite.width)
            body_height = self.layout.px(self.p["label_size"] * self.settings["components"]["line_height"] + 2 * label["padding"][1])
            start = (x + sprite.width if anchor[0] >= x+sprite.width/2 else x, y + round(body_height * zoom / 2))
            elbow = (round(start[0] + (anchor[0] - start[0]) * label["leader_fraction"]), start[1])
            rect=[x,y,x+sprite.width,y+sprite.height]
            raw_xy=self.raw_anchors["head" if event.get("limb") in {"head","body"} else slot][point_index]
            real_anchor=self.layout.source_point(raw_xy[0]*self.layout.source[0],raw_xy[1]*self.layout.source[1]) if np.isfinite(raw_xy).all() else anchor
            real_start=(x+sprite.width if real_anchor[0]>=x+sprite.width/2 else x,start[1])
            real_elbow=(round(real_start[0]+(real_anchor[0]-real_start[0])*label["leader_fraction"]),real_start[1])
            path=[start,elbow,anchor]
            real_path=[real_start,real_elbow,real_anchor]
            color = (*tuple(int(self.c["lime"][i:i+2], 16) for i in (1, 3, 5)), round(255 * alpha))
            leader = Image.new("RGBA", image.size)
            ld = ImageDraw.Draw(leader)
            ld.line(path, fill=color, width=max(1, self.layout.px(label["leader_width"])), joint="curve")
            radius = self.layout.px(label["point_radius"])
            ld.ellipse((anchor[0] - radius, anchor[1] - radius, anchor[0] + radius, anchor[1] + radius), fill=color, outline=self.c["ink"], width=max(1, self.layout.px(label["point_outline"])))
            image.alpha_composite(leader)
            image.alpha_composite(opacity(sprite, alpha), (x, y))
            self.audit["labels"].append({"slot": slot, "event": event["id"], "rect": [x, y, x + sprite.width, y + sprite.height], "leader": [list(p) for p in path], "actual_hand_leader": [list(p) for p in real_path], "point": list(anchor), "point_clamped": anchor != projected_anchor, "relaxed": bool(placement and placement["selected"][slot]["relaxed"]), "forced": bool(placement and placement["selected"][slot]["forced"]), "fallback": False})

    def forbidden(self, seconds: float) -> list[dict[str, Any]]:
        zones = []
        vx, vy, vw, vh = self.layout.video
        def zone(name: str, rect: Sequence[float]) -> None:
            zones.append({"name": name, "rect": list(self.layout.rect(rect))})
        index = min(len(self.points["pose"])-1, max(0, round(seconds*float(self.points.get("fps",30)))))
        for name, value in body_zones(np.asarray(self.points["pose"])[index], self.layout.source, self.visibility).items():
            if name == "face":
                x0,y0,x1,y1=value
                zones.append({"name": name, "rect": list(self.layout.source_rect((x0,y0,x1-x0,y1-y0)))})
        ratio = self.settings["subtitle_exclusion_ratio"]
        if self.layout.strip_video:
            zone("subtitle_strip", self.layout.strip_video)
        else:
            zone("subtitles", (vx, vy + vh * (1-ratio), vw, vh * ratio))
        for rect in self.blur:
            zones.append({"name": "blur", "rect": list(self.layout.source_rect(rect))})
        margin = float(self.settings.get("layout", {}).get("safe_margin", 24))
        for rect in ((vx,vy,vw,margin),(vx,vy,margin,vh),(vx+vw-margin,vy,margin,vh),(vx,vy+vh-margin,vw,margin)):
            zone("edge", rect)
        if self.layout.kind == "h":
            cx,cy,cw,ch=self.p["card"]
            zone("card",(self.card_x(seconds),cy,cw,ch))
            zone("commentary",self.p["commentary"])
            # Use the capsule's actual bounds rather than its maximum text slot.
            wi,_gap=window_at(self.windows,seconds)
            status,position=self.panels.status(wi,round(seconds*10),False)
            shift=self.p["card"][0]-self.card_x(seconds)
            zones.append({"name":"status","rect":[position[0]+self.layout.px(shift),position[1],position[0]+self.layout.px(shift)+status.width,position[1]+status.height]})
            zone("quote", (self.p["commentary"][0],self.p["quote_y"],self.p["commentary"][2],self.p["quote_size"]*self.settings["components"]["line_height"]+2*self.p["quote_padding"][1]))
        return zones

    def debug_layout(self, clip: Path, out: Path) -> None:
        directory = out / "layout_debug"
        directory.mkdir(exist_ok=True)
        capture = cv2.VideoCapture(str(clip))
        try:
            for shot in self.shots:
                seconds = (float(shot["t0"]) + float(shot["t1"])) / 2
                capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("cannot decode layout debug midpoint")
                index = min(len(self.points["pose"])-1, round(seconds * float(self.points.get("fps", 30))))
                image = self.frame(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), seconds, index)
                draw = ImageDraw.Draw(image)
                font = self.fonts.font("mono", 12)
                for zone in self.audit["forbidden"]:
                    draw.rectangle(zone["rect"], outline="red", width=2)
                    draw.text(tuple(zone["rect"][:2]), zone["name"], fill="red", font=font)
                for position in self.p["label_slots"]:
                    x, y = self.layout.point(*position)
                    draw.rectangle((x,y,x+self.layout.px(self.p["label_width"]),y+self.layout.px(80)),outline="#4C82FF",width=2)
                placement=self.placements[int(shot["index"])]
                for values in placement["candidates"].values():
                    for candidate in values[:24]:
                        draw.rectangle(candidate["rect"],outline="#4C82FF",width=1)
                for slot,position in placement["positions"].items():
                    x,y=position;w,h=placement["sizes"][slot]
                    draw.rectangle((x,y,x+w,y+h),outline="#C8FF2E",width=3)
                for label in self.audit["labels"]:
                    draw.rectangle(label["rect"], outline="#C8FF2E", width=3)
                image.save(directory / f"{self.layout.kind}_shot_{shot['index']:03d}.png")
        finally:
            capture.release()

    def _subtitles(self, image: Image.Image, seconds: float) -> None:
        if any("subtitle" in window for window in self.windows):
            text = next((str(window.get("subtitle", "")) for window in self.windows if float(window["t0"]) <= seconds < float(window["t1"])), "")
        else:
            text = next((str(segment.get("text", "")) for segment in self.transcript.get("segments", []) if float(segment["t0"]) <= seconds < float(segment["t1"])), "")
        if not text:
            return
        cfg = self.settings["subtitles"]
        vx, vy, vw, vh = self.layout.video
        if text not in self.subtitle_cache:
            px, py = cfg["padding"]
            fit = self.fonts.fit(text, "sans", cfg["size"], vw * cfg["max_width_ratio"] - 2 * px, 2)
            width = round(max(fit.font.getlength(row) for row in fit.lines)) + 2 * self.layout.px(px)
            height = round(len(fit.lines) * fit.size * self.settings["components"]["line_height"]) + 2 * self.layout.px(py)
            layer = Image.new("RGBA", (width, height), self.c["ink"])
            draw_fitted(layer, self.layout.point(px, py), fit, self.c, line_height=self.settings["components"]["line_height"])
            self.subtitle_cache[text] = layer
        layer = self.subtitle_cache[text]
        image.alpha_composite(layer, (self.layout.px(vx + vw / 2) - layer.width // 2, self.layout.px(vy + vh - cfg["bottom"]) - layer.height))

    def frame(self, source: Image.Image, seconds: float, point_index: int) -> Image.Image:
        if self.reframe_plan.get("enabled"):
            self.layout=replace(self.layout,crop=crop_at(self.reframe_plan,seconds))
        self.audit = {"seconds": seconds, "output": list(self.layout.output), "labels": [], "forbidden": self.forbidden(seconds)}
        cores = body_zones(np.asarray(self.points["pose"])[point_index], self.layout.source, self.visibility)
        face = cores.get("face_raw")
        self.audit["leader_zones"] = [{"name":"face","rect":list(self.layout.source_rect((face[0],face[1],face[2]-face[0],face[3]-face[1])))}] if face else []
        self.audit["midline"] = self.layout.source_point(cores["midline"],0)[0] if "midline" in cores else None
        self.audit["endpoint_exemption"] = self.layout.px(60)
        source = blur_frame(source, self.blur, self.layout.source, self.settings["blur_radius"])
        original_source=source
        if self.layout.crop:
            cx,cy,cw,ch=self.layout.crop
            sx,sy=source.width/self.layout.source[0],source.height/self.layout.source[1]
            source=source.crop((round(cx*sx),round(cy*sy),round((cx+cw)*sx),round((cy+ch)*sy)))
        x, y, width, height = self.layout.video
        image = Image.new("RGBA", self.layout.output, self.c["ink"])
        image.paste(source.resize((self.layout.px(width), self.layout.px(height)), Image.Resampling.BICUBIC), self.layout.point(x, y))
        if self.layout.strip_video and self.layout.strip_y is not None:
            sy=original_source.height/self.layout.source[1]
            strip=original_source.crop((0,round(self.layout.strip_y*sy),original_source.width,original_source.height))
            sx,sy,sw,sh=self.layout.strip_video
            image.paste(strip.resize((self.layout.px(sw),self.layout.px(sh)),Image.Resampling.BICUBIC),self.layout.point(sx,sy))
        image.alpha_composite(self.panels.base)
        index, gap = window_at(self.windows, seconds)
        attenuation = self.settings["gap_opacity"] if gap else 1.0
        if index is not None:
            elapsed = seconds - float(self.windows[index]["t0"])
            analysis=opacity(self.panels.analysis(index, elapsed), attenuation)
            offset=self.layout.px(self.card_x(seconds)-self.p["card"][0]) if self.layout.kind=="h" else 0
            image.alpha_composite(analysis,(offset,0))
            animation = self.settings["animation"]
            sentence, amount, offset = commentary_at(index, elapsed, animation)
            image.alpha_composite(opacity(self.panels.quote(sentence), amount * attenuation))
            layer, position = self.panels.commentary(sentence)
            position = (position[0], position[1] + self.layout.px(offset))
            image.alpha_composite(opacity(layer, amount * attenuation), position)
        else:
            layer, position = self.panels.commentary(None)
            image.alpha_composite(layer, position)
        pose = np.asarray(self.points["pose"])
        lost = not np.isfinite(pose[point_index, :, :2]).any()
        if "target_index" in self.points:
            lost = lost or int(self.points["target_index"][point_index]) < 0
        shot = str(self.windows[index].get("shot", "target")) if index is not None else "target"
        self._labels(image, seconds, point_index, lost, shot)
        if self.subtitles:
            self._subtitles(image, seconds)
        status, position = self.panels.status(index, round(seconds * 10), lost)
        if self.layout.kind == "h":
            position=(position[0]+self.layout.px(self.p["card"][0]-self.card_x(seconds)),position[1])
        image.alpha_composite(status, position)
        return image.convert("RGB")


def model_short_name(identifier: str, settings: Mapping[str, Any]) -> str:
    for pattern, name in settings.get("model_names", {}).items():
        if fnmatch.fnmatchcase(identifier, pattern):
            return str(name)
    return re.sub(r"-(?:\d{8}|\d{4}-\d{2}-\d{2})$", "", identifier.split("/")[-1])


def _sources(out: Path, settings: Mapping[str, Any], config: Mapping[str, Any], lang: str) -> str:
    tr = load_translations(lang)["ui"]
    def model(name: str, fallback: str) -> str:
        path = out / name
        value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        prefix = "jev" if name == "judge_meta.json" else "narrate"
        for raw_path in sorted((out / "raw").glob(f"{prefix}_W*.json")):
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            if raw.get("model"):
                return model_short_name(str(raw["model"]), settings)
        return model_short_name(str(value.get("model", fallback)), settings)
    return " · ".join((tr["action_source"], tr["judge_source"].format(model=model("judge_meta.json", str(config["jev"]["model"]))), tr["narrate_source"].format(model=model("narrate_meta.json", str(config["narrate"]["model"]))), tr["disclaimer"]))


def _signature(out: Path, values: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(values, sort_keys=True, ensure_ascii=False).encode())
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        digest.update(str(path.relative_to(Path(__file__).parent)).encode())
        digest.update(path.read_bytes())
    for name in ("clip.mp4", "windows.json", "keypoints.npz", "actions.json", "judgments.json", "narration.json", "transcript.json", "judge_meta.json", "narrate_meta.json", "shots.json"):
        path = out / name
        if path.exists():
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def run(clip: Path, out: Path, windows: Sequence[Mapping[str, Any]], points: Mapping[str, Any], *, config: Mapping[str, Any], speaker: str, lang: str = "zh", title: str | None = None, layout: str = "both", blur: Sequence[Sequence[float]] = (), subtitles: bool = True, source_size: tuple[int, int] | None = None, force: bool = False, debug_layout: bool = False, reframe: str = "auto") -> dict[str, Any]:
    settings = config["render"]
    executable = _ffmpeg()
    available = subprocess.run([executable, "-hide_banner", "-encoders"], capture_output=True, text=True, check=True).stdout
    encoders = [name for name in ("libx264", "h264_videotoolbox", "mpeg4") if name in available]
    if not encoders:
        raise RuntimeError("no supported ffmpeg video encoder")
    title = title or load_translations(lang)["ui"]["title"].format(speaker=speaker)
    def read(name: str, default: Any) -> Any:
        path = out / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    judgments, narration = read("judgments.json", {}), read("narration.json", {})
    actions = read("actions.json", [])
    events = actions.get("events", []) if isinstance(actions, Mapping) else actions
    transcript = read("transcript.json", {})
    previous_meta = read("render_meta.json", {})
    results: dict[str, Any] = dict(previous_meta)
    for kind in (("h", "v") if layout == "both" else (layout,)):
        signature = _signature(out, {"render": settings, "kind": kind, "title": title, "lang": lang, "blur": blur, "subtitles": subtitles, "source_size": source_size,"reframe":reframe})
        destination = out / f"output_{kind}.mp4"
        if destination.exists() and not force and not debug_layout and previous_meta.get(kind, {}).get("signature") == signature:
            results[kind] = {**previous_meta[kind], "cached": True}
            print(f"render {kind}: cached {destination}", flush=True)
            continue
        started = time.perf_counter()
        encoder_errors = []
        for encoder in encoders:
            capture = cv2.VideoCapture(str(clip))
            if not capture.isOpened():
                raise RuntimeError(f"cannot decode {clip}")
            input_fps = capture.get(cv2.CAP_PROP_FPS)
            total_input = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fps = float(settings["fps"])
            total = round(total_input / input_fps * fps)
            effective=copy.deepcopy(settings)
            shot_list=read("shots.json", [])
            plan={}
            if kind=="v" and reframe=="auto" and frame_size[0]/frame_size[1]>4/3:
                subtitle_result=detect_subtitles(clip,settings["reframe"]) if not subtitles else {"bounds":None}
                (out / "burned_subtitles.json").write_text(json.dumps(subtitle_result,indent=2))
                plan=plan_crops(*(source_size or frame_size),shot_list,points,subtitle_result["bounds"],settings["reframe"])
                mode=subtitle_mode(subtitle_result,*(source_size or frame_size),settings["reframe"]) if not subtitles else {"mode":"center","oversized":False}
                plan.update(mode)
                if mode["mode"] == "off":
                    plan["enabled"]=False
                else:
                    effective["v"].update(effective["v_reframe"])
                    effective["components"].update(effective["reframe_components"])
                    if mode["mode"] == "strip":
                        for shot in plan["shots"]:
                            shot["height"]=mode["strip_y"]
            (out / f"reframe_{kind}.json").write_text(json.dumps(plan,indent=2))
            geometry = Layout.create(kind, effective, source_size or frame_size,crop_at(plan,0),strip_y=plan.get("strip_y") if plan.get("mode")=="strip" else None)
            painter = Composer(effective, geometry, windows, points, events, judgments, narration, transcript, title=title, sources=_sources(out, settings, config, lang), lang=lang, blur=blur, subtitles=subtitles, config=config, shots=shot_list,reframe_plan=plan)
            (out / f"panel_spacing_{kind}.json").write_text(json.dumps({"geometry":{"picture":geometry.video,"strip":geometry.strip_video,"strip_y":geometry.strip_y},"spacing":getattr(painter.panels,"spacing",None),"positions":painter.panels.p},indent=2))
            (out / f"layout_{kind}.json").write_text(json.dumps(list(painter.placements.values()), ensure_ascii=False, indent=2))
            if debug_layout:
                painter.debug_layout(clip, out)
            raw = out / f"render_{kind}_noaudio.mp4"
            log_path = out / f"render_{kind}_ffmpeg.log"
            command = [executable, "-y", "-hide_banner", "-loglevel", "warning", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{geometry.output[0]}x{geometry.output[1]}", "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", encoder, "-pix_fmt", "yuv420p"]
            if encoder == "libx264":
                command += ["-crf", str(settings["crf"]), "-preset", str(settings["preset"])]
            else:
                command += ["-b:v", str(settings["bitrate"])]
                if encoder == "h264_videotoolbox":
                    command += ["-allow_sw", "1"]
            command.append(str(raw))
            with log_path.open("w", encoding="utf-8") as log:
                trace = (out / f"layout_trace_{kind}.jsonl").open("w", encoding="utf-8")
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log, stdout=subprocess.DEVNULL)
                last_source = -1
                frame = None
                try:
                    for frame_index in range(total):
                        seconds = frame_index / fps
                        source_index = min(total_input - 1, int(seconds * input_fps))
                        while last_source < source_index:
                            ok, frame = capture.read()
                            if not ok:
                                raise RuntimeError("video decoding ended before expected frame count")
                            last_source += 1
                        rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        point_index = min(len(points["pose"]) - 1, round(seconds * float(points.get("fps", input_fps))))
                        composed = painter.frame(rgb, seconds, point_index)
                        trace.write(json.dumps({"frame": frame_index, **painter.audit}) + "\n")
                        process.stdin.write(composed.tobytes())
                        if frame_index % max(1, round(fps * 5)) == 0:
                            print(f"render {kind}: {seconds:.1f}/{total / fps:.1f}s elapsed={time.perf_counter() - started:.1f}s", flush=True)
                except BrokenPipeError:
                    pass
                except BaseException:
                    process.kill()
                    raise
                finally:
                    trace.close()
                    capture.release()
                    try:
                        process.stdin.close()
                    except BrokenPipeError:
                        pass
                    process.wait()
            if process.returncode == 0:
                break
            encoder_errors.append({"encoder": encoder, "log": log_path.read_text(encoding="utf-8")[-2000:]})
        else:
            raise RuntimeError(f"all video encoders failed: {encoder_errors}")
        subprocess.run([executable, "-y", "-hide_banner", "-loglevel", "error", "-i", str(raw), "-i", str(clip), "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-movflags", "+faststart", str(destination)], check=True)
        raw.unlink()
        results[kind] = {"file": str(destination), "resolution": list(geometry.output), "fps": fps, "frames": total, "elapsed_s": time.perf_counter() - started, "encoder": encoder, "encoder_failures": encoder_errors, "signature": signature, "cached": False}
        (out / "render_meta.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"render {kind}: complete in {results[kind]['elapsed_s']:.1f}s", flush=True)
    return results
