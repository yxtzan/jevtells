"""Pillow frame composition and checked ffmpeg pipe encoding with source audio."""

from __future__ import annotations

import hashlib
import fnmatch
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..actions.features import smooth_zero_phase
from ..i18n import load_translations
from ..stages.prepare import _ffmpeg
from .animation import commentary_at, window_at
from .geometry import Layout
from .labels import schedule
from .panels import Panels, opacity
from .text import Fonts, draw_fitted


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
    def __init__(self, settings: Mapping[str, Any], layout: Layout, windows: Sequence[Mapping[str, Any]], points: Mapping[str, Any], actions: Sequence[Mapping[str, Any]], judgments: Mapping[str, Any], narration: Mapping[str, Any], transcript: Mapping[str, Any], *, title: str, sources: str, lang: str, blur: Sequence[Sequence[float]], subtitles: bool, config: Mapping[str, Any]) -> None:
        self.settings, self.layout, self.windows, self.points = settings, layout, windows, points
        self.events, self.judgments, self.transcript = actions, judgments, transcript
        self.blur, self.subtitles = blur, subtitles
        self.tr = load_translations(lang)
        self.fonts = Fonts(settings, layout.scale)
        self.panels = Panels(settings, layout, self.fonts, self.tr, windows, judgments, narration, title, sources)
        self.c, self.p = settings["colors"], settings[layout.kind]
        self.label_cache: dict[tuple[str, str], Image.Image] = {}
        self.subtitle_cache: dict[str, Image.Image] = {}
        self.visibility = float(config.get("detection", {}).get("pose_visibility_threshold", 0.5))
        smoothing = config.get("smoothing", {})
        smooth_window, polynomial = int(smoothing.get("window_length", 7)), int(smoothing.get("polyorder", 2))
        pose = np.asarray(points.get("pose", []), dtype=float)
        hands = np.asarray(points.get("hands", []), dtype=float)
        self.anchors: dict[str, np.ndarray] = {}
        for slot, side in (("left", 1), ("right", 0)):
            wrist = int(settings["label"]["wrist_points"][side])
            xy = pose[:, wrist, :2].copy()
            if pose.shape[-1] > 3:
                xy[pose[:, wrist, 3] < self.visibility] = np.nan
            if hands.ndim == 4 and hands.shape[1] > side:
                palm = hands[:, side, settings["label"]["palm_points"], :2]
                valid = np.isfinite(palm).all(axis=(1, 2))
                xy[valid] = palm[valid].mean(axis=1)
            self.anchors[slot] = smooth_zero_phase(xy, smooth_window, polynomial) if smoothing.get("enabled", True) else xy
        head = pose[:, int(settings["label"]["head_point"]), :2]
        self.anchors["head"] = smooth_zero_phase(head, smooth_window, polynomial)

    def label_sprite(self, event: Mapping[str, Any], slot: str) -> Image.Image:
        cache_key = (str(event["id"]), slot)
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
        fit = self.fonts.fit(text, "sans_black", self.p["label_size"], self.p["label_width"] - 2 * label["padding"][0], 1)
        px, py = (self.layout.px(v) for v in label["padding"])
        width = round(fit.font.getlength(text)) + 2 * px
        height = round(fit.size * g["line_height"]) + 2 * py
        magnitude = event.get("magnitude", event.get("amplitude"))
        details = [self.tr["ui"]["magnitude"].format(value=self.tr["facts"]["magnitudes"].get(magnitude, magnitude))] if magnitude else []
        judgment = self.judgments.get(str(event.get("window"))) or {}
        probability = judgment.get("actions", {}).get(str(event["id"]))
        details.append(self.tr["ui"]["expressive"].format(value=f"{probability:.2f}" if isinstance(probability, (int, float)) else self.tr["ui"]["missing"]))
        badge_text = " · ".join(details)
        badge_fit = self.fonts.fit(badge_text, "sans", self.p["badge_size"], self.p["label_width"] - 2 * label["badge_padding"][0] - label["badge_indent"], 1)
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
            xy = self.anchors["head" if event.get("limb") in {"head", "body"} else slot][point_index]
            if not np.isfinite(xy).all():
                continue
            anchor = self.layout.source_point(xy[0] * self.layout.source[0], xy[1] * self.layout.source[1])
            vx, vy, vw, vh = self.layout.video
            if not self.layout.px(vx) <= anchor[0] <= self.layout.px(vx + vw) or not self.layout.px(vy) <= anchor[1] <= self.layout.px(vy + vh):
                continue
            if anchor[1] >= self.layout.px(vy + vh * (1 - self.settings["subtitle_exclusion_ratio"])):
                continue
            if self.layout.kind == "h":
                cx, cy, cw, ch = self.p["card"]
                if self.layout.px(cx) <= anchor[0] <= self.layout.px(cx + cw) and self.layout.px(cy) <= anchor[1] <= self.layout.px(cy + ch):
                    continue
            sprite = self.label_sprite(event, slot)
            if zoom < 1:
                sprite = sprite.resize((round(sprite.width * zoom), round(sprite.height * zoom)), Image.Resampling.LANCZOS)
            x, y = self.layout.point(*self.p["label_slots"][0 if slot == "left" else 1])
            # Keep the fixed slot but cap its right edge for long English text.
            x = min(x, image.width - sprite.width)
            body_height = self.layout.px(self.p["label_size"] * self.settings["components"]["line_height"] + 2 * label["padding"][1])
            start = (x + sprite.width if slot == "left" else x, y + round(body_height * zoom / 2))
            elbow = (round(start[0] + (anchor[0] - start[0]) * label["leader_fraction"]), start[1])
            color = (*tuple(int(self.c["lime"][i:i+2], 16) for i in (1, 3, 5)), round(255 * alpha))
            leader = Image.new("RGBA", image.size)
            ld = ImageDraw.Draw(leader)
            ld.line([start, elbow, anchor], fill=color, width=max(1, self.layout.px(label["leader_width"])), joint="curve")
            radius = self.layout.px(label["point_radius"])
            ld.ellipse((anchor[0] - radius, anchor[1] - radius, anchor[0] + radius, anchor[1] + radius), fill=color, outline=self.c["ink"], width=max(1, self.layout.px(label["point_outline"])))
            image.alpha_composite(leader)
            image.alpha_composite(opacity(sprite, alpha), (x, y))

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
        source = blur_frame(source, self.blur, self.layout.source, self.settings["blur_radius"])
        x, y, width, height = self.layout.video
        image = Image.new("RGBA", self.layout.output, self.c["ink"])
        image.paste(source.resize((self.layout.px(width), self.layout.px(height)), Image.Resampling.BICUBIC), self.layout.point(x, y))
        image.alpha_composite(self.panels.base)
        index, gap = window_at(self.windows, seconds)
        attenuation = self.settings["gap_opacity"] if gap else 1.0
        if index is not None:
            elapsed = seconds - float(self.windows[index]["t0"])
            image.alpha_composite(opacity(self.panels.analysis(index, elapsed), attenuation))
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
    for name in ("clip.mp4", "windows.json", "keypoints.npz", "actions.json", "judgments.json", "narration.json", "transcript.json", "judge_meta.json", "narrate_meta.json"):
        path = out / name
        if path.exists():
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def run(clip: Path, out: Path, windows: Sequence[Mapping[str, Any]], points: Mapping[str, Any], *, config: Mapping[str, Any], speaker: str, lang: str = "zh", title: str | None = None, layout: str = "both", blur: Sequence[Sequence[float]] = (), subtitles: bool = True, source_size: tuple[int, int] | None = None, force: bool = False) -> dict[str, Any]:
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
        signature = _signature(out, {"render": settings, "kind": kind, "title": title, "lang": lang, "blur": blur, "subtitles": subtitles, "source_size": source_size})
        destination = out / f"output_{kind}.mp4"
        if destination.exists() and not force and previous_meta.get(kind, {}).get("signature") == signature:
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
            geometry = Layout.create(kind, settings, source_size or frame_size)
            painter = Composer(settings, geometry, windows, points, events, judgments, narration, transcript, title=title, sources=_sources(out, settings, config, lang), lang=lang, blur=blur, subtitles=subtitles, config=config)
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
                        process.stdin.write(composed.tobytes())
                        if frame_index % max(1, round(fps * 5)) == 0:
                            print(f"render {kind}: {seconds:.1f}/{total / fps:.1f}s elapsed={time.perf_counter() - started:.1f}s", flush=True)
                except BrokenPipeError:
                    pass
                except BaseException:
                    process.kill()
                    raise
                finally:
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
