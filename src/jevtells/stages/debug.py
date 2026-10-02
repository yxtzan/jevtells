"""Render the landmark debug video and preserve the source audio."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]


def _font() -> ImageFont.FreeTypeFont:
    """Load the downloaded Chinese font for overlay text."""
    path = Path(__file__).resolve().parents[3] / "assets/fonts/NotoSansCJKsc-Regular.otf"
    if not path.exists():
        raise FileNotFoundError(f"font missing: {path}; run scripts/download_models.py")
    return ImageFont.truetype(str(path), 26)


def _draw(frame: np.ndarray, points: dict[str, Any], frame_index: int, fps: float, windows: list[dict[str, Any]]) -> np.ndarray:
    """Draw pose, both hands, labels, and the active window on one frame."""
    height, width = frame.shape[:2]
    pose = points["pose"][frame_index]
    hands = points["hands"][frame_index]
    for first, second in POSE_EDGES:
        if np.isfinite(pose[[first, second], :2]).all():
            start = tuple((pose[first, :2] * [width, height]).astype(int))
            end = tuple((pose[second, :2] * [width, height]).astype(int))
            cv2.line(frame, start, end, (255, 220, 80), 3)
    colors = [(80, 100, 255), (255, 100, 80)]
    labels = ["L", "R"]
    for side in range(2):
        for point in hands[side]:
            if np.isfinite(point[:2]).all():
                x, y = (point[:2] * [width, height]).astype(int)
                cv2.circle(frame, (int(x), int(y)), 4, colors[side], -1)
        if np.isfinite(hands[side, 0, :2]).all():
            x, y = (hands[side, 0, :2] * [width, height]).astype(int)
            cv2.putText(frame, labels[side], (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[side], 2)
    timestamp = frame_index / fps
    active = next((window for window in windows if window["t0"] <= timestamp <= window["t1"]), None)
    text = f"t={timestamp:.2f}  {active['id'] if active else '-'}  shot={active['shot'] if active else 'target'}"
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = _font()
    draw.text((20, 18), text, fill=(255, 220, 80), font=font)
    if active and active["subtitle"]:
        draw.text((20, height - 48), active["subtitle"], fill=(255, 255, 255), font=font)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def run(clip: Path, out: Path, windows: list[dict[str, Any]], points: dict[str, Any], encoder: str, force: bool = False) -> Path:
    """Render debug.mp4 through an ffmpeg raw-video pipe and mux source audio."""
    destination = out / "debug.mp4"
    if destination.exists() and not force:
        return destination
    capture = cv2.VideoCapture(str(clip))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw = out / "debug_noaudio.mp4"
    command = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", encoder, "-pix_fmt", "yuv420p"]
    if encoder == "libx264":
        command.extend(["-crf", "18"])
    else:
        command.extend(["-allow_sw", "1", "-b:v", "8M"])
    command.append(str(raw))
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            process.stdin.write(_draw(frame, points, frame_index, fps, windows).tobytes())
            frame_index += 1
    finally:
        capture.release()
        process.stdin.close()
        process.wait()
    subprocess.run(["ffmpeg", "-y", "-i", str(raw), "-i", str(clip), "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", str(destination)], check=True)
    raw.unlink()
    return destination
