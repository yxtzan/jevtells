"""Render the landmark debug video and preserve the source audio."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ..resources import asset_path


POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]


def _font() -> ImageFont.FreeTypeFont:
    """Load the downloaded Chinese font for overlay text."""
    path = asset_path("assets/fonts/NotoSansCJKsc-Regular.otf")
    if path.exists():
        return ImageFont.truetype(str(path), 26)
    # Debug rendering remains useful on machines that did not download the
    # optional CJK font. OpenCV overlays below still render all English labels.
    return ImageFont.load_default()


def _draw(
    frame: np.ndarray,
    points: dict[str, Any],
    frame_index: int,
    fps: float,
    windows: list[dict[str, Any]],
    detections: dict[str, Any] | None = None,
    actions: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Draw pose, both hands, labels, and the active window on one frame."""
    height, width = frame.shape[:2]
    pose = points["pose"][frame_index]
    hands = points.get("hands", np.empty((len(points["pose"]), 0, 21, 3)))[frame_index]
    threshold = float((config or {}).get("detection", {}).get("pose_visibility_threshold", 0.5))
    # Draw all people in a thin neutral box first, then make the tracked target
    # prominent. This remains valid when the target is lost (all target points
    # are NaN) and prevents an accidental jump to the host from being hidden.
    target_index = -1
    if "target_index" in points:
        values = np.asarray(points["target_index"]).reshape(-1)
        if frame_index < len(values):
            target_index = int(values[frame_index])
    if detections:
        poses_all = np.asarray(detections.get("poses_all", detections.get("poses", [])), dtype=float)
        if poses_all.ndim == 4 and frame_index < len(poses_all):
            frame_poses = poses_all[frame_index]
            for person_index, candidate in enumerate(frame_poses):
                visible = np.isfinite(candidate[:, :2]).all(axis=1)
                if candidate.shape[1] > 3:
                    visible &= ~np.isfinite(candidate[:, 3]) | (candidate[:, 3] >= threshold)
                if not visible.any():
                    continue
                xy = candidate[visible, :2] * [width, height]
                x0, y0 = xy.min(axis=0).astype(int)
                x1, y1 = xy.max(axis=0).astype(int)
                is_target = person_index == target_index and target_index >= 0
                colour = (0, 230, 255) if is_target else (180, 180, 180)
                thickness = 3 if is_target else 1
                cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), colour, thickness)
                if is_target:
                    cv2.putText(frame, "TARGET", (int(x0), max(25, int(y0) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)
    for first, second in POSE_EDGES:
        visible = np.isfinite(pose[[first, second], :2]).all(axis=1)
        if pose.shape[1] > 3:
            visible &= ~np.isfinite(pose[[first, second], 3]) | (pose[[first, second], 3] >= threshold)
        if visible.all():
            start = tuple((pose[first, :2] * [width, height]).astype(int))
            end = tuple((pose[second, :2] * [width, height]).astype(int))
            cv2.line(frame, start, end, (255, 220, 80), 3)
    colors = [(80, 100, 255), (255, 100, 80)]
    labels = ["L", "R"]
    for side in range(min(2, len(hands))):
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
    result = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    if not np.isfinite(pose[:, :2]).any():
        cv2.putText(result, "target=lost", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 80, 255), 2, cv2.LINE_AA)
    if actions:
        events = actions.get("events", actions) if isinstance(actions, dict) else actions
        if isinstance(events, list):
            active_events = [event for event in events if float(event.get("start", event.get("t0", 0.0))) <= timestamp <= float(event.get("end", event.get("t1", 0.0)))]
            for offset, event in enumerate(active_events[:3]):
                cv2.putText(result, str(event.get("type", "gesture")), (width - 260, 35 + 30 * offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 140, 255), 2, cv2.LINE_AA)
    return result


def run(
    clip: Path,
    out: Path,
    windows: list[dict[str, Any]],
    points: dict[str, Any],
    encoder: str,
    force: bool = False,
    detections: dict[str, Any] | None = None,
    actions: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> Path:
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
            process.stdin.write(_draw(frame, points, frame_index, fps, windows, detections=detections, actions=actions, config=config).tobytes())
            frame_index += 1
    finally:
        capture.release()
        process.stdin.close()
        process.wait()
    subprocess.run(["ffmpeg", "-y", "-i", str(raw), "-i", str(clip), "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", str(destination)], check=True)
    raw.unlink()
    return destination
