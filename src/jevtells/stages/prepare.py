"""Normalize input video and extract 16 kHz mono audio."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

from ..config import config_value


def _ffmpeg() -> str:
    """Return the ffmpeg executable or raise a clear setup error."""
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("ffmpeg is not on PATH; install it with brew install ffmpeg")
    return executable


def _encoder(executable: str) -> str:
    """Choose libx264 first, then the macOS VideoToolbox encoder."""
    output = subprocess.run([executable, "-hide_banner", "-encoders"], capture_output=True, text=True, check=True).stdout
    if "libx264" in output:
        return "libx264"
    if "h264_videotoolbox" in output:
        return "h264_videotoolbox"
    raise RuntimeError("ffmpeg has neither libx264 nor h264_videotoolbox")


def run(inp: Path, out: Path, force: bool = False, start: float = 0.0, duration: float | None = None, config: Mapping[str, Any] | None = None) -> tuple[Path, Path, dict[str, object]]:
    """Create normalized clip and audio files with real video conversion."""
    out.mkdir(parents=True, exist_ok=True)
    clip = out / "clip.mp4"
    audio = out / "audio.wav"
    executable = _ffmpeg()
    encoder = _encoder(executable)
    video_command = [executable, "-y", "-ss", str(start), "-i", str(inp)]
    if duration is not None:
        video_command.extend(["-t", str(duration)])
    max_width = int(config_value(config or {}, "video.max_width", config_value(config or {}, "max_width", 1920)))
    max_fps = float(config_value(config or {}, "video.max_fps", config_value(config or {}, "max_fps", 30)))
    video_command.extend(["-vf", f"scale='min({max_width},iw)':-2,fps='min({max_fps:g},source_fps)'", "-c:v", encoder, "-pix_fmt", "yuv420p"])
    if encoder == "libx264":
        video_command.extend(["-crf", "18"])
    else:
        video_command.extend(["-allow_sw", "1", "-b:v", "8M"])
    video_command.append(str(clip))
    if force or not clip.exists():
        subprocess.run(video_command, check=True)
    if force or not audio.exists():
        subprocess.run([executable, "-y", "-i", str(clip), "-ar", "16000", "-ac", "1", str(audio)], check=True)
    return clip, audio, {"ffmpeg": executable, "encoder": encoder, "converted": True}
