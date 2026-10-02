"""Command-line entry point for the JevTells pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from .stages import asr, debug, pose, prepare, segment, shots, state, voice


def _arguments() -> argparse.Namespace:
    """Parse the public run command arguments."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("input")
    run_parser.add_argument("-o", "--output")
    run_parser.add_argument("--speaker", required=True)
    run_parser.add_argument("--scene", default="unknown")
    run_parser.add_argument("--srt")
    run_parser.add_argument("--start", type=float, default=0.0)
    run_parser.add_argument("--duration", type=float)
    run_parser.add_argument("--force", action="store_true")
    run_parser.add_argument("--until", default="state")
    run_parser.add_argument("--from", dest="from_stage")
    return parser.parse_args()


def _snapshot(video: Path, destination: Path, seconds: float, name: str) -> None:
    """Extract one PNG snapshot with ffmpeg."""
    subprocess.run(["ffmpeg", "-y", "-ss", str(seconds), "-i", str(video), "-frames:v", "1", str(destination / name)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    """Run the requested JevTells pipeline."""
    arguments = _arguments()
    if arguments.command != "run":
        raise SystemExit("usage: jevtells run INPUT --speaker NAME [--until state]")
    input_path = Path(arguments.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    clip_id = f"{input_path.stem}_s{arguments.start:g}_d{arguments.duration:g}" if arguments.duration else f"{input_path.stem}_s{arguments.start:g}_dauto"
    output = Path(arguments.output).parent if arguments.output else Path("work") / clip_id
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    stage_times: dict[str, float] = {}
    stage_start = time.perf_counter()
    clip, audio, encoding = prepare.run(input_path, output, arguments.force, arguments.start, arguments.duration)
    stage_times["prepare"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    keypoints = pose.run(clip, output, arguments.force)
    stage_times["pose"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    transcript = asr.run(audio, output, arguments.srt, arguments.force)
    stage_times["asr"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    voice_features = voice.run(audio, output, arguments.force)
    stage_times["voice"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    shot_list = shots.run(clip, output, arguments.force)
    stage_times["shots"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    windows = segment.run(transcript, shot_list, output, arguments.force)
    stage_times["segment"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    state.run(windows, voice_features, arguments.scene, arguments.speaker, output, arguments.force, keypoints)
    stage_times["state"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    debug.run(clip, output, windows, keypoints, str(encoding["encoder"]), arguments.force)
    stage_times["debug"] = time.perf_counter() - stage_start
    snapshots = output / "snapshots"
    snapshots.mkdir(exist_ok=True)
    duration = float(shot_list[0]["t1"]) if shot_list else 1.0
    _snapshot(output / "debug.mp4", snapshots, duration * 0.25, "25.png")
    _snapshot(output / "debug.mp4", snapshots, duration * 0.75, "75.png")
    _snapshot(output / "debug.mp4", snapshots, duration * 0.50, "a4_left_right.png")
    metadata: dict[str, Any] = {"parameters": vars(arguments), "elapsed_s": time.perf_counter() - started, "stage_times_s": stage_times, "encoding": encoding, "language": transcript.get("language"), "detection_rates": {"pose": float(keypoints["pose_present"].mean()), "left_hand": float(keypoints["hand_present"][:, 0].mean()), "right_hand": float(keypoints["hand_present"][:, 1].mean())}}
    (output / "run_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    if arguments.output:
        Path(arguments.output).write_bytes((output / "debug.mp4").read_bytes())
    print(f"output={output} windows={len(windows)} elapsed={metadata['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
