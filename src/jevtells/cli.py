"""Command-line entry point for the JevTells pipeline."""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import cv2

from .config import load_config
from .stages import asr, debug, people, pose, prepare, segment, shots, state, track, voice


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="jevtells")
    subparsers = parser.add_subparsers(dest="command")

    people_parser = subparsers.add_parser("people", help="number detected people in a representative frame")
    people_parser.add_argument("input")
    people_parser.add_argument("--at", type=float, default=0.0)
    people_parser.add_argument("-o", "--output")
    people_parser.add_argument("--config")
    people_parser.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser("run", help="run the JevTells data pipeline")
    run_parser.add_argument("input")
    run_parser.add_argument("-o", "--output")
    run_parser.add_argument("--speaker", required=True)
    run_parser.add_argument("--scene", default=None)
    run_parser.add_argument("--srt")
    run_parser.add_argument("--start", type=float, default=0.0)
    run_parser.add_argument("--duration", type=float)
    run_parser.add_argument("--force", action="store_true")
    run_parser.add_argument("--until", default="state", choices=("prepare", "pose", "people", "track", "asr", "voice", "shots", "segment", "actions", "state", "debug"))
    run_parser.add_argument("--from", dest="from_stage")
    run_parser.add_argument("--target", action="append", default=[], metavar="N@SECONDS", help="target person number at a time anchor; repeat for cuts")
    run_parser.add_argument("--config")
    return parser.parse_args()


def _parse_targets(values: list[str] | None) -> list[tuple[int, float]]:
    """Parse repeated ``N@seconds`` target anchors."""

    anchors: list[tuple[int, float]] = []
    for raw in values or []:
        text = str(raw).strip()
        if "@" in text:
            person, seconds = text.split("@", 1)
        else:
            person, seconds = text, "0"
        try:
            number = int(person)
            when = float(seconds)
        except ValueError as error:
            raise ValueError(f"--target must be N or N@SECONDS, got {raw!r}") from error
        if number < 1 or when < 0:
            raise ValueError(f"--target requires N>=1 and seconds>=0, got {raw!r}")
        anchors.append((number, when))
    return anchors or [(1, 0.0)]


def _invoke(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Call a stage while tolerating the older M1 function signatures."""

    signature = inspect.signature(function)
    accepts_var_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    if accepts_var_kwargs:
        return function(*args, **kwargs)
    accepted = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return function(*args, **accepted)


def _clip_id(input_path: Path, start: float, duration: float | None) -> str:
    duration_text = f"{duration:g}" if duration is not None else "auto"
    return f"{input_path.stem}_s{start:g}_d{duration_text}"


def _load_npz(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _run_pose(clip: Path, output: Path, force: bool, config: Mapping[str, Any]) -> dict[str, Any]:
    result = _invoke(pose.run, clip, output, force=force, config=config)
    if isinstance(result, Mapping):
        return dict(result)
    destination = Path(result) if result is not None else output / "detections.npz"
    return _load_npz(destination)


def _run_track(detections: dict[str, Any], output: Path, anchors: list[tuple[int, float]], force: bool, config: Mapping[str, Any]) -> dict[str, Any]:
    result = _invoke(track.run, detections, output, anchors=anchors, force=force, config=config)
    if isinstance(result, Mapping):
        return dict(result)
    return _load_npz(Path(result) if result is not None else output / "keypoints.npz")


def _run_actions(points: dict[str, Any], windows: list[dict[str, Any]], output: Path, force: bool, config: Mapping[str, Any], clip: Path | None = None) -> dict[str, Any]:
    """Run the optional M2 action stage with either supported call shape."""

    try:
        from .stages import actions as action_stage
    except ImportError:
        return {"events": []}
    function = getattr(action_stage, "run", None)
    if function is None:
        return {"events": []}
    kwargs = {"points": points, "keypoints": points, "windows": windows, "out": output, "force": force, "config": config, "clip": clip}
    signature = inspect.signature(function)
    named = {name: value for name, value in kwargs.items() if name in signature.parameters}
    positional: list[Any] = []
    for parameter in signature.parameters.values():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if parameter.name in named and parameter.default is inspect.Parameter.empty:
            positional.append(named.pop(parameter.name))
        elif parameter.name not in named and parameter.default is inspect.Parameter.empty:
            break
    try:
        result = function(*positional, **named)
    except TypeError:
        result = function(points, windows, output, force=force, config=config, clip=clip)
    if isinstance(result, Mapping):
        return dict(result)
    if result is None and (output / "actions.json").exists():
        return json.loads((output / "actions.json").read_text(encoding="utf-8"))
    return {"events": []}


def _snapshot(video: Path, destination: Path, seconds: float, name: str) -> bool:
    """Extract one frame when the requested timestamp exists."""

    if not video.exists():
        return False
    executable = shutil.which("ffmpeg") or "ffmpeg"
    destination.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run([executable, "-y", "-ss", str(seconds), "-i", str(video), "-frames:v", "1", str(destination / name)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return completed.returncode == 0 and (destination / name).exists()


def _video_duration(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()
    return float(frames / fps) if frames else 0.0


def _people_command(arguments: argparse.Namespace) -> None:
    input_path = Path(arguments.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    config = load_config(arguments.config)
    output = Path(arguments.output) if arguments.output else Path("work") / _clip_id(input_path, 0.0, None)
    output.mkdir(parents=True, exist_ok=True)
    clip, _audio, _encoding = _invoke(prepare.run, input_path, output, force=arguments.force, start=0.0, duration=None)
    detections_path = output / "detections.npz"
    if detections_path.exists() and not arguments.force:
        detections = _load_npz(detections_path)
    else:
        detections = _run_pose(clip, output, arguments.force, config)
    records = _invoke(people.run, clip, output, at=arguments.at, detections=detections, force=arguments.force, config=config)
    for record in records:
        print(f"{record['number']}: detector_index={record['detector_index']} box={record['box']} t={record['time']:.2f}s")
    print(f"people={output / f'people_{arguments.at:04.1f}s.png'}")


def main() -> None:
    arguments = _arguments()
    if arguments.command == "people":
        _people_command(arguments)
        return
    if arguments.command != "run":
        raise SystemExit("usage: jevtells run INPUT --speaker NAME [--until state]")

    input_path = Path(arguments.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    config = load_config(arguments.config)
    anchors = _parse_targets(arguments.target)
    target_force = arguments.force or bool(arguments.target)
    scene = arguments.scene if arguments.scene is not None else str(config.get("scene", "unknown"))
    clip_id = _clip_id(input_path, arguments.start, arguments.duration)
    output = Path(arguments.output).parent if arguments.output else Path("work") / clip_id
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    stage_times: dict[str, float] = {}

    stage_start = time.perf_counter()
    clip, audio, encoding = _invoke(prepare.run, input_path, output, arguments.force, arguments.start, arguments.duration, config=config)
    stage_times["prepare"] = time.perf_counter() - stage_start
    if arguments.until == "prepare":
        return

    stage_start = time.perf_counter()
    detections = _run_pose(clip, output, arguments.force, config)
    stage_times["pose"] = time.perf_counter() - stage_start
    detections_path = output / "detections.npz"
    if not detections_path.exists() and (output / "keypoints.npz").exists():
        detections = _load_npz(output / "keypoints.npz")
    if arguments.until == "pose":
        return
    if not arguments.target:
        from .stages.track import sorted_person_indices
        poses_all = detections.get("poses_all", detections.get("poses", []))
        if getattr(poses_all, "ndim", 0) == 4 and len(poses_all):
            first_people = sorted_person_indices(
                poses_all,
                0,
                float(detections.get("width", 1.0)),
                float(detections.get("height", 1.0)),
            )
            if len(first_people) != 1:
                raise SystemExit("multiple people detected; run `jevtells people INPUT` and pass --target N[@SECONDS]")
    if arguments.until == "people":
        records = _invoke(people.run, clip, output, at=0.0, detections=detections, force=arguments.force, config=config)
        for record in records:
            print(f"{record['number']}: detector_index={record['detector_index']} box={record['box']} t={record['time']:.2f}s")
        print(f"people={output / 'people_00.0s.png'}")
        return

    stage_start = time.perf_counter()
    points = _run_track(detections, output, anchors, target_force, config)
    stage_times["track"] = time.perf_counter() - stage_start
    if arguments.until == "track":
        return
    stage_start = time.perf_counter()
    transcript = _invoke(asr.run, audio, output, arguments.srt, arguments.force, model_name=config.get("models", {}).get("whisper", "small"))
    stage_times["asr"] = time.perf_counter() - stage_start
    if arguments.until == "asr":
        return
    stage_start = time.perf_counter()
    voice_features = _invoke(voice.run, audio, output, arguments.force, config=config)
    stage_times["voice"] = time.perf_counter() - stage_start
    if arguments.until == "voice":
        return
    stage_start = time.perf_counter()
    shot_list = _invoke(shots.run, clip, output, target_force, points=points, config=config)
    stage_times["shots"] = time.perf_counter() - stage_start
    if arguments.until == "shots":
        return
    stage_start = time.perf_counter()
    windows = _invoke(segment.run, transcript, shot_list, output, target_force, config=config)
    stage_times["segment"] = time.perf_counter() - stage_start
    if arguments.until == "segment":
        return
    stage_start = time.perf_counter()
    actions_result = _run_actions(points, windows, output, target_force, config, clip=clip)
    stage_times["actions"] = time.perf_counter() - stage_start
    if arguments.until == "actions":
        return
    stage_start = time.perf_counter()
    _invoke(state.run, windows, voice_features, scene, arguments.speaker, output, target_force, points, actions=actions_result, config=config)
    stage_times["state"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    debug_kwargs = {"detections": detections, "actions": actions_result, "config": config}
    _invoke(debug.run, clip, output, windows, points, str(encoding["encoder"]), target_force, **debug_kwargs)
    stage_times["debug"] = time.perf_counter() - stage_start
    snapshots = output / "snapshots"
    duration = _video_duration(output / "debug.mp4") or _video_duration(clip)
    for seconds, name in ((2.0, "target_02s.png"), (5.0, "target_05s.png"), (7.0, "target_07s.png"), (12.0, "target_12s.png"), (21.0, "target_21s.png")):
        if seconds < duration:
            _snapshot(output / "debug.mp4", snapshots, seconds, name)
    if duration > 0:
        _snapshot(output / "debug.mp4", snapshots, duration * 0.25, "25.png")
        _snapshot(output / "debug.mp4", snapshots, duration * 0.75, "75.png")
        _snapshot(output / "debug.mp4", snapshots, duration * 0.50, "a4_left_right.png")
    metadata: dict[str, Any] = {"parameters": vars(arguments), "elapsed_s": time.perf_counter() - started, "stage_times_s": stage_times, "encoding": encoding, "language": transcript.get("language"), "target_anchors": anchors, "duration_s": duration, "occupied_hands": actions_result.get("occupied_hands", {}) if isinstance(actions_result, Mapping) else {}}
    (output / "run_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if arguments.output:
        Path(arguments.output).write_bytes((output / "debug.mp4").read_bytes())
    print(f"output={output} windows={len(windows)} elapsed={metadata['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
