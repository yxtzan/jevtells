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
from . import render
from .render.geometry import parse_blurs
from .stages.speakers import mark_windows, parse_intervals
from .stages import asr, debug, judge, narrate, people, pose, prepare, review, scene as scene_stage, segment, shots, state, track, voice


_STAGE_ORDER = ("prepare", "pose", "people", "track", "asr", "voice", "shots", "segment", "actions", "scene", "state", "debug", "judge", "narrate", "render")
_TARGET_STAGES = {"track", "shots", "segment", "actions", "state", "debug", "judge", "narrate", "render"}


def _targets_changed(output: Path, anchors: list[tuple[int, float]]) -> bool:
    path = output / "track_meta.json"
    if not path.exists():
        return True
    try:
        saved = json.loads(path.read_text(encoding="utf-8")).get("anchors", [])
        return saved != [[number, seconds] for number, seconds in anchors]
    except (OSError, ValueError, TypeError):
        return True


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="jevtells")
    subparsers = parser.add_subparsers(dest="command")
    doctor_parser=subparsers.add_parser("doctor",help="check Python, ffmpeg, models, fonts and key without writing files")
    doctor_parser.add_argument("--config")
    subparsers.add_parser("download",help="download MediaPipe models and OFL fonts to JEVTELLS_HOME or current directory")

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
    run_parser.add_argument("--lang", choices=("zh", "en"), default="zh")
    run_parser.add_argument("--srt")
    run_parser.add_argument("--start", type=float, default=0.0)
    run_parser.add_argument("--duration", type=float)
    run_parser.add_argument("--force", action="store_true")
    run_parser.add_argument("--until", default="render", choices=_STAGE_ORDER, help="last stage to run (default: render); state also writes debug.mp4")
    run_parser.add_argument("--from", dest="from_stage", choices=_STAGE_ORDER)
    run_parser.add_argument("--target", action="append", default=[], metavar="N@SECONDS", help="target person number at a time anchor; repeat for cuts")
    run_parser.add_argument("--config")
    run_parser.add_argument("--others-speaking", action="append", default=[], metavar="START-END")
    run_parser.add_argument("--blur", action="append", default=[], metavar="X,Y,W,H")
    run_parser.add_argument("--subtitles", choices=("on", "off"), default="on")
    run_parser.add_argument("--layout", choices=("h", "v", "both"), default="both")
    run_parser.add_argument("--title")
    run_parser.add_argument("--debug-layout", action="store_true", help="draw per-shot target bounds, forbidden zones and label positions")
    run_parser.add_argument("--reframe", choices=("auto","off"), default="auto", help="portrait full-height 4:3 crop (default: auto)")
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
    result = _invoke(track.run, detections, output, anchors=anchors, force=force, config=config, clip=output / "clip.mp4")
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


def _finish_run(
    arguments: argparse.Namespace,
    output: Path,
    clip: Path,
    encoding: dict[str, Any],
    transcript: dict[str, Any],
    anchors: list[tuple[int, float]],
    actions_result: dict[str, Any],
    started: float,
    stage_times: dict[str, float],
    windows: list[dict[str, Any]],
) -> None:
    """Save snapshots and accounting at each completed M3 stop point."""
    snapshots = output / "snapshots"
    duration = _video_duration(output / "debug.mp4") or _video_duration(clip)
    for seconds, name in ((2.0, "target_02s.png"), (5.0, "target_05s.png"), (7.0, "target_07s.png"), (12.0, "target_12s.png"), (21.0, "target_21s.png")):
        if seconds < duration:
            _snapshot(output / "debug.mp4", snapshots, seconds, name)
    if duration > 0:
        _snapshot(output / "debug.mp4", snapshots, duration * 0.25, "25.png")
        _snapshot(output / "debug.mp4", snapshots, duration * 0.75, "75.png")
        _snapshot(output / "debug.mp4", snapshots, duration * 0.50, "a4_left_right.png")
    api_stats: dict[str, Any] = {}
    for name in ("scene", "judge", "narrate"):
        path = output / ("judge_meta.json" if name == "judge" else "narrate_meta.json" if name == "narrate" else "scene.json")
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                api_stats[name] = {key: payload.get(key) for key in ("calls", "prompt_tokens", "completion_tokens", "reasoning_tokens", "response_records", "length_retries", "cost", "model", "failed", "retries", "fallbacks", "skipped", "windows") if key in payload}
                if name == "scene" and "model" in payload:
                    api_stats[name]["cost"] = payload.get("cost")
            except (OSError, ValueError, TypeError):
                api_stats[name] = {"error": "metadata_unavailable"}
    costs = [float(item["cost"]) for item in api_stats.values() if isinstance(item, Mapping) and item.get("cost") is not None]
    api_stats["total_cost"] = sum(costs) if costs else None
    metadata: dict[str, Any] = {"parameters": vars(arguments), "elapsed_s": time.perf_counter() - started, "stage_times_s": stage_times, "encoding": encoding, "language": transcript.get("language"), "target_anchors": anchors, "duration_s": duration, "occupied_hands": actions_result.get("occupied_hands", {}) if isinstance(actions_result, Mapping) else {}, "api": api_stats}
    if (output / "track_meta.json").exists():
        metadata["tracking"] = json.loads((output / "track_meta.json").read_text(encoding="utf-8"))
    metadata["models_used"] = {}
    for stage, prefix in (("judge", "jev"), ("narrate", "narrate")):
        identifiers = set()
        for path in (output / "raw").glob(f"{prefix}_W*.json"):
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("model"):
                identifiers.add(str(raw["model"]))
        metadata["models_used"][stage] = sorted(identifiers)
    render_meta = output / "render_meta.json"
    if render_meta.exists():
        metadata["render"] = json.loads(render_meta.read_text(encoding="utf-8"))
    metadata["narration_retries"] = api_stats.get("narrate", {}).get("retries", 0)
    metadata["narration_fallbacks"] = api_stats.get("narrate", {}).get("fallbacks", 0)
    (output / "run_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if arguments.output and arguments.until != "render" and (output / "debug.mp4").exists():
        Path(arguments.output).write_bytes((output / "debug.mp4").read_bytes())
    print(f"output={output} windows={len(windows)} elapsed={metadata['elapsed_s']:.1f}s total_cost={api_stats['total_cost']}")


def main() -> None:
    arguments = _arguments()
    if arguments.command=="doctor":
        from .doctor import run as doctor_run
        raise SystemExit(doctor_run(load_config(arguments.config)))
    if arguments.command=="download":
        from .download import main as download_main
        download_main()
        return
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
    clip_id = _clip_id(input_path, arguments.start, arguments.duration)
    output = Path(arguments.output).parent if arguments.output else Path("work") / clip_id
    output.mkdir(parents=True, exist_ok=True)
    targets_changed = _targets_changed(output, anchors)
    intervals = parse_intervals(getattr(arguments, "others_speaking", []))
    blur_rectangles = parse_blurs(getattr(arguments, "blur", []))
    others_changed = False

    def force_stage(stage: str) -> bool:
        rerun_from = arguments.from_stage
        return bool(arguments.force or (rerun_from and _STAGE_ORDER.index(stage) >= _STAGE_ORDER.index(rerun_from)) or (targets_changed and stage in _TARGET_STAGES) or (others_changed and stage in {"judge", "narrate", "render"}))

    started = time.perf_counter()
    stage_times: dict[str, float] = {}

    def render_clip(clip: Path, windows: list[dict[str, Any]], points: dict[str, Any]) -> None:
        capture = cv2.VideoCapture(str(input_path))
        source_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        capture.release()
        stage_start = time.perf_counter()
        render.run(clip, output, windows, points, config=config, speaker=arguments.speaker, lang=arguments.lang, title=getattr(arguments, "title", None), layout=getattr(arguments, "layout", "both"), blur=blur_rectangles, subtitles=getattr(arguments, "subtitles", "on") == "on", source_size=source_size, force=force_stage("render"), debug_layout=getattr(arguments, "debug_layout", False), reframe=getattr(arguments,"reframe","auto"))
        stage_times["render"] = time.perf_counter() - stage_start

    if arguments.from_stage == "render":
        if arguments.until != "render":
            raise ValueError("--from render requires --until render")
        if targets_changed:
            raise ValueError("target anchors changed; rerun from track before rendering")
        def read_cache(name: str) -> Any:
            return json.loads((output / name).read_text(encoding="utf-8"))
        windows = read_cache("windows.json")
        marked = mark_windows(windows, intervals, float(config.get("other_speaker_overlap", 0.5)))
        if marked != windows:
            raise ValueError("other-speaker intervals changed; rerun from judge before rendering")
        narration_meta = read_cache("narrate_meta.json")
        if narration_meta.get("lang") != arguments.lang:
            raise ValueError("narration language changed; rerun from narrate before rendering")
        clip = output / "clip.mp4"
        render_clip(clip, windows, _load_npz(output / "keypoints.npz"))
        _finish_run(arguments, output, clip, {"stage": "render"}, read_cache("transcript.json"), anchors, read_cache("actions.json"), started, stage_times, windows)
        return

    stage_start = time.perf_counter()
    clip, audio, encoding = _invoke(prepare.run, input_path, output, force_stage("prepare"), arguments.start, arguments.duration, config=config)
    stage_times["prepare"] = time.perf_counter() - stage_start
    if arguments.until == "prepare":
        return

    stage_start = time.perf_counter()
    detections = _run_pose(clip, output, force_stage("pose"), config)
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
    points = _run_track(detections, output, anchors, force_stage("track"), config)
    stage_times["track"] = time.perf_counter() - stage_start
    if arguments.until == "track":
        return
    stage_start = time.perf_counter()
    transcript = _invoke(asr.run, audio, output, arguments.srt, force_stage("asr"), model_name=config.get("models", {}).get("whisper", "small"))
    stage_times["asr"] = time.perf_counter() - stage_start
    if arguments.until == "asr":
        return
    stage_start = time.perf_counter()
    voice_features = _invoke(voice.run, audio, output, force_stage("voice"), config=config)
    stage_times["voice"] = time.perf_counter() - stage_start
    if arguments.until == "voice":
        return
    stage_start = time.perf_counter()
    shot_list = _invoke(shots.run, clip, output, force_stage("shots"), points=points, config=config)
    stage_times["shots"] = time.perf_counter() - stage_start
    if arguments.until == "shots":
        return
    stage_start = time.perf_counter()
    windows = _invoke(segment.run, transcript, shot_list, output, force_stage("segment"), config=config)
    marked = mark_windows(windows, intervals, float(config.get("other_speaker_overlap", 0.5)))
    others_changed = marked != windows
    if others_changed:
        windows = marked
        (output / "windows.json").write_text(json.dumps(windows, ensure_ascii=False, indent=2), encoding="utf-8")
    stage_times["segment"] = time.perf_counter() - stage_start
    if arguments.until == "segment":
        return
    stage_start = time.perf_counter()
    actions_result = _run_actions(points, windows, output, force_stage("actions"), config, clip=clip)
    stage_times["actions"] = time.perf_counter() - stage_start
    if arguments.until == "actions":
        return

    def finish() -> None:
        _finish_run(arguments, output, clip, encoding, transcript, anchors, actions_result, started, stage_times, windows)

    stage_start = time.perf_counter()
    scene_result = _invoke(scene_stage.run, clip, output, force_stage("scene"), config=config, explicit=arguments.scene)
    scene_text = scene_result.get("scene", "unknown") if isinstance(scene_result, Mapping) else str(scene_result)
    stage_times["scene"] = time.perf_counter() - stage_start
    if arguments.until == "scene":
        finish()
        return
    stage_start = time.perf_counter()
    states = _invoke(state.run, windows, voice_features, scene_text, arguments.speaker, output, force_stage("state"), points, actions=actions_result, config=config, transcript=transcript)
    stage_times["state"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    debug_kwargs = {"detections": detections, "actions": actions_result, "config": config}
    _invoke(debug.run, clip, output, windows, points, str(encoding["encoder"]), force_stage("debug"), **debug_kwargs)
    stage_times["debug"] = time.perf_counter() - stage_start
    if arguments.until in {"state", "debug"}:
        finish()
        return
    stage_start = time.perf_counter()
    judgments = _invoke(judge.run, states, output, force_stage("judge"), config=config)
    stage_times["judge"] = time.perf_counter() - stage_start
    if arguments.until == "judge":
        finish()
        return
    stage_start = time.perf_counter()
    narration = _invoke(narrate.run, states, judgments, output, force_stage("narrate"), config=config, lang=arguments.lang)
    stage_times["narrate"] = time.perf_counter() - stage_start
    review.write(states, judgments, narration, output, lang=arguments.lang)
    if arguments.until == "render":
        render_clip(clip, windows, points)
    finish()


if __name__ == "__main__":
    main()
