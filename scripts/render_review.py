"""Extract evidence from encoded outputs; build design comparisons and overviews."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def probe(video: Path) -> dict[str, Any]:
    return json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)], capture_output=True, text=True, check=True).stdout)


def extract(video: Path, seconds: float, destination: Path) -> None:
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", str(seconds), "-i", str(video), "-frames:v", "1", "-update", "1", str(destination)], check=True)
    if not destination.exists():
        raise RuntimeError(f"no frame at {seconds}s in {video}")


def comparison(frame: Path, reference: Path, destination: Path) -> None:
    left = Image.open(frame).convert("RGB")
    right = Image.open(reference).convert("RGB").resize(left.size, Image.Resampling.LANCZOS)
    font = ImageFont.truetype(str(ROOT / "assets/fonts/NotoSansCJKsc-Bold.otf"), 22)
    header = 40
    image = Image.new("RGB", (left.width * 2, left.height + header), "#0B0B0B")
    image.paste(left, (0, header))
    image.paste(right, (left.width, header))
    draw = ImageDraw.Draw(image)
    draw.text((16, 8), "实际成片 · 31.63s", font=font, fill="white", anchor="lt")
    draw.text((left.width + 16, 8), "设计参考（示例数据）", font=font, fill="white", anchor="lt")
    image.save(destination)


def overview(video: Path, destination: Path, kind: str) -> int:
    metadata = probe(video)
    duration = float(metadata["format"]["duration"])
    count = math.ceil(duration)
    tile_w, tile_h = (320, 180) if kind == "h" else (270, 360)
    columns, label_height = (5 if kind == "h" else 6), 24
    font = ImageFont.truetype(str(ROOT / "assets/fonts/NotoSansMono-Regular.ttf"), 16)
    image = Image.new("RGB", (columns * tile_w, math.ceil(count / columns) * (tile_h + label_height)), "#0B0B0B")
    draw = ImageDraw.Draw(image)
    capture = cv2.VideoCapture(str(video))
    try:
        for seconds in range(count):
            capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"overview decoding failed at {seconds}s")
            tile = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            x = (seconds % columns) * tile_w
            y = (seconds // columns) * (tile_h + label_height)
            image.paste(tile, (x, y + label_height))
            draw.text((x + 8, y + 2), f"{seconds:02d}.0s", font=font, fill="#C8FF2E", anchor="lt")
    finally:
        capture.release()
    image.save(destination)
    return count


def transitions(clip_dir: Path, destination: Path) -> list[dict[str, Any]]:
    """Save exact encoded frames around each window's commentary boundary."""
    windows = json.loads((clip_dir / 'windows.json').read_text())
    records = []
    for kind in ('h', 'v'):
        video = clip_dir / f'output_{kind}.mp4'
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS)
        try:
            for index, window in enumerate(windows[1:], 1):
                boundary = float(window['t0'])
                first = math.ceil(boundary * fps - 1e-8)
                folder = destination / f'transition_{kind}_{index:02d}'
                folder.mkdir(exist_ok=True)
                entries = []
                tiles = []
                for offset in range(-5, math.ceil(.4 * fps) + 2):
                    frame_number = first + offset
                    if frame_number < 0:
                        continue
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(f'transition decoding failed at frame {frame_number}')
                    seconds = frame_number / fps
                    filename = folder / f'frame_{frame_number:05d}.png'
                    cv2.imwrite(str(filename), frame)
                    elapsed = seconds - boundary
                    phase = 'previous' if elapsed < 0 else 'fade-out' if elapsed < .15 else 'fade-in' if elapsed < .4 else 'settled'
                    entries.append({'frame': frame_number, 'seconds': seconds, 'elapsed': elapsed, 'phase': phase, 'file': str(filename)})
                    tile = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((240, 320) if kind == 'v' else (320, 180))
                    tiles.append(tile)
                width, height = tiles[0].size
                sheet = Image.new('RGB', (5 * width, math.ceil(len(tiles) / 5) * (height + 24)), '#0B0B0B')
                draw = ImageDraw.Draw(sheet)
                for position, (tile, entry) in enumerate(zip(tiles, entries)):
                    x, y = (position % 5) * width, (position // 5) * (height + 24)
                    sheet.paste(tile, (x, y + 24))
                    draw.text((x + 4, y + 4), f"{entry['seconds']:.3f}s {entry['phase']}", fill='white')
                sheet.save(folder / 'contact_sheet.png')
                records.append({'kind': kind, 'boundary': boundary, 'frames': entries, 'contact_sheet': str(folder / 'contact_sheet.png')})
        finally:
            capture.release()
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip_dir", type=Path)
    parser.add_argument("--at", type=float, action="append", required=True)
    parser.add_argument("--reference-v", type=Path)
    parser.add_argument("--reference-h", type=Path)
    parser.add_argument("--transitions", action="store_true", help="save 5 frames before and all fade frames after every commentary boundary")
    arguments = parser.parse_args()
    destination = arguments.clip_dir / "render_review"
    destination.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for kind in ("h", "v"):
        video = arguments.clip_dir / f"output_{kind}.mp4"
        metadata = probe(video)
        results[kind] = {"ffprobe": metadata, "snapshots": []}
        for seconds in arguments.at:
            frame = destination / f"{kind}_{seconds:g}s.png"
            extract(video, seconds, frame)
            results[kind]["snapshots"].append(str(frame))
        sheet = destination / f"overview_{kind}.png"
        results[kind]["overview_frames"] = overview(video, sheet, kind)
        results[kind]["overview"] = str(sheet)
        reference = getattr(arguments, f"reference_{kind}")
        if reference:
            frame = destination / f"{kind}_31.63s.png"
            if not frame.exists():
                extract(video, 31.63, frame)
            compare = destination / f"compare_{kind}.png"
            comparison(frame, reference, compare)
            results[kind]["compare"] = str(compare)
    if arguments.transitions:
        results['transitions'] = transitions(arguments.clip_dir, destination)
    results["files"] = sorted(str(path) for path in destination.rglob("*.png"))
    (destination / "manifest.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({kind: {"snapshots": len(results[kind]["snapshots"]), "overview_frames": results[kind]["overview_frames"]} for kind in ("h", "v")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
