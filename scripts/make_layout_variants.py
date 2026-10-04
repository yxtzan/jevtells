"""Create the M5 left/right/small/cut fixtures without changing the source."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def generate(source: Path, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    filters = {
        "left": "crop=980:720:300:0,pad=1280:720:0:0:color=0x202020",
        "right": "crop=980:720:0:0,pad=1280:720:300:0:color=0x202020",
        "small": "scale=640:360,pad=1280:720:320:180:color=0x202020",
    }
    results = []
    for name, transform in filters.items():
        path = output / f"jensen_{name}.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "12", "-i", str(source), "-t", "22", "-vf", transform, "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)], check=True)
        results.append(path)
    cut = output / "jensen_cut.mp4"
    graph = "[0:v]trim=start=0:end=5,setpts=PTS-STARTPTS[v0];[0:a]atrim=start=0:end=5,asetpts=PTS-STARTPTS[a0];[1:v]trim=start=5:end=10,setpts=PTS-STARTPTS[v1];[1:a]atrim=start=5:end=10,asetpts=PTS-STARTPTS[a1];[2:v]trim=start=22:end=27,setpts=PTS-STARTPTS[v2];[2:a]atrim=start=22:end=27,asetpts=PTS-STARTPTS[a2];[v0][a0][v1][a1][v2][a2]concat=n=3:v=1:a=1[v][a]"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(results[0]), "-i", str(results[1]), "-i", str(source), "-filter_complex", graph, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", str(cut)], check=True)
    return [*results, cut]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "samples/jensen_panel.mov")
    parser.add_argument("--output", type=Path, default=ROOT / "samples/synthetic")
    args = parser.parse_args()
    for path in generate(args.input, args.output):
        print(path)


if __name__ == "__main__":
    main()
