"""Create a palette-optimized GIF from a published video, using only ffmpeg.

Example: python scripts/make_demo_gif.py input.mp4 output.gif --start 10.5 \
    --duration 8 --width 800 --fps 12
Use a portrait input for a portrait GIF; the source aspect ratio is preserved.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile


def positive(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be greater than zero')
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--duration', type=positive, default=8)
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--fps', type=positive, default=12)
    parser.add_argument('--colors', type=int, choices=range(4, 257), default=256, metavar='4..256')
    args = parser.parse_args()
    if args.start < 0 or args.width <= 0:
        parser.error('start must be nonnegative and width must be positive')
    if not args.input.is_file():
        parser.error(f'input does not exist: {args.input}')
    if args.input.resolve() == args.output.resolve():
        parser.error('input and output must be different files')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Light temporal denoising removes compression noise that otherwise makes
    # the GIF large and flickers in smooth backdrops; retain the full palette.
    filters = f'fps={args.fps:g},scale={args.width}:-1:flags=lanczos,hqdn3d=2:2:6:6'
    source = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
              '-ss', str(args.start), '-t', str(args.duration), '-i', str(args.input)]
    with tempfile.TemporaryDirectory(prefix='jevtells-gif-') as temporary:
        palette = Path(temporary) / 'palette.png'
        subprocess.run([*source, '-vf', filters + f',palettegen=stats_mode=diff:max_colors={args.colors}',
                        '-frames:v', '1', str(palette)], check=True)
        # A stable global palette and ordered dithering keep static card areas
        # steady between frames, while diff_mode limits work to changed regions.
        subprocess.run([*source, '-i', str(palette), '-lavfi',
                        filters + ' [video];[video][1:v]paletteuse=dither=bayer:bayer_scale=2:diff_mode=rectangle',
                        '-loop', '0', str(args.output)], check=True)
    print(f'{args.output}: {args.output.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
