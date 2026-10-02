# JevTells · 肢体潜台词

> Upload a talking-head video, get it back annotated with gestures, intent and Jev judgments.
>
> 上传一段说话人视频，得到一段标注了手势、意图和 Jev 判定的视频。

**Status: v0.1.0 — data layer only.** The tool currently extracts subtitles, body and hand keypoints, voice features and a per-window state, and renders a debug video. Gesture labels (v0.2), Jev judgments (v0.3) and the final annotated video (v0.4) are in progress.

## How it works

| Step | Tool | Runs |
|---|---|---|
| Body & hand keypoints, every frame | MediaPipe | locally |
| Subtitles with word timestamps | faster-whisper | locally |
| Loudness, pitch, speech rate | librosa | locally |
| Keypoints → named gestures (raise, press down, open palm …) | rule engine | locally |
| Confidence / focus / tension, intent, emotion | Jev via OpenRouter | API |
| One-line narration and key quote | any LLM via OpenRouter | API |
| Annotated output video | Pillow + ffmpeg | locally |

The only key you need is an OpenRouter API key (from v0.3).

## Requirements

- macOS (tested on Apple Silicon) or Linux
- Python 3.11
- ffmpeg **with libx264**
  - macOS: install [Homebrew](https://brew.sh), then `brew install ffmpeg`
  - Ubuntu / Debian: `sudo apt install ffmpeg`
  - Check: `ffmpeg -hide_banner -encoders | grep libx264` should print a line
- About 2 GB of disk space for Python packages and models

## Install

```bash
git clone https://github.com/yxtzan/jevtells.git
cd jevtells
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/download_models.py
```

The speech-to-text model downloads automatically on the first run.

## Usage (v0.1.0)

```bash
jevtells run path/to/video.mp4 --speaker "Speaker name" --until state
```

Results go to `work/<clip_id>/`: `transcript.json`, `keypoints.npz`, `windows.json`, `states.json`, `debug.mp4` and more.

| Option | Meaning |
|---|---|
| `--start`, `--duration` | Analyse only part of the video (seconds) |
| `--srt FILE` | Use your own subtitles instead of automatic transcription |
| `--force` | Recompute every stage (otherwise cached results are reused) |
| `--from STAGE`, `--until STAGE` | Run only part of the pipeline |

A 30-second clip takes about one minute on an Apple M4 MacBook Air.

## Troubleshooting

- **`Could not create an NSOpenGLPixelFormat`** (MediaPipe) or **VideoToolbox error `-12903`** (encoding): you are running inside a sandbox without GPU access, typically an AI coding agent. Run the command from a normal terminal.
- **`libx264` not found**: your ffmpeg is a minimal build. Install it with Homebrew or apt as above.

## Disclaimer

Scores are uncalibrated model judgments for demonstration only. They are not a psychological assessment, and this tool does not detect lies or "read minds".

分数仅为未经校准的模型判断，仅供演示，不构成任何心理评估。

Jev is a model by TypeSafe AI. JevTells is an independent community project and is not affiliated with TypeSafe AI.

## License

MIT
