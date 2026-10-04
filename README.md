# JevTells · 肢体潜台词

> Upload a talking-head video, get it back annotated with gestures, intent and Jev judgments.
>
> 上传一段说话人视频，得到一段标注了手势、意图和 Jev 判定的视频。

**Status: v0.4.0 — first complete version.** Feed in a video, pick the person to analyse, and get back two annotated videos: landscape 16:9 and portrait 3:4. They show gesture labels that follow the hands, Jev's scores and trends, intent and emotion, and a one-line narration grounded in the measured data.

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
cp .env.example .env    # then open .env and paste your OpenRouter key after OPENROUTER_API_KEY=
```

The speech-to-text model downloads automatically on the first run. Get an OpenRouter key at [openrouter.ai](https://openrouter.ai); it is only read from `.env`, which is never committed.

## Usage (v0.4.0)

**1. See who is in the frame.** Every detected person gets a number, left to right:

```bash
jevtells people path/to/video.mp4            # writes work/<clip_id>/people_00.0s.png
jevtells people path/to/video.mp4 --at 12    # numbering at another moment
```

**2. Run with the person you want to analyse.** `--until render` produces the finished videos:

```bash
jevtells run path/to/video.mp4 --target 2 --speaker "Speaker name" \
  --title "AI 读 Speaker" --until render
```

This writes `output_h.mp4` (1920×1080) and `output_v.mp4` (1080×1440) to `work/<clip_id>/`.

If the video cuts or zooms and the numbering changes, add more anchors in the form `NUMBER@SECONDS`. For example, follow person 2 from the start and person 1 from 12 s on:

```bash
jevtells run path/to/video.mp4 --target 2@0 --target 1@12 --speaker "Speaker name" --until render
```

When the main person can't be found, the tool leaves the frame unlabelled (`target=lost`) instead of jumping to someone else.

Results go to `work/<clip_id>/`:

| File | Content |
|---|---|
| `output_h.mp4`, `output_v.mp4` | The annotated videos |
| `debug.mp4` | Skeleton, hands, target box and current gesture drawn on the video |
| `actions.json` | Detected gestures with time, hand and magnitude |
| `actions_review/` | One start / middle / end strip per gesture, for checking by eye |
| `states.json` | The five-field state per time window (scene, speaker, subtitle, voice, gestures) |
| `judgments.json` | Jev's confidence / focus / tension scores (0–1), intent and emotion per window |
| `narration.json` | One-line narration and key quote per window |
| `judge_review.md` | Everything above side by side, one row per window, for checking by eye |
| `scene.json`, `raw/` | Automatic scene description and raw API responses |
| `transcript.json`, `windows.json`, `keypoints.npz` | Intermediate data |

| Option | Meaning |
|---|---|
| `--target N` or `N@SECONDS` | Who to analyse; repeat for cuts |
| `--lang zh` / `--lang en` | Narration language (default `zh`); switching only re-runs the narration |
| `--scene TEXT` | Describe the scene yourself instead of the automatic description |
| `--layout h` / `v` / `both` | Which video to render (default `both`) |
| `--title TEXT` | Headline of the portrait video (default `AI 读<speaker>`) |
| `--others-speaking START-END` | Seconds when someone else is talking; repeatable. Those sentences get no scores or narration |
| `--blur X,Y,W,H` | Blur a rectangle (source pixels) in the finished video, e.g. a watermark; repeatable |
| `--subtitles on` / `off` | Draw subtitles (default `on`); use `off` if the video already has burned-in subtitles |
| `--start`, `--duration` | Analyse only part of the video (seconds) |
| `--srt FILE` | Use your own subtitles instead of automatic transcription |
| `--force` | Recompute every stage (otherwise cached results are reused) |
| `--from STAGE`, `--until STAGE` | Run only part of the pipeline |
| `--config FILE` | Use your own thresholds instead of `config/default.yaml` |

Changing `--target` only re-runs tracking and the stages after it; body detection is cached. To change only the look (title, blur, layout, subtitles), use `--from render --until render`: no API calls are made. Visual settings (colours, sizes, timings, which gesture types are shown) live in the `render` section of `config/default.yaml`. A 30-second clip takes about one minute on an Apple M4 MacBook Air.

**Cost.** Jev, the narration model and the scene model all run through OpenRouter. A 30-second clip cost about **$0.002** in October 2026. Cached stages are free; re-running `judge`, `narrate` or `scene` calls the API again. `run_meta.json` lists the calls and cost per stage. Question wording and the narration prompt are editable in `config/jev_questions.yaml` and `config/narrate_prompt.md`.

**Limitations.** The tool assumes the person you chose is the one speaking; mark other speakers with `--others-speaking`. Scores come from a general-purpose model and are uncalibrated. Gesture rules were tuned on a small set of clips, so expect some wrong labels on very different footage.

## Troubleshooting

- **`Could not create an NSOpenGLPixelFormat`** (MediaPipe) or **VideoToolbox error `-12903`** (encoding): you are running inside a sandbox without GPU access, typically an AI coding agent. Run the command from a normal terminal.
- **`libx264` not found**: your ffmpeg is a minimal build. Install it with Homebrew or apt as above.
- **`缺少 OPENROUTER_API_KEY`** (missing key): copy `.env.example` to `.env` and paste your key after `OPENROUTER_API_KEY=`.

## Disclaimer

Scores are uncalibrated model judgments for demonstration only. They are not a psychological assessment, and this tool does not detect lies or "read minds".

分数仅为未经校准的模型判断，仅供演示，不构成任何心理评估。

Jev is a model by TypeSafe AI. JevTells is an independent community project and is not affiliated with TypeSafe AI.

## License

MIT
