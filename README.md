# JevTells · 肢体潜台词

> Upload a talking-head video, get it back annotated with gestures, intent and Jev judgments.
>
> 上传一段说话人视频，得到一段标注了手势、意图和 Jev 判定的视频。

**Status: v0.5.0.** Feed in a video, pick the person to analyse, and get back two annotated videos: landscape 16:9 and portrait 3:4.

- **What's on screen:** gesture labels that follow the hands, Jev's scores and trends, intent and emotion, and a one-line narration grounded in the measured data.
- **Layout adapts to the shot:**
  - Labels and the score card move to the empty side of the frame.
  - The portrait video zooms in on the speaker. When the video already has burned-in subtitles too wide to fit, they are moved into a strip below the picture so no text is cut off.

## How it works

| Step | Tool | Runs |
|---|---|---|
| Body & hand keypoints, every frame | MediaPipe | locally |
| Subtitles with word timestamps | faster-whisper | locally |
| Loudness, pitch, speech rate | librosa | locally |
| Keypoints → named gestures (raise, press down, open palm …) | rule engine | locally |
| Confidence / focus / tension, intent, emotion | Jev via OpenRouter | API |
| Narration line and key quote for every sentence | Gemini 3.8 Flash via OpenRouter (configurable) | API |
| Annotated output video | Pillow + ffmpeg | locally |

The only key you need is an OpenRouter API key.

## Requirements

- macOS (tested on Apple Silicon) or Linux
- Python 3.11
- ffmpeg **with libx264**
  - macOS: install [Homebrew](https://brew.sh), then `brew install ffmpeg`
  - Ubuntu / Debian: `sudo apt install ffmpeg`
- About 2 GB of disk space for Python packages and models

## Install

```bash
git clone https://github.com/yxtzan/jevtells.git
cd jevtells
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
jevtells download       # MediaPipe models and fonts
cp .env.example .env    # then open .env and paste your OpenRouter key after OPENROUTER_API_KEY=
jevtells doctor         # checks Python, ffmpeg, models, fonts and whether the key is set
```

- **Speech-to-text model:** downloads automatically on the first run.
- **API key:** get one at [openrouter.ai](https://openrouter.ai). It is only read from `.env`, which is never committed. `doctor` only reports whether the key is set, never its value.
- **Running outside the repository:** `pip install .` also works, and config, translations and the narration prompt are bundled. The tool then looks for models and fonts in this order:
  1. `$JEVTELLS_HOME`
  2. the current directory
  3. `~/.jevtells`

## Usage

**1. See who is in the frame.** Every detected person gets a number, left to right:

```bash
jevtells people path/to/video.mp4            # writes work/<clip_id>/people_00.0s.png
jevtells people path/to/video.mp4 --at 12    # numbering at another moment
```

**2. Run with the person you want to analyse:**

```bash
jevtells run path/to/video.mp4 --target 2 --speaker "Speaker name" --title "AI 读 Speaker"
```

This writes `output_h.mp4` (1920×1080) and `output_v.mp4` (1080×1440) to `work/<clip_id>/`.

**If the shot changes.** When the video cuts or zooms and the numbering changes, add more anchors in the form `NUMBER@SECONDS`. For example, follow person 2 from the start and person 1 from 12 s on:

```bash
jevtells run path/to/video.mp4 --target 2@0 --target 1@12 --speaker "Speaker name"
```

When the main person can't be found, the tool leaves the frame unlabelled (`target=lost`) rather than jumping to someone else. In distant shots, where hands are too small to measure reliably, no gesture labels are shown.

Results go to `work/<clip_id>/`:

| File | Content |
|---|---|
| `output_h.mp4`, `output_v.mp4` | The annotated videos |
| `debug.mp4` | Skeleton, hands, target box and current gesture (with `--until state`) |
| `actions.json` | Detected gestures with time, hand and magnitude |
| `actions_review/` | One start / middle / end strip per gesture, for checking by eye |
| `states.json` | The five-field state per time window (scene, speaker, subtitle, voice, gestures) |
| `judgments.json` | Jev's confidence / focus / tension scores (0–1), intent and emotion per window |
| `narration.json` | One-line narration and key quote per window |
| `judge_review.md` | Everything above side by side, one row per window, for checking by eye |
| `shots.json`, `layout_h.json`, `layout_v.json`, `reframe_v.json` | Shots, label positions and the portrait crop |
| `scene.json`, `raw/` | Automatic scene description and raw API responses |
| `transcript.json`, `windows.json`, `keypoints.npz` | Intermediate data |

| Option | Meaning |
|---|---|
| `--target N` or `N@SECONDS` | Who to analyse; repeat for cuts |
| `--lang zh` / `--lang en` | Narration language (default `zh`); switching only re-runs the narration |
| `--scene TEXT` | Describe the scene yourself instead of the automatic description |
| `--layout h` / `v` / `both` | Which video to render (default `both`) |
| `--title TEXT` | Headline of the portrait video (default `AI 读<speaker>`) |
| `--reframe auto` / `off` | Portrait only: zoom in on the speaker with a full-height 4:3 crop (default `auto`) |
| `--others-speaking START-END` | Seconds when someone else is talking; repeatable. Those sentences get no scores or narration |
| `--blur X,Y,W,H` | Blur a rectangle (source pixels) in the finished video, e.g. a watermark; repeatable |
| `--subtitles on` / `off` | Draw subtitles (default `on`); use `off` if the video already has burned-in subtitles |
| `--debug-layout` | Save one picture per shot showing the speaker area, no-go zones and chosen label positions |
| `--start`, `--duration` | Analyse only part of the video (seconds) |
| `--srt FILE` | Use your own subtitles instead of automatic transcription |
| `--force` | Recompute every stage (otherwise cached results are reused) |
| `--from STAGE`, `--until STAGE` | Run only part of the pipeline; `run` goes all the way to `render` by default |
| `--config FILE` | Use your own settings instead of the bundled `config/default.yaml` |

**What gets re-run:**
- Changing `--target` re-runs tracking and the stages after it. Body detection stays cached.
- To change only the look (title, blur, layout, subtitles, reframe), use `--from render`. No API calls are made, and the videos are re-rendered automatically after an update to the rendering code.

**Where settings live.** Visual settings (colours, sizes, timings, which gesture types are shown) are in the `render` section of `config/default.yaml`. The narration model and its settings are in the `narrate` section.

**Speed.** On an Apple M4 MacBook Air, a first run of a 30-second clip takes a few minutes. Re-rendering takes about 20 seconds per layout.

**Cost.** Jev, the narration model and the scene model all run through OpenRouter. In October 2026 a 30-second clip cost about **$0.005**, mostly the narration. Cached stages are free. `run_meta.json` lists the calls and cost per stage. Question wording and the narration prompt are editable in `config/jev_questions.yaml` and `config/narrate_prompt.md`.

**How the narration is written:**
1. Code first works out the facts for each sentence: gestures, voice, score changes, and clip-wide highs and lows.
2. The model writes the lines for the whole clip in one go, using only those facts.
3. Every line is checked before it is accepted: length, no numbers, the quoted phrase must come from the subtitle, no repeated openings.

## Checking results on several videos

| Script | What it does |
|---|---|
| `scripts/batch_eval.py MANIFEST.yaml` | Runs a list of clips (see `config/eval_manifest.example.yaml`) and writes a summary: tracking rate, gestures per type, labels shown, API cost, narration retries, render time, plus warnings for anything unusual |
| `scripts/check_layout.py work/<clip_id>` | Checks every frame of the finished videos for labels overlapping the face, subtitles, the score card or each other, and for hidden labels or leader lines |
| `scripts/render_review.py work/<clip_id> --at SECONDS …` | Frame grabs and per-second overviews of the finished videos |
| `scripts/make_layout_variants.py` | Makes off-centre, distant and cut test clips from a video you have, for testing the layout |

## Limitations

- **Who is speaking:** the tool assumes the person you chose is the one speaking. Mark other speakers with `--others-speaking`.
- **Scores:** they come from a general-purpose model and are uncalibrated.
- **Gestures:** the rules were tuned on a small set of clips, so expect some wrong labels on very different footage.
- **Subtitle timing:** burned-in subtitles follow their own timing, while the analysis follows the audio. The narration can therefore run a second or two ahead of the subtitle on screen.

## Troubleshooting

- **Run `jevtells doctor` first.** It lists what is missing and how to fix it.
- **`Could not create an NSOpenGLPixelFormat`** (MediaPipe) or **VideoToolbox error `-12903`** (encoding): you are running inside a sandbox without GPU access, typically an AI coding agent. Run the command from a normal terminal.
- **`libx264` not found:** your ffmpeg is a minimal build. Install it with Homebrew or apt as above.
- **`缺少 OPENROUTER_API_KEY`** (missing key): copy `.env.example` to `.env` and paste your key after `OPENROUTER_API_KEY=`.

## Disclaimer

Scores are uncalibrated model judgments for demonstration only. They are not a psychological assessment, and this tool does not detect lies or "read minds".

分数仅为未经校准的模型判断，仅供演示，不构成任何心理评估。

Jev is a model by TypeSafe AI. JevTells is an independent community project and is not affiliated with TypeSafe AI.

## License

MIT
