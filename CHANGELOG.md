# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.5.0] - 2026-10-05

### Added

- **Shot detection and per-shot speaker bounds.** Distant shots are recognised. Gestures detected there stay in `actions.json`, but they are not sent to Jev and not drawn.
- **Automatic label placement, decided once per shot.**
  - Labels avoid the face, subtitles, blurred areas, the score card and each other.
  - Leader lines never cross the face or the body's midline.
  - A shown label always has its leader line and hand dot.
- **Landscape score card that moves to the empty side at cuts.** It stays on the right unless the left is clearly emptier.
- **Portrait smart reframe** (`--reframe auto|off`, default `auto`). The portrait video is a full-height 4:3 crop around the speaker. When burned-in subtitles are too wide for the crop, the picture is zoomed and the original subtitle band is shown full-width below it.
- **`jevtells doctor`** checks Python, ffmpeg encoders, models, fonts and whether the key is set.
- **`jevtells download`** fetches the models and fonts.
- **Installable with `pip install .`**, with config, translations and the prompt bundled. Models and fonts are looked up in `$JEVTELLS_HOME`, then the current directory, then `~/.jevtells`.
- **`--debug-layout`** saves one picture per shot.
- **New scripts:** `scripts/check_layout.py` (frame-by-frame layout audit), `scripts/batch_eval.py` (multi-clip summary with warnings) and `scripts/make_layout_variants.py` (off-centre, distant and cut test clips).

### Changed

- **`run` renders the videos by default.**
- **Narration is written for the whole clip in one call**, by Gemini 3.8 Flash with low reasoning.
  - Lines contain no numbers; changes are described in words.
  - The quoted phrase must translate part of the subtitle and may not be a label name.
  - Adjacent lines may not repeat an opening or a highlight.
  - Failed lines are retried together, and the fallback reads as a plain sentence.
- **Commentary transitions:** the old line fades out completely before the new one fades in.
- **Fixed heights:** the narration area keeps one height for the whole clip, and leftover space in the portrait layout is spread evenly.
- **Footer** shows short model names. Full model IDs are kept in `run_meta.json`.
- **Rendering cache:** videos are re-rendered automatically when the rendering code changes, and `--from render` always re-renders.
- **OpenRouter errors:** rate limits, server errors and timeouts are retried with backoff. Error responses are saved without credentials, and requests rejected before reaching a provider are recorded as unbilled.

### Known issues

- Burned-in subtitles follow their own timing, so the narration can run a second or two ahead of the on-screen subtitle.
- Label positions are fixed per shot, so the leader line gets long when a hand moves far within the shot.
- Gesture rules and layout have so far been checked on one speaker plus synthetic variants. Validation on new footage comes before v1.0.

## [0.4.0] - 2026-10-05

### Added

- `render` stage: annotated landscape (1920×1080) and portrait (1080×1440) videos following `docs/design.md`. Gesture labels follow the hands; animated scores with whole-clip trend lines and change vs the previous sentence; top-3 intent probabilities; emotion timeline; narration headline with highlighted key phrase.
- `--layout`, `--title`, `--others-speaking`, `--blur`, `--subtitles`.
- Narration rebuilt on computed facts: code derives highlights (clip-wide highs and lows, changes, trends, intent and emotion shifts), the model may only use those, and every line is checked before it is accepted, with up to two retries and a template fallback.
- `scripts/render_review.py` for frame grabs, side-by-side comparisons and per-second overviews.
- Additional fonts (Noto Sans CJK SC, Noto Serif CJK SC, Noto Sans Mono) in the download script.

### Known issues

- Label positions are fixed per side, so they can land far from a speaker standing at the edge of the frame.
- `run` still stops after `state` by default; add `--until render` for the videos.
- Old and new narration lines overlap briefly when the sentence changes.

## [0.3.0] - 2026-10-04

### Added

- OpenRouter client (`clients/openrouter.py`) for Jev decisions and chat models: key read from `.env` only, retries on 429 / 5xx, key redacted from errors, cost recorded per call.
- `judge` stage: Jev scores confidence, focus and tension, picks intent and emotion, and rates how expressive each gesture is, once per time window.
- `scene` stage: one-sentence scene description from the middle frame (skipped when `--scene` is given).
- `narrate` stage: one-line narration and a verbatim quote per window, with tone checks and quote validation.
- Chinese and English labels (`i18n/`), `--lang zh|en`.
- `judge_review.md` for reviewing every window by eye; per-stage API calls and cost in `run_meta.json`.
- Editable `config/jev_questions.yaml` and `config/narrate_prompt.md`; `scripts/jev_smoke.py` for a one-call API check.

### Changed

- `fist` now needs a tightly closed hand and `point` needs the other three fingers curled, so cupped and open hands no longer count.
- Small motions stay in `actions.json` but are left out of the state sent to Jev.
- Tracking settings use the same key names as the config file.

### Known issues

- Narration lines are accurate but plain and repetitive; to be rewritten in 0.4.0.
- Speech by someone other than the target is attributed to the target.
- In wide shots, a still wrist can drift enough to be reported as a raise.

## [0.2.0] - 2026-10-03

### Added

- `jevtells people`: numbers every person in a frame (`people_<seconds>s.png`) so you can pick who to analyse.
- `--target N@SECONDS` with multiple anchors for cuts; a separate tracking stage, so changing the target does not re-run body detection. A lost target stays unlabelled instead of jumping to another person.
- Gesture detection: raise, press down, beat, spread, gather, nod, shake, open palm, palms up, point, fist; per-gesture review strips (`actions_review/`) and `actions_summary.txt`.
- Voice description with loudness, pitch variation (pyin), speech rate and pauses.
- All thresholds in `config/default.yaml`, overridable with `--config`.

### Changed

- Geometry is computed in pixels and normalised by shoulder width; low-visibility landmarks are ignored and not drawn.
- Left / right hands are matched to the body by the best overall assignment.
- Debug video shows every detected person, the highlighted target and the current gesture.

### Fixed

- The main speaker is no longer picked as the largest person, which locked onto the wrong person in 0.1.0.
- Gestures are no longer reported for hands at the frame edge or poorly visible, for a hand holding a microphone (detected automatically as an occupied hand), or for long static postures.

### Known issues

- `fist` and `point` still fire on some cupped or open hands.
- `lean_in` is disabled by default: with one camera it can't be told apart from zooming or turning.
- A lost target is not re-acquired automatically; add another `--target` anchor.

## [0.1.0] - 2026-10-03

### Added

- Data pipeline with per-stage caching (`--force`, `--from`, `--until`): clip preparation (ffmpeg), body and hand keypoints (MediaPipe, CPU), transcription with word timestamps (faster-whisper) or SRT import, loudness and pitch features (librosa), shot labels, sentence-based time windows (2–5 s), and the five-field state for each window.
- Debug video with skeleton, hand landmarks (speaker's own left / right), window and shot overlay, original audio.
- Model and font download script.
- Project specification and milestone briefs in `docs/`.

### Known issues

- The main speaker is picked as the largest person in the first frame and can lock onto the wrong person. To be fixed in 0.2.0 with `jevtells people` and `--target`.
- The voice description covers loudness only.
- Gesture measurement is a placeholder (wrist travel); gesture labels arrive in 0.2.0.

[Unreleased]: https://github.com/yxtzan/jevtells/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/yxtzan/jevtells/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/yxtzan/jevtells/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/yxtzan/jevtells/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yxtzan/jevtells/releases/tag/v0.1.0
