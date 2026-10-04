# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
