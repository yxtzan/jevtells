# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/yxtzan/jevtells/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/yxtzan/jevtells/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yxtzan/jevtells/releases/tag/v0.1.0
