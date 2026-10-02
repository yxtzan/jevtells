# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/yxtzan/jevtells/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yxtzan/jevtells/releases/tag/v0.1.0
