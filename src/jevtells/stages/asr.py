"""Speech recognition and SRT import."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _parse_srt(path: Path) -> dict[str, Any]:
    """Parse SRT cues into the project transcript schema."""
    segments: list[dict[str, Any]] = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip()):
        match = re.search(r"(\d\d):(\d\d):(\d\d),(\d+) --> (\d\d):(\d\d):(\d\d),(\d+)\n(.+)", block, re.S)
        if not match:
            continue
        start = int(match.group(1)) * 3600 + int(match.group(2)) * 60 + int(match.group(3)) + int(match.group(4)) / 1000
        end = int(match.group(5)) * 3600 + int(match.group(6)) * 60 + int(match.group(7)) + int(match.group(8)) / 1000
        segments.append({"t0": start, "t1": end, "text": match.group(9).strip().replace("\n", " "), "words": []})
    return {"language": "unknown", "segments": segments, "subtitle_source": "srt"}


def run(wav: Path, out: Path, srt: str | None = None, force: bool = False, model_name: str = "small", destination_name: str = "transcript.json") -> dict[str, Any]:
    """Transcribe audio with faster-whisper or import the supplied SRT."""
    destination = out / destination_name
    if destination.exists() and not force:
        return json.loads(destination.read_text())
    if srt:
        transcript = _parse_srt(Path(srt))
    else:
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel(model_name, device="cpu", compute_type="int8")
            segments, info = model.transcribe(str(wav), word_timestamps=True, vad_filter=True)
            records: list[dict[str, Any]] = []
            for segment in segments:
                records.append({"t0": float(segment.start), "t1": float(segment.end), "text": segment.text.strip(), "words": [{"t0": float(word.start), "t1": float(word.end), "w": word.word.strip()} for word in (segment.words or [])]})
            transcript = {"language": info.language, "segments": records}
        except Exception as error:
            raise RuntimeError(f"faster-whisper transcription failed: {error}") from error
    destination.write_text(json.dumps(transcript, ensure_ascii=False, indent=2))
    return transcript
