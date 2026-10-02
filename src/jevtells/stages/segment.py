"""Create speech and silence windows from transcript segments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schemas import Window


def run(transcript: dict[str, Any], shots: list[dict[str, Any]], out: Path, force: bool = False) -> list[dict[str, Any]]:
    """Split long segments at word boundaries and retain silence windows."""
    destination = out / "windows.json"
    if destination.exists() and not force:
        return json.loads(destination.read_text())
    windows: list[dict[str, Any]] = []
    previous = ""
    index = 0
    normalized: list[dict[str, Any]] = []
    for segment in transcript.get("segments", []):
        current = dict(segment)
        current["words"] = sorted(current.get("words") or [], key=lambda item: float(item["t0"]))
        if normalized:
            last = normalized[-1]
            combined_duration = float(current["t1"]) - float(last["t0"])
            last_duration = float(last["t1"]) - float(last["t0"])
            current_duration = float(current["t1"]) - float(current["t0"])
            if combined_duration <= 5.0 and (last_duration < 2.0 or current_duration < 2.0):
                last["t1"] = current["t1"]
                last["text"] = f"{last.get('text', '')} {current.get('text', '')}".strip()
                last["words"].extend(current["words"])
                continue
        normalized.append(current)
    for segment in normalized:
        words = segment.get("words") or []
        start = float(segment["t0"])
        end = float(segment["t1"])
        while end - start > 5.0:
            candidates = [float(word["t1"]) for word in words if start < float(word["t1"]) <= start + 5.0]
            remaining_candidates = [cut for cut in candidates if end - cut >= 2.0]
            cut = max(remaining_candidates or candidates or [start + 5.0])
            selected = [word for word in words if start <= float(word["t0"]) and float(word["t1"]) <= cut]
            text = " ".join(str(word.get("w", "")) for word in selected) or str(segment.get("text", ""))
            window = Window(id=f"W{index:02d}", index=index, t0=start, t1=cut, subtitle=text, prev_subtitle=previous)
            windows.append(window.model_dump())
            previous = window.subtitle
            index += 1
            start = cut
        selected = [word for word in words if start <= float(word["t0"]) and float(word["t1"]) <= end]
        text = " ".join(str(word.get("w", "")) for word in selected) or str(segment.get("text", ""))
        window = Window(id=f"W{index:02d}", index=index, t0=start, t1=end, subtitle=text, prev_subtitle=previous)
        windows.append(window.model_dump())
        previous = window.subtitle
        index += 1
    if not windows:
        duration = max((float(shot["t1"]) for shot in shots), default=0.0)
        if duration > 2.0:
            windows.append(Window(id="W00", index=0, t0=0.0, t1=duration, subtitle="", prev_subtitle="", kind="silence").model_dump())
    destination.write_text(json.dumps(windows, ensure_ascii=False, indent=2))
    return windows
