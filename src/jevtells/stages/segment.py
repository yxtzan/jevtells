"""Create speech and silence windows from transcript segments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schemas import Window
from ..config import config_value, load_config


def run(transcript: dict[str, Any], shots: list[dict[str, Any]], out: Path, force: bool = False, config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Split long segments at word boundaries and retain silence windows."""
    destination = out / "windows.json"
    if destination.exists() and not force:
        return json.loads(destination.read_text())
    settings = config or load_config()
    min_window = float(config_value(settings, "windows.min_seconds", config_value(settings, "min_window", 2.0)))
    max_window = float(config_value(settings, "windows.max_seconds", config_value(settings, "max_window", 5.0)))
    windows: list[dict[str, Any]] = []
    previous = ""
    index = 0
    normalized: list[dict[str, Any]] = []
    for segment in transcript.get("segments", []):
        current = dict(segment)
        current["words"] = sorted(current.get("words") or [], key=lambda item: float(item["t0"]))
        if normalized and transcript.get("subtitle_source") not in {"ocr", "srt"}:
            last = normalized[-1]
            combined_duration = float(current["t1"]) - float(last["t0"])
            last_duration = float(last["t1"]) - float(last["t0"])
            current_duration = float(current["t1"]) - float(current["t0"])
            if combined_duration <= max_window and (last_duration < min_window or current_duration < min_window):
                last["t1"] = current["t1"]
                last["text"] = f"{last.get('text', '')} {current.get('text', '')}".strip()
                last["words"].extend(current["words"])
                continue
        normalized.append(current)
    for segment in normalized:
        words = segment.get("words") or []
        start = float(segment["t0"])
        end = float(segment["t1"])
        while end - start > max_window:
            candidates = [float(word["t1"]) for word in words if start < float(word["t1"]) <= start + max_window]
            remaining_candidates = [cut for cut in candidates if end - cut >= 2.0]
            cut = max(remaining_candidates or candidates or [start + max_window])
            selected = [word for word in words if start <= float(word["t0"]) and float(word["t1"]) <= cut]
            text = " ".join(str(word.get("w", "")) for word in selected) or str(segment.get("text", ""))
            window = Window(id=f"W{index:02d}", index=index, t0=start, t1=cut, subtitle=text, subtitle_translation=segment.get("subtitle_translation", ""), prev_subtitle=previous)
            windows.append(window.model_dump())
            previous = window.subtitle
            index += 1
            start = cut
        selected = [word for word in words if start <= float(word["t0"]) and float(word["t1"]) <= end]
        text = " ".join(str(word.get("w", "")) for word in selected) or str(segment.get("text", ""))
        window = Window(id=f"W{index:02d}", index=index, t0=start, t1=end, subtitle=text, subtitle_translation=segment.get("subtitle_translation", ""), prev_subtitle=previous)
        windows.append(window.model_dump())
        previous = window.subtitle
        index += 1
    if not windows:
        duration = max((float(shot["t1"]) for shot in shots), default=0.0)
        if duration > min_window:
            windows.append(Window(id="W00", index=0, t0=0.0, t1=duration, subtitle="", prev_subtitle="", kind="silence").model_dump())
    destination.write_text(json.dumps(windows, ensure_ascii=False, indent=2))
    return windows


def merge_short_windows(windows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Merge only adjacent equal presence/speaker classes, retaining literal text."""
    rows = [dict(w) for w in windows]
    minimum = float(config_value(config, 'windows.min_seconds', 2.0))
    maximum = float(config_value(config, 'windows.max_seconds', 5.0))
    def category(w):
        return bool(w.get('speaker_other')), bool(w.get('target_offscreen'))
    index = 0
    while index < len(rows):
        if rows[index]['t1']-rows[index]['t0'] >= minimum:
            index += 1
            continue
        neighbours = [j for j in (index-1,index+1) if 0 <= j < len(rows) and category(rows[j]) == category(rows[index])]
        if not neighbours:
            index += 1
            continue
        # Prefer a result within max_window, then the closest earlier window.
        other = min(neighbours, key=lambda j: (max(rows[j]['t1'],rows[index]['t1'])-min(rows[j]['t0'],rows[index]['t0']) > maximum, abs(j-index), j))
        a,b = sorted((index,other))
        left,right = rows[a],rows[b]
        duration_left, duration_right = left['t1']-left['t0'],right['t1']-right['t0']
        left['t1'] = right['t1']
        for field in ('subtitle','subtitle_translation'):
            left[field] = ' '.join(v for v in (left.get(field,''),right.get(field,'')) if v)
        if left.get('target_presence_ratio') is not None and right.get('target_presence_ratio') is not None:
            left['target_presence_ratio'] = (duration_left*left['target_presence_ratio']+duration_right*right['target_presence_ratio'])/(duration_left+duration_right)
        rows.pop(b)
        index = max(0,a-1)
    for index,row in enumerate(rows):
        row.update(id=f'W{index:02d}', index=index, prev_subtitle=rows[index-1]['subtitle'] if index else '')
        row['hold_previous_panel'] = bool(index and any(category(row)) and row['t1']-row['t0'] < float(config_value(config,'windows.panel_hold_seconds',1.0)))
    return rows
