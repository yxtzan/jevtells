"""Human-readable per-window review artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _score(judgment: Mapping[str, Any], key: str) -> str:
    value = judgment.get("scores", {}).get(key) if isinstance(judgment.get("scores"), Mapping) else None
    if not isinstance(value, Mapping):
        return "—"
    return f"{value.get('value', '—')} (raw={value.get('raw', '—')})"


def write(states: Mapping[str, Any], judgments: Mapping[str, Any], narration: Mapping[str, Any], out: Path, *, lang: str = "zh") -> Path:
    destination = Path(out) / "judge_review.md"
    lines = ["# Judge review", "", "| window | time | subtitle | measured_actions | voice | confidence | focus | tension | intent (confidence) | emotion (confidence) | narration | quote |", "|---|---:|---|---|---|---:|---:|---:|---|---|---|---|"]
    windows_path = Path(out) / "windows.json"
    windows = {str(item.get("id")): item for item in json.loads(windows_path.read_text(encoding="utf-8"))} if windows_path.exists() else {}
    for window_id, state in states.items():
        judgment = judgments.get(window_id, {}) if isinstance(judgments, Mapping) else {}
        line = narration.get(window_id, {}) if isinstance(narration, Mapping) else {}
        window = windows.get(str(window_id), {})
        intent = judgment.get("intent") if isinstance(judgment, Mapping) else None
        emotion = judgment.get("emotion") if isinstance(judgment, Mapping) else None
        intent_text = f"{intent.get('label', '—')} ({intent.get('confidence', '—')})" if isinstance(intent, Mapping) else "—"
        emotion_text = f"{emotion.get('label', '—')} ({emotion.get('confidence', '—')})" if isinstance(emotion, Mapping) else "—"
        actions = "; ".join(str(item) for item in state.get("measured_actions", [])) if isinstance(state, Mapping) else "—"
        subtitle = str(state.get("subtitle", {}).get("current", "")) if isinstance(state.get("subtitle"), Mapping) else ""
        voice = str(state.get("voice", "")) if isinstance(state, Mapping) else ""
        safe = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
        lines.append("| " + " | ".join((safe(window_id), f"{float(window.get('t0', 0.0)):.2f}–{float(window.get('t1', 0.0)):.2f}s", safe(subtitle), safe(actions), safe(voice), _score(judgment, "confidence"), _score(judgment, "focus"), _score(judgment, "tension"), safe(intent_text), safe(emotion_text), safe(line.get("line", "—")), safe(line.get("quote", "—")))) + " |")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


__all__ = ["write"]
