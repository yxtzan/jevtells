"""Compute commentary facts from measured events and actual Jev results."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..i18n import load_translations
from .state import action_text


SCORE_IDS = ("confidence", "focus", "tension")


def score_value(judgment: Mapping[str, Any] | None, key: str) -> float | None:
    score = (judgment or {}).get("scores", {}).get(key)
    value = score.get("value") if isinstance(score, Mapping) else None
    return float(value) if isinstance(value, (int, float)) else None


def events_for_state(state: Mapping[str, Any], events: Sequence[Mapping[str, Any]], window_id: str) -> list[Mapping[str, Any]]:
    """Only include the exact events that contributed to this state's text."""
    descriptions = list(state.get("measured_actions", []))
    selected: list[Mapping[str, Any]] = []
    for event in events:
        if event.get("window") != window_id or event.get("magnitude", event.get("amplitude")) == "small":
            continue
        description = action_text(dict(event))
        if description in descriptions:
            selected.append(event)
            descriptions.remove(description)
    return selected


def gesture_facts(state: Mapping[str, Any], events: Sequence[Mapping[str, Any]], window_id: str, judgment: Mapping[str, Any] | None) -> list[str]:
    translations = load_translations("zh")
    limbs = translations["facts"]["limbs"]
    magnitudes = translations["facts"]["magnitudes"]
    selected = events_for_state(state, events, window_id)
    used: set[int] = set()
    ranked: list[tuple[tuple[float, ...], str]] = []
    for index, event in enumerate(selected):
        if index in used:
            continue
        pair = None
        for other_index in range(index + 1, len(selected)):
            other = selected[other_index]
            if other_index in used or other["type"] != event["type"]:
                continue
            if {event.get("limb"), other.get("limb")} != {"left_hand", "right_hand"}:
                continue
            if min(float(event["t1"]), float(other["t1"])) > max(float(event["t0"]), float(other["t0"])):
                pair = other_index
                break
        group = [event] if pair is None else [event, selected[pair]]
        if pair is not None:
            used.add(pair)
        limb = "both_hands" if pair is not None else str(event.get("limb", ""))
        prefix = limbs.get(limb, "")
        if pair is not None:
            prefix += translations["facts"]["simultaneous"]
        action = translations["actions"].get(str(event["type"]), str(event["type"]))
        if limb == "both_hands" and action.startswith("双手"):
            action = action[2:]
        details: list[str] = []
        magnitude = event.get("magnitude", event.get("amplitude"))
        if magnitude:
            details.append("幅度" + magnitudes.get(str(magnitude), str(magnitude)))
        probabilities = [(judgment or {}).get("actions", {}).get(str(item.get("id"))) for item in group]
        for probability in probabilities:
            if isinstance(probability, (int, float)):
                details.append(f"表达欲 {probability:.2f}")
        description = prefix + action + ("（" + "，".join(details) + "）" if details else "")
        motion = event["type"] not in {"open_palm", "fist", "point", "palms_up"}
        rank = (float(limb == "both_hands"), float(motion), float({"large": 2, "medium": 1}.get(str(magnitude), 0)))
        ranked.append((rank, description))
    return [description for _rank, description in sorted(ranked, key=lambda item: item[0], reverse=True)]


def compute_highlights(judgments: Mapping[str, Any], windows: Sequence[Mapping[str, Any]], config: Mapping[str, Any] | None = None) -> dict[str, list[str]]:
    settings = (config or {}).get("narrate", {}).get("facts", {})
    delta_threshold = float(settings.get("delta_threshold", 0.08))
    trend_deltas = int(settings.get("trend_deltas", 2))
    epsilon = float(settings.get("epsilon", 1e-9))
    min_extrema_windows = int(settings.get("min_extrema_windows", 2))
    tr = load_translations("zh")
    result: dict[str, list[str]] = {str(window["id"]): [] for window in windows}
    extrema: dict[str, tuple[float, float] | None] = {}
    means: dict[str, float] = {}
    for key in SCORE_IDS:
        values = [score_value(judgments.get(str(w["id"])), key) for w in windows if not w.get("speaker_other")]
        values = [value for value in values if value is not None]
        extrema[key] = (min(values), max(values)) if len(values) >= min_extrema_windows else None
        means[key] = sum(values) / len(values) if values else 0.0
    extreme_candidates: list[tuple[float, str, str]] = []
    history: list[Mapping[str, Any]] = []
    previous: Mapping[str, Any] | None = None
    for window in windows:
        identifier = str(window["id"])
        judgment = judgments.get(identifier)
        if window.get("speaker_other") or not isinstance(judgment, Mapping):
            history.clear()
            previous = None
            continue
        history.append(judgment)
        highlights = result[identifier]
        for key in SCORE_IDS:
            value = score_value(judgment, key)
            if value is None:
                continue
            name = tr["facts"]["scores"][key]
            limits = extrema[key]
            if limits and limits[1] - limits[0] > epsilon:
                if abs(value - limits[1]) <= epsilon:
                    extreme_candidates.append((abs(value - means[key]), identifier, f"{name} {value:.2f}，全场最高"))
                elif abs(value - limits[0]) <= epsilon:
                    extreme_candidates.append((abs(value - means[key]), identifier, f"{name} {value:.2f}，全场最低"))
            before = score_value(previous, key)
            if before is not None and abs(value - before) + epsilon >= delta_threshold:
                highlights.append(f"{name}比上一句 {value - before:+.2f}")
                if value < before:
                    highlights.append(f"{name}回落")
            recent = [score_value(item, key) for item in history[-(trend_deltas + 1):]]
            if len(recent) == trend_deltas + 1 and all(v is not None for v in recent):
                differences = [b - a for a, b in zip(recent, recent[1:])]
                if all(delta > epsilon for delta in differences):
                    highlights.append(f"{name}持续走高")
                elif all(delta < -epsilon for delta in differences):
                    highlights.append(f"{name}持续走低")
        if previous:
            for field, category, label in (("intent", "intents", "意图"), ("emotion", "emotions", "情绪")):
                old = (previous.get(field) or {}).get("label")
                new = (judgment.get(field) or {}).get("label")
                if old and new and old != new:
                    highlights.append(f"{label}由「{tr[category].get(old, old)}」转为「{tr[category].get(new, new)}」")
        previous = judgment
    for _distance, identifier, text in sorted(extreme_candidates, key=lambda item: -item[0])[:math.ceil(len(windows) / 3)]:
        result[identifier].insert(0, text)
    return result


def build_facts(states: Mapping[str, Any], judgments: Mapping[str, Any], windows: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]] = (), voice: Mapping[str, Any] | None = None, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    highlights = compute_highlights(judgments, windows, config)
    tr = load_translations("zh")
    result: dict[str, Any] = {}
    opening_index = 0
    for index, window in enumerate(windows, 1):
        identifier = str(window["id"])
        state = states[identifier]
        judgment = judgments.get(identifier) or {}
        choices: dict[str, Any] = {}
        for field, category in (("intent", "intents"), ("emotion", "emotions")):
            choice = judgment.get(field) or {}
            label = choice.get("label")
            choices[field] = {"label": tr[category].get(label, label), "confidence": choice.get("confidence")}
        voice_metrics = (voice or {}).get("metrics_by_window", {}).get(identifier, {})
        result[identifier] = {
            "index": index, "total": len(windows),
            "subtitle": state.get("subtitle", {}).get("current", ""),
            "previous_subtitle": state.get("subtitle", {}).get("previous", ""),
            "gestures": gesture_facts(state, events, identifier, judgment),
            "voice": {"description": state.get("voice", ""), **voice_metrics},
            "judgments": {"scores": {tr["facts"]["scores"][key]: score_value(judgment, key) for key in SCORE_IDS}, **choices},
            "highlights": highlights[identifier],
            "speaker_other": bool(window.get("speaker_other")),
            "opening_style": ("动作开头 / movement", "引语开头 / quotation", "数据变化开头 / measured change", "声音开头 / voice")[opening_index % 4],
        }
        if not window.get("speaker_other"):
            opening_index += 1
    return result


def run(states: Mapping[str, Any], judgments: Mapping[str, Any], out: Path, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    def read(name: str, default: Any) -> Any:
        path = out / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    windows = read("windows.json", [{"id": identifier} for identifier in states])
    actions = read("actions.json", [])
    events = actions.get("events", []) if isinstance(actions, Mapping) else actions
    facts = build_facts(states, judgments, windows, events, read("voice_features.json", {}), config)
    (out / "narrate_facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    return facts
