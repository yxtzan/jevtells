"""Jev decisions for each state window."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ..clients.openrouter import OpenRouterClient
from ..schemas import Judgment
from .state import action_text
from .speakers import other_ids


_SCORE_IDS = ("confidence", "focus", "tension")
_SCORE_LABELS = ("very_low", "low", "medium", "high", "very_high")


def _questions_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "jev_questions.yaml"


def load_questions(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    location = Path(path) if path else _questions_path()
    value = yaml.safe_load(location.read_text(encoding="utf-8")) if location.exists() else {}
    questions = value.get("questions", value) if isinstance(value, Mapping) else {}
    return {str(key): dict(item) for key, item in questions.items() if isinstance(item, Mapping)}


def _actions_questions(state: Mapping[str, Any], base: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    # Jev's alpha endpoint expects a keyed object.  Keep the question IDs as
    # keys and pass the criteria field through unchanged.
    questions = {str(key): dict(value) for key, value in base.items() if key != "action_expressive"}
    template = base.get("action_expressive", {})
    actions = state.get("measured_actions", [])
    if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
        actions = []
    for index, action in enumerate(actions):
        if action == "no notable gestures":
            continue
        question = dict(template)
        question["instructions"] = str(template["instructions"]).format(k=index, action=action)
        questions[f"action_{index}_expressive"] = question
    return questions


def _walk(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
        for nested in value.values():
            found = _walk(nested, key)
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            found = _walk(nested, key)
            if found is not None:
                return found
    return None


def _answer(raw: Mapping[str, Any], question_id: str) -> Any:
    for container_key in ("answers", "decisions", "results", "responses", "data"):
        container = raw.get(container_key)
        if isinstance(container, Mapping) and question_id in container:
            return container[question_id]
        if isinstance(container, Sequence) and not isinstance(container, (str, bytes)):
            for item in container:
                if isinstance(item, Mapping) and str(item.get("id", item.get("question_id", ""))) == question_id:
                    return item
    return _walk(raw, question_id)


def _field(answer: Any, *names: str) -> Any:
    if isinstance(answer, Mapping):
        for name in names:
            if name in answer:
                return answer[name]
    return answer


def _score(answer: Any) -> dict[str, Any] | None:
    if answer is None:
        return None
    raw_value = _field(answer, "value", "score", "level", "choice")
    legend = answer.get("legend") if isinstance(answer, Mapping) else None
    numeric: float | None = None
    try:
        numeric = float(raw_value)
    except (TypeError, ValueError):
        if isinstance(raw_value, str) and raw_value in _SCORE_LABELS:
            # Jev's score legend uses the 0..4 keys shown in the real smoke
            # response, so ``very_low`` maps to 0 rather than 1.
            numeric = float(_SCORE_LABELS.index(raw_value))
    if numeric is None:
        return {"value": None, "raw": raw_value}
    is_label = isinstance(raw_value, str) and raw_value in _SCORE_LABELS
    denominator = 4.0 if is_label else (5.0 if numeric > 1.0 else 1.0)
    if isinstance(legend, Mapping):
        numeric_keys: list[float] = []
        for key in legend:
            try:
                numeric_keys.append(float(key))
            except (TypeError, ValueError):
                continue
        if numeric_keys:
            denominator = max(numeric_keys)
    # A legend describes the raw scale even when the score is below one.
    normalised = numeric / denominator if denominator > 0 else 0.0
    return {"value": max(0.0, min(1.0, normalised)), "raw": raw_value}


def _choice(answer: Any) -> dict[str, Any] | None:
    if answer is None:
        return None
    label = _field(answer, "label", "choice", "value", "selected")
    confidence = _field(answer, "confidence", "probability", "score_confidence")
    probs = _field(answer, "probs", "probabilities", "distribution")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None
    return {"label": label, "confidence": confidence, "probs": probs if isinstance(probs, Mapping) else {}}


def normalise_judgment(raw: Mapping[str, Any], questions: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]], action_ids: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Map alpha response variants into the stable §6.8 schema."""

    scores = {key: _score(_answer(raw, key)) for key in _SCORE_IDS}
    intent = _choice(_answer(raw, "intent"))
    emotion = _choice(_answer(raw, "emotion"))
    actions: dict[str, Any] = {}
    iterable = questions.items() if isinstance(questions, Mapping) else ((question.get("id", ""), question) for question in questions)
    for identifier, question in iterable:
        identifier = str(identifier)
        if identifier.startswith("action_"):
            answer = _answer(raw, identifier)
            value = _field(answer, "probability", "prob", "value", "score", "noul")
            action_id = (action_ids or {}).get(identifier, identifier)
            try:
                actions[action_id] = float(value) if value is not None else None
            except (TypeError, ValueError):
                actions[action_id] = None
    return {"scores": scores, "intent": intent, "emotion": emotion, "actions": actions}


def _action_ids(state: Mapping[str, Any], window_id: str, out: Path) -> dict[str, str]:
    """Link each measured-action question to its stable actions.json event ID."""
    path = out / "actions.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("events", []) if isinstance(payload, Mapping) else payload
    available = [(str(event["id"]), action_text(event)) for event in events if event.get("window") == window_id and event.get("id")]
    result: dict[str, str] = {}
    for index, description in enumerate(state.get("measured_actions", [])):
        for candidate, (identifier, text) in enumerate(available):
            if text == description:
                result[f"action_{index}_expressive"] = identifier
                available.pop(candidate)
                break
    return result


def _stats(result: Mapping[str, Any]) -> dict[str, Any]:
    usage = result.get("usage", {})
    usage = dict(usage) if isinstance(usage, Mapping) else {}
    return {"calls": 1, "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)), "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)), "cost": result.get("cost")}


def run(states: Mapping[str, Any] | None, out: Path, force: bool = False, config: Mapping[str, Any] | None = None, client: OpenRouterClient | None = None) -> dict[str, Any]:
    out = Path(out)
    destination = out / "judgments.json"
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    if states is None:
        states = json.loads((out / "states.json").read_text(encoding="utf-8"))
    windows_path = out / "windows.json"
    skipped = other_ids(json.loads(windows_path.read_text(encoding="utf-8"))) if windows_path.exists() else set()
    base = load_questions()
    model = str((config or {}).get("jev", {}).get("model", "typesafe/jev-1.13"))
    active_client = client or OpenRouterClient()
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    judgments: dict[str, Any] = {}
    aggregate = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "failed": 0}
    for window_id, state in states.items():
        if window_id in skipped:
            judgments[window_id] = None
            (raw_dir / f"jev_{window_id}.json").unlink(missing_ok=True)
            continue
        questions = _actions_questions(state, base)
        aggregate["calls"] += 1
        try:
            result = active_client.decide(state, questions, model)
            raw = result.get("raw", result.get("response", result)) if isinstance(result, Mapping) else result
            if not isinstance(raw, Mapping):
                raise ValueError("Jev response is not an object")
            (raw_dir / f"jev_{window_id}.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
            judgments[window_id] = normalise_judgment(raw, questions, _action_ids(state, window_id, out))
            Judgment.model_validate(judgments[window_id])
            stats = _stats(result if isinstance(result, Mapping) else {})
            aggregate["prompt_tokens"] += int(stats["prompt_tokens"] or 0)
            aggregate["completion_tokens"] += int(stats["completion_tokens"] or 0)
            if stats["cost"] is not None:
                aggregate["cost"] += float(stats["cost"])
        except Exception as error:
            # Do not serialize exception text: transport errors can contain provider data.
            aggregate["failed"] += 1
            judgments[window_id] = {"scores": {key: None for key in _SCORE_IDS}, "intent": None, "emotion": None, "actions": {}, "error": type(error).__name__}
            if (config or {}).get("stop_on_api_error", False):
                destination.write_text(json.dumps(judgments, ensure_ascii=False, indent=2), encoding="utf-8")
                (out / "judge_meta.json").write_text(json.dumps({**aggregate, "model": model, "error_window": window_id}, ensure_ascii=False, indent=2), encoding="utf-8")
                raise
    if aggregate["cost"] == 0.0:
        aggregate["cost"] = None
    for value in judgments.values():
        if value is not None:
            Judgment.model_validate(value)
    destination.write_text(json.dumps(judgments, ensure_ascii=False, indent=2), encoding="utf-8")
    aggregate.update({"model": model, "skipped": {identifier: "speaker_other: window overlaps other speech >= configured threshold" for identifier in sorted(skipped)}})
    (out / "judge_meta.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    judgments["_meta"] = aggregate
    return judgments


__all__ = ["load_questions", "normalise_judgment", "run"]
