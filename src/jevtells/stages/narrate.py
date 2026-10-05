"""Generate, validate and audit commentary from code-computed facts only."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..clients.openrouter import OpenRouterClient, OpenRouterError
from ..schemas import Narration
from . import narrate_facts
from .narrate_validation import RED_FLAGS as _RED_FLAGS, quote_tokens, validation_errors, repeated_highlights
from ..resources import data_path


def _content(raw: Mapping[str, Any]) -> str:
    choices = raw.get("choices", [])
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        value = choices[0].get("message", {}).get("content", "")
        if isinstance(value, list):
            return "".join(str(part.get("text", "")) for part in value if isinstance(part, Mapping))
        return str(value)
    return str(raw.get("output", raw.get("text", "")))


def _parse(text: str) -> dict[str, str]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {"line": "", "quote": ""}
    return {key: str(value.get(key, "")).strip() for key in ("line", "quote")} if isinstance(value, Mapping) else {"line": "", "quote": ""}


def _quote(text: str, subtitle: str) -> str:
    if text in subtitle and 2 <= len(quote_tokens(text)) <= 8:
        return text
    for sentence in re.split(r"(?<=[。！？.!?；;])\s*", subtitle):
        if 2 <= len(quote_tokens(sentence.strip())) <= 8:
            return sentence.strip()
    tokens = list(re.finditer(r"[a-zA-Z0-9]+(?:['’\-][a-zA-Z0-9]+)*|[\u3400-\u9fff]", subtitle))
    return subtitle[tokens[0].start():tokens[min(7, len(tokens) - 1)].end()] if len(tokens) >= 2 else ""


def _fallback(facts: Mapping[str, Any], lang: str, previous: Sequence[str]) -> dict[str, str]:
    from ..i18n import load_translations
    tr = load_translations(lang)
    quote = _quote("", str(facts.get("subtitle", "")))
    highlights = narrate_facts.verbal_highlights(facts.get("highlights", []), lang)
    gestures = [re.sub(r"（.*", "", item) for item in facts.get("gestures", [])] if lang == "zh" else []
    choices = facts.get("judgments", {})
    intent = (choices.get("intent") or {}).get("label")
    emotion = (choices.get("emotion") or {}).get("label")
    if lang == "en":
        zh = load_translations("zh")
        for field, category in (("intent", "intents"), ("emotion", "emotions")):
            label = intent if field == "intent" else emotion
            label = next((tr[category][key] for key, value in zh[category].items() if value == label), label)
            if field == "intent":
                intent = label
            else:
                emotion = label
    candidates = [a + "，" + b for a in gestures for b in highlights] + highlights
    if intent and emotion:
        candidates.append(tr["narration"]["fallback"].format(intent=intent, emotion=emotion))
    for line in candidates:
        parsed = {"line": line, "quote": quote}
        if not validation_errors(parsed, facts, previous, "", lang, allow_repeated_opening=True):
            return parsed
    raise ValueError("no factual fallback fits the commentary contract")


def _parse_batch(text: str) -> dict[str, dict[str, str]]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(value, Mapping) or not isinstance(value.get("lines"), list):
        return {}
    result = {}
    for item in value["lines"]:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
            continue
        identifier = item["id"]
        if identifier in result:
            return {}  # Ambiguous IDs cannot silently overwrite a line.
        result[identifier] = {key: str(item.get(key, "")).strip() for key in ("line", "quote")}
    return result


def run(states: Mapping[str, Any] | None, judgments: Mapping[str, Any] | None, out: Path, force: bool = False, config: Mapping[str, Any] | None = None, lang: str = "zh", client: OpenRouterClient | None = None) -> dict[str, Any]:
    out = Path(out)
    destination = out / "narration.json"
    if destination.exists() and not force:
        meta_path = out / "narrate_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if meta.get("lang") == lang and meta.get("facts_version") == 4:
            return json.loads(destination.read_text(encoding="utf-8"))
    if states is None:
        states = json.loads((out / "states.json").read_text(encoding="utf-8"))
    if judgments is None:
        judgments = json.loads((out / "judgments.json").read_text(encoding="utf-8"))
    settings = (config or {}).get("narrate", {})
    model = str(settings.get("model", "google/gemini-3.8-flash"))
    reasoning = settings.get("reasoning", {"effort": "low"})
    max_tokens = int(settings.get("max_tokens", 4000))
    retries = min(2, max(0, int(settings.get("validation_retries", 2))))
    template = data_path("config/narrate_prompt.md").read_text(encoding="utf-8")
    facts_by_id = narrate_facts.run(states, judgments, out, config)
    active = {key: {**facts, "id": key, "verbal_highlights": narrate_facts.verbal_highlights(facts.get("highlights", []), lang)} for key, facts in facts_by_id.items() if not facts["speaker_other"]}
    active_client = client or OpenRouterClient()
    request_records = []
    if isinstance(active_client, OpenRouterClient):
        active_client.audit_dir = out / "raw"
        observer = getattr(active_client, "on_response", None)
        def account(event):
            request_records.append({key: value for key, value in event.items() if key != "body"})
            if observer:
                observer(event)
        active_client.on_response = account
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    generation = time.time_ns()
    results = {key: None for key in facts_by_id}
    stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "response_records": [], "cost": 0.0, "failed": 0, "retries": 0, "fallbacks": 0, "windows": {key: {"retries": 0, "fallback": False, "failures": []} for key in active}, "skipped": {key: "speaker_other" for key in facts_by_id if key not in active}}
    accepted = {}
    pending = list(active)
    failures = {}
    def save(error=False):
        if request_records:
            stats["http_calls"] = len(request_records)
            stats["request_records"] = request_records
            stats["cost"] = None if any(r["cost"] is None for r in request_records) else sum(r["cost"] for r in request_records)
            stats["known_cost"] = sum(r["cost"] or 0 for r in request_records)
        if not error:
            destination.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / ("narrate_error_meta.json" if error else "narrate_meta.json")).write_text(json.dumps({**stats, "model": model, "reasoning": reasoning, "max_tokens": max_tokens, "lang": lang, "facts_version": 4}, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(retries + 1):
        if not pending:
            break
        stats["retries"] = attempt
        for key in pending:
            stats["windows"][key]["retries"] = attempt
        prompt = template.format(facts_json=json.dumps(list(active.values()), ensure_ascii=False), fixed_lines=json.dumps(accepted, ensure_ascii=False), failures_json=json.dumps(failures, ensure_ascii=False), requested_ids=json.dumps(pending), language_name="English" if lang == "en" else "Chinese")
        try:
            request_max_tokens = max_tokens
            for length_attempt in range(2):
                stats["calls"] += 1
                response = active_client.chat([{"role": "user", "content": prompt}], model, response_format={"type": "json_object"}, max_tokens=request_max_tokens, temperature=0.2, reasoning=reasoning)
                raw = response.get("raw", response.get("response", response))
                raw_path = raw_dir / f"narrate_batch_{generation}_{attempt}_{length_attempt}.json"
                raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
                stats.setdefault("raw_responses", []).append(str(raw_path))
                usage = response.get("usage") or raw.get("usage") or {}
                stats["prompt_tokens"] += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
                stats["completion_tokens"] += int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
                details = usage.get("completion_tokens_details") or {}
                reasoning_tokens = details.get("reasoning_tokens")
                stats["reasoning_tokens"] = (None if reasoning_tokens is None or stats["reasoning_tokens"] is None
                                             else stats["reasoning_tokens"] + int(reasoning_tokens))
                finish_reasons = [choice.get("finish_reason") for choice in raw.get("choices", [])]
                cost = response.get("cost")
                stats["response_records"].append({"call": stats["calls"], "validation_attempt": attempt, "length_attempt": length_attempt, "max_tokens": request_max_tokens, "reasoning_tokens": reasoning_tokens, "finish_reason": finish_reasons[0] if finish_reasons else None, "finish_reasons": finish_reasons, "cost": cost, "raw_path": str(raw_path)})
                stats.setdefault("costs", []).append({"cost": cost, "reason": None if cost is not None else "response has no usage.cost"})
                stats["cost"] = None if stats["cost"] is None or cost is None else stats["cost"] + float(cost)
                if "length" not in finish_reasons:
                    candidates = _parse_batch(_content(raw))
                    break
                stats["failed"] += 1
                stats["truncated_responses"] = stats.get("truncated_responses", 0) + 1
                candidates = {}
                if length_attempt == 0:
                    stats["length_retries"] = stats.get("length_retries", 0) + 1
                    request_max_tokens *= 2
        except Exception as error:
            stats["failed"] += 1
            stats["error"] = type(error).__name__
            stats["cost"] = None
            stats.setdefault("costs", []).append({"cost": None, "reason": "API failed; see raw error records"})
            if (config or {}).get("stop_on_api_error", False) or isinstance(error, OpenRouterError):
                save(error=True)
                raise
            save()
            break
        if "length" in finish_reasons:
            # The doubled-limit retry also truncated. Do not restart this
            # same generation at the original limit in a validation round.
            break
        failures = {}
        fixed = dict(accepted)
        previous = []
        ordered = list(active)
        for index, key in enumerate(ordered):
            if key in fixed:
                previous.append(fixed[key]["line"])
                continue
            parsed = candidates.get(key, {})
            errors = validation_errors(parsed, active[key], previous, str(states[key].get("speaker", "")), lang)
            line = str(parsed.get("line", ""))
            future = [fixed[k]["line"] for k in ordered[index+1:] if k in fixed]
            if any(line[:4] == old[:4] for old in future):
                errors.append("first four characters conflict with a fixed later line")
            if future and repeated_highlights(line, future[0]):
                errors.append("highlight claim repeats the adjacent fixed later line")
            if lang == "zh":
                if sum(old[:2] == line[:2] for old in previous+future) >= 2:
                    errors.append("first two characters conflict with fixed lines")
                opening = re.search(r"(.{2})「", line)
                before = [re.search(r"(.{2})「", old) for old in previous+future]
                if opening and sum(match is not None and match[1] == opening[1] for match in before) >= 2:
                    errors.append("two characters before 「 conflict with fixed lines")
                next_opening = re.search(r"(.{2})「", future[0]) if future else None
                if opening and next_opening and opening[1] == next_opening[1]:
                    errors.append("two characters before 「 conflict with an adjacent fixed line")
            if errors:
                failures[key] = errors
                stats["windows"][key]["failures"].append(errors)
            else:
                accepted[key] = parsed
                previous.append(parsed["line"])
        pending = list(failures)
    previous = []
    for key, facts in active.items():
        parsed = accepted.get(key)
        if parsed is not None and validation_errors(parsed, facts, previous, str(states[key].get("speaker", "")), lang):
            parsed = None
        if parsed is None:
            try:
                parsed = _fallback(facts, lang, previous)
            except ValueError:
                # No measured facts or no possible exact quotation: keep
                # this window unavailable instead of fabricating a sentence.
                stats["windows"][key]["unavailable"] = "no factual fallback satisfies all validation rules"
                stats["windows"][key]["validation_errors"] = [stats["windows"][key]["unavailable"]]
                stats["unavailable"] = stats.get("unavailable",0)+1
                continue
            stats["fallbacks"] += 1
            stats["windows"][key]["fallback"] = True
        Narration.model_validate(parsed)
        results[key] = parsed
        stats["windows"][key]["validation_errors"] = validation_errors(parsed, facts, previous, str(states[key].get("speaker", "")), lang, allow_repeated_opening=stats["windows"][key]["fallback"])
        previous.append(parsed["line"])
    save()
    return results


__all__ = ["run", "validation_errors"]
