"""Generate, validate and audit commentary from code-computed facts only."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..clients.openrouter import OpenRouterClient
from ..schemas import Narration
from . import narrate_facts
from .narrate_validation import RED_FLAGS as _RED_FLAGS, quote_tokens, validation_errors


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
    subtitle = str(facts.get("subtitle", ""))
    quote = _quote("", subtitle)
    if lang == "en":
        # Use the original subtitle; do not fabricate a translation of facts.
        base = f'Saying 「{quote}」'
        candidates = [base, f'Words 「{quote}」', f'Phrase 「{quote}」', f'Sentence {facts["index"]}: 「{quote}」']
    else:
        gestures = [re.sub(r"（.*", "", item) for item in facts.get("gestures", [])]
        highlights = [item.replace("「", "").replace("」", "") for item in facts.get("highlights", [])]
        voice = facts.get("voice", {})
        rate = voice.get("speech_rate_wps") if isinstance(voice, Mapping) else None
        voice_line = f"语速 {rate:.1f} 词/秒" if isinstance(rate, (int, float)) else "继续表达本句内容"
        candidates = [a + "，" + b for a in gestures for b in highlights] + gestures + highlights + [voice_line]
        candidates += [f'第{facts["index"]}句，' + value for value in candidates]
    for line in candidates:
        parsed = {"line": line.rstrip("。."), "quote": quote}
        # A short/empty subtitle may have no possible 2-token quotation; this
        # is explicitly recorded, never filled with made-up words.
        errors = validation_errors(parsed, facts, previous, "", lang)
        if not errors or errors == ["quote must be an exact subtitle substring with 2–8 tokens"]:
            return parsed
    raise ValueError("no factual fallback fits the commentary contract")


def run(states: Mapping[str, Any] | None, judgments: Mapping[str, Any] | None, out: Path, force: bool = False, config: Mapping[str, Any] | None = None, lang: str = "zh", client: OpenRouterClient | None = None) -> dict[str, Any]:
    out = Path(out)
    destination = out / "narration.json"
    if destination.exists() and not force:
        meta_path = out / "narrate_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if meta.get("lang") == lang and meta.get("facts_version") == 1:
            return json.loads(destination.read_text(encoding="utf-8"))
    if states is None:
        states = json.loads((out / "states.json").read_text(encoding="utf-8"))
    if judgments is None:
        judgments = json.loads((out / "judgments.json").read_text(encoding="utf-8"))
    settings = (config or {}).get("narrate", {})
    model = str(settings.get("model", "google/gemini-3.1-flash-lite"))
    retries = int(settings.get("validation_retries", 2))
    template = (Path(__file__).resolve().parents[3] / "config/narrate_prompt.md").read_text(encoding="utf-8")
    facts_by_id = narrate_facts.run(states, judgments, out, config)
    active_client = client or OpenRouterClient()
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    stats: dict[str, Any] = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "failed": 0, "retries": 0, "fallbacks": 0, "windows": {}, "skipped": {}}
    previous: list[str] = []
    for identifier, facts in facts_by_id.items():
        if facts["speaker_other"]:
            results[identifier] = None
            stats["skipped"][identifier] = "speaker_other: no judge/narrator calls for this window"
            for path in raw_dir.glob(f"narrate_{identifier}*.json"):
                path.unlink()
            continue
        for path in raw_dir.glob(f"narrate_{identifier}*.json"):
            path.unlink()
        prompt = template.format(facts_json=json.dumps(facts, ensure_ascii=False), previous_lines=json.dumps(previous, ensure_ascii=False), language_name="English" if lang == "en" else "Chinese")
        if lang == "zh":
            prompt += "\nChinese length counts ALL characters, including spaces, Latin letters and brackets. Aim for 20–28 characters. Translate the phrase in line into concise Chinese; quote remains original."
        failures: list[list[str]] = []
        parsed: dict[str, Any] = {}
        used_fallback = False
        actual_retries = 0
        for attempt in range(retries + 1):
            if attempt:
                actual_retries += 1
                stats["retries"] += 1
            stats["calls"] += 1
            try:
                response = active_client.chat([{"role": "user", "content": prompt}], model, response_format={"type": "json_object"}, max_tokens=int(settings.get("max_tokens", 120)), temperature=0.2)
                raw = response.get("raw", response.get("response", response))
                raw_path = raw_dir / f'narrate_{identifier}{"" if attempt == 0 else f"_retry{attempt}"}.json'
                raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
                usage = response.get("usage") or {}
                stats["prompt_tokens"] += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
                stats["completion_tokens"] += int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
                if response.get("cost") is not None:
                    stats["cost"] += float(response["cost"])
                parsed = _parse(_content(raw))
                errors = validation_errors(parsed, facts, previous, str(states[identifier].get("speaker", "")), lang)
                if not errors:
                    break
                failures.append(errors)
                prompt += "\nValidation failed. Fix these errors in your next JSON:\n" + json.dumps(errors, ensure_ascii=False)
            except Exception as error:
                # Transport retries already belong to the OpenRouter client.
                stats["failed"] += 1
                failures.append([type(error).__name__])
                if (config or {}).get("stop_on_api_error", False):
                    destination.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                    (out / "narrate_meta.json").write_text(json.dumps({**stats, "model": model, "lang": lang, "error_window": identifier, "error": type(error).__name__}, ensure_ascii=False, indent=2), encoding="utf-8")
                    raise
                break
        else:
            parsed = {}
        if not parsed or validation_errors(parsed, facts, previous, str(states[identifier].get("speaker", "")), lang):
            parsed = _fallback(facts, lang, previous)
            used_fallback = True
            stats["fallbacks"] += 1
        Narration.model_validate(parsed)
        results[identifier] = parsed
        previous.append(parsed["line"])
        stats["windows"][identifier] = {"retries": actual_retries, "fallback": used_fallback, "failures": failures, "validation_errors": validation_errors(parsed, facts, previous[:-1], str(states[identifier].get("speaker", "")), lang)}
    if stats["cost"] == 0.0:
        stats["cost"] = None
    destination.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "narrate_meta.json").write_text(json.dumps({**stats, "model": model, "lang": lang, "facts_version": 1}, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


__all__ = ["run", "validation_errors"]
