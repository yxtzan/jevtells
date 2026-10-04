"""Neutral, short narration generated from state and Jev judgments."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from ..clients.openrouter import OpenRouterClient
from ..schemas import Narration


_RED_FLAGS = re.compile(r"撒谎|说谎|心虚|装|骗子|欺骗|心理|道德|动机|身份|私生活|lie|lying|liar|deceiv|guilty|dishonest|motive|psycholog|identity", re.IGNORECASE)


def _content(raw: Mapping[str, Any]) -> str:
    choices = raw.get("choices", [])
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        message = choices[0].get("message", {})
        value = message.get("content", "") if isinstance(message, Mapping) else ""
        if isinstance(value, list):
            return "".join(str(part.get("text", "")) for part in value if isinstance(part, Mapping))
        return str(value)
    return str(raw.get("output", raw.get("text", "")))


def _fallback(state: Mapping[str, Any], judgment: Mapping[str, Any], lang: str) -> str:
    actions = state.get("measured_actions", [])
    action = str(actions[0]) if isinstance(actions, list) and actions else ""
    intent = judgment.get("intent") if isinstance(judgment, Mapping) else None
    label = intent.get("label") if isinstance(intent, Mapping) else None
    action_names = {"raise": ("抬手", "raises a hand"), "press_down": ("下压", "presses down"), "open_palm": ("张开手掌", "opens a palm"), "fist": ("握拳", "forms a fist"), "point": ("指点", "points"), "palms_up": ("摊手", "turns a palm up"), "spread": ("展开双手", "spreads both hands"), "gather": ("收拢双手", "gathers both hands")}
    action_type = next((name for name in action_names if name in action), None)
    chinese_action = action_names.get(action_type, ("继续表达", "continues speaking"))[0]
    english_action = action_names.get(action_type, ("继续表达", "continues speaking"))[1]
    intent_names = {"state_position": "陈述立场", "explain": "解释说明", "give_example": "举例", "tell_story": "讲述经历", "emphasize": "强调重点", "respond_challenge": "回应质疑", "deflect": "过渡转移", "humor": "使用幽默", "ask": "提出问题", "transition": "过渡铺垫"}
    if lang == "en":
        return f"The speaker {english_action} while presenting the current point."[:120]
    return f"说话人{chinese_action}，{intent_names.get(str(label), '围绕当前话题推进')}。"[:32]


def _quote(text: str, subtitle: str) -> str:
    if not subtitle:
        return ""
    if text and text in subtitle:
        return text
    pieces = re.split(r"(?<=[。！？.!?；;])\s*", subtitle)
    return next((piece.strip() for piece in pieces if piece.strip()), subtitle[:80])


def _valid_line(line: str, lang: str) -> bool:
    if not line or _RED_FLAGS.search(line):
        return False
    if lang == "en":
        return len(line.split()) <= 16
    return len(re.sub(r"\s", "", line)) <= 32


def _parse(text: str) -> dict[str, str]:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {"line": "", "quote": ""}
    return {"line": str(parsed.get("line", "")), "quote": str(parsed.get("quote", ""))} if isinstance(parsed, Mapping) else {"line": "", "quote": ""}


def run(states: Mapping[str, Any] | None, judgments: Mapping[str, Any] | None, out: Path, force: bool = False, config: Mapping[str, Any] | None = None, lang: str = "zh", client: OpenRouterClient | None = None) -> dict[str, Any]:
    out = Path(out)
    destination = out / "narration.json"
    if destination.exists() and not force:
        meta_path = out / "narrate_meta.json"
        cached_lang = json.loads(meta_path.read_text(encoding="utf-8")).get("lang", lang) if meta_path.exists() else lang
        if cached_lang == lang:
            return json.loads(destination.read_text(encoding="utf-8"))
    if states is None:
        states = json.loads((out / "states.json").read_text(encoding="utf-8"))
    if judgments is None:
        judgments = json.loads((out / "judgments.json").read_text(encoding="utf-8"))
    settings = (config or {}).get("narrate", {})
    model = str(settings.get("model", "google/gemini-3.1-flash-lite")) if isinstance(settings, Mapping) else "google/gemini-3.1-flash-lite"
    template_path = Path(__file__).resolve().parents[3] / "config" / "narrate_prompt.md"
    template = template_path.read_text(encoding="utf-8")
    active_client = client or OpenRouterClient()
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "failed": 0}
    previous = ""
    for window_id, state in states.items():
        judgment = judgments.get(window_id, {}) if isinstance(judgments, Mapping) else {}
        subtitle = str(state.get("subtitle", {}).get("current", "")) if isinstance(state.get("subtitle"), Mapping) else ""
        prompt = template.format(state_json=json.dumps(state, ensure_ascii=False), judgment_json=json.dumps(judgment, ensure_ascii=False), previous_line=previous, language="English" if lang == "en" else "Chinese")
        stats["calls"] += 1
        try:
            response = active_client.chat(
                [{"role": "user", "content": prompt}], model,
                response_format={"type": "json_object"},
                max_tokens=int(settings.get("max_tokens", 120)) if isinstance(settings, Mapping) else 120,
                temperature=0.2,
            )
            raw = response.get("raw", response.get("response", response))
            (raw_dir / f"narrate_{window_id}.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
            parsed = _parse(_content(raw if isinstance(raw, Mapping) else {}))
            if not _valid_line(parsed["line"], lang):
                parsed["line"] = _fallback(state, judgment, lang)
            parsed["quote"] = _quote(parsed.get("quote", ""), subtitle)
            result[window_id] = parsed
            previous = parsed["line"]
            usage = response.get("usage", {}) if isinstance(response, Mapping) else {}
            usage = usage if isinstance(usage, Mapping) else {}
            stats["prompt_tokens"] += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
            stats["completion_tokens"] += int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
            if response.get("cost") is not None:
                stats["cost"] += float(response["cost"])
        except Exception as error:
            stats["failed"] += 1
            result[window_id] = {"line": _fallback(state, judgment, lang), "quote": _quote("", subtitle), "error": type(error).__name__}
            previous = result[window_id]["line"]
    if stats["cost"] == 0.0:
        stats["cost"] = None
    for value in result.values():
        Narration.model_validate(value)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "narrate_meta.json").write_text(json.dumps({**stats, "model": model, "lang": lang}, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


__all__ = ["run"]
