"""Deterministic checks for factual, short, varied on-screen commentary."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from ..i18n import load_translations


RED_FLAGS = re.compile(r"撒谎|说谎|心虚|装|骗子|欺骗|心理|道德|动机|身份|私生活|\blie\b|lying|liar|deceiv|guilty|dishonest|motive|psycholog|identity", re.IGNORECASE)


def quote_tokens(text: str) -> list[str]:
    """English words; unspaced CJK uses one character per token."""
    return re.findall(r"[a-zA-Z0-9]+(?:['’\-][a-zA-Z0-9]+)*|[\u3400-\u9fff]", text)


def validation_errors(parsed: Mapping[str, Any], facts: Mapping[str, Any], previous: Sequence[str], speaker: str, lang: str, *, allow_repeated_opening: bool = False) -> list[str]:
    line, quote = str(parsed.get("line", "")).strip(), str(parsed.get("quote", "")).strip()
    errors: list[str] = []
    if not line or (len(line) > 32 if lang == "zh" else len(line.split()) > 16):
        errors.append(f"line exceeds its limit: {len(line)} characters / {len(line.split())} words. Chinese maximum is 32 TOTAL characters, including spaces, Latin letters and brackets; English maximum is 16 words. Use 20–28 Chinese characters, translate the phrase inside 「」 into short Chinese, keep quote in the original language")
    if re.search(r"[0-9]", line):
        errors.append("line must not contain Arabic digits")
    if line.endswith(("。", ".")) or "\n" in line:
        errors.append("one sentence without a final period is required")
    if RED_FLAGS.search(line):
        errors.append("line contains a prohibited tone or private/mental-state claim")
    if line.startswith(("他", "她", "在")) or (speaker and line.casefold().startswith(speaker.casefold())) or re.match(r"^(he|she|they|the speaker|in|at)\b", line, re.IGNORECASE):
        errors.append("line starts with the name, a pronoun or a scene description")
    if line.count("「") > 1 or line.count("」") > 1 or line.count("「") != line.count("」"):
        errors.append("line must contain at most one matched pair of 「」")
    highlights = [str(value).replace(" ", "") for value in facts.get("highlights", [])]
    compact = line.replace(" ", "")
    for match in re.finditer(r"最高|最低|持续|回落|转为", compact):
        marker = match[0]
        context = compact[max(0, match.start() - 20):match.start()]
        metrics = re.findall(r"自信度?|专注度?|紧张度?", context)
        metric = metrics[-1] if metrics else None
        if metric and not metric.endswith("度"):
            metric += "度"
        if marker in {"最高", "最低", "回落"}:
            required = "全场" + marker if marker != "回落" else marker
            if not any(required in highlight and (metric is None or metric in highlight) for highlight in highlights):
                errors.append(f"claim absent from facts.highlights: {metric or ''}{required}")
        elif marker == "持续":
            trend = re.match(r"持续(?:走高|走低)", compact[match.start():])
            if not trend or not any(trend[0] in highlight and (metric is None or metric in highlight) for highlight in highlights):
                errors.append("continuous trend absent from facts.highlights")
        else:
            # Corner brackets are optional wording; the category and new
            # label must match the code-proven transition, including when
            # the line omits the old label to stay under 32 characters.
            categories = re.findall(r"意图|情绪", context)
            category = categories[-1] if categories else None
            suffix = compact[match.end():].lstrip("「")
            proven = [item for item in highlights if "转为「" in item and (category is None or item.startswith(category))]
            if not any(suffix.startswith(item.split("转为「", 1)[1].split("」", 1)[0]) for item in proven):
                errors.append("transition absent from facts.highlights")
    if lang == "en":
        # The highlights are Chinese facts, so map their exact claim types.
        names = {"confidence": "自信度", "focus": "专注度", "tension": "紧张度"}
        pattern = r"\b(?:highest|lowest)\b|continues? (?:rising|falling)|steadily (?:rises?|falls?)|(?:falls?|drops?) back|(?:shifts?|turns?) to"
        en, zh = load_translations("en"), load_translations("zh")
        for match in re.finditer(pattern, line, re.IGNORECASE):
            phrase = match[0].lower()
            metric_names = re.findall(r"confidence|focus|tension", line[:match.start()].lower())
            metric = names.get(metric_names[-1]) if metric_names else None
            if phrase.endswith(" to"):
                suffix = line[match.end():].strip(' 「"').lower()
                categories = re.findall(r"intent|emotion", line[:match.start()].lower())
                category = {"intent": "意图", "emotion": "情绪"}.get(categories[-1]) if categories else None
                found = False
                for highlight in highlights:
                    if "转为「" not in highlight or (category and not highlight.startswith(category)):
                        continue
                    target = highlight.split("转为「", 1)[1].split("」", 1)[0]
                    group = "intents" if highlight.startswith("意图") else "emotions"
                    labels = [en[group][key].lower() for key, value in zh[group].items() if value == target]
                    found = found or any(suffix.startswith(label) for label in labels)
                if not found:
                    errors.append("English transition absent from facts.highlights")
                continue
            required = "全场最高" if phrase == "highest" else "全场最低" if phrase == "lowest" else "回落" if phrase.endswith(" back") else "持续走低" if "fall" in phrase else "持续走高"
            if not any(required in item and (metric is None or metric in item) for item in highlights):
                errors.append("English trend/extreme absent from facts.highlights")
    if not allow_repeated_opening and any(line[:4] == old[:4] for old in previous):
        errors.append(f"first four characters repeat an earlier line: {line[:4]!r}. Change the opening, still using only the given facts")
    if lang == "zh":
        if not allow_repeated_opening and sum(old[:2] == line[:2] for old in previous) >= 2:
            errors.append("first two characters may appear at most twice across the video")
        before_quote = re.search(r"(.{2})「", line)
        if before_quote:
            opening = before_quote[1]
            earlier = [re.search(r"(.{2})「", old) for old in previous]
            if sum(match is not None and match[1] == opening for match in earlier) >= 2:
                errors.append("two characters before 「 may appear at most twice across the video")
            if earlier and earlier[-1] and earlier[-1][1] == opening:
                errors.append("two characters before 「 must differ in adjacent lines")
    subtitle = str(facts.get("subtitle", ""))
    if not quote or quote not in subtitle or not 2 <= len(quote_tokens(quote)) <= 8:
        errors.append("quote must be an exact subtitle substring with 2–8 tokens")
    return errors
