"""Deterministic checks for factual, short, varied on-screen commentary."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from ..i18n import load_translations


RED_FLAGS = re.compile(r"撒谎|说谎|心虚|装|骗子|欺骗|心理|道德|动机|身份|私生活|\blie\b|lying|liar|deceiv|guilty|dishonest|motive|psycholog|identity", re.IGNORECASE)


def quote_tokens(text: str) -> list[str]:
    """English words; unspaced CJK uses one character per token."""
    return re.findall(r"[a-zA-Z0-9]+(?:['’\-][a-zA-Z0-9]+)*|[\u3400-\u9fff]", text)


def display_names() -> set[str]:
    names = set()
    for language in ("zh", "en"):
        tr = load_translations(language)
        for category in ("intents", "emotions", "scores"):
            names.update(str(value).casefold() for value in tr[category].values())
        names.update(str(value).casefold() for value in tr["facts"]["scores"].values())
    return names | {"意图转变", "情绪转变", "指标变化", "数据变化", "intent transition", "emotion transition", "metric change"}


def highlight_claims(line: str) -> set[str]:
    """Normalize metric trends/extremes and label transitions for adjacency."""
    claims = set()
    zh, en = load_translations("zh"), load_translations("en")
    for language, tr in (("zh", zh), ("en", en)):
        for name in tr["facts"]["scores"].values():
            prefix = re.escape(name) + (r"\s*" if language == "en" else "")
            for match in re.finditer(prefix + r"(?:与(?:自信度|专注度|紧张度))?(?:降至|升至|达到|为|处于)?(?:全场)?(?:最高|最低|持续走高|持续走低|明显上升|明显下降|有所上升|有所下降|continues rising|continues falling|reaches the highest level|reaches the lowest level|rises markedly|falls markedly|rises somewhat|falls somewhat)", line, re.I):
                wording = match[0]
                trend = next((term for term in ("最高", "最低", "持续走高", "持续走低", "明显上升", "明显下降", "有所上升", "有所下降") if term in wording), wording.casefold())
                claims.add(name.casefold() + ":" + trend)
        for category, group in (("意图", "intents"), ("情绪", "emotions")):
            for label in tr[group].values():
                pattern = re.escape(category) + r"转为\s*" + re.escape(label) if language == "zh" else r"(?:shifts|turns) to\s+" + re.escape(label)
                if re.search(pattern, line, re.I):
                    claims.add(category + ":" + label.casefold())
    return claims


def repeated_highlights(line: str, previous: str) -> bool:
    return bool(highlight_claims(line) & highlight_claims(previous))


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
    # Preserve the factual movement constraint: a bilateral action needs
    # the measured, overlapping pair, not two different single-hand events.
    if lang == "zh" and "gestures" in facts:
        for match in re.finditer(r"双(?:手|掌)(?:同时)?(?:向下)?(下压|抬起|抬手|张开)", line):
            action = {"抬起": "抬手", "抬手": "抬手", "下压": "下压", "张开": "张开手掌"}[match[1]]
            if not any("双手同时" + action in str(gesture) for gesture in facts["gestures"]):
                errors.append("bilateral movement absent from facts.gestures")
    if line.startswith(("他", "她", "在")) or (speaker and line.casefold().startswith(speaker.casefold())) or re.match(r"^(he|she|they|the speaker|in|at)\b", line, re.IGNORECASE):
        errors.append("line starts with the name, a pronoun or a scene description")
    if line.count("「") > 1 or line.count("」") > 1 or line.count("「") != line.count("」"):
        errors.append("line must contain at most one matched pair of 「」")
    names = display_names()
    for phrase in re.findall(r"「([^」]*)」", line):
        if phrase.strip().casefold() in names:
            errors.append("corner quote contains a display label or summary term; use a translated subtitle phrase")
        if lang == "zh" and (not re.search(r"[\u3400-\u9fff]", phrase) or re.search(r"[A-Za-z]", phrase)):
            errors.append("corner quote must be a Chinese translation of the current subtitle phrase")
    if any(line.casefold().startswith(name) for name in names if name in {
            value.casefold() for language in ("zh", "en") for group in ("intents", "emotions")
            for value in load_translations(language)[group].values()}):
        errors.append("line starts with an intent or emotion display label")
    if previous and repeated_highlights(line, previous[-1]):
        errors.append("highlight claim repeats the adjacent previous line")
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
