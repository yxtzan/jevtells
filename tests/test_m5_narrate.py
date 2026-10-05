import math

from jevtells.stages.narrate_facts import build_facts, compute_highlights
from jevtells.stages.narrate_validation import validation_errors


def test_extrema_budget_keeps_farthest_from_measured_mean():
    windows = [{"id": str(i)} for i in range(7)]
    values = [.1, .5, .5, .5, .5, .5, .9]
    judgments = {str(i): {"scores": {"confidence": {"value": value}}} for i, value in enumerate(values)}
    facts = compute_highlights(judgments, windows)
    extremes = [(identifier, line) for identifier, items in facts.items() for line in items if "全场" in line]
    assert len(extremes) <= math.ceil(len(windows) / 3)
    assert {identifier for identifier, line in extremes} == {"0", "6"}
    assert "自信度比上一句 +0.40" in facts["1"]


def test_two_character_openings_and_quote_verbs_are_limited():
    facts = {"subtitle": "hello world", "highlights": []}
    def errors(line, previous):
        return validation_errors({"line": line, "quote": "hello world"}, facts, previous, "Demo", "zh")
    assert any("first two" in e for e in errors("双手收拢继续", ["双手下压回应", "双手展开强调"]))
    assert any("adjacent" in e for e in errors("指尖变化，阐述「核心」", ["掌面向外，阐述「变化」"]))
    assert any("at most twice" in e for e in errors("指尖变化，强调「核心」", ["掌面向外，强调「变化」", "抬手强调「细节」", "动作收束"]))
    assert errors("指尖变化，提及「核心」", ["掌面向外，强调「变化」"]) == []


def test_opening_suggestions_alternate_for_active_windows():
    windows = [{"id": str(i), "speaker_other": i == 0} for i in range(6)]
    states = {str(i): {"subtitle": {"current": "hello world"}} for i in range(6)}
    facts = build_facts(states, {}, windows)
    styles = [facts[str(i)]["opening_style"] for i in range(1, 6)]
    assert all(a != b for a, b in zip(styles, styles[1:]))
    assert len(set(styles)) == 4
