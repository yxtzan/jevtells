"""Factual highlights, exact measured gestures and every narration guardrail."""

import json
from pathlib import Path

import pytest

from jevtells.stages.narrate import run
from jevtells.stages.narrate_facts import build_facts, compute_highlights, gesture_facts
from jevtells.stages.narrate_validation import validation_errors
from jevtells.stages.speakers import mark_windows, parse_intervals


def judgment(value, intent="explain", emotion="calm"):
    return {"scores": {key: {"value": value} for key in ("confidence", "focus", "tension")}, "intent": {"label": intent, "confidence": 0.8}, "emotion": {"label": emotion, "confidence": 0.7}, "actions": {}}


def test_other_speaker_overlap_union_and_half_boundary():
    windows = [{"id": "W00", "t0": 0, "t1": 4}, {"id": "W01", "t0": 4, "t1": 8}]
    intervals = parse_intervals(["0-1.5", "1-2", "6.01-8"])
    assert intervals == [(0, 2), (6.01, 8)]
    marked = mark_windows(windows, intervals)
    assert marked[0]["speaker_other"] is True
    assert marked[1]["speaker_other"] is False
    assert "speaker_other" not in windows[0]


@pytest.mark.parametrize("value", ["2-1", "-1-2", "NaN-4", "1-inf", "1", "1-1"])
def test_invalid_other_speaker_ranges(value):
    with pytest.raises(ValueError):
        parse_intervals([value])


def test_highlights_extrema_delta_continuous_and_label_transition():
    windows = [{"id": f"W{i}"} for i in range(4)]
    scores = {"W0": judgment(.2), "W1": judgment(.28), "W2": judgment(.5, "emphasize", "firm"), "W3": judgment(.3)}
    facts = compute_highlights(scores, windows)
    assert "自信度 0.20，全场最低" in facts["W0"]
    assert "自信度比上一句 +0.08" in facts["W1"]
    assert "自信度 0.50，全场最高" in facts["W2"]
    assert "紧张度持续走高" in facts["W2"]
    assert "意图由「解释说明」转为「强调重点」" in facts["W2"]
    assert "情绪由「平静」转为「坚定」" in facts["W2"]
    assert "自信度回落" in facts["W3"]
    assert "自信度持续走低" not in facts["W3"]


def test_highlights_skip_other_speakers_missing_values_and_ties():
    windows = [{"id": "W0"}, {"id": "W1", "speaker_other": True}, {"id": "W2"}, {"id": "W3"}]
    scores = {"W0": judgment(.5), "W1": judgment(1.0, "ask"), "W2": judgment(.5), "W3": None}
    facts = compute_highlights(scores, windows)
    assert facts == {"W0": [], "W1": [], "W2": [], "W3": []}


def test_gestures_merge_only_overlapping_state_events_and_drop_small():
    events = [{"id": "a", "window": "W0", "limb": "left_hand", "type": "press_down", "t0": 1, "t1": 2, "magnitude": "large"}, {"id": "b", "window": "W0", "limb": "right_hand", "type": "press_down", "t0": 1.5, "t1": 2.5, "magnitude": "large"}, {"id": "c", "window": "W0", "limb": "left_hand", "type": "raise", "t0": 3, "t1": 4, "magnitude": "small"}, {"id": "d", "window": "W0", "limb": "right_hand", "type": "raise", "t0": 3, "t1": 4, "magnitude": "large"}]
    state = {"measured_actions": ["left_hand: press_down, 1.0s, large", "right_hand: press_down, 1.0s, large", "left_hand: raise, 1.0s, small"]}
    result = gesture_facts(state, events, "W0", {"actions": {"a": .73, "b": .76}})
    assert result == ["双手同时下压（幅度大，表达欲 0.73，表达欲 0.76）"]
    events[1]["t0"], events[1]["t1"] = 2, 3
    assert len(gesture_facts(state, events, "W0", {})) == 2


def test_facts_indices_are_one_based_and_other_highlights_are_empty():
    states = {"W0": {"subtitle": {"current": "hello world", "previous": ""}}, "W1": {"subtitle": {"current": "next phrase", "previous": "hello world"}}}
    windows = [{"id": "W0", "speaker_other": True}, {"id": "W1"}]
    result = build_facts(states, {"W0": judgment(1), "W1": judgment(.5)}, windows)
    assert result["W0"]["index"] == 1 and result["W1"]["total"] == 2
    assert result["W0"]["highlights"] == []
    assert result["W1"]["previous_subtitle"] == "hello world"


@pytest.mark.parametrize("line,quote,previous,highlights,lang,reason", [
    ("字" * 33, "hello world", [], [], "zh", "exceeds"),
    (" ".join(["word"] * 17), "hello world", [], [], "en", "exceeds"),
    ("继续表达。", "hello world", [], [], "zh", "period"),
    ("他说谎而且心虚", "hello world", [], [], "zh", "prohibited"),
    ("黄仁勋解释芯片", "hello world", [], [], "zh", "starts"),
    ("他解释芯片", "hello world", [], [], "zh", "starts"),
    ("她解释芯片", "hello world", [], [], "zh", "starts"),
    ("在白色背景前解释", "hello world", [], [], "zh", "starts"),
    ("The speaker explains", "hello world", [], [], "en", "starts"),
    ("说到「一」和「二」", "hello world", [], [], "zh", "matched pair"),
    ("说到「芯片", "hello world", [], [], "zh", "matched pair"),
    ("自信度全场最高", "hello world", [], [], "zh", "absent"),
    ("紧张度全场最低", "hello world", [], [], "zh", "absent"),
    ("专注度持续走高", "hello world", [], [], "zh", "absent"),
    ("紧张度回落", "hello world", [], [], "zh", "absent"),
    ("意图转为「强调重点」", "hello world", [], [], "zh", "absent"),
    ("专注度全场最高", "hello world", [], ["紧张度 0.9，全场最高"], "zh", "absent"),
    ("左手下压讲述内容", "hello world", ["左手下压回应问题"], [], "zh", "repeat"),
    ("继续表达", "different words", [], [], "zh", "substring"),
    ("继续表达", "hello", [], [], "zh", "2–8"),
    ("继续表达", "one two three four five six seven eight nine", [], [], "zh", "2–8"),
    ("Confidence is highest", "hello world", [], [], "en", "absent"),
])
def test_each_commentary_validation_rule(line, quote, previous, highlights, lang, reason):
    facts = {"subtitle": "hello world one two three four five six seven eight nine", "highlights": highlights}
    errors = validation_errors({"line": line, "quote": quote}, facts, previous, "黄仁勋", lang)
    assert any(reason in error for error in errors), errors


def test_grounded_superlative_and_transition_pass():
    facts = {"subtitle": "hello world", "highlights": ["自信度 0.92，全场最高", "意图由「解释说明」转为「强调重点」"]}
    for line in ("双手下压，自信度升至全场最高", "意图转为「强调重点」", "意图转为强调重点"):
        assert validation_errors({"line": line, "quote": "hello world"}, facts, [], "黄仁勋", "zh") == []


def test_invalid_commentary_retries_twice_then_records_fallback(tmp_path):
    class BadClient:
        calls = 0
        def chat(self, messages, model, **kwargs):
            self.calls += 1
            if self.calls > 1:
                assert "Validation failed" in messages[0]["content"]
            return {"raw": {"choices": [{"message": {"content": '{"line":"他说谎","quote":"invented words"}'}}]}, "usage": {"prompt_tokens": 5, "completion_tokens": 4}, "cost": .001}
    client = BadClient()
    states = {"W0": {"speaker": "Demo", "subtitle": {"current": "hello world", "previous": ""}, "measured_actions": []}}
    output = run(states, {"W0": judgment(.5)}, tmp_path, client=client)
    metadata = json.loads((tmp_path / "narrate_meta.json").read_text())
    assert client.calls == 3
    assert metadata["retries"] == 2 and metadata["fallbacks"] == 1
    assert metadata["cost"] == .003 and metadata["prompt_tokens"] == 15
    assert validation_errors(output["W0"], json.loads((tmp_path / "narrate_facts.json").read_text())["W0"], [], "Demo", "zh") == []


def test_other_speaker_never_calls_judge_or_narrator_and_stays_null(tmp_path):
    from jevtells.stages.judge import run as judge_run
    class NoCalls:
        def decide(self, *args, **kwargs):
            pytest.fail("other speaker reached Jev")
        def chat(self, *args, **kwargs):
            pytest.fail("other speaker reached narrator")
    (tmp_path / "windows.json").write_text(json.dumps([{"id": "W00", "speaker_other": True}]))
    states = {"W00": {"subtitle": {"current": "other speaker", "previous": ""}, "measured_actions": []}}
    judgments = judge_run(states, tmp_path, client=NoCalls())
    narration = run(states, judgments, tmp_path, client=NoCalls())
    assert judgments["W00"] is narration["W00"] is None
    assert json.loads((tmp_path / "judge_meta.json").read_text())["calls"] == 0
    assert json.loads((tmp_path / "narrate_meta.json").read_text())["calls"] == 0
    assert "W00" in json.loads((tmp_path / "narrate_meta.json").read_text())["skipped"]


def test_wrong_metric_extreme_and_wrong_emotion_transition_are_rejected():
    facts = {"subtitle": "hello world", "highlights": ["紧张度 0.90，全场最高", "情绪由「兴奋」转为「坚定」"]}
    for line in ["专注度达到全场最高", "情绪转为平静", "意图转为坚定"]:
        assert validation_errors({"line": line, "quote": "hello world"}, facts, [], "Demo", "zh")
    assert validation_errors({"line": "情绪转为坚定", "quote": "hello world"}, facts, [], "Demo", "zh") == []


def test_explicit_api_failure_stops_and_preserves_accounting(tmp_path):
    from jevtells.stages.judge import run as judge_run
    class Failed:
        def decide(self, *args, **kwargs):
            raise TimeoutError("offline")
    with pytest.raises(TimeoutError):
        judge_run({"W0": {"measured_actions": []}, "W1": {"measured_actions": []}}, tmp_path, config={"stop_on_api_error": True}, client=Failed())
    assert json.loads((tmp_path / "judge_meta.json").read_text())["calls"] == 1


def test_english_trend_direction_and_transition_label_are_grounded():
    facts = {"subtitle": "hello world", "highlights": ["专注度持续走高", "情绪由「兴奋」转为「坚定」"]}
    for line in ["Focus continues falling", "Emotion shifts to calm"]:
        assert validation_errors({"line": line, "quote": "hello world"}, facts, [], "Demo", "en")
    for line in ["Focus continues rising", "Emotion shifts to firm"]:
        assert validation_errors({"line": line, "quote": "hello world"}, facts, [], "Demo", "en") == []
