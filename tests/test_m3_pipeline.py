"""Offline coverage for M3 normalization, narration safeguards, and i18n."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
import pytest

from jevtells.i18n import load_translations
from jevtells.stages.judge import normalise_judgment
from jevtells.stages.narrate import run as narrate


def _keys(value, prefix=""):
    result = set()
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.add(path)
            result |= _keys(child, path)
    return result


def test_i18n_keys_match_and_cover_jev_options():
    zh = yaml.safe_load(Path("i18n/zh.yaml").read_text(encoding="utf-8"))
    en = yaml.safe_load(Path("i18n/en.yaml").read_text(encoding="utf-8"))
    assert _keys(zh) == _keys(en)
    questions = yaml.safe_load(Path("config/jev_questions.yaml").read_text(encoding="utf-8"))["questions"]
    translations = load_translations("zh")
    for key in questions["intent"]["criteria"]:
        assert key in translations["intents"]
    for key in questions["emotion"]["criteria"]:
        assert key in translations["emotions"]


def test_score_normalization_uses_actual_legend_and_noul():
    raw = {
        "answers": {
            "confidence": {"type": "score", "score": 3.5, "legend": {"0": "very_low", "4": "very_high"}},
            "focus": {"type": "score", "score": 0.75},
            "tension": {"type": "score", "score": "high", "legend": {"0": "very_low", "1": "low", "2": "medium", "3": "high", "4": "very_high"}},
            "intent": {"type": "choice", "choice": "explain", "confidence": 0.8, "probabilities": {"explain": 0.8}},
            "emotion": {"type": "choice", "choice": "calm", "confidence": 0.7},
            "action_0_expressive": {"type": "noul", "noul": 0.63},
        }
    }
    result = normalise_judgment(raw, {"action_0_expressive": {}})
    assert result["scores"]["confidence"]["value"] == 0.875
    assert result["scores"]["focus"]["value"] == 0.75
    assert result["scores"]["tension"]["value"] == 0.75
    assert result["actions"]["action_0_expressive"] == 0.63


class _BadNarrator:
    def chat(self, messages, model, **kwargs):
        return {"raw": {"choices": [{"message": {"content": json.dumps({"line": "他说谎而且很心虚。", "quote": "不存在"}, ensure_ascii=False)}}]}, "usage": {}, "cost": 0.0}


def test_narration_redline_and_quote_fallback(tmp_path):
    states = {"W00": {"scene": "office", "speaker": "Speaker", "subtitle": {"current": "第一句。第二句", "previous": ""}, "voice": "medium", "measured_actions": []}}
    judgments = {"W00": {"scores": {}, "intent": {"label": "explain", "confidence": 0.8}, "emotion": {"label": "calm", "confidence": 0.7}, "actions": {}}}
    result = narrate(states, judgments, tmp_path, force=True, config={"narrate": {"model": "test/model"}}, lang="zh", client=_BadNarrator())
    assert "撒谎" not in result["W00"]["line"]
    assert result["W00"]["quote"] == "第一句。"


@pytest.mark.parametrize("raw_score,expected", [(0, 0.0), (0.5, 0.125), (1, 0.25), (3.5, 0.875), (4, 1.0), ("low", 0.25), ("very_high", 1.0)])
def test_legend_normalization_covers_low_scores(raw_score, expected):
    raw = {"answers": {"tension": {"score": raw_score, "legend": {"0": "very_low", "1": "low", "2": "medium", "3": "high", "4": "very_high"}}}}
    value = normalise_judgment(raw, {})["scores"]["tension"]
    assert value == {"value": expected, "raw": raw_score}


def test_editable_action_questions_reference_measured_actions():
    from jevtells.stages.judge import _actions_questions, load_questions
    questions = _actions_questions({"measured_actions": ["left_hand: raise, 0.4s, medium"]}, load_questions())
    assert "action_expressive" not in questions
    assert questions["action_0_expressive"]["type"] == "noul"
    assert "`measured_actions[0]`" in questions["action_0_expressive"]["instructions"]
    assert questions["action_0_expressive"]["criteria"] == {"true": "yes", "false": "no"}
    assert not any(key.startswith("action_") for key in _actions_questions({"measured_actions": ["no notable gestures"]}, load_questions()))


def test_judgment_expressivity_uses_action_event_ids(tmp_path):
    from jevtells.stages.judge import run
    (tmp_path / "actions.json").write_text(json.dumps([{"id": "A014", "window": "W00", "limb": "left_hand", "type": "raise", "t0": 0.2, "t1": 0.6, "magnitude": "medium"}]))
    class Client:
        def decide(self, state, questions, model):
            return {"raw": {"answers": {"action_0_expressive": {"noul": 0.63}}}, "usage": {}, "cost": 0.001}
    states = {"W00": {"measured_actions": ["left_hand: raise, 0.4s, medium"]}}
    result = run(states, tmp_path, client=Client())
    assert result["W00"]["actions"] == {"A014": 0.63}
    assert json.loads((tmp_path / "judgments.json").read_text())["W00"]["actions"] == {"A014": 0.63}


def test_failed_calls_are_counted_and_windows_remain_nullable(tmp_path):
    from jevtells.stages.judge import run
    class Client:
        def decide(self, *args, **kwargs):
            raise TimeoutError("provider unavailable")
        def chat(self, *args, **kwargs):
            raise TimeoutError("provider unavailable")
    states = {"W00": {"subtitle": {"current": "Hello.", "previous": ""}, "measured_actions": []}}
    judgments = run(states, tmp_path, client=Client())
    assert judgments["W00"]["scores"] == {"confidence": None, "focus": None, "tension": None}
    assert judgments["_meta"]["calls"] == judgments["_meta"]["failed"] == 1
    narrate(states, judgments, tmp_path, client=Client())
    meta = json.loads((tmp_path / "narrate_meta.json").read_text())
    assert meta["calls"] == meta["failed"] == 1


def test_scene_cache_retains_complete_api_accounting(tmp_path, monkeypatch):
    from jevtells.stages import scene
    monkeypatch.setattr(scene, "_middle_frame", lambda clip: b"image")
    class Client:
        calls = 0
        def chat(self, messages, model, **kwargs):
            self.calls += 1
            return {"raw": {"choices": [{"message": {"content": "An indoor room with a fixed camera."}}]}, "usage": {"prompt_tokens": 10, "completion_tokens": 7}, "cost": 0.002}
    client = Client()
    first = scene.run(Path("clip.mp4"), tmp_path, client=client)
    second = scene.run(Path("clip.mp4"), tmp_path, client=client)
    assert first == second
    assert client.calls == 1
    assert (second["calls"], second["prompt_tokens"], second["completion_tokens"], second["cost"]) == (1, 10, 7, 0.002)


def test_cli_cache_from_stage_and_language_do_not_repeat_unrelated_api_calls(tmp_path, monkeypatch):
    from argparse import Namespace
    import numpy as np
    from jevtells import cli
    from jevtells.stages import judge, scene
    from jevtells.stages import narrate as narration_stage
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "clip.mov"
    input_path.write_bytes(b"clip")
    args = Namespace(command="run", input=str(input_path), output=None, speaker="Demo", scene=None, lang="zh", srt=None, start=0.0, duration=None, force=False, until="narrate", from_stage=None, target=["1@0"], config=None)
    monkeypatch.setattr(cli, "_arguments", lambda: args)
    monkeypatch.setattr(cli.prepare, "run", lambda *a, **kw: (input_path, input_path, {"encoder": "libx264"}))
    monkeypatch.setattr(cli, "_run_pose", lambda *a, **kw: {})
    def tracked(detections, output, anchors, force, config):
        (output / "track_meta.json").write_text(json.dumps({"anchors": anchors, "lost_frames": 0}))
        return {"t": [], "pose": np.empty((0, 33, 4))}
    monkeypatch.setattr(cli, "_run_track", tracked)
    monkeypatch.setattr(cli.asr, "run", lambda *a, **kw: {"language": "en", "segments": []})
    monkeypatch.setattr(cli.voice, "run", lambda *a, **kw: {})
    monkeypatch.setattr(cli.shots, "run", lambda *a, **kw: [])
    monkeypatch.setattr(cli.segment, "run", lambda *a, **kw: [{"id": "W00", "t0": 0.0, "t1": 2.0, "subtitle": "Hello."}])
    monkeypatch.setattr(cli, "_run_actions", lambda *a, **kw: {"events": []})
    monkeypatch.setattr(cli.debug, "run", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_video_duration", lambda *a: 0.0)
    monkeypatch.setattr(scene, "_middle_frame", lambda clip: b"image")
    calls = []
    class Client:
        def decide(self, state, questions, model):
            calls.append("judge")
            return {"raw": {"answers": {}}, "usage": {"input_tokens": 4, "output_tokens": 3}, "cost": 0.001}
        def chat(self, messages, model, **kwargs):
            stage = "scene" if kwargs.get("images") else "narrate"
            calls.append(stage)
            text = "An indoor room with a fixed camera." if stage == "scene" else json.dumps({"line": "The speaker explains the current point." if "Language: English" in messages[0]["content"] else "说话人解释当前话题。", "quote": "Hello."}, ensure_ascii=False)
            return {"raw": {"choices": [{"message": {"content": text}}]}, "usage": {"prompt_tokens": 10, "completion_tokens": 7}, "cost": 0.002}
    for module in (judge, scene, narration_stage):
        monkeypatch.setattr(module, "OpenRouterClient", Client)
    cli.main()
    assert calls == ["scene", "judge", "narrate"]
    cli.main()
    assert calls == ["scene", "judge", "narrate"]
    args.from_stage = "judge"
    cli.main()
    assert calls == ["scene", "judge", "narrate", "judge", "narrate"]
    args.from_stage = None
    args.lang = "en"
    cli.main()
    assert calls == ["scene", "judge", "narrate", "judge", "narrate", "narrate"]
    folder = tmp_path / "work" / "clip_s0_dauto"
    assert json.loads((folder / "narration.json").read_text())["W00"]["line"].startswith("The speaker")
    meta = json.loads((folder / "run_meta.json").read_text())["api"]
    assert meta["scene"]["calls"] == 1
    assert meta["scene"]["prompt_tokens"] == 10
    args.until = "debug"
    args.force = True
    cli.main()
    assert calls == ["scene", "judge", "narrate", "judge", "narrate", "narrate", "scene"]
    args.until = "judge"
    args.force = False
    args.from_stage = "judge"
    cli.main()
    assert calls[-1] == "judge" and len(calls) == 8
    final_meta = json.loads((folder / "run_meta.json").read_text())
    assert final_meta["parameters"]["until"] == "judge"
    assert final_meta["api"]["judge"]["calls"] == 1
