import copy

import numpy as np
import pytest
from PIL import Image

from jevtells.config import load_config
from jevtells.render.animation import commentary_at
from jevtells.render.geometry import Layout
from jevtells.render.renderer import Composer


def test_footer_short_names_match_actual_version_and_strip_unknown_date():
    from jevtells.render.renderer import model_short_name
    settings = load_config()["render"]
    assert model_short_name("typesafe/jev-1.13-20260917", settings) == "Jev 1.13"
    assert model_short_name("google/gemini-3.1-flash-lite", settings) == "Gemini Flash Lite"
    assert model_short_name("vendor/model-20261005", settings) == "model"
    assert model_short_name("vendor/model-2026-10-05", settings) == "model"
    assert model_short_name("vendor/model-v2", settings) == "model-v2"


def test_commentary_height_is_measured_once_for_the_whole_video():
    from jevtells.i18n import load_translations
    from jevtells.render.panels import Panels
    from jevtells.render.text import Fonts
    settings = load_config()["render"]
    layout = Layout.create("v", settings, (1280, 720))
    windows = [{"id": "W0"}, {"id": "W1"}]
    def panels(lines):
        return Panels(settings, layout, Fonts(settings, 1), load_translations("zh"), windows, {}, {f"W{i}": {"line": line, "quote": "hello world"} for i, line in enumerate(lines)}, "标题", "来源")
    short = panels(["双手收拢", "语速改变"])
    mixed = panels(["双手收拢", "复杂计算架构中的每一个层级都经历了彻底革新并重新构建和改变自身工作方式"])
    assert short.p["commentary"][3] < mixed.p["commentary"][3] <= 98
    assert short.p["quote_y"] < mixed.p["quote_y"]
    assert mixed.commentary(0)[0].height == mixed.commentary(1)[0].height
    assert mixed.quote(0).getbbox()[1] == mixed.quote(1).getbbox()[1]
    assert settings["v"]["commentary"][3] == 98


def test_cut_clears_old_labels_and_card_moves_only_at_boundary():
    config=load_config();settings=config["render"]
    points={"pose":np.tile(np.array([.5,.5,0,1]),(90,33,1)),"hands":np.full((90,2,21,3),np.nan),"fps":30}
    shots=[{"index":1,"t0":0,"t1":1,"label":"target","far":False,"target_box":[100,100,500,700]}, {"index":2,"t0":1,"t1":3,"label":"target","far":False,"target_box":[750,100,1130,700]}]
    actions=[{"id":"old","t0":.1,"t1":.8,"type":"raise","limb":"right_hand","magnitude":"large","shot_index":1},{"id":"new","t0":1,"t1":2,"type":"raise","limb":"left_hand","magnitude":"large","shot_index":2}]
    painter=Composer(settings,Layout.create("h",settings,(1280,720)),[{"id":"W0","t0":0,"t1":3}],points,actions,{}, {},{},title="Title",sources="Source",lang="en",blur=[],subtitles=False,config=config,shots=shots)
    assert painter.card_x(.9)==968
    assert painter.card_x(1)==968
    assert painter.card_x(1.2)==pytest.approx(500)
    assert painter.card_x(1.4)==pytest.approx(32)
    assert painter.card_x(2.5)==32
    source=Image.new("RGB",(1280,720))
    painter.frame(source,1.2,36)
    assert painter.audit["labels"]==[]
    painter.frame(source,1.5,45)
    assert [label["event"] for label in painter.audit["labels"]]==["new"]


def test_run_defaults_to_render_and_state_remains_available(monkeypatch, capsys):
    import sys
    from jevtells.cli import _arguments
    base = ["jevtells", "run", "sample.mp4", "--speaker", "Demo"]
    monkeypatch.setattr(sys, "argv", base)
    assert _arguments().until == "render"
    monkeypatch.setattr(sys, "argv", base + ["--until", "state"])
    assert _arguments().until == "state"
    monkeypatch.setattr(sys, "argv", ["jevtells", "run", "--help"])
    with pytest.raises(SystemExit):
        _arguments()
    assert "default: render" in capsys.readouterr().out


def test_render_signature_changes_when_source_changes(tmp_path, monkeypatch):
    from pathlib import Path
    import jevtells.render.renderer as renderer
    source = tmp_path / "renderer.py"
    source.write_text("initial code")
    monkeypatch.setattr(renderer, "__file__", str(source))
    first = renderer._signature(tmp_path, {"title": "same"})
    assert renderer._signature(tmp_path, {"title": "same"}) == first
    source.write_text("changed code")
    assert renderer._signature(tmp_path, {"title": "same"}) != first
    first = renderer._signature(tmp_path, {"title": "same"})
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/geometry.py").write_text("new source")
    assert renderer._signature(tmp_path, {"title": "same"}) != first


@pytest.mark.parametrize("kind", ["h", "v"])
def test_commentary_and_quote_switch_together_without_overlap(kind, monkeypatch):
    config = load_config()
    settings = copy.deepcopy(config["render"])
    layout = Layout.create(kind, settings, (1280, 720))
    windows = [{"id": "W0", "t0": 0, "t1": 2}, {"id": "W1", "t0": 2, "t1": 4}]
    points = {"pose": np.full((1, 33, 4), np.nan), "hands": np.full((1, 2, 21, 3), np.nan)}
    painter = Composer(settings, layout, windows, points, [], {}, {}, {}, title="Title", sources="Source", lang="en", blur=[], subtitles=False, config=config)
    seen = []
    def sentence(index):
        seen.append(("line", index))
        return Image.new("RGBA", (2, 2), (255, 255, 255, 255)), (0, 0)
    def quote(index):
        seen.append(("quote", index))
        return Image.new("RGBA", layout.output)
    monkeypatch.setattr(painter.panels, "commentary", sentence)
    monkeypatch.setattr(painter.panels, "quote", quote)
    for elapsed, expected in [(0, 0), (.1, 0), (.15, 1), (.2, 1), (.4, 1)]:
        seen.clear()
        painter.frame(Image.new("RGB", layout.source), 2 + elapsed, 0)
        assert seen == [("quote", expected), ("line", expected)]
    assert commentary_at(1, .15, settings["animation"]) == (1, 0, 12)
    assert commentary_at(1, .4, settings["animation"]) == (1, 1, 0)
    # Examine both sides of each boundary as well as sub-frame timestamps.
    for elapsed in np.linspace(0, .5, 501):
        index, alpha, offset = commentary_at(1, elapsed, settings["animation"])
        assert 0 <= alpha <= 1
        if index == 0:
            assert elapsed < .15 and offset == 0
        else:
            assert elapsed >= .15
