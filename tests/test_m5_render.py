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
