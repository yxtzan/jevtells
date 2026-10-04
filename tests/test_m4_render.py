"""Render geometry, animation, label scheduling and an actual 3-second encode."""

import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from jevtells.config import load_config
from jevtells.render.animation import ease_out_cubic, interpolate, window_at
from jevtells.render.geometry import Layout, parse_blurs
from jevtells.render.labels import schedule, visibility
from jevtells.render.renderer import Composer, blur_frame, run
from jevtells.render.text import Fonts, draw_fitted


def test_layout_scaling_and_source_blur_coordinates():
    settings = load_config()["render"]
    h = Layout.create("h", settings, (1280, 720))
    v = Layout.create("v", settings, (1280, 720))
    assert h.scale == 1.5 and v.scale == 1
    assert h.source_rect((1042, 15, 226, 60)) == (1563, 22, 1902, 112)
    assert v.source_rect((1042, 15, 226, 60)) == (879, 209, 1070, 259)
    assert v.source_point(640, 360) == (540, 500)
    small = copy.deepcopy(settings)
    small["v"]["output"] = [540, 720]
    assert Layout.create("v", small, (1280, 720)).source_point(640, 360) == (270, 250)
    small["v"]["output"] = [600, 720]
    with pytest.raises(ValueError):
        Layout.create("v", small, (1280, 720))


@pytest.mark.parametrize("value", ["1,2,3", "-1,2,3,4", "1,2,0,4", "NaN,2,3,4"])
def test_invalid_blur_rectangles(value):
    with pytest.raises(ValueError):
        parse_blurs([value])


def test_cubic_easing_interpolation_and_window_gaps():
    assert ease_out_cubic(-1) == 0 and ease_out_cubic(2) == 1
    assert ease_out_cubic(.5) == .875
    assert interpolate(.2, .6, .875) == pytest.approx(.55)
    assert interpolate(.2, None, .5) is None
    windows = [{"t0": 1, "t1": 2}, {"t0": 4, "t1": 5}]
    assert window_at(windows, .5) == (None, False)
    assert window_at(windows, 3) == (0, True)
    assert window_at(windows, 4) == (1, False)


def event(identifier="a", limb="right_hand", kind="raise", start=1, end=2, magnitude="large"):
    return {"id": identifier, "limb": limb, "type": kind, "t0": start, "t1": end, "magnitude": magnitude}


def test_label_fade_times_hold_and_scale():
    animation = load_config()["render"]["animation"]
    item = event()
    assert visibility(item, .9, animation) == (0, 1)
    assert visibility(item, 1, animation) == (0, .9)
    assert visibility(item, 1.075, animation) == pytest.approx((.875, .9875))
    assert visibility(item, 2.79, animation) == (1, 1)
    assert visibility(item, 2.9, animation)[0] == pytest.approx(.5)
    assert visibility(item, 3, animation)[0] == 0


def test_label_scheduler_one_per_side_and_filters():
    settings = load_config()["render"]
    events = [event(), event("b", start=1.2, end=2.5), event("c", "left_hand"), event("s", "left_hand", magnitude="small"), event("f", "right_hand", "fist")]
    labels = schedule(events, 1.5, settings)
    assert set(labels) == {"left", "right"}
    assert labels["left"][0]["id"] == "b"
    assert labels["right"][0]["id"] == "c"
    assert schedule(events, 1.5, settings, lost=True) == {}
    assert schedule(events, 1.5, settings, shot="other") == {}
    assert set(schedule([event(limb="both_hands", kind="spread")], 1.5, settings)) == {"left", "right"}


def test_text_fitting_shrinks_before_wrapping_and_keeps_bounds():
    fonts = Fonts(load_config()["render"], 1)
    text = "四个汉字"
    original = fonts.font("sans", 30).getlength(text)
    shrunk = fonts.fit(text, "sans", 30, original * .9)
    assert len(shrunk.lines) == 1 and 24 <= shrunk.size < 30
    wrapped = fonts.fit(text, "sans", 30, original * .55)
    assert wrapped.size == 24 and len(wrapped.lines) == 2
    assert all(wrapped.font.getlength(line) <= original * .55 for line in wrapped.lines)
    with pytest.raises(ValueError):
        fonts.fit("很多汉字" * 20, "sans", 30, 60, 2)


def test_source_blur_only_changes_the_requested_region_before_resize():
    data = np.indices((80, 100)).sum(axis=0) % 2 * 255
    source = Image.fromarray(np.repeat(data[:, :, None], 3, axis=2).astype("uint8"))
    result = np.asarray(blur_frame(source, [(20, 20, 20, 20)], (100, 80), 5))
    assert np.array_equal(result[:20], np.asarray(source)[:20])
    assert result[22:38, 22:38].std() < 10
    # Original-pixel coordinates map correctly when prepare has downscaled.
    half = source.resize((50, 40), Image.Resampling.NEAREST)
    result_half = np.asarray(blur_frame(half, [(40, 40, 40, 40)], (200, 160), 10))
    assert np.array_equal(result_half[:10], np.asarray(half)[:10])


def test_other_speaker_panel_is_null_and_future_emotions_stay_blank():
    from jevtells.i18n import load_translations
    from jevtells.render.panels import Panels
    settings = load_config()["render"]
    layout = Layout.create("v", settings, (1280, 720))
    windows = [{"id": "W0", "t0": 0, "t1": 2, "speaker_other": True}, {"id": "W1", "t0": 3, "t1": 5}]
    panels = Panels(settings, layout, Fonts(settings, 1), load_translations("zh"), windows, {"W0": None, "W1": {"emotion": {"label": "firm"}}}, {"W0": None}, "标题", "未经人工校准 · 仅供演示")
    image = panels.panel(0)
    # The second, future emotion has no blue fill in its allocated arc cell.
    assert tuple(image.getpixel((910, 1276)))[:3] != (47, 107, 255)
    assert image.getbbox() is not None


def test_subtitles_on_draws_and_muxes_real_three_second_clip(tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=320x180:r=30:d=3", "-f", "lavfi", "-i", "sine=frequency=1000:duration=3", "-c:v", "libx264", "-c:a", "aac", "-shortest", str(clip)], check=True)
    config = copy.deepcopy(load_config())
    config["render"]["h"]["output"] = [640, 360]
    windows = [{"id": "W0", "t0": 0, "t1": 3}]
    points = {"pose": np.full((90, 33, 4), np.nan), "hands": np.full((90, 2, 21, 3), np.nan), "fps": 30}
    for name, data in {"windows.json": windows, "judgments.json": {"W0": None}, "narration.json": {"W0": {"line": "继续表达", "quote": "Subtitle proof"}}, "transcript.json": {"segments": [{"t0": 0, "t1": 3, "text": "Subtitle proof text"}]}}.items():
        (tmp_path / name).write_text(json.dumps(data, ensure_ascii=False))
    output = run(clip, tmp_path, windows, points, config=config, speaker="Demo", layout="h", subtitles=True)
    assert output["h"]["frames"] == 90
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(tmp_path / "output_h.mp4")], capture_output=True, text=True, check=True).stdout)
    assert any(stream["codec_type"] == "audio" for stream in probe["streams"])
    assert float(probe["format"]["duration"]) == pytest.approx(3, abs=.05)
    import cv2
    capture = cv2.VideoCapture(str(tmp_path / "output_h.mp4"))
    capture.set(cv2.CAP_PROP_POS_MSEC, 1500)
    ok, frame = capture.read()
    capture.release()
    assert ok and frame.shape[:2] == (360, 640)
    assert np.count_nonzero(frame[320:348, 180:460] > 170) > 30


def test_mixed_mono_ui_has_cjk_glyphs_and_pure_numbers_stay_mono():
    fonts = Fonts(load_config()["render"], 1)
    assert "NotoSansCJK" in str(fonts.fit("窗口 8/8", "mono", 16, 300).font.path)
    assert "NotoSansMono" in str(fonts.fit("W08 0.92", "mono", 16, 300).font.path)


def test_cli_render_only_never_visits_analysis_or_api_stages(tmp_path, monkeypatch):
    from argparse import Namespace
    from jevtells import cli
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "sample.mov"
    source.write_bytes(b"test input")
    output = tmp_path / "work" / "sample_s0_dauto"
    output.mkdir(parents=True)
    for name, value in {"track_meta.json": {"anchors": [[1, 0.0]]}, "windows.json": [{"id": "W00", "t0": 0, "t1": 3, "speaker_other": False}], "narrate_meta.json": {"lang": "zh", "retries": 0, "fallbacks": 0}, "transcript.json": {}, "actions.json": []}.items():
        (output / name).write_text(json.dumps(value))
    arguments = Namespace(command="run", input=str(source), output=None, speaker="Demo", scene=None, lang="zh", srt=None, start=0, duration=None, force=False, until="render", from_stage="render", target=["1@0"], config=None, others_speaking=[], blur=[], subtitles="off", layout="h", title="Changed title")
    monkeypatch.setattr(cli, "_arguments", lambda: arguments)
    for module in (cli.prepare, cli.pose, cli.track, cli.asr, cli.voice, cli.scene_stage, cli.judge, cli.narrate):
        monkeypatch.setattr(module, "run", lambda *a, **kw: pytest.fail("render-only touched analysis/API"))
    monkeypatch.setattr(cli, "_load_npz", lambda path: {})
    monkeypatch.setattr(cli, "_video_duration", lambda path: 0)
    class Video:
        def get(self, property):
            return 1280 if property == cli.cv2.CAP_PROP_FRAME_WIDTH else 720
        def release(self):
            pass
    monkeypatch.setattr(cli.cv2, "VideoCapture", lambda path: Video())
    calls = []
    monkeypatch.setattr(cli.render, "run", lambda *args, **kwargs: calls.append(kwargs))
    cli.main()
    assert len(calls) == 1 and calls[0]["title"] == "Changed title"
    assert set(json.loads((output / "run_meta.json").read_text())["stage_times_s"]) == {"render"}


def test_static_panel_is_drawn_once_after_transition(monkeypatch):
    from jevtells.render.panels import Panels
    from jevtells.i18n import load_translations
    settings = load_config()["render"]
    layout = Layout.create("v", settings, (1280, 720))
    painter = Panels(settings, layout, Fonts(settings, 1), load_translations("zh"), [{"id": "W0", "t0": 0, "t1": 2}], {"W0": None}, {"W0": None}, "标题", "免责声明")
    original = painter._metrics
    calls = []
    def metrics(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(painter, "_metrics", metrics)
    first = painter.analysis(0, .4)
    assert painter.analysis(0, .7) is first
    assert painter.analysis(0, 1.1) is first
    assert len(calls) == 1


def test_subtitles_use_short_windows_instead_of_long_asr_segments():
    config = load_config()
    settings = config["render"]
    layout = Layout.create("h", settings, (1280, 720))
    points = {"pose": np.full((1, 33, 4), np.nan), "hands": np.full((1, 2, 21, 3), np.nan)}
    painter = Composer(settings, layout, [{"id": "W0", "t0": 0, "t1": 3, "subtitle": "A short caption"}], points, [], {"W0": None}, {"W0": None}, {"segments": [{"t0": 0, "t1": 30, "text": "long transcript " * 100}]}, title="Title", sources="Disclaimer", lang="en", blur=[], subtitles=True, config=config)
    image = painter.frame(Image.new("RGB", (1280, 720)), 1, 0)
    assert "A short caption" in painter.subtitle_cache
    assert image.size == (1920, 1080)


def test_probability_bars_interpolate_with_the_scores():
    from jevtells.render.panels import Panels
    from jevtells.i18n import load_translations
    settings = load_config()["render"]
    geometry = Layout.create("v", settings, (1280, 720))
    windows = [{"id": "W0", "t0": 0, "t1": 2}, {"id": "W1", "t0": 3, "t1": 5}]
    judgments = {"W0": {"intent": {"label": "ask", "probs": {"ask": 1.0}}}, "W1": {"intent": {"label": "ask", "probs": {"ask": .2}}}}
    panels = Panels(settings, geometry, Fonts(settings, 1), load_translations("zh"), windows, judgments, {}, "标题", "免责声明")
    initial = panels.analysis(1, 0)
    final = panels.analysis(1, .3)
    p, g = settings["v"], settings["components"]
    y = p["lower_y"] + g["heading_gap"] + p["choice_heading_size"] + p["choice_top"] + p["choice_size"] + p["choice_bottom"] + 4
    assert initial.getpixel((300, int(y)))[:3] == (200, 255, 46)
    assert final.getpixel((300, int(y)))[:3] == (38, 38, 38)


def test_wrapped_quotation_highlights_every_line_without_overflow():
    settings = load_config()["render"]
    fonts = Fonts(settings, 1)
    fitted = fonts.fit("「必须完整高亮的跨行短语」", "sans", 30, 190, highlight_padding=4)
    assert fitted.size == 24 and len(fitted.lines) == 2
    assert not fitted.lines[1].startswith("「")
    image = Image.new("RGB", (210, 100), settings["colors"]["ink"])
    draw_fitted(image, (4, 4), fitted, settings["colors"], line_height=1.2, highlight_padding=4)
    pixels = np.asarray(image)
    lime = np.array([200, 255, 46])
    highlighted = np.all(pixels == lime, axis=2)
    second_y = 4 + round(fitted.size * 1.2)
    assert highlighted[4:second_y].any()
    assert highlighted[second_y:].any()
    assert np.where(highlighted)[1].max() <= 194
