"""Bright-background burned captions must not become a second subtitle track."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from jevtells.config import load_config
from jevtells.render.reframe import SUBTITLE_DETECTION_VERSION, detect_subtitles


def encoded_clip(tmp_path, caption):
    image = Image.new('RGB', (1280, 720), '#d8d8d8')
    if caption:
        font = ImageFont.truetype(load_config()['render']['fonts']['sans'], 32)
        ImageDraw.Draw(image).text((360, 625), 'Bright outlined captions', font=font,
                                   fill='white', stroke_width=1, stroke_fill='#444444')
    image.save(tmp_path/'frame.png')
    clip = tmp_path/'clip.mp4'
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-loop', '1', '-i', str(tmp_path/'frame.png'),
                    '-t', '1', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(clip)], check=True)
    return clip


@pytest.mark.parametrize('caption', [False, True])
def test_actual_bright_background_requires_recognizable_caption(tmp_path, caption):
    result = detect_subtitles(encoded_clip(tmp_path, caption), load_config()['render']['reframe'])
    assert bool(result['bounds']) == caption
    if caption:
        assert 345 <= result['bounds'][0] <= 375
        assert result['bounds'][1] >= 650
        assert len({s['seconds'] for s in result['samples']}) >= 2


@pytest.mark.parametrize('mode', ['single', 'low_confidence', 'short_text', 'different_baselines'])
def test_ocr_fallback_rejects_unconfirmed_text(tmp_path, monkeypatch, mode):
    import rapidocr_onnxruntime
    calls = 0
    def recognize(image):
        nonlocal calls
        calls += 1
        y = calls * 30 if mode == 'different_baselines' else 120
        rows = [[[[350,y],[700,y],[700,y+30],[350,y+30]],
                 'OK' if mode == 'short_text' else 'Recognized caption',
                 .6 if mode == 'low_confidence' else .99]]
        return (rows if mode != 'single' or calls == 1 else None), None
    monkeypatch.setattr(rapidocr_onnxruntime, 'RapidOCR', lambda **kw: recognize)
    result = detect_subtitles(encoded_clip(tmp_path, False), load_config()['render']['reframe'])
    assert result['bounds'] is None


def test_inspect_invalidates_old_negative_cache(tmp_path, monkeypatch):
    from jevtells.ui import server
    from jevtells.render import reframe
    (tmp_path/'samples').mkdir()
    source = tmp_path/'samples/short.mov'
    source.write_bytes(b'video fixture')
    cache = tmp_path/'work/.ui_probe/short.json'
    cache.parent.mkdir(parents=True)
    old_signature = [source.stat().st_size, source.stat().st_mtime_ns]
    cache.write_text(json.dumps({'signature':old_signature, 'detection':{'bounds':None}}))
    monkeypatch.setattr(server, 'media_info', lambda p: {'duration':4})
    monkeypatch.setattr(server, 'checks', lambda cfg: [])
    calls = []
    def detected(*args):
        calls.append(args)
        return {'bounds':[350,700]}
    monkeypatch.setattr(reframe, 'detect_subtitles', detected)
    client = server.create_app(tmp_path).test_client()
    for _ in range(2):
        response = client.get('/api/inspect?input=samples/short.mov', base_url='http://127.0.0.1')
        assert response.status_code == 200
        assert response.json['subtitles']['bounds'] == [350,700]
    assert len(calls) == 1
    assert json.loads(cache.read_text())['signature'] == [*old_signature,SUBTITLE_DETECTION_VERSION]


def test_wide_ocr_caption_uses_its_probe_spacing_for_portrait_strip():
    from jevtells.render.reframe import subtitle_mode
    detection = {'sample_seconds':1.0, 'strip_y':580, 'samples':[
        {'seconds':0.0, 'bounds':[50,1230]},
        {'seconds':1.0, 'bounds':[50,1230]},
    ]}
    settings = {'sample_seconds':.5}
    assert subtitle_mode(detection,1280,720,settings)['mode'] == 'strip'
    # A genuine gap in observations is still insufficient confirmation.
    detection['samples'][1]['seconds'] = 3.0
    assert not subtitle_mode(detection,1280,720,settings)['oversized']
