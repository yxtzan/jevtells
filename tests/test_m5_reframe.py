import copy

import numpy as np
import pytest

from jevtells.config import load_config
from jevtells.render.geometry import Layout
from jevtells.render.reframe import crop_at, crop_left, plan_crops


def test_burned_subtitle_edges_are_detected_from_actual_encoded_glyphs(tmp_path):
    import subprocess
    from PIL import Image, ImageDraw, ImageFont
    from jevtells.render.reframe import detect_subtitles
    frame=Image.new("RGB",(1280,720),"#888888")
    draw=ImageDraw.Draw(frame)
    draw.rectangle((250,620,1030,700),fill="black")
    font=ImageFont.truetype(load_config()["render"]["fonts"]["sans"],24)
    draw.text((300,628),"保留完整的中英字幕行",font=font,fill="yellow")
    draw.text((300,660),"Keep both subtitle lines complete",font=font,fill="white")
    frame.save(tmp_path/"frame.png")
    clip=tmp_path/"clip.mp4"
    subprocess.run(["ffmpeg","-y","-v","error","-loop","1","-i",str(tmp_path/"frame.png"),"-t","0.1","-c:v","libx264","-pix_fmt","yuv420p",str(clip)],check=True)
    result=detect_subtitles(clip,load_config()["render"]["reframe"])
    assert result["bounds"] is not None
    assert 290 <= result["bounds"][0] <= 310
    assert 600 <= result["bounds"][1] <= 1030


def test_crops_center_clamp_and_keep_detected_subtitle_span():
    assert crop_left(1280,720,640,[200,1080])==160
    assert crop_left(1280,720,50,[0,800])==0
    assert crop_left(1280,720,1250,[500,1280])==320
    assert crop_left(1280,720,700,[100,1000])==100
    assert crop_left(1280,720,1250,None)==pytest.approx(262.4)
    assert crop_left(1280,720,640,[50,1230])==160


def test_crop_coordinate_transform_and_off_geometry_regression():
    settings=copy.deepcopy(load_config()["render"])
    off=Layout.create("v",settings,(1280,720))
    assert off.video==(0,196,1080,608)
    assert off.source_point(640,360)==(540,500)
    settings["v"].update(settings["v_reframe"])
    auto=Layout.create("v",settings,(1280,720),(160,0,960,720))
    assert auto.video==(0,150,1080,810)
    assert auto.source_point(160,0)==(0,150)
    assert auto.source_point(1120,720)==(1080,960)
    assert auto.source_rect([300,100,100,100])==(158,262,270,375)


def test_narrow_source_disables_crop_and_cut_jumps_without_pan():
    config=load_config()["render"]["reframe"]
    assert not plan_crops(800,720,[],{},None,config)["enabled"]
    shots=[{"index":1,"t0":0,"t1":5,"target_center_x":400},{"index":2,"t0":5,"t1":10,"target_center_x":900}]
    plan=plan_crops(1280,720,shots,{},[300,1100],config)
    assert crop_at(plan,4.9)[0]==140
    assert crop_at(plan,5)[0]==300


def test_in_shot_pan_waits_a_second_and_uses_600ms_easing():
    config={**load_config()["render"]["reframe"],"max_offset":.5}
    pose=np.tile(np.array([.95,.5,0,1]),(90,33,1))
    plan=plan_crops(1280,720,[{"index":1,"t0":0,"t1":3,"target_center_x":480}],{"pose":pose,"fps":30},None,config)
    assert len(plan["shots"][0]["pans"])==1
    assert crop_at(plan,.9)[0]==0
    assert crop_at(plan,1.3)[0]==pytest.approx(160)
    assert crop_at(plan,1.6)[0]==320
