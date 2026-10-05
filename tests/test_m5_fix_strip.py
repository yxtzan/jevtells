import copy
import pytest
from jevtells.config import load_config
from jevtells.render.geometry import Layout
from jevtells.render.reframe import subtitle_mode, subtitle_strip_y


def test_subtitle_mode_fits_oversized_and_single_noise():
    settings=load_config()['render']['reframe']
    def sample(t,width):return {'seconds':t,'bounds':[0,width]}
    assert subtitle_mode({'samples':[sample(0,900),sample(.5,900)]},1280,720,settings)['mode']=='center'
    assert subtitle_mode({'samples':[sample(0,1176),sample(.5,1176)]},1280,720,settings)['mode']=='strip'
    assert subtitle_mode({'samples':[sample(0,900),sample(.5,1176),sample(1,900)]},1280,720,settings)['mode']=='center'
    assert subtitle_mode({'samples':[sample(0,1176),sample(1,1176)]},1280,720,settings)['mode']=='center'
    for mode in ['center','off']:
        assert subtitle_mode({'samples':[sample(0,1176),sample(.5,1176)]},1280,720,{**settings,'oversized_subtitles':mode})['mode']==mode


def test_subtitle_strip_y_uses_earliest_text_minus_six():
    assert subtitle_strip_y(720,[])==576
    assert subtitle_strip_y(720,[620,650])==576
    assert subtitle_strip_y(720,[565,625])==559


def test_strip_picture_and_subtitle_coordinate_transforms():
    settings=copy.deepcopy(load_config()['render']);settings['v'].update(settings['v_reframe'])
    layout=Layout.create('v',settings,(1280,720),(160,0,960,576),strip_y=576)
    assert layout.video==pytest.approx((0,166.25,1080,648))
    assert layout.strip_video==pytest.approx((0,822.25,1080,121.5))
    assert layout.source_point(160,0)==(0,166)
    assert layout.source_point(1120,576)==(1080,814)
    assert layout.source_strip_point(0,576)==(0,822)
    assert layout.source_strip_point(1280,720)==(1080,944)
    assert layout.strip_video[1]-(layout.video[1]+layout.video[3])==8


def test_label_positions_cannot_enter_subtitle_strip():
    from jevtells.render.placement import select_positions
    picture=[0,166,1080,814];strip=[0,822,1080,944]
    result=select_positions(picture,None,[strip],{'left':[120,60],'right':[120,60]},{'left':[200,600],'right':[800,600]},{'left':[24,250],'right':[800,250]})
    assert not result['fallback']
    for slot,(x,y) in result['positions'].items():
        assert y+60<=picture[3]-24
