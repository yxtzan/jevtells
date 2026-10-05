import copy
import pytest
from jevtells.config import load_config
from jevtells.render.geometry import Layout
from jevtells.render.panels import Panels
from jevtells.render.text import Fonts
from jevtells.i18n import load_translations


@pytest.mark.parametrize('reframe',[False,True])
def test_portrait_distributes_fixed_extra_space_and_choice_gaps(reframe):
    settings=copy.deepcopy(load_config()['render'])
    if reframe:
        settings['v'].update(settings['v_reframe'])
        settings['components'].update(settings['reframe_components'])
    layout=Layout.create('v',settings,(1280,720),(160,0,960,720) if reframe else None)
    windows=[{'id':'W0'},{'id':'W1'}]
    panels=Panels(settings,layout,Fonts(settings,1),load_translations('zh'),windows,{}, {'W0':{'line':'双手下压','quote':'hello world'},'W1':{'line':'专注度明显上升','quote':'next phrase'}},'标题','来源')
    assert panels.p['choice_top']>=6
    assert panels.p['choice_bottom']>=8
    assert panels.p['arc_top']>=8
    assert 0<panels.spacing['added_per_gap']<=32
    assert panels.spacing['free_height']==pytest.approx(3*panels.spacing['added_per_gap']+panels.spacing['remaining_bottom'])
    before=dict(panels.p)
    panels.analysis(0,1);panels.analysis(1,1)
    assert panels.p==before
