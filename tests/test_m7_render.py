import copy
from PIL import Image
from jevtells.render.geometry import Layout
from jevtells.render.panels import Panels
from jevtells.render.text import Fonts
from jevtells.i18n import load_translations
from jevtells.config import load_config
from jevtells.render.layout_audit import frame_violations


def test_two_person_commentary_and_cards_render_unknown_offscreen_and_names():
    config=load_config();settings=copy.deepcopy(config['render']);settings['min_judgment_confidence']=.4
    windows=[{'id':'W00','t0':0.,'t1':2.,'two_person':True,'speaker':'Tim','speaker_color':'#C8FF2E','target_offscreen':True},{'id':'W01','t0':2.,'t1':4.,'two_person':True,'speaker':None,'speaker_unknown':True,'speaker_color':'#9A9A9A'}]
    for kind in ['h','v']:
        layout=Layout.create(kind,settings,(1280,720));panels=Panels(settings,layout,Fonts(settings,layout.scale),load_translations('zh'),windows,{}, {},'AI 读访谈','来源')
        for i in range(2):
            layer,_=panels.commentary(i);assert layer.getbbox()
            assert panels.panel(i).getbbox()


def test_audit_detects_opponent_core_and_person_specific_leader_midline():
    from jevtells.render.layout_audit import leader_violations
    frame={'labels':[{'slot':'B:left','person':'B','event':'B1','rect':[10,10,20,20],'leader':[[5,5],[30,5]],'midline':15}], 'person_forbidden':{'B':[12,12,25,25]},'leader_zones':[],'endpoint_exemption':0}
    assert frame_violations(frame)[0]['zones']==['other_core']
    assert leader_violations(frame)[0]['zone']=='midline'
