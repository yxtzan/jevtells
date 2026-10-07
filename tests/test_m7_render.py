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


def test_two_people_keep_both_label_tracks_visible_with_distinct_colors():
    import numpy as np
    from jevtells.render.two_person import TwoPersonComposer
    from jevtells.render.layout_audit import frame_violations
    config=load_config();settings=copy.deepcopy(config['render']);settings['min_judgment_confidence']=.4
    n=90;tracks={};all_poses=[]
    for p,name in enumerate(['A','B']):
        pose=np.zeros((n,33,4));pose[:,:,3]=1;pose[:,:,0]=.25+.45*p;pose[:,:,1]=.35
        pose[:,0:11,1]=.22;pose[:,11,:2]=[.15+.45*p,.38];pose[:,12,:2]=[.35+.45*p,.38];pose[:,23,:2]=[.16+.45*p,.75];pose[:,24,:2]=[.34+.45*p,.75]
        pose[:,15,:2]=[.12+.45*p,.52];pose[:,16,:2]=[.38+.45*p,.52]
        tracks[name]={'pose':pose,'hands':np.full((n,2,21,3),np.nan),'target_index':np.full(n,p),'pose_present':np.ones(n,bool),'fps':30.,'t':np.arange(n)/30}
        all_poses.append(pose)
    points={**tracks['A'],'poses_all':np.stack(all_poses,axis=1)}
    events=[{'id':name+'1','person':name,'t0':.2,'t1':2.,'type':'raise','limb':'right_hand','magnitude':'medium','shot_index':1} for name in tracks]
    windows=[{'id':'W00','t0':0.,'t1':3.,'two_person':True,'speaker':None,'speaker_unknown':True,'speaker_color':'#9A9A9A'}]
    layout=Layout.create('h',settings,(1280,720));composer=TwoPersonComposer(settings,layout,windows,points,events,{}, {},{},title='访谈',sources='来源',lang='zh',blur=[],subtitles=False,config=config,shots=[{'index':1,'t0':0.,'t1':3.,'label':'target','far':False}],person_tracks=tracks)
    composer.frame(Image.new('RGB',(1280,720),'#555555'),1.,30)
    assert {label['person'] for label in composer.audit['labels']}=={'A','B'}
    assert len(composer.audit['expected_labels'])==len(composer.audit['labels'])==2
    assert not frame_violations(composer.audit)
