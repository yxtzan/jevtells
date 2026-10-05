import copy
import numpy as np
from PIL import Image
from jevtells.config import load_config
from jevtells.stages.framing import multiple_people
from jevtells.render.reframe import plan_crops
from jevtells.render.renderer import Composer
from jevtells.render.geometry import Layout
from jevtells.render.layout_audit import frame_violations,leader_violations


def person(x):
    pose=np.full((33,4),np.nan)
    pose[:11]=[x,.25,0,1]
    pose[11]=[x-.05,.35,0,1];pose[12]=[x+.05,.35,0,1]
    pose[23]=[x-.04,.85,0,1];pose[24]=[x+.04,.85,0,1]
    return pose


def test_multiperson_requires_large_bodies_and_half_frame_coverage():
    poses=np.array([[person(.25),person(.75)]]*10)
    assert multiple_people(poses,{})['multi_person']
    poses[5:,1]=np.nan
    assert multiple_people(poses,{})['multi_person']
    poses[4:,1]=np.nan
    assert not multiple_people(poses,{})['multi_person']


def painter(kind='h',other=False):
    config=load_config();settings=copy.deepcopy(config['render'])
    if kind=='v':settings['v'].update(settings['v_reframe'])
    pose=np.array([person(.2)]*30)
    points={'pose':pose,'poses_all':np.array([[person(.2),person(.8)]]*30),'hands':np.full((30,2,21,3),np.nan),'fps':30}
    shots=[{'index':1,'t0':0,'t1':1,'label':'target','far':True,'multi_person':True}]
    plan=plan_crops(1280,720,shots,points,None,settings['reframe']) if kind=='v' else {}
    windows=[{'id':'W0','t0':0,'t1':1,'speaker_other':other}]
    return Composer(settings,Layout.create(kind,settings,(1280,720)),windows,points,[],{}, {},{},title='Title',sources='Source',lang='zh',blur=[],subtitles=False,config=config,shots=shots,reframe_plan=plan)


def test_multibody_portrait_fits_full_frame_without_extra_subtitle_strip():
    p=painter('v');p.frame(Image.new('RGB',(1280,720)),.5,15)
    assert p.layout.crop is None and p.layout.strip_video is None
    assert p.layout.video==(0,251,1080,608)


def test_portrait_framing_animates_inside_shot_and_cuts_directly():
    shots = [
        {'index':1,'t0':0,'t1':1,'multi_person':True,'cut_at_start':False},
        {'index':2,'t0':1,'t1':2,'multi_person':False,'cut_at_start':False},
        {'index':3,'t0':2,'t1':3,'multi_person':False,'cut_at_start':True},
    ]
    plan = plan_crops(1280,720,shots,{'fps':30},None,load_config()['render']['reframe'])
    assert plan['shots'][0]['multi_person'] and not plan['shots'][1].get('multi_person')
    assert plan['shots'][0]['width'] == 1280
    assert plan['shots'][1]['transition']['seconds'] == .6
    assert 'transition' not in plan['shots'][2]
    assert plan['shots'][2]['width'] < 1280


def test_both_sides_face_collision_uses_mini_card_and_other_speech_collapses():
    p=painter();p.frame(Image.new('RGB',(1280,720)),.5,15)
    assert p.audit['cards'][0]['mode']=='mini'
    assert not frame_violations(p.audit)
    p=painter(other=True);p.frame(Image.new('RGB',(1280,720)),.5,15)
    assert p.audit['cards'][0]['mode']=='collapsed'
    assert not frame_violations(p.audit)


def test_face_audit_includes_other_people_and_cards():
    trace={'forbidden':[{'name':'face:1','rect':[20,20,40,40]}],'cards':[{'mode':'full','rect':[0,0,30,30]}],'labels':[]}
    assert frame_violations(trace)[0]['slot']=='card'
    trace={'leader_zones':[{'name':'face','rect':[200,200,250,250]},{'name':'face','rect':[20,20,40,40]}],'labels':[{'slot':'a','event':'x','leader':[[0,0],[50,50]],'rect':[0,0,5,5]}]}
    assert leader_violations(trace)[0]['zone']=='face'


def test_placement_checks_other_faces_at_every_display_sample():
    from jevtells.render.placement import select_positions
    from jevtells.render.layout_audit import line_flags
    face = [180,80,260,160]
    samples = {'left': [{'anchor':[140,220],'midline':250,'faces':[face],'face':None}],
               'right': [{'anchor':[380,220],'midline':250,'faces':[face],'face':None}]}
    result = select_positions([0,0,500,400],None,[],{'left':[80,40],'right':[80,40]},
                              {'left':[140,220],'right':[380,220]},samples=samples,
                              midline=250,step=36,margin=20,gap=10,
                              max_face_crossing_ratio=0,fixed_leader_width=True)
    assert all(not line_flags(v['leader'],face,None,60)[0] for v in result['selected'].values())
