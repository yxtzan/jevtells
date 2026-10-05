import numpy as np
from jevtells.stages.framing import person_mask, faces_at


def person(x=.5):
    p = np.full((33,4), np.nan)
    p[:11] = [x,.25,0,1]
    p[11:13] = [x,.4,0,1]
    p[23:25] = [x,.8,0,1]
    return p


def test_person_standard_rejects_edges_missing_head_and_small_body():
    good = person()
    edge = person(.01)
    headless = person(); headless[:11] = np.nan
    short = person(); short[11:25,1] = .4
    hidden = person(); hidden[[2,5],3] = .49
    poses = np.array([good,edge,headless,short,hidden])
    assert person_mask(poses,{}).tolist() == [True,False,False,False,False]
    assert len(faces_at(poses,(1280,720),.5)) == 1
    one_eye = good.copy(); one_eye[5] = np.nan
    assert person_mask(np.array([one_eye]),{})[0]


def test_multi_timeline_ignores_short_changes_and_resets_at_cuts():
    from jevtells.stages.framing import multi_timeline
    p = np.array([[person(.25), person(.75)]]*100)
    p[10:20,1] = np.nan
    p[50:,1] = np.nan
    states = multi_timeline(p,10,[],{'multi_smooth_seconds':.1})
    assert states[:50].all() and not states[50:].any()
    p[40:,1] = np.nan
    assert not multi_timeline(p,10,[40],{})[40:].any()


def test_zoom_eases_for_six_hundred_ms():
    from jevtells.render.reframe import plan_crops, crop_at
    shots = [{'index':1,'t0':0,'t1':2,'multi_person':True},
             {'index':2,'t0':2,'t1':5,'multi_person':False}]
    plan = plan_crops(1280,720,shots,{'fps':30},None,{})
    widths = [crop_at(plan,t)[2] for t in [2,2.1,2.3,2.5,2.6]]
    assert widths[0] == 1280 and widths[-1] == 960
    assert widths == sorted(widths,reverse=True)
    assert widths[0]-widths[1] < widths[1]-widths[2]


def test_short_windows_merge_only_within_speaker_and_presence_boundaries():
    from jevtells.stages.segment import merge_short_windows
    rows = [{'t0':0,'t1':.3,'subtitle':'对'}, {'t0':.3,'t1':3,'subtitle':'原文'},
            {'t0':3,'t1':3.4,'subtitle':'是','speaker_other':True},
            {'t0':3.4,'t1':4,'subtitle':'下句'}, {'t0':4,'t1':6,'subtitle':'末句'},
            {'t0':6,'t1':6.2,'subtitle':'离画','target_offscreen':True}]
    result = merge_short_windows(rows,{})
    assert [w['subtitle'] for w in result] == ['对 原文','是','下句 末句','离画']
    assert result[1]['hold_previous_panel'] and result[-1]['hold_previous_panel']
    from jevtells.render.animation import panel_window_at
    assert panel_window_at(result,3.2) == (0,False)
    assert panel_window_at(result,3.5) == (2,False)


def test_srt_does_not_merge_before_speaker_classification(tmp_path):
    from jevtells.stages.segment import run, merge_short_windows
    from jevtells.stages.speakers import mark_windows
    transcript = {'subtitle_source':'srt','segments':[{'t0':0,'t1':.3,'text':'是'},{'t0':.3,'t1':2.6,'text':'原句'}]}
    result = run(transcript,[],tmp_path)
    result = merge_short_windows(mark_windows(result,[(0,.3)]),{})
    assert len(result) == 2 and result[0]['speaker_other']


def test_partial_edge_people_allow_full_right_card():
    import copy
    from jevtells.config import load_config
    from jevtells.render.geometry import Layout
    from jevtells.render.renderer import Composer
    config = load_config(); settings = copy.deepcopy(config['render'])
    center = person(.5)
    center[11,0] = .4; center[12,0] = .6
    points = {'pose':np.array([center]*30), 'poses_all':np.array([[person(.01),center,person(.99)]]*30),
              'fps':30,'hands':np.full((30,2,21,3),np.nan)}
    shots = [{'index':1,'t0':0,'t1':1,'label':'target','far':False}]
    painter = Composer(settings,Layout.create('h',settings,(1280,720)),[{'id':'W0','t0':0,'t1':1}],
                       points,[],{},{},{},title='Title',sources='Source',lang='zh',blur=[],subtitles=False,config=config,shots=shots)
    assert painter.card_modes[1] == 'full' and painter.card_sides[1] == 'right'
    assert painter.card_face_collisions[1] == {'right':0,'left':0}


def test_actions_discard_only_onsets_inside_cut_settling_period():
    from jevtells.stages.actions import discard_after_cuts
    events = [{'t0':t,'t1':t+.5,'type':'beat'} for t in [3.9,4,4.1,4.3,5]]
    kept,discarded = discard_after_cuts(events,[4],.3)
    assert [e['t0'] for e in kept] == [3.9,4.3,5]
    assert [e['t0'] for e in discarded] == [4,4.1]
    assert all(e['reason']=='settling_after_cut' and e['cut_seconds']==4 for e in discarded)


def test_short_interjection_keeps_rendered_card_and_commentary():
    import copy
    from PIL import Image
    from jevtells.config import load_config
    from jevtells.stages.segment import merge_short_windows
    from jevtells.render.geometry import Layout
    from jevtells.render.renderer import Composer
    cfg = load_config(); settings = copy.deepcopy(cfg['render'])
    rows = merge_short_windows([{'t0':0,'t1':3,'subtitle':'主角原话'},
                               {'t0':3,'t1':3.4,'subtitle':'是','speaker_other':True},
                               {'t0':3.4,'t1':6,'subtitle':'主角续话'}],cfg)
    pose = person();pose[11,0] = .4;pose[12,0] = .6
    points = {'pose':np.array([pose]*180),'fps':30,'hands':np.full((180,2,21,3),np.nan)}
    judgments = {'W00':{'scores':{k:{'value':.6} for k in ['confidence','focus','tension']}}}
    painter = Composer(settings,Layout.create('h',settings,(1280,720)),rows,points,[],judgments,
                       {'W00':{'line':'原解说保持显示','quote':'主角原话'}},{},title='Title',sources='Source',
                       lang='zh',blur=[],subtitles=False,config=cfg)
    first = painter.frame(Image.new('RGB',(1280,720)),2.9,87)
    held = painter.frame(Image.new('RGB',(1280,720)),3.2,96)
    assert painter.audit['cards'][0]['mode'] == 'full'
    assert first.crop((0,0,1450,150)).tobytes() == held.crop((0,0,1450,150)).tobytes()
    assert first.crop((1450,230,1900,750)).tobytes() == held.crop((1450,230,1900,750)).tobytes()


def test_zoom_strip_subpixel_start_renders_without_zero_size_resize():
    import copy
    from PIL import Image
    from jevtells.config import load_config
    from jevtells.render.geometry import Layout
    from jevtells.render.renderer import Composer
    from jevtells.render.reframe import plan_crops
    cfg = load_config(); settings = copy.deepcopy(cfg['render'])
    settings['v'].update(settings['v_reframe'])
    pose = person();points = {'pose':np.array([pose]*120),'fps':30,'hands':np.full((120,2,21,3),np.nan)}
    shots = [{'index':1,'t0':0,'t1':2,'multi_person':True,'far':True},
             {'index':2,'t0':2,'t1':4,'multi_person':False,'far':True}]
    plan = plan_crops(1280,720,shots,points,None,settings['reframe'])
    plan.update(mode='strip',strip_y=576);plan['shots'][1]['height'] = 576
    painter = Composer(settings,Layout.create('v',settings,(1280,720)),[{'id':'W0','t0':0,'t1':4}],
                       points,[],{},{},{},title='Title',sources='Source',lang='zh',blur=[],subtitles=False,
                       config=cfg,shots=shots,reframe_plan=plan)
    image = painter.frame(Image.new('RGB',(1280,720)),2+1/30,61)
    assert image.size == (1080,1440)


def test_merging_avoids_long_span_when_original_cues_allow_two_windows():
    from jevtells.stages.segment import merge_short_windows
    cues = [{'t0':a,'t1':b,'subtitle':text} for a,b,text in [(0,1.5,'甲'),(1.5,2.8,'乙'),(2.8,4.4,'丙'),(4.4,6.1,'丁')]]
    result = merge_short_windows(cues,{})
    assert len(result) == 2
    assert all(2 <= w['t1']-w['t0'] <= 5 for w in result)
    assert ''.join(w['subtitle'].replace(' ','') for w in result) == '甲乙丙丁'


def test_merge_cannot_cross_speaker_gap_without_caption():
    from jevtells.stages.segment import merge_short_windows
    rows = [{'t0':0,'t1':.5,'subtitle':'前'}, {'t0':1.5,'t1':3.5,'subtitle':'后'}]
    assert len(merge_short_windows(rows,{},boundaries=[.7,1.2])) == 2


def test_presence_metadata_drift_does_not_invalidate_paid_analysis():
    from jevtells.cli import _analysis_windows_changed
    old = [{'id':'W00','t0':0,'t1':3,'subtitle':'原话','speaker_other':False,'target_offscreen':False,'target_presence_ratio':.999999999999998}]
    changed = [{**old[0],'target_presence_ratio':1.0}]
    assert not _analysis_windows_changed(changed,old)
    changed[0]['target_offscreen'] = True
    assert _analysis_windows_changed(changed,old)
    assert _analysis_windows_changed(old,None)
