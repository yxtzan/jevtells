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
