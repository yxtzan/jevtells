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
