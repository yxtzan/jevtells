import json

import cv2
import numpy as np

from jevtells.config import load_config
from jevtells.stages.appearance import clothing_histograms, track_appearance
from jevtells.stages.track import run


def person(center=.5):
    pose = np.full((33, 4), np.nan, np.float32)
    for i, point in [(11, (center-.12,.2)), (12, (center+.12,.2)), (23, (center-.12,.8)), (24, (center+.12,.8))]:
        pose[i] = [*point, 0, 1]
    return pose


def descriptor(color):
    return clothing_histograms(np.full((100,100,3), color, np.uint8), person())


def fixture(colors, centers=None, anchors=((1,0),), cuts=(10,)):
    poses=np.array([[person(c) for c in row] for row in (centers or [[.5]*len(row) for row in colors])])
    descriptors=np.array([[descriptor(color) for color in row] for row in colors])
    return track_appearance(poses, descriptors, anchors, cuts, 10, 100, 100, load_config())


WHITE=(240,240,240)
DARK=(20,20,20)


def test_hard_cut_to_other_person_stays_offscreen():
    indices, meta=fixture([[WHITE]]*10+[[DARK]]*30)
    assert np.all(indices[:10]==0) and np.all(indices[10:]==-1)
    assert meta['shots'][1]['decisions'][0]['result']=='target_offscreen'


def test_hard_cut_back_recognizes_same_person():
    indices, meta=fixture([[WHITE]]*10+[[DARK]]*10+[[WHITE]]*10,cuts=(10,20))
    assert np.all(indices[10:20]==-1) and np.all(indices[20:]==0)
    assert meta['shots'][2]['decisions'][0]['result']=='recognized'


def test_two_people_select_appearance_and_follow_detector_slot_changes():
    colors=[[WHITE]]*10+[[DARK,WHITE] if i%2 else [WHITE,DARK] for i in range(10)]
    centers=[[.5]]*10+[[.25,.75] if i%2 else [.75,.25] for i in range(10)]
    # Pad initial frames to the same detector K.
    colors[:10]=[[WHITE,DARK]]*10;centers[:10]=[[.25,.75]]*10
    indices,_=fixture(colors,centers)
    assert indices[10:].tolist()==[i%2 for i in range(10)]


def test_similar_clothing_without_margin_rejects_both():
    indices,meta=fixture([[WHITE,DARK]]*10+[[WHITE,WHITE]]*10,centers=[[.25,.75]]*20)
    assert np.all(indices[10:]==-1)
    assert meta['shots'][1]['decisions'][0]['second_similarity']==1


def test_anchor_overrides_automatic_offscreen_decision():
    indices,meta=fixture([[WHITE]]*10+[[DARK]]*10,anchors=((1,0),(1,1.5)))
    # Future explicit reference is retained too; anchor must still be recorded.
    assert np.all(indices[15:]==0)
    assert meta['shots'][1]['decisions'][-1]['result']=='anchor'


def test_occluded_target_can_be_recovered_without_learning_host():
    poses=np.array([[person(),person(.85)]]*30)
    appearances=np.array([[descriptor(WHITE),descriptor(DARK)]]*30)
    poses[10:20,0]=np.nan;appearances[10:20,0]=np.nan
    indices,meta=track_appearance(poses,appearances,[(1,0)],[],10,100,100,load_config())
    assert np.all(indices[10:20]==-1) and np.all(indices[20:]==0)
    assert any(r['result']=='recognized' for r in meta['recoveries'])


def test_first_anchor_tracks_backward_within_shot():
    indices,_=fixture([[WHITE]]*20,anchors=((1,.5),),cuts=())
    assert np.all(indices==0)


def test_encoded_video_track_uses_shared_cuts_and_writes_metadata(tmp_path):
    clip=tmp_path/'clip.mp4'
    writer=cv2.VideoWriter(str(clip),cv2.VideoWriter_fourcc(*'mp4v'),10,(100,100))
    for color in [WHITE]*10+[DARK]*10+[WHITE]*10:
        writer.write(np.full((100,100,3),color,np.uint8))
    writer.release()
    poses=np.array([[person()]]*30)
    data={'poses_all':poses,'hands_all':np.full((30,1,21,3),np.nan),'fps':10,'width':100,'height':100}
    result=run(data,tmp_path,anchors=[(1,0)],config=load_config())
    assert result['target_index'].tolist()==[0]*10+[-1]*10+[0]*10
    assert not result['hand_present'][10:20].any()
    meta=json.loads((tmp_path/'track_meta.json').read_text())
    assert [s['decisions'][0]['result'] for s in meta['shots']]==['anchor','target_offscreen','recognized']
