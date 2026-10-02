import numpy as np
from jevtells.stages.segment import run
from jevtells.utils.geometry import match_hands_to_pose

def test_short_sentences_merge(tmp_path):
    tr={'segments':[{'t0':0,'t1':1,'text':'one','words':[{'t0':0,'t1':1,'w':'one'}]},{'t0':1,'t1':3.5,'text':'two','words':[{'t0':1,'t1':3.5,'w':'two'}]}]}
    d=run(tr,[{'t0':0,'t1':4}],tmp_path,True); assert len(d)==2

def test_long_sentence_splits_at_word_boundary(tmp_path):
    words=[{'t0':i,'t1':i+1,'w':str(i)} for i in range(8)]
    d=run({'segments':[{'t0':0,'t1':8,'text':' '.join(str(i) for i in range(8)),'words':words}]},[],tmp_path,True)
    assert len(d)==2 and d[0]['t1']==5 and d[1]['t0']==5

def test_silence_window(tmp_path):
    d=run({'segments':[]},[{'t0':0,'t1':6,'label':'target'}],tmp_path,True); assert d[0]['kind']=='silence'

def test_left_right_matches_pose_wrist_nearest():
    pose=np.full((33,4),np.nan); pose[15,:2]=[.2,.5]; pose[16,:2]=[.8,.5]
    hands=np.array([[.79,.5,.1],[.21,.5,.1]])
    assert match_hands_to_pose(hands,pose).tolist()==[1,0]
