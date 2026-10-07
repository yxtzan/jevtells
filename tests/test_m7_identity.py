import numpy as np
from jevtells.stages.face_identity import associate_faces, cosine
from jevtells.config import load_config
from jevtells.download import ASSETS


def test_face_assignment_is_exclusive_and_rejects_distant_pose():
    poses=np.zeros((3,33,4));poses[:,0,:2]=[[.2,.3],[.21,.3],[.9,.9]]
    faces=np.zeros((1,15));faces[0,:4]=[10,20,20,20];faces[0,8:10]=[20,30]
    assigned=associate_faces(poses,faces,100,100,load_config()['two_person'])
    assert len(assigned)==1 and 2 not in assigned


def test_cosine_unobservable_is_rejected_and_models_registered():
    assert cosine(np.array([np.nan,0]),[np.array([1,0])])==-1
    assert cosine(np.array([1,0]),[np.array([1,0])])==1
    config=load_config()
    for name in ('face_detector','face_recognizer'):
        assert 'models/'+config['models'][name] in ASSETS


def test_hungarian_keeps_exclusive_assignments_and_rejects_ambiguity():
    from jevtells.stages.two_person_track import exclusive_assignment
    scores=np.array([[.9,.8],[.85,.1]])
    assert exclusive_assignment(scores,np.ones_like(scores,bool)).tolist()==[1,0]
    assert exclusive_assignment(scores,np.zeros_like(scores,bool)).tolist()==[-1,-1]
    assert exclusive_assignment(scores,np.array([[True,False],[True,False]])).tolist()==[0,-1]
