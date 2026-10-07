import numpy as np
import pytest
from jevtells.stages.auto_start import find_start
from jevtells.config import load_config


def test_auto_start_requires_solo_speech_and_aligns_real_cut():
    n=150;poses=np.zeros((n,2,33,4));poses[:]=np.nan
    poses[:,0,:,0]=.4;poses[:,0,:,1]=np.linspace(.1,.8,33);poses[:,0,:,3]=1
    poses[:30,1]=poses[:30,0];poses[:30,1,:,0]=.7
    points={'fps':30.,'target_index':np.zeros(n,int)};detections={'poses_all':poses,'width':1280,'height':720}
    windows=[{'t0':0.,'t1':1.5,'speaker':'B'},{'t0':1.5,'t1':5.,'speaker':'A'}]
    assert find_start(points,detections,[30,45],windows,'A',[],load_config())==(1.5,1.5)
    with pytest.raises(ValueError): find_start(points,detections,[],[{'t0':0.,'t1':5.,'speaker':None}],'A',[],load_config())
    assert find_start(points,detections,[30],None,'A',[(0.,1.)],load_config())==(1.,1.)
