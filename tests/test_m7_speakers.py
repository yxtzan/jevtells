import numpy as np
import pytest
from jevtells.stages.active_speaker import classify,parse_maps
from jevtells.config import load_config


def tracks(n):
    return {name:{'target_index':np.full(n,p),'pose_present':np.ones(n,bool),'fps':30.,'t':np.arange(n)/30} for p,name in enumerate(['A','B'])}


def test_correlated_speaker_and_ambiguous_mouths():
    n=120;t=np.arange(n)/30;signal=.03+.02*np.sin(t*8);opening=np.stack([signal,np.full(n,.01)],axis=1)
    window={'t0':0.,'t1':4.};settings=load_config()['two_person']
    assert classify(window,tracks(n),opening,t,signal*2,30,settings)['speaker']=='A'
    opening[:,1]=opening[:,0]
    assert classify(window,tracks(n),opening,t,signal*2,30,settings)['speaker'] is None


def test_still_observable_listener_means_offscreen_other_but_small_faces_unknown():
    n=120;t=np.arange(n)/30;opening=np.full((n,2),np.nan);opening[:,0]=.01
    result=classify({'t0':0.,'t1':4.},tracks(n),opening,t,np.full(n,.1),30,load_config()['two_person'])
    assert (result['speaker'],result['reason'])==('B','offscreen')
    opening[:]=np.nan
    assert classify({'t0':0.,'t1':4.},tracks(n),opening,t,np.full(n,.1),30,load_config()['two_person'])['speaker'] is None


def test_silence_never_selects_a_speaker_and_maps_validate():
    n=120;t=np.arange(n)/30;opening=np.full((n,2),np.nan);opening[:,0]=.01
    assert classify({'t0':0.,'t1':4.},tracks(n),opening,t,np.zeros(n),30,load_config()['two_person'])['speaker'] is None
    assert parse_maps(['0-4=A','10.3-15.7=B'],['A','B'])==[(0.,4.,'A'),(10.3,15.7,'B')]
    for values in (['0-4=C'],['0-4=A','3-5=B']):
        with pytest.raises(ValueError): parse_maps(values,['A','B'])
