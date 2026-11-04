import json
from pathlib import Path
import numpy as np
import pytest
from jevtells.config import load_config
from jevtells.stages import face_identity, auto_start, jev_images, target_desc, judge
from jevtells.clients import openrouter, budget


def poses(n=150,k=2):
    x=np.full((n,k,33,4),np.nan)
    x[:,0,:,0]=.4;x[:,0,:,1]=np.linspace(.1,.8,33);x[:,0,:,3]=1
    return x


def test_face_recovers_changed_shirt_and_rejects_mismatched_face():
    p=poses(4);p[:,1]=p[:,0];p[:,1,:,0]=.7
    features=np.full((4,2,128),np.nan);ref=np.eye(128)[0]
    features[1,0]=ref;features[2,0]=-ref;features[3,0]=ref;features[3,1]=ref
    config=load_config();config['face_identity']['face_probe_frames']=1
    result,meta=face_identity.refine_single(p,features,[ref],np.array([0,-1,0,0]),[(1,0)],30,1280,720,config)
    assert list(result)==[0,0,-1,-1]
    assert meta['methods']==['anchor','face','absent','absent']


def test_face_unobservable_uses_clothing_and_assignment_is_exclusive():
    p=poses(1);emb=np.full((1,2,128),np.nan)
    result,_=face_identity.refine_single(p,emb,[np.eye(128)[0]],[0],[],30,1280,720,load_config())
    assert list(result)==[0]
    faces=np.array([[10,10,100,100,0,0,0,0,512,72,0,0,0,0,1]])
    assigned=face_identity.associate_faces(np.repeat(p[0,:1],2,axis=0),faces,1280,720,load_config()['face_identity'])
    assert len(assigned)==1


def test_auto_start_requires_solo_and_does_not_align_into_others_speech():
    p=poses();p[:30,1]=p[:30,0];p[:30,1,:,0]=.7
    data={'poses_all':p,'width':1280,'height':720};points={'fps':30,'target_index':np.zeros(150,int)}
    assert auto_start.find_start(points,data,[30,45],[(0,1.5)],load_config())==(1.5,1.5)
    with pytest.raises(ValueError,match='sustained'):
        auto_start.find_start(points,data,[],[(0,5)],load_config())


def test_auto_start_rebases_anchors_and_bounds_duration():
    p=poses(150);data={'poses_all':p,'width':1280,'height':720}
    points={'fps':30,'target_index':np.zeros(150,int)}
    assert auto_start.rebase([(1,0),(1,2),(1,4)],points,data,1.5,2)==[(1,0.),(1,.5)]


def test_jev_compliant_size_preserves_aspect_with_padding():
    settings=load_config()['jev_images']
    for shape in ((10,10),(1000,50),(50,3000),(720,1280)):
        output=jev_images.compliant(np.ones((*shape,3),np.uint8),settings)
        h,w=output.shape[:2]
        assert 256<=h<=1568 and 256<=w<=1568 and h*w>=78400


def test_uuid_assets_and_box_wire_format(tmp_path):
    p=tmp_path/'x.jpg';import cv2
    cv2.imwrite(str(p),np.ones((280,280,3),np.uint8))
    state,assets=jev_images.bind({'frames':[str(p)],'speaker':'X'})
    import uuid
    uuid.UUID(state['frames'][0])
    seen={}
    def transport(url,headers,body,timeout):
        seen.update(json.loads(body));return 200,json.dumps({'answers':{},'usage':{'cost':.0001}})
    openrouter.decide(state,{'b':{'type':'box','instructions':'target in frames'}},assets=assets,api_key='test',transport=transport)
    assert seen['assets']==assets and seen['questions']['b']=={'type':'box','instructions':'target in frames'}


def test_description_no_presence_never_requests_box(tmp_path):
    class Client:
        def decide(self,state,questions,*a,**kw):
            assert list(questions)==['present']
            return {'raw':{'answers':{'present':{'noul':.2}}},'cost':0}
    import cv2
    video=tmp_path/'v.mp4'
    # Decode is replaced with an actual pixel array; this test checks routing only.
    from unittest.mock import patch
    data={'fps':30,'width':1280,'height':720,'poses_all':poses(1)}
    with patch.object(jev_images,'read_frame',return_value=np.ones((720,1280,3),np.uint8)):
        with pytest.raises(ValueError,match='No unambiguous'):
            target_desc.select(video,data,'white shirt',0,tmp_path,load_config(),Client())


def test_box_refuses_no_overlap_and_ambiguous_people():
    p=poses(1)[0];p[1]=p[0]
    answer={'box':{'left':.3,'right':.5,'top':.1,'bottom':.8}}
    with pytest.raises(ValueError,match='unique'):
        target_desc.match_box(answer,p,1280,720,load_config()['target_desc'],.5)


def test_budget_records_every_response_and_blocks_unknown_cost(tmp_path,monkeypatch):
    ledger=tmp_path/'ledger';monkeypatch.setenv('JEVTELLS_API_LEDGER',str(ledger))
    budget.before_request();budget.record({'status':200,'cost':.01,'model':'test'})
    assert json.loads(ledger.read_text())['total_cost']==.01
    budget.record({'status':401,'cost':None})
    with pytest.raises(RuntimeError,match='unknown'):
        budget.before_request()


def test_face_temporal_probe_rejects_isolated_lookalike_score():
    p=poses(20);features=np.full((20,2,128),np.nan);ref=np.eye(128)[0]
    features[:,0]=-ref;features[3,0]=ref
    indices,meta=face_identity.refine_single(p,features,[ref],np.full(20,-1),[],30,1280,720,load_config())
    assert (indices==-1).all()
    features[:,0]=ref
    indices,_=face_identity.refine_single(p,features,[ref],np.full(20,-1),[],30,1280,720,load_config())
    assert (indices==0).all()


def test_default_state_has_exactly_five_fields_and_images_are_opt_in(tmp_path,monkeypatch):
    from jevtells.stages import state
    windows=[{'id':'W00','t0':0,'t1':2,'subtitle':'hello'}]
    config=load_config()
    plain=state.run(windows,{},'room','A',tmp_path,config=config)
    assert set(plain['W00'])=={'scene','speaker','subtitle','voice','measured_actions'}
    config['jev_images']['enabled']=True
    monkeypatch.setattr(jev_images,'extract',lambda *a,**kw:{'W00':['a.jpg','b.jpg','c.jpg']})
    enriched=state.run(windows,{},'room','A',tmp_path,config=config)
    assert enriched['W00']['frames']==['a.jpg','b.jpg','c.jpg']
    config['jev_images']['enabled']=False
    assert 'frames' not in state.run(windows,{},'room','A',tmp_path,config=config)['W00']
