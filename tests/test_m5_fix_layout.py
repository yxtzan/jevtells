import numpy as np
from jevtells.render.body_zones import body_zones
from jevtells.render.placement import select_positions
from jevtells.render.layout_audit import leader_violations, segment_intersects_rect


def test_face_padding_and_invisible_hips_use_bottom():
    pose = np.full((33,4),np.nan)
    pose[:11]=[.5,.2,0,1]
    pose[11]=[.4,.4,0,1];pose[12]=[.6,.4,0,1]
    pose[23]=[.4,.8,0,.2];pose[24]=[.6,.8,0,.2]
    zones=body_zones(pose,(1000,700))
    assert np.allclose(zones['face'],[470,110,530,170])
    assert np.allclose(zones['torso'],[400,280,600,700])


def test_same_side_preferred_and_opposite_leader_cannot_cross_core():
    target=[350,200,650,650]
    result=select_positions([0,0,1000,700],target,[target],{'left':[150,60],'right':[150,60]},{'left':[100,350],'right':[800,350]},{'left':[24,100],'right':[826,100]},protected=[target],soft=[[24,200,300,500]])
    assert not result['fallback']
    assert result['positions']['left'][0]<500
    assert result['positions']['right'][0]>500
    assert all(v['same_side'] for v in result['selected'].values())


def test_frame_audit_counts_actual_face_and_torso_leaders():
    frame={'midline':80,'labels':[{'slot':'left','event':'a','leader':[[0,50],[20,50],[100,50]]}],'leader_zones':[{'name':'face','rect':[40,20,60,80]},{'name':'torso','rect':[70,20,90,80]}]}
    assert [e['zone'] for e in leader_violations(frame)]==['face','midline']
    assert segment_intersects_rect([0,50],[100,50],[40,20,60,80])
    assert not segment_intersects_rect([0,0],[100,0],[40,20,60,80])
