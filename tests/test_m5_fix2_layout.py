from jevtells.render.layout_audit import line_flags
from jevtells.render.placement import select_positions


def test_endpoint_circle_exempts_face_and_midline_only_near_hand():
    assert line_flags([[0,50],[35,50],[100,50]],[80,40,105,60],90,60)==(False,False)
    assert line_flags([[0,50],[35,50],[100,50]],[20,40,35,60],30,60)==(True,True)
    # Euclidean clipping applies to diagonal paths as well.
    assert line_flags([[0,0],[40,0],[100,100]],[95,95,105,105],98,60)==(False,False)


def solve(samples, forbidden=(), exemption=60):
    return select_positions([0,0,500,400],[200,100,300,350],forbidden,
            {'left':[70,40],'right':[70,40]}, {'left':[140,200],'right':[380,200]},
            samples=samples,midline=250,step=36,margin=20,gap=10,exemption=exemption)


def test_95_percent_threshold_uses_all_true_display_frames():
    normal={'anchor':[140,200],'face':None,'midline':250}
    impossible={'anchor':[140,200],'face':[0,0,500,400],'midline':250}
    other={'anchor':[380,200],'face':None,'midline':250}
    valid=solve({'left':[normal]*19+[impossible],'right':[other]*20}, exemption=0)
    assert valid['selected']['left']['line_pass_rate']==.95
    assert not valid['selected']['left']['relaxed']
    invalid=solve({'left':[normal]*18+[impossible]*2,'right':[other]*20}, exemption=0)
    assert invalid['selected']['left']['relaxed']
    assert invalid['selected']['left']['same_side']
    assert not invalid['fallback']


def test_hand_side_uses_midline_instead_of_left_right_anatomical_names():
    data={'left':[{'anchor':[400,200],'midline':250,'face':None}],
          'right':[{'anchor':[350,280],'midline':250,'face':None}]}
    result=solve(data)
    assert all(v['same_side'] for v in result['selected'].values())
    assert all((v['rect'][0]+v['rect'][2])/2>250 for v in result['selected'].values())


def test_forced_position_is_face_minimal_and_keeps_hand_side():
    data={'left':[{'anchor':[140,200],'face':None,'midline':250}],
          'right':[{'anchor':[380,200],'face':None,'midline':250}]}
    result=solve(data,[[0,0,500,400]])
    assert result['forced'] and not result['fallback']
    assert all(v['same_side'] for v in result['selected'].values())
