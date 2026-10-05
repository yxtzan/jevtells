from jevtells.render.placement import card_side


def test_card_defaults_right_unless_left_space_at_least_one_and_half():
    assert card_side([330,100,960,700],1280,280,24)=='right'
    assert card_side([270,100,1020,700],1280,280,24)=='right'
    assert card_side([270,100,1100,700],1280,280,24)=='left'
    assert card_side([100,100,500,700],1280,280,24)=='right'
    assert card_side([750,100,1130,700],1280,280,24)=='left'
