from jevtells.render.layout_audit import frame_violations, intersects
from jevtells.render.placement import card_side, crosses, leader, select_positions


def solve(extra=()):
    target=[350,100,650,500]
    return select_positions([0,0,1000,700],target,[target,[0,560,1000,700],*extra],{"left":[120,60],"right":[120,60]},{"left":[420,350],"right":[580,350]},{"left":[40,196],"right":[760,92]})


def test_labels_choose_both_sides_and_are_deterministic():
    result=solve()
    assert not result["fallback"]
    assert result["positions"]["left"][0]+120<=350
    assert result["positions"]["right"][0]>=650
    assert solve()==result


def test_labels_stack_on_one_side_without_crossing():
    result=solve([[0,0,350,700],[350,0,650,100]])
    assert not result["fallback"]
    a,b=result["positions"].values()
    assert a[0]>=650 and b[0]>=650
    assert abs(a[1]-b[1])>=72
    pa,pb=leader([a[0],a[1],a[0]+120,a[1]+60],[420,350]),leader([b[0],b[1],b[0]+120,b[1]+60],[580,350])
    assert not any(crosses(x,y,z,w) for x,y in zip(pa,pa[1:]) for z,w in zip(pb,pb[1:]))


def test_no_legal_location_uses_explicit_legacy_fallback():
    result=solve([[0,0,1000,700]])
    assert result["fallback"] and result["positions"]=={"left":[40,196],"right":[760,92]}
    assert result["fallback_slots"]==["left","right"]


def test_card_side_uses_available_space_and_cubic_slide():
    from jevtells.render.animation import ease_in_out_cubic
    assert card_side([100,100,500,600],1280,280,24)=="right"
    assert card_side([750,100,1130,600],1280,280,24)=="left"
    assert card_side([200,100,1200,600],1280,280,24)=="left"
    assert card_side(None,1280,280,24)=="right"
    assert ease_in_out_cubic(0)==0 and ease_in_out_cubic(1)==1
    assert ease_in_out_cubic(.5)==.5 and ease_in_out_cubic(.25)==.0625


def test_layout_audit_catches_collisions_and_records_fallbacks():
    frame = {"labels": [{"slot": "left", "event": "a", "rect": [10,10,40,40], "fallback": False}, {"slot": "right", "event": "b", "rect": [20,20,50,50], "fallback": True}], "forbidden": [{"name": "target", "rect": [30,30,60,60]}]}
    failures = frame_violations(frame)
    assert failures[0]["zones"] == ["target"] and not failures[0]["fallback"]
    assert failures[1]["zones"] == ["target", "label:left"] and failures[1]["fallback"]
    assert intersects([0,0,10,10], [10,0,20,10]) is False
