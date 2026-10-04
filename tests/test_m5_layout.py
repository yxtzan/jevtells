from jevtells.render.layout_audit import frame_violations, intersects


def test_layout_audit_catches_collisions_and_records_fallbacks():
    frame = {"labels": [{"slot": "left", "event": "a", "rect": [10,10,40,40], "fallback": False}, {"slot": "right", "event": "b", "rect": [20,20,50,50], "fallback": True}], "forbidden": [{"name": "target", "rect": [30,30,60,60]}]}
    failures = frame_violations(frame)
    assert failures[0]["zones"] == ["target"] and not failures[0]["fallback"]
    assert failures[1]["zones"] == ["target", "label:left"] and failures[1]["fallback"]
    assert intersects([0,0,10,10], [10,0,20,10]) is False
