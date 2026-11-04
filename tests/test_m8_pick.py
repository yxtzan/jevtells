import json
from pathlib import Path
import pytest
from jevtells.stages import pick


def test_pick_roundtrip_and_deduplicated_source_time_anchors(tmp_path):
    source = tmp_path/'video.mov'; source.write_bytes(b'real file identity')
    target = tmp_path/'target.json'
    pick.save(target, [(2,4.),(1,0.),(3,4.)], source)
    assert pick.load(target, source) == [(1,0.),(3,4.)]
    source.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        pick.load(target, source)


def test_explicit_target_has_priority_even_over_invalid_pick_file(tmp_path):
    (tmp_path/'target.json').write_text('invalid')
    assert pick.resolve(['2@1'], tmp_path, tmp_path/'other', tmp_path/'x') == ([(2,1.)], None)


def test_run_finds_canonical_pick_with_duration_specific_output(tmp_path):
    source = tmp_path/'v'; source.touch(); canonical = tmp_path/'canonical'
    pick.save(canonical/'target.json', [(2,0.)], source)
    assert pick.resolve([],tmp_path/'duration52',canonical,source)[0] == [(2,0.)]


def test_pick_headless_explains_manual_fallback(monkeypatch):
    monkeypatch.setenv('SSH_CONNECTION','test')
    with pytest.raises(RuntimeError,match='people.*--target'):
        pick.require_gui()
