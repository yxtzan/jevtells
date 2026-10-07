import sys
import pytest
from jevtells.cli import _arguments, _parse_persons


def args(monkeypatch, *values):
    monkeypatch.setattr(sys, 'argv', ['jevtells', 'run', 'input.mov', *values])
    return _arguments()


def test_repeated_person_anchors_and_two_person_limit():
    assert _parse_persons(['A=1@0', 'B=2@2', 'A=1@12']) == {'A': [(1,0.),(1,12.)], 'B': [(2,2.)]}
    for value in (['A=1@nan'], ['A=0@1'], ['=1@0'], ['A=1'], ['A=1@0','B=2@0','C=3@0']):
        with pytest.raises(ValueError): _parse_persons(value)


@pytest.mark.parametrize('option', [['--target','1'],['--speaker','A'],['--others-speaking','0-2']])
def test_person_rejects_legacy_flags(monkeypatch, option):
    with pytest.raises(SystemExit): args(monkeypatch, '--person','A=1@0',*option)


def test_one_person_is_exact_legacy_arguments(monkeypatch):
    one=args(monkeypatch,'--person','Tim=1@0','--person','Tim=1@4.1')
    old=args(monkeypatch,'--target','1@0.0','--target','1@4.1','--speaker','Tim')
    assert (one.target,one.speaker)==(old.target,old.speaker)
