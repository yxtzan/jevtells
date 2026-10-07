import json
import pytest
from jevtells.clients import budget


def test_budget_tracks_each_response_and_refuses_unresolved_or_exhausted_cost(tmp_path,monkeypatch):
    path=tmp_path/'ledger.jsonl';monkeypatch.setenv('JEVTELLS_API_LEDGER',str(path));monkeypatch.setenv('JEVTELLS_API_LIMIT','.30')
    budget.before_request();budget.record({'cost':.20,'model':'test'});budget.before_request()
    budget.record({'cost':.08,'model':'test'})
    with pytest.raises(RuntimeError,match='exhausted'): budget.before_request()
    rows=[json.loads(s) for s in path.read_text().splitlines()];assert rows[-1]['total_cost']==pytest.approx(.28)
    path.unlink();budget.record({'cost':None})
    with pytest.raises(RuntimeError,match='unknown'): budget.before_request()
