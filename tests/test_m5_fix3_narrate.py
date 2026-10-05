import json

import pytest

from jevtells.clients.openrouter import OpenRouterClient, OpenRouterError
from jevtells.stages.narrate import run


def response(finish="stop", tokens=13):
    return json.dumps({
        "choices": [{"finish_reason": finish, "message": {"content": json.dumps({
            "lines": [{"id": "W0", "line": "逐层解释芯片设计", "quote": "hello world"}]
        })}}],
        "usage": {"cost": .001, "prompt_tokens": 20, "completion_tokens": 30,
                  "completion_tokens_details": {"reasoning_tokens": tokens}},
    })


@pytest.mark.parametrize("reasoning", [None, {"effort": "medium"}])
def test_length_retry_doubles_limit_without_spending_validation_retry(tmp_path, reasoning):
    requests, events = [], []
    def transport(url, headers, body, timeout):
        requests.append(json.loads(body))
        return 200, response("length" if len(requests) == 1 else "stop")
    settings = {"validation_retries": 0}
    if reasoning is not None:
        settings["reasoning"] = reasoning
    client = OpenRouterClient(api_key="secret", transport=transport, on_response=events.append)
    result = run({"W0": {"subtitle": {"current": "hello world"}}}, {}, tmp_path,
                 config={"narrate": settings}, client=client)
    assert result["W0"]["line"] == "逐层解释芯片设计"
    assert [r["max_tokens"] for r in requests] == [4000, 8000]
    assert all(r["reasoning"] == (reasoning or {"effort": "low"}) for r in requests)
    meta = json.loads((tmp_path / "narrate_meta.json").read_text())
    assert meta["fallbacks"] == 0 and meta["failed"] == 1
    assert meta["calls"] == 2 and meta["retries"] == 0 and meta["length_retries"] == 1
    assert meta["reasoning_tokens"] == 26 and meta["cost"] == .002
    assert [r["finish_reason"] for r in meta["response_records"]] == ["length", "stop"]
    assert [e["reasoning_tokens"] for e in events] == [13, 13]
    assert [e["max_tokens"] for e in events] == [4000, 8000]
    assert len(list((tmp_path / "raw").glob("narrate_batch_*.json"))) == 2


def test_second_truncation_is_never_accepted_or_retried_indefinitely(tmp_path):
    requests = []
    def transport(url, headers, body, timeout):
        requests.append(json.loads(body))
        return 200, response("length")
    result = run({"W0": {"subtitle": {"current": "hello world"}}}, {}, tmp_path,
                 config={"narrate": {"validation_retries": 0}},
                 client=OpenRouterClient(api_key="secret", transport=transport))
    assert len(requests) == 2 and result["W0"] is None
    meta = json.loads((tmp_path / "narrate_meta.json").read_text())
    assert meta["truncated_responses"] == meta["failed"] == 2
    assert meta["unavailable"] == 1 and meta["cost"] == .002


@pytest.mark.parametrize("metadata,cost,expected", [
    ({"provider_name": None}, None, 0.0),
    ({}, None, None),
    ({"provider_name": "Provider"}, None, None),
    ({"provider_name": None}, .002, .002),
])
def test_only_explicit_unrouted_400_is_zero_and_known_cost_takes_precedence(tmp_path, metadata, cost, expected):
    events, calls = [], []
    body = {"error": {"message": "denied", "metadata": metadata}}
    if cost is not None:
        body["usage"] = {"cost": cost}
    def transport(*args):
        calls.append(args)
        return 400, json.dumps(body)
    with pytest.raises(OpenRouterError):
        run({"W0": {"subtitle": {"current": "hello world"}}}, {}, tmp_path,
            client=OpenRouterClient(api_key="secret", transport=transport, on_response=events.append))
    assert len(calls) == 1 and events[0]["cost"] == expected
    assert events[0]["finish_reason"] is None and events[0]["reasoning_tokens"] is None
    assert json.loads((tmp_path / "narrate_error_meta.json").read_text())["cost"] == expected
    saved = json.loads(next((tmp_path / "raw").glob("openrouter_error_*.json")).read_text())
    assert saved["cost"] == expected
    if expected == 0:
        assert "未路由，按 0 计" in saved["cost_reason"]
