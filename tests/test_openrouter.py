"""Offline contract tests for the OpenRouter client."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jevtells.clients.openrouter import (
    CHAT_URL,
    DECIDE_URL,
    MISSING_KEY,
    OpenRouterClient,
    OpenRouterError,
)


def _transport_sequence(responses, calls):
    def transport(url, headers, body, timeout):
        calls.append((url, headers, json.loads(body.decode("utf-8")), timeout))
        response = responses.pop(0) if len(responses) > 1 else responses[0]
        return response

    return transport


def test_decide_request_shape_and_cost():
    calls = []
    response = {"id": "jev-1", "usage": {"cost": 0.0012, "total_tokens": 11}, "decisions": []}
    client = OpenRouterClient(api_key="test-secret", transport=_transport_sequence([(200, json.dumps(response))], calls), sleep=lambda _: None)
    result = client.decide({"scene": "office", "speaker": "A"}, [{"id": "confidence", "type": "score"}], "typesafe/jev-1.13")
    assert calls[0][0] == DECIDE_URL
    assert calls[0][2] == {
        "model": "typesafe/jev-1.13",
        "state": {"scene": "office", "speaker": "A"},
        "questions": {"confidence": {"type": "score", "instructions": "", "criteria": []}},
    }
    assert calls[0][1]["Authorization"] == "Bearer test-secret"
    assert result["response"] == response
    assert result["raw"] == response
    assert result["usage"]["total_tokens"] == 11
    assert result["cost"] == 0.0012


def test_chat_request_shape_and_image(tmp_path: Path):
    image = tmp_path / "frame.png"
    image.write_bytes(b"fake-png")
    calls = []
    client = OpenRouterClient(api_key="secret", transport=_transport_sequence([(200, '{"choices": [], "usage": {}}')], calls), sleep=lambda _: None)
    client.chat([{"role": "user", "content": "describe"}], "vision/model", image=image, response_format={"type": "json_object"})
    assert calls[0][0] == CHAT_URL
    payload = calls[0][2]
    assert payload["model"] == "vision/model"
    assert payload["response_format"] == {"type": "json_object"}
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retryable_responses_retry_three_times(status):
    calls = []
    sleeps = []
    responses = [(status, '{"error":"temporary"}')] * 4
    client = OpenRouterClient(
        api_key="secret",
        transport=_transport_sequence(responses, calls),
        sleep=sleeps.append,
        attempts=4,
    )
    with pytest.raises(OpenRouterError) as raised:
        client.decide({}, [], "model")
    assert raised.value.status == status
    assert len(calls) == 4  # initial attempt plus three retries
    assert sleeps == [2, 4, 8]


def test_bad_request_does_not_retry_and_redacts_key():
    calls = []
    key = "super-secret-key"
    body = json.dumps({"error": f"invalid token {key}"})
    client = OpenRouterClient(api_key=key, transport=_transport_sequence([(400, body)], calls), sleep=lambda _: None)
    with pytest.raises(OpenRouterError) as raised:
        client.chat([], "model")
    assert len(calls) == 1
    assert raised.value.status is None or raised.value.status == 400
    assert key not in str(raised.value)


def test_missing_key_error_is_actionable(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(OpenRouterError, match="缺少 OPENROUTER_API_KEY") as raised:
        OpenRouterClient(env_path=tmp_path / "missing.env").decide({}, [], "model")
    assert str(raised.value) == MISSING_KEY
