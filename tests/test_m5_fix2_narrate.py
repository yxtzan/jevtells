import json
import socket

import pytest

from jevtells.clients.openrouter import OpenRouterClient, OpenRouterError
from jevtells.stages.narrate import _fallback, run
from jevtells.stages.narrate_validation import display_names, validation_errors


def test_http_failures_save_complete_redacted_body_and_per_attempt_cost(tmp_path):
    sleeps, events, calls = [], [], []
    secret = 'sensitive-test-token'
    body = json.dumps({'error': 'x' * 6000 + secret, 'usage': {'cost': .002}})
    def transport(*args):
        calls.append(args)
        return (429, body) if len(calls) < 4 else (200, '{"usage":{"cost":0.001}}')
    client = OpenRouterClient(api_key=secret, transport=transport, sleep=sleeps.append,
                              audit_dir=tmp_path, on_response=events.append)
    client.chat([], 'model')
    assert sleeps == [2, 4, 8]
    assert [e['cost'] for e in events] == [.002, .002, .002, .001]
    saved = list(tmp_path.glob('openrouter_error_*.json'))
    assert len(saved) == 3
    for path in saved:
        text = path.read_text()
        assert secret not in text and 'Authorization' not in text
        record = json.loads(text)
        assert record['status'] == 429 and len(record['body']) > 6000


@pytest.mark.parametrize('status', [400, 401, 403])
def test_permanent_http_error_stops_and_retains_null_cost(tmp_path, status):
    events, calls = [], []
    def transport(*args):
        calls.append(args)
        return status, '{"error":"denied"}'
    with pytest.raises(OpenRouterError):
        OpenRouterClient(api_key='secret', transport=transport, audit_dir=tmp_path,
                         on_response=events.append).chat([], 'model')
    assert len(calls) == 1 and events[0]['cost'] is None
    assert events[0]['cost_reason'] and events[0]['status'] == status
    assert len(list(tmp_path.glob('*.json'))) == 1


def test_timeout_retries_and_records_null_cost(tmp_path):
    sleeps, events = [], []
    def transport(*args):
        raise socket.timeout('timeout containing secret')
    with pytest.raises(OpenRouterError):
        OpenRouterClient(api_key='secret', transport=transport, sleep=sleeps.append,
                         audit_dir=tmp_path, on_response=events.append).chat([], 'model')
    assert sleeps == [2, 4, 8] and len(events) == 4
    assert all(e['status'] is None and e['cost'] is None for e in events)
    assert all('secret' not in p.read_text() for p in tmp_path.glob('*.json'))


def test_all_display_names_and_summary_terms_are_not_quotes_or_openings():
    facts = {'subtitle': 'hello world', 'highlights': []}
    for label in display_names():
        errors = validation_errors({'line': f'谈及「{label}」的变化', 'quote': 'hello world'}, facts, [], '', 'zh')
        assert any('display label' in error for error in errors), label
    for opening in ['强调重点', '解释说明', '坚定', 'explain', 'firm']:
        assert any('starts with an intent' in e for e in validation_errors(
            {'line': opening + '，谈到芯片设计', 'quote': 'hello world'}, facts, [], '', 'zh'))


def test_adjacent_highlight_rejection_also_selects_alternative_fallback():
    facts = {'subtitle': 'hello world', 'gestures': ['右手张开手掌'],
             'highlights': ['自信度持续走低', '专注度比上一句 +0.10']}
    old = '谈及软件训练，自信度持续走低'
    errors = validation_errors({'line': '张开手掌，自信度持续走低', 'quote': 'hello world'}, facts, [old], '', 'zh')
    assert any('adjacent previous' in e for e in errors)
    assert _fallback(facts, 'zh', [old])['line'] == '右手张开手掌，专注度明显上升'


def test_truncated_response_is_rejected_even_if_content_parses(tmp_path):
    class Client:
        calls = 0
        def chat(self, *args, **kwargs):
            self.calls += 1
            data = {'lines': [{'id':'W0','line':'逐层解释芯片设计','quote':'hello world'}]}
            return {'raw': {'choices': [{'finish_reason': 'length' if self.calls == 1 else 'stop',
                      'message': {'content': json.dumps(data)}}]}, 'cost': .001}
    client = Client()
    run({'W0': {'subtitle': {'current': 'hello world'}}}, {}, tmp_path, client=client)
    meta = json.loads((tmp_path/'narrate_meta.json').read_text())
    assert meta['truncated_responses'] == 1 and meta['calls'] == 2 and meta['fallbacks'] == 0
