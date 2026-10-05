import json
from jevtells.stages.narrate import run, _fallback
from jevtells.stages.narrate_facts import verbal_highlights
from jevtells.stages.narrate_validation import validation_errors


def test_batch_retries_only_failed_lines_and_freezes_accepted(tmp_path):
    class Client:
        calls = 0
        def chat(self, messages, model, **kwargs):
            self.calls += 1
            if self.calls == 1:
                data = [{'id': 'W0', 'line': '解释芯片设计', 'quote': 'hello world'}, {'id': 'W1', 'line': '数值升到0.9', 'quote': 'next phrase'}]
            else:
                assert '解释芯片设计' in messages[0]['content']
                assert 'Arabic digits' in messages[0]['content']
                data = [{'id': 'W1', 'line': '强调新的重点', 'quote': 'next phrase'}]
            return {'raw': {'choices': [{'message': {'content': json.dumps({'lines': data})}}]}, 'cost': .001}
    states = {key: {'subtitle': {'current': sub}} for key, sub in [('W0', 'hello world'), ('W1', 'next phrase')]}
    client = Client()
    result = run(states, {}, tmp_path, client=client)
    meta = json.loads((tmp_path / 'narrate_meta.json').read_text())
    assert client.calls == 2 and meta['retries'] == 1 and meta['fallbacks'] == 0
    assert result['W0']['line'] == '解释芯片设计'


def test_qualitative_highlights_thresholds_and_fallback_has_no_digits():
    highlights = ['自信度比上一句 +0.10', '专注度比上一句 -0.08', '紧张度 0.90，全场最高']
    assert verbal_highlights(highlights) == ['自信度明显上升', '专注度有所下降', '紧张度升至全场最高']
    facts = {'subtitle': 'hello world', 'highlights': highlights, 'gestures': ['双手同时下压（幅度大，表达欲 0.8）']}
    line = _fallback(facts, 'zh', ['双手同时展开'])
    assert line['line'] == '双手同时下压，自信度明显上升'
    assert validation_errors(line, facts, [], '', 'zh') == []
    assert any('digits' in e for e in validation_errors({'line':'升到0.8','quote':'hello world'}, facts, [], '', 'zh'))
