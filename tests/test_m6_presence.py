import json

import numpy as np
import pytest

from jevtells.i18n import load_translations
from jevtells.stages import judge, narrate
from jevtells.stages.presence import mark_presence, presence_ratio, skip_reasons, split_presence_changes


def test_less_than_half_presence_skips_but_exact_half_does_not():
    points={'fps':2,'t':np.arange(4)/2,'pose_present':[True,False,False,True]}
    windows=[{'id':'half','t0':0,'t1':1},{'id':'absent','t0':.5,'t1':1.5}]
    result=mark_presence(windows,points)
    assert not result[0]['target_offscreen'] and result[1]['target_offscreen']
    assert presence_ratio(points,.25,.75)==.5


def test_identity_edit_splits_mixed_sentence_without_duplicate_words():
    windows=[{'id':'W00','index':0,'t0':0,'t1':4,'subtitle':'one two three four'}]
    points={'fps':1,'pose_present':[True,False,False,True]}
    shots=[{'t0':0,'t1':1},{'t0':1,'t1':3},{'t0':3,'t1':4}]
    transcript={'segments':[{'words':[{'t0':i,'t1':i+1,'w':word} for i,word in enumerate(['one','two','three','four'])]}]}
    result=split_presence_changes(windows,shots,points,transcript)
    assert [w['subtitle'] for w in result]==['one','two three','four']
    assert [w['target_offscreen'] for w in result]==[False,True,False]
    assert result[2]['prev_subtitle']=='two three'


def test_other_speaker_takes_precedence_over_offscreen():
    assert skip_reasons([{'id':'W00','speaker_other':True,'target_offscreen':True}])=={'W00':'speaker_other'}


def test_no_identity_change_preserves_sentence_segmentation():
    windows=[{'id':'W00','index':0,'t0':0,'t1':4,'subtitle':'keep'}]
    result=split_presence_changes(windows,[{'t0':0,'t1':2},{'t0':2,'t1':4}],{'fps':1,'pose_present':[True]*4},{})
    assert len(result)==1 and result[0]['subtitle']=='keep'


@pytest.mark.parametrize('other',[False,True])
def test_offscreen_window_makes_no_judge_or_narration_request(tmp_path,other):
    class Forbidden:
        def decide(self,*args,**kwargs): raise AssertionError('offscreen Jev request')
        def chat(self,*args,**kwargs): raise AssertionError('offscreen narrator request')
    windows=[{'id':'W00','t0':0,'t1':2,'target_offscreen':True,'speaker_other':other}]
    states={'W00':{'scene':'interview','speaker':'target','subtitle':{'current':'question','previous':''},'voice':'quiet','measured_actions':[]}}
    (tmp_path/'windows.json').write_text(json.dumps(windows))
    judgments=judge.run(states,tmp_path,client=Forbidden(),config={'stop_on_api_error':True})
    lines=narrate.run(states,judgments,tmp_path,client=Forbidden(),config={'stop_on_api_error':True})
    assert judgments['W00'] is None and lines['W00'] is None
    reason='speaker_other' if other else 'target_offscreen'
    for name in ['judge_meta.json','narrate_meta.json']:
        meta=json.loads((tmp_path/name).read_text())
        assert meta['calls']==0 and meta['skipped']['W00']==reason


def test_offscreen_message_available_in_both_languages():
    assert load_translations('zh')['ui']['target_offscreen']=='（主角不在画面，本句不做判定）'
    assert load_translations('en')['ui']['target_offscreen']=='(Target offscreen; no judgment)'


@pytest.mark.parametrize('lang',['zh','en'])
@pytest.mark.parametrize('other',[False,True])
def test_rendered_offscreen_message_is_gray_and_other_speaker_wins(monkeypatch,lang,other):
    from jevtells.config import load_config
    from jevtells.render.geometry import Layout
    from jevtells.render.text import Fonts
    from jevtells.render import panels as module
    settings=load_config()['render'];layout=Layout.create('h',settings,(1280,720))
    tr=load_translations(lang)
    windows=[{'id':'W00','t0':0,'t1':2,'target_offscreen':True,'speaker_other':other}]
    panels=module.Panels(settings,layout,Fonts(settings,layout.scale),tr,windows,{'W00':None},{'W00':None},'Target','Source')
    seen=[];original=module.draw_fitted
    def capture(image,position,fitted,*args,**kwargs):
        seen.append((''.join(fitted.lines),kwargs.get('fill')))
        return original(image,position,fitted,*args,**kwargs)
    monkeypatch.setattr(module,'draw_fitted',capture)
    panels.commentary(0)
    assert (tr['ui']['other_speaker' if other else 'target_offscreen'],settings['colors']['muted']) in seen


def test_split_windows_refresh_cached_voice_metrics_from_sample_times(tmp_path):
    from jevtells.stages.voice import refresh_window_metrics, window_voice_metrics
    features={'t':[0,.5,1,1.5], 'rms_db':[-10,-10,-30,-30], 'f0_hz':[100]*4, 'voiced':[True]*4, 'baseline_rms_db':-20, 'hop_seconds':.5, 'metrics_by_window':{'stale':{'loudness_delta_db':99}}}
    windows=[{'id':'W00','t0':0,'t1':1},{'id':'W01','t0':1,'t1':2}]
    transcript={'segments':[{'words':[{'t0':0,'t1':.5,'w':'one'},{'t0':1,'t1':1.5,'w':'two'}]}]}
    result=refresh_window_metrics(features,windows,transcript,tmp_path)
    assert set(result['metrics_by_window'])=={'W00','W01'}
    for window in windows:
        measured = dict(result['metrics_by_window'][window['id']])
        assert measured.pop('speech_rate_band')=='适中'
        assert measured==window_voice_metrics(features,window,transcript)
    assert result['t']==features['t'] and features['metrics_by_window']=={'stale':{'loudness_delta_db':99}}
