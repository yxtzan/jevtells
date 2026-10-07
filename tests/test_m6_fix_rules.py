import copy
import numpy as np
from jevtells.config import load_config
from jevtells.stages.narrate_validation import validation_errors
from jevtells.stages.narrate_facts import build_facts
from jevtells.stages.voice import window_voice_metrics, refresh_window_metrics
from jevtells.stages.state import _actions
from jevtells.actions.features import extract_features
from jevtells.actions.rules import _fist_valid
from jevtells.render.labels import schedule


def errors(line,**kwargs):
    return validation_errors({'line':line,'quote':'转载'}, {'subtitle':'然后会被转载','highlights':[],'gestures':[],**kwargs}, [], '', 'zh')


def test_first_judged_window_rejects_direction_even_with_extreme_fact():
    assert errors('紧张度降至全场最低',first_judged=True,highlights=['紧张度 0.2，全场最低'])
    assert not errors('紧张度处于全场最低',first_judged=True,highlights=['紧张度 0.2，全场最低'])


def test_narration_rejects_quantified_chinese_rate_and_counts():
    assert errors('语速达到每秒五字')
    assert errors('左手抬起三次')
    assert not errors('语速偏快，继续谈论「转载」')


def test_large_motion_claim_requires_identical_action_and_hand():
    events=[{'limb':'right_hand','type':'raise','magnitude':'large'},{'limb':'right_hand','type':'press_down','magnitude':'medium'}]
    assert errors('右手大动作下压，谈及「转载」',measured_events=events)
    assert not errors('右手大幅抬起，谈及「转载」',measured_events=events)
    assert errors('左手大幅抬起，谈及「转载」',measured_events=events)


def test_chinese_corner_quote_must_be_unspaced_substring():
    assert errors('谈到「转 载」时语速偏快')
    assert errors('谈到「转发」时语速偏快')
    assert not errors('谈到「转载」时语速偏快')


def test_chinese_rate_counts_characters_and_english_counts_words(tmp_path):
    zh={'language':'zh','segments':[{'words':[{'t0':0,'t1':1,'w':'短视频'},{'t0':1,'t1':2,'w':'平台'}]}]}
    en={'language':'en','segments':[{'words':[{'t0':0,'t1':1,'w':'video'},{'t0':1,'t1':2,'w':'platform'}]}]}
    assert window_voice_metrics({}, {'t0':0,'t1':2},zh)['speech_rate']==2.5
    assert window_voice_metrics({}, {'t0':0,'t1':2},en)['speech_rate']==1


def test_unobserved_hands_are_not_no_gestures():
    points={'t':np.arange(10)/10,'hands':np.full((10,2,21,3),np.nan)}
    assert _actions(points,{'t0':0,'t1':1},{'events':[]},load_config())==['hands mostly not visible']
    assert errors('没有明显动作，继续讨论转载',hands_mostly_not_visible=True)


def test_shot_boundary_breaks_smoothing_and_velocity():
    pose=np.tile([.3,.3,0,1.],(60,33,1));pose[:,11,0]=.2;pose[:,12,0]=.4;pose[30:,:,0]+=.3
    points={'pose':pose,'hands':np.full((60,2,21,3),np.nan),'t':np.arange(60)/30,'fps':30,'width':1280,'height':720,'cut_frames':[30]}
    features=extract_features(points)
    assert np.isnan(features['wrist_velocity_px_s'][30]).all()
    assert np.nanmax(np.abs(features['wrist_velocity_px_s']))<1e-8


def test_pinch_with_two_extended_fingers_cannot_be_fist():
    features={'hand_open':np.zeros((10,2)),'fingertip_palm_ratio':np.ones((10,2,4))*.5,'pinch_tip_palm_ratio':np.ones((10,2))*.1,'finger_straight':np.tile([0,1,1,0,0],(10,2,1))}
    assert not _fist_valid(features,np.arange(10)/10,{'t0':0,'t1':.9,'side':'left hand'},load_config()['actions'])


def test_new_same_hand_label_crossfades_during_previous_hold():
    settings=load_config()['render']
    events=[{'id':'a','limb':'left_hand','type':'raise','magnitude':'large','t0':0,'t1':1,'shot_index':1},{'id':'b','limb':'left_hand','type':'press_down','magnitude':'medium','t0':1.3,'t1':2,'shot_index':1}]
    event,alpha,scale=schedule(events,1.35,settings)['right']
    assert event['id']=='b' and event['_replaces']['id']=='a'
    assert abs(event['_replace_amount']-.5)<1e-8 and alpha==scale==1


def test_uncertain_choices_and_transitions_are_absent_from_facts():
    states={k:{'subtitle':{'current':'原文字幕'},'measured_actions':[]} for k in ['W0','W1']}
    judgments={'W0':{'intent':{'label':'explain','confidence':.9},'emotion':{'label':'calm','confidence':.9}},'W1':{'intent':{'label':'ask','confidence':.39},'emotion':{'label':'firm','confidence':.39}}}
    facts=build_facts(states,judgments,[{'id':'W0'},{'id':'W1'}],config=load_config())
    assert 'intent' not in facts['W1']['judgments'] and 'emotion' not in facts['W1']['judgments']
    assert facts['W1']['highlights']==[]


def test_factual_fallback_survives_uncertain_emotion_without_highlights():
    from jevtells.stages.narrate import _fallback
    facts = {'subtitle': '好我觉得为什么这个我会回应啊',
             'gestures': ['双手同时抬手（幅度大）'], 'highlights': [],
             'judgments': {'intent': {'label': '提问'}}}
    result = _fallback(facts, 'zh', [])
    assert result['line'] == '双手同时抬手'
    assert not validation_errors(result, facts, [], '', 'zh')
