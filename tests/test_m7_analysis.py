import json
import numpy as np
from jevtells.stages import state,presence,narrate_facts
from jevtells.config import load_config


def test_two_person_state_contains_only_current_speaker_actions_and_five_fields(tmp_path):
    windows=[{'id':'W00','t0':0.,'t1':2.,'subtitle':'answer','two_person':True,'speaker':'B'}]
    events=[{'id':'A1','person':'A','t0':0.,'t1':2.,'mid':1.,'limb':'left_hand','type':'raise','magnitude':'large','window':'W00'}, {'id':'B1','person':'B','t0':0.,'t1':2.,'mid':1.,'limb':'right_hand','type':'press_down','magnitude':'large','window':'W00'}]
    points={name:{'t':[],'pose':np.empty((0,33,4))} for name in ['A','B']}
    states=state.run(windows,{},'interview','A',tmp_path,True,points,{'events':events},load_config(),{})
    assert set(states['W00'])=={'scene','speaker','subtitle','voice','measured_actions'}
    assert states['W00']['speaker']=='B'
    assert all('raise' not in a for a in states['W00']['measured_actions'])
    assert any('press_down' in a for a in states['W00']['measured_actions'])
    facts=narrate_facts.build_facts(states,{},windows,events,config=load_config())
    assert facts['W00']['speaker']=='B'
    assert [e['type'] for e in facts['W00']['measured_events']]==['press_down']


def test_unknown_and_offscreen_have_explicit_skip_reasons():
    assert presence.skip_reasons([{'id':'W00','speaker_unknown':True},{'id':'W01','target_offscreen':True}])=={'W00':'speaker_unknown','W01':'target_offscreen'}
