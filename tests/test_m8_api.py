"""Experiment failures must remain visible without losing completed evidence."""
import json
import runpy
from pathlib import Path

import numpy as np


def test_description_rejections_are_recorded_and_resumable(tmp_path,monkeypatch):
    module=runpy.run_path('scripts/m8_experiments.py')
    from jevtells.stages import target_desc
    from jevtells.config import load_config
    source=tmp_path/'source';source.mkdir()
    np.savez(source/'detections.npz',poses_all=np.zeros((1,1,33,4)),fps=30,width=1280,height=720)
    calls=[]
    def reject(*args,**kwargs):
        calls.append(args[2]);raise ValueError('No unambiguous person')
    monkeypatch.setattr(target_desc,'select',reject)
    results=module['evaluate_descriptions'](source,'tim',tmp_path/'out',load_config(),True)
    assert len(results)==6 and len(calls)==6
    assert all(r['status']=='rejected' and r['selected'] is None for r in results)
    assert all(r['reason']=='No unambiguous person' for r in results)
    again=module['evaluate_descriptions'](source,'tim',tmp_path/'out',load_config(),True)
    assert again==results and len(calls)==6


def test_unknown_failure_reservation_counts_against_total_budget(tmp_path,monkeypatch):
    import pytest
    from jevtells.clients import budget
    ledger=tmp_path/'ledger.jsonl'
    monkeypatch.setenv('JEVTELLS_API_LEDGER',str(ledger))
    monkeypatch.setenv('JEVTELLS_API_LIMIT','0.30')
    monkeypatch.setenv('JEVTELLS_API_RESERVE','0.05')
    ledger.write_text(json.dumps({'cost':None,'reserved_cost':.05})+'\n')
    budget.before_request()
    budget.record({'status':200,'cost':.21})
    row=json.loads(ledger.read_text().splitlines()[-1])
    assert row['total_cost']==.21 and row['reserved_cost_total']==.05
    with pytest.raises(RuntimeError,match='exhausted'):
        budget.before_request()


def test_face_frames_use_safe_dimensions_without_changing_body_frames(tmp_path,monkeypatch):
    from jevtells.stages import jev_images
    from jevtells.config import load_config
    config=load_config();shapes=[]
    monkeypatch.setattr(jev_images,'select_frames',lambda *a:[0])
    monkeypatch.setattr(jev_images,'read_frame',lambda *a:np.zeros((720,1280,3),np.uint8))
    def crop(frame,pose,hands,settings,face_only=False):
        shapes.append((settings['min_side'],settings['min_pixels']))
        return jev_images.compliant(np.ones((50,70,3),np.uint8),settings),[0,0,70,50]
    monkeypatch.setattr(jev_images,'crop_target',crop)
    points={'pose':np.zeros((1,33,4)),'hands':np.zeros((1,2,21,3)),'fps':30}
    jev_images.extract('clip',tmp_path,[{'id':'W00'}],points,{},config['jev_images'],True)
    jev_images.extract('clip',tmp_path,[{'id':'W00'}],points,{},config['jev_images'],False)
    assert shapes==[(320,102400),(256,78400)]


def test_comparison_summary_uses_completed_windows_and_actual_confidences():
    module=runpy.run_path('scripts/m8_experiments.py')
    def row(intent,emotion,ci,ce):
        return {'status':'completed','judgment':{'intent':{'label':intent,'confidence':ci},'emotion':{'label':emotion,'confidence':ce}},'cost':.001,'elapsed_s':1.5}
    groups={'1.13-text':{'W00':row('explain','calm',.8,.6),'W01':{'status':'skipped'}},
            '1.14-text':{'W00':row('explain','firm',.9,.7),'W01':{'status':'skipped'}},
            '1.14-images':{'W00':row('emphasize','firm',.95,.3),'W01':{'status':'skipped'}}}
    summary,diff=module['summarize_comparison'](groups,.4)
    assert summary['1.13-text']['average_choice_confidence']==.7
    assert summary['1.14-images']['not_obvious_windows']==1
    assert summary['1.14-images']['completed']==1
    assert diff['all_three']['intent_different']==1 and diff['all_three']['emotion_different']==1
    assert diff['pairwise']['1.14-text vs 1.14-images']['emotion_different']==0
