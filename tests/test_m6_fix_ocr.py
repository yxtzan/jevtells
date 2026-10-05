import numpy as np
import pytest
from jevtells.stages.ocr import clean_text, frame_spans, resolve_source, select_lines
from jevtells.stages.segment import run


def test_subtitle_source_priority_and_explicit_override():
    assert resolve_source('auto','x.srt','off',True)=='srt'
    assert resolve_source('auto',None,'off',True)=='ocr'
    assert resolve_source('auto',None,'off',False)=='asr'
    assert resolve_source('asr',None,'off',True)=='asr'
    with pytest.raises(ValueError): resolve_source('srt',None,'on',True)


def test_ocr_bilingual_rows_and_cjk_spacing():
    rows=[([[0,0],[80,0],[80,20],[0,20]],'中文 字幕',.98),([[0,30],[180,30],[180,50],[0,50]],'Every layer changes',.99)]
    assert select_lines(rows,'en',.8)==('Every layer changes','中文字幕',.99)
    assert select_lines(rows,'zh',.8)==('中文字幕','Every layer changes',.98)
    assert clean_text('转 载 AI 平台')=='转载 AI 平台'


def test_ocr_frame_boundaries_follow_caption_change():
    a=np.zeros((10,10),dtype='uint8');b=a.copy();b[1:5,1:5]=1
    assert frame_spans([a,a,b,b,a],{'change_ratio':.35})==[(0,2),(2,4),(4,5)]


def test_ocr_cues_remain_separate_and_translation_survives(tmp_path):
    transcript={'subtitle_source':'ocr','segments':[{'t0':0,'t1':1,'text':'第一句','subtitle_translation':'First'},{'t0':1,'t1':2,'text':'第二句'}]}
    result=run(transcript,[],tmp_path)
    assert len(result)==2 and result[0]['subtitle_translation']=='First'
