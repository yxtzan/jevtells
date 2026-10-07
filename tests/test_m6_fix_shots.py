import cv2
import numpy as np
from jevtells.stages.shots import detect_hard_cuts


def test_single_frame_flash_is_not_a_cut_and_real_edit_remains(tmp_path):
    path=tmp_path/'flash.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),30,(160,90))
    for index in range(90):
        color=(0,0,255) if index<60 else (0,255,0)
        if index==30:color=(255,0,0)
        writer.write(np.full((90,160,3),color,np.uint8))
    writer.release()
    assert detect_hard_cuts(path,{'persistence_frames':3})['cuts']==[60]


def test_shared_cut_detector_merges_short_shot_into_previous(tmp_path):
    path=tmp_path/'short.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),30,(160,90))
    for color in [(0,0,255)]*30+[(255,0,0)]*6+[(0,255,0)]*30:
        writer.write(np.full((90,160,3),color,np.uint8))
    writer.release()
    assert detect_hard_cuts(path,{'persistence_frames':3})['cuts']==[36]
