"""Synthetic tests for M1 segmentation and wrist matching."""

import numpy as np

from jevtells.stages.segment import run
from jevtells.utils.geometry import match_hands_to_pose


def test_short_sentences_merge(tmp_path):
    """Keep short adjacent sentence windows available for downstream merging."""
    transcript = {"segments": [{"t0": 0, "t1": 1, "text": "one", "words": [{"t0": 0, "t1": 1, "w": "one"}]}, {"t0": 1, "t1": 3.5, "text": "two", "words": [{"t0": 1, "t1": 3.5, "w": "two"}]}]}
    windows = run(transcript, [{"t0": 0, "t1": 4}], tmp_path, True)
    assert len(windows) == 1
    assert windows[0]["t1"] == 3.5


def test_long_sentence_splits_at_word_boundary(tmp_path):
    """Split a long segment no later than five seconds at a word end."""
    words = [{"t0": index, "t1": index + 1, "w": str(index)} for index in range(8)]
    transcript = {"segments": [{"t0": 0, "t1": 8, "text": " ".join(str(index) for index in range(8)), "words": words}]}
    windows = run(transcript, [], tmp_path, True)
    assert len(windows) == 2
    assert windows[0]["t1"] == 5
    assert windows[1]["t0"] == 5


def test_silence_window(tmp_path):
    """Create a silence window when no transcript segments exist."""
    windows = run({"segments": []}, [{"t0": 0, "t1": 6, "label": "target"}], tmp_path, True)
    assert windows[0]["kind"] == "silence"


def test_left_right_matches_pose_wrist_nearest():
    """Match wrists to pose indices 15 and 16 without handedness labels."""
    pose = np.full((33, 4), np.nan)
    pose[15, :2] = [0.2, 0.5]
    pose[16, :2] = [0.8, 0.5]
    hands = np.array([[0.79, 0.5, 0.1], [0.21, 0.5, 0.1]])
    assert match_hands_to_pose(hands, pose).tolist() == [1, 0]
