"""Synthetic M2 tracking and gesture tests (no video or model dependency)."""

import numpy as np

from jevtells.actions.features import extract_features, interpolate_short_gaps
from jevtells.actions.rules import detect_actions
from jevtells.stages.track import bbox_iou, select_anchor, track_people
from jevtells.utils.geometry import match_hands_to_pose


def _person(center_x: float, center_y: float = 0.5, width: float = 0.2) -> np.ndarray:
    pose = np.full((33, 4), np.nan, dtype=float)
    pose[:, 3] = 1.0
    pose[:, 0] = center_x
    pose[:, 1] = center_y
    pose[[11, 23], :2] = [center_x - width / 2, center_y]
    pose[[12, 24], :2] = [center_x + width / 2, center_y]
    pose[15, :2] = [center_x - width / 3, center_y + 0.2]
    pose[16, :2] = [center_x + width / 3, center_y + 0.2]
    return pose


def test_optimal_hand_assignment_tries_both_permutations():
    pose = _person(0.5)
    hands = np.asarray([[0.83, 0.7, 0.0], [0.17, 0.7, 0.0]])
    assert match_hands_to_pose(hands, pose).tolist() == [1, 0]


def test_track_iou_and_missing_target_does_not_switch_person():
    frames = []
    for index in range(5):
        left = _person(0.30 + index * 0.005)
        right = _person(0.72)
        if index == 2:
            left[:] = np.nan  # target disappears while host remains detected
        frames.append([left, right])
    poses = np.asarray(frames)
    indices = track_people(poses, [(1, 0.0)], fps=1, width=1000, height=1000)
    assert indices[:2].tolist() == [0, 0]
    assert indices[2:].tolist() == [-1, -1, -1]
    assert bbox_iou(np.asarray([0, 0, 10, 10]), np.asarray([1, 1, 11, 11])) > 0


def test_lost_target_is_not_reidentified_as_a_different_person():
    target = _person(0.30)
    absent = np.full_like(target, np.nan)
    replacement = _person(0.34)
    poses = np.asarray([[target, _person(0.75)], [absent, absent], [replacement, absent]])
    indices = track_people(poses, [(1, 0.0)], fps=1, width=1000, height=1000)
    assert indices.tolist() == [0, -1, -1]


def test_multi_anchor_switches_only_at_explicit_anchor():
    poses = np.asarray([[_person(0.25), _person(0.75)] for _ in range(5)])
    indices = track_people(poses, [(1, 0.0), (2, 3.0)], fps=1)
    assert indices[:3].tolist() == [0, 0, 0]
    assert indices[3:].tolist() == [1, 1]


def test_short_gap_interpolation_preserves_long_gap():
    values = np.asarray([[0.0], [np.nan], [2.0], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan], [8.0]])
    filled = interpolate_short_gaps(values, max_gap=1)
    assert filled[1, 0] == 1.0
    assert np.isnan(filled[3:8]).all()


def test_raise_and_posture_filter_rules():
    frames = 20
    y = np.r_[np.linspace(0.8, 0.4, 10), np.full(10, 0.4)]
    wrists = np.zeros((frames, 2, 2), dtype=float)
    wrists[:, :, 1] = y[:, None]
    features = {"t": np.arange(frames) / 10, "fps": 10.0, "wrist_norm": wrists}
    events = detect_actions(features)
    assert any(event["type"] == "raise" for event in events)
    open_score = np.ones((40, 2))
    posture_events = detect_actions({"t": np.arange(40) / 10, "fps": 10.0, "hand_open": open_score})
    assert not any(event["type"] == "open_palm" for event in posture_events)


def test_extract_features_pixels_and_shoulder_normalisation():
    pose = np.stack([_person(0.5) for _ in range(4)])
    hands = np.full((4, 2, 21, 3), np.nan)
    result = extract_features({"pose": pose, "hands": hands, "t": np.arange(4) / 2, "fps": 2.0, "width": 1000, "height": 500})
    assert np.isclose(result["shoulder_width_px"][0], 200.0)
    assert result["shoulder_width_relative"][0] == 1.0


def test_open_palm_and_fist_are_distinct_short_states():
    t = np.arange(12) / 10.0
    open_events = detect_actions({"t": t, "hand_open": np.ones((12, 2))})
    fist_events = detect_actions({"t": t, "hand_open": np.zeros((12, 2))})
    assert any(event["type"] == "open_palm" for event in open_events)
    assert any(event["type"] == "fist" for event in fist_events)


def test_spread_gather_beat_and_nod_rules():
    t = np.arange(20) / 10.0
    distance = np.r_[np.linspace(0.2, 0.8, 6), np.linspace(0.8, 0.3, 6), np.full(8, 0.3)]
    wrist_y = np.array([0.5, 0.4, 0.6, 0.4, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    wrists = np.zeros((20, 2, 2), dtype=float)
    wrists[:, :, 1] = wrist_y[:, None]
    wrists[:, 0, 0] = 0.2
    wrists[:, 1, 0] = 0.2 + distance
    features = {
        "t": t,
        "fps": 10.0,
        "wrist_norm": wrists,
        "two_hands_distance": distance,
        "nose_relative": np.column_stack([np.zeros(20), np.array([0.5, 0.4, 0.6, 0.4, 0.6] + [0.5] * 15)]),
    }
    events = detect_actions(features)
    types = {event["type"] for event in events}
    assert {"spread", "gather", "beat", "nod"}.issubset(types)
