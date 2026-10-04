"""Synthetic regression tests for the M3 action-rule carry-overs."""

import numpy as np

from jevtells.actions.rules import detect_actions
from jevtells.stages.state import run as write_state


def _shape_features(frames: int = 12) -> dict[str, np.ndarray | float]:
    return {
        "t": np.arange(frames, dtype=float) / 10.0,
        "fps": 10.0,
        "hand_open": np.zeros((frames, 2), dtype=float),
        "hand_point_ratio": np.ones((frames, 2), dtype=float),
    }


def test_fist_requires_all_fingertips_near_palm():
    good = _shape_features()
    good["fingertip_palm_ratio"] = np.full((12, 2, 4), 0.5)
    assert any(event["type"] == "fist" for event in detect_actions(good))

    cupped = dict(good)
    cupped["fingertip_palm_ratio"] = np.array(good["fingertip_palm_ratio"], copy=True)
    cupped["fingertip_palm_ratio"][4, 0, 2] = 1.5
    assert not any(event["type"] == "fist" and event["side"] == "left hand" for event in detect_actions(cupped))


def test_point_requires_curled_non_index_fingers():
    points = _shape_features()
    straight = np.zeros((12, 2, 5), dtype=float)
    straight[:, 0, 0] = 1.0
    points["finger_straight"] = straight
    tips = np.ones((12, 2, 5), dtype=float)
    roots = np.full((12, 2, 5), 2.0, dtype=float)
    tips[:, 0, 0] = 3.0
    roots[:, 0, 0] = 1.0
    points["finger_tip_wrist_distance"] = tips
    points["finger_root_wrist_distance"] = roots
    assert any(event["type"] == "point" for event in detect_actions(points))

    open_middle = dict(points)
    open_middle["finger_tip_wrist_distance"] = np.array(tips, copy=True)
    open_middle["finger_tip_wrist_distance"][:, 0, 1] = 3.0
    assert not any(event["type"] == "point" for event in detect_actions(open_middle))


def test_small_motion_is_kept_in_actions_but_omitted_from_state(tmp_path):
    windows = [{"id": "W00", "t0": 0.0, "t1": 2.0, "subtitle": "hello"}]
    actions = {
        "events": [
            {"type": "raise", "limb": "left_hand", "t0": 0.2, "t1": 0.4, "magnitude": "small"},
            {"type": "raise", "limb": "left_hand", "t0": 0.8, "t1": 1.0, "magnitude": "medium"},
            {"type": "fist", "limb": "left_hand", "t0": 1.2, "t1": 1.4, "magnitude": None},
        ]
    }
    result = write_state(windows, {}, "unknown", "speaker", tmp_path, force=True, actions=actions, config={"state_min_motion_magnitude": "medium"})
    measured = result["W00"]["measured_actions"]
    assert not any("small" in item for item in measured)
    assert any("medium" in item for item in measured)
    assert any("fist" in item for item in measured)


def test_point_geometry_must_cover_eighty_percent_of_merged_span():
    points = _shape_features(frames=11)
    straight = np.zeros((11, 2, 5))
    straight[:, 0, 0] = 1.0
    points["finger_straight"] = straight
    tips = np.ones((11, 2, 5))
    roots = np.full((11, 2, 5), 2.0)
    tips[:, 0, 0] = 3.0
    points["finger_root_wrist_distance"] = roots
    points["finger_tip_wrist_distance"] = tips
    passing = dict(points)
    passing["finger_tip_wrist_distance"] = tips.copy()
    passing["finger_tip_wrist_distance"][5:7, 0, 1] = 3.0
    events = detect_actions(passing, config={"merge_gap": 0.45})
    assert any(event["type"] == "point" and event["side"] == "left hand" and event["t0"] == 0.0 and event["t1"] == 1.0 for event in events)
    failing = dict(points)
    failing["finger_tip_wrist_distance"] = tips.copy()
    failing["finger_tip_wrist_distance"][4:7, 0, 1] = 3.0
    assert not any(event["type"] == "point" and event["side"] == "left hand" for event in detect_actions(failing, config={"merge_gap": 0.45}))
