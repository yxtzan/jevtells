import json
import subprocess

import numpy as np

from jevtells.config import load_config
from jevtells.render.labels import schedule
from jevtells.stages.actions import mark_far
from jevtells.stages.shots import _framing_boundaries, run, target_bounds
from jevtells.stages.state import _actions


def test_far_events_remain_auditable_but_leave_state_and_labels():
    events = [{"id": "a", "t0": 1, "t1": 2, "type": "raise", "limb": "right_hand", "magnitude": "large"}, {"id": "b", "t0": 3, "t1": 4, "type": "raise", "limb": "left_hand", "magnitude": "large"}]
    marked = mark_far(events, [{"index": 1, "t0": 0, "t1": 3, "far": True}, {"index": 2, "t0": 3, "t1": 5, "far": False}])
    assert marked[0]["far"] and not marked[1]["far"]
    assert "far" not in events[0]
    assert _actions({}, {"t0": 0, "t1": 3}, {"events": marked}) == ["no notable gestures"]
    assert schedule(marked, 1.5, load_config()["render"]) == {}
    assert schedule(marked, 3.5, load_config()["render"])


def test_zoom_scale_changes_need_sustained_evidence():
    assert _framing_boundaries(np.array([1] * 20 + [0] + [1] * 3 + [0] * 20), 5) == [24]


def test_hsv_cuts_and_short_shot_merge_on_actual_encoded_clip(tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=red:s=320x180:r=30:d=1", "-f", "lavfi", "-i", "color=blue:s=320x180:r=30:d=0.2", "-f", "lavfi", "-i", "color=green:s=320x180:r=30:d=1", "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]", "-map", "[v]", "-c:v", "libx264", str(clip)], check=True)
    result = run(clip, tmp_path, config=load_config())
    assert [shot["t0"] for shot in result] == [0, 1]
    assert result[1]["t1"] == 2.2 and result[1]["cut_at_start"]
    assert result[0]["target_box"] is None and result[0]["far"] is None
    assert json.loads((tmp_path / "shots_meta.json").read_text())["histogram_cuts"] == [1, 1.2]


def test_target_activity_bounds_use_percentiles_visibility_and_padding():
    pose = np.full((20, 33, 4), np.nan)
    pose[:, :10, :2] = [.5, .5]
    pose[:, :10, 3] = .9
    pose[:, 10, :2] = [1, 1]
    pose[:, 10, 3] = .1
    assert target_bounds({"pose": pose}, 0, 20, 1000, 500, .5, 100, .15) == [485, 235, 515, 265]
