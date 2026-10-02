"""MediaPipe pose and hand landmark extraction."""

from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from ..utils.geometry import match_hands_to_pose


def _landmarks_to_array(landmarks: Any, width: int = 3) -> np.ndarray:
    """Convert MediaPipe landmark objects to a float array."""
    values = np.full((len(landmarks), width), np.nan, dtype=np.float32)
    for index, landmark in enumerate(landmarks):
        values[index, 0] = landmark.x
        values[index, 1] = landmark.y
        if width > 2:
            values[index, 2] = getattr(landmark, "z", np.nan)
        if width > 3:
            values[index, 3] = getattr(landmark, "visibility", np.nan)
    return values


def _choose_pose(poses: list[np.ndarray], previous: np.ndarray | None) -> np.ndarray:
    """Choose the largest pose, then track it by nearest center."""
    if not poses:
        return np.full((33, 4), np.nan, dtype=np.float32)
    boxes = []
    for pose in poses:
        points = pose[:, :2]
        finite = points[np.isfinite(points).all(axis=1)]
        boxes.append((float(np.ptp(finite[:, 0]) * np.ptp(finite[:, 1])) if len(finite) else 0.0, finite.mean(axis=0) if len(finite) else np.array([0.5, 0.5])))
    if previous is None or not np.isfinite(previous[:, :2]).any():
        return poses[int(np.argmax([box[0] for box in boxes]))]
    old_points = previous[:, :2]
    old_points = old_points[np.isfinite(old_points).all(axis=1)]
    old_center = old_points.mean(axis=0) if len(old_points) else np.array([0.5, 0.5])
    return min(poses, key=lambda pose: float(np.linalg.norm(_pose_center(pose) - old_center)))


def _pose_center(pose: np.ndarray) -> np.ndarray:
    """Return the center of finite pose points."""
    points = pose[:, :2]
    points = points[np.isfinite(points).all(axis=1)]
    return points.mean(axis=0) if len(points) else np.array([0.5, 0.5])


def run(clip: Path, out: Path, force: bool = False, model_dir: Path | None = None) -> dict[str, Any]:
    """Extract real pose and hand landmarks into keypoints.npz."""
    destination = out / "keypoints.npz"
    if destination.exists() and not force:
        return dict(np.load(destination, allow_pickle=False))
    root = model_dir or Path(__file__).resolve().parents[3] / "models"
    pose_model = root / "pose_landmarker_full.task"
    hand_model = root / "hand_landmarker.task"
    if not pose_model.exists() or not hand_model.exists():
        raise FileNotFoundError(f"MediaPipe models missing: {pose_model}, {hand_model}; run scripts/download_models.py")
    base = mp.tasks.BaseOptions
    running = mp.tasks.vision.RunningMode.VIDEO
    pose_options = mp.tasks.vision.PoseLandmarkerOptions(base_options=base(model_asset_path=str(pose_model), delegate=base.Delegate.CPU), running_mode=running, num_poses=2)
    hand_options = mp.tasks.vision.HandLandmarkerOptions(base_options=base(model_asset_path=str(hand_model), delegate=base.Delegate.CPU), running_mode=running, num_hands=2)
    pose_detector = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)
    hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)
    capture = cv2.VideoCapture(str(clip))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    poses: list[np.ndarray] = []
    hands: list[np.ndarray] = []
    pose_present: list[bool] = []
    hand_present: list[list[bool]] = []
    previous: np.ndarray | None = None
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp = round(frame_index * 1000 / fps)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pose_result = pose_detector.detect_for_video(image, timestamp)
        hand_result = hand_detector.detect_for_video(image, timestamp)
        candidates = [_landmarks_to_array(item, 4) for item in pose_result.pose_landmarks]
        selected = _choose_pose(candidates, previous)
        previous = selected
        pose_wrist = selected
        hand_candidates = [_landmarks_to_array(item, 3) for item in hand_result.hand_landmarks]
        wrist_points = np.array([item[0] for item in hand_candidates], dtype=np.float32) if hand_candidates else np.empty((0, 3), dtype=np.float32)
        sides = match_hands_to_pose(wrist_points, pose_wrist)
        shoulder_width = float(np.linalg.norm(selected[11, :2] - selected[12, :2])) if np.isfinite(selected[11, :2]).all() and np.isfinite(selected[12, :2]).all() else 0.0
        hand_frame = np.full((2, 21, 3), np.nan, dtype=np.float32)
        present = [False, False]
        for hand_index, side in enumerate(sides):
            if side < 0 or shoulder_width <= 0:
                continue
            anchor = selected[15 + side, :2]
            if float(np.linalg.norm(wrist_points[hand_index, :2] - anchor)) <= 0.75 * shoulder_width:
                hand_frame[side] = hand_candidates[hand_index]
                present[side] = True
        poses.append(selected)
        hands.append(hand_frame)
        pose_present.append(bool(np.isfinite(selected).any()))
        hand_present.append(present)
        frame_index += 1
    capture.release()
    data = {"fps": fps, "width": width, "height": height, "n_frames": len(poses), "t": np.arange(len(poses)) / fps, "pose": np.asarray(poses), "hands": np.asarray(hands), "pose_present": np.asarray(pose_present), "hand_present": np.asarray(hand_present)}
    np.savez(destination, **data)
    return data
