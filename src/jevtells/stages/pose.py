"""MediaPipe CPU pose and hand detection (all people, no target selection)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import cv2
import mediapipe as mp
import numpy as np
from ..resources import asset_path


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


def run(clip: Path, out: Path, force: bool = False, model_dir: Path | None = None, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Detect every person and hand into ``detections.npz``.

    This stage intentionally performs no target choice. ``track.run`` reads
    this cache and can therefore be rerun with different ``--target`` anchors
    without another MediaPipe pass.
    """
    destination = out / "detections.npz"
    if destination.exists() and not force:
        with np.load(destination, allow_pickle=False) as loaded:
            return {key: loaded[key] for key in loaded.files}
    settings = dict(config or {})
    detection = settings.get("detection", {}) if isinstance(settings.get("detection", {}), Mapping) else {}
    models = settings.get("models", {}) if isinstance(settings.get("models", {}), Mapping) else {}
    max_people = int(detection.get("max_people", settings.get("max_people", 4)))
    max_hands = int(detection.get("max_hands", settings.get("max_hands", 4)))
    pose_name=str(models.get("pose", settings.get("pose_model", "pose_landmarker_full.task")))
    hand_name=str(models.get("hands", settings.get("hand_model", "hand_landmarker.task")))
    pose_model=model_dir / pose_name if model_dir else asset_path(Path("models")/pose_name)
    hand_model=model_dir / hand_name if model_dir else asset_path(Path("models")/hand_name)
    if not pose_model.exists() or not hand_model.exists():
        raise FileNotFoundError(f"MediaPipe models missing: {pose_model}, {hand_model}; run scripts/download_models.py")
    base = mp.tasks.BaseOptions
    running = mp.tasks.vision.RunningMode.VIDEO
    pose_options = mp.tasks.vision.PoseLandmarkerOptions(base_options=base(model_asset_path=str(pose_model), delegate=base.Delegate.CPU), running_mode=running, num_poses=max_people)
    hand_options = mp.tasks.vision.HandLandmarkerOptions(base_options=base(model_asset_path=str(hand_model), delegate=base.Delegate.CPU), running_mode=running, num_hands=max_hands)
    pose_detector = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)
    hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)
    capture = cv2.VideoCapture(str(clip))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    poses: list[np.ndarray] = []
    hands: list[np.ndarray] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp = round(frame_index * 1000 / fps)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pose_result = pose_detector.detect_for_video(image, timestamp)
        hand_result = hand_detector.detect_for_video(image, timestamp)
        pose_frame = np.full((max_people, 33, 4), np.nan, dtype=np.float32)
        for index, landmark_list in enumerate(list(pose_result.pose_landmarks)[:max_people]):
            pose_frame[index] = _landmarks_to_array(landmark_list, 4)
        hand_frame = np.full((max_hands, 21, 3), np.nan, dtype=np.float32)
        for index, landmark_list in enumerate(list(hand_result.hand_landmarks)[:max_hands]):
            hand_frame[index] = _landmarks_to_array(landmark_list, 3)
        poses.append(pose_frame)
        hands.append(hand_frame)
        frame_index += 1
    capture.release()
    data: dict[str, Any] = {
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "n_frames": len(poses),
        "t": np.arange(len(poses), dtype=float) / float(fps),
        "poses_all": np.asarray(poses, dtype=np.float32),
        "hands_all": np.asarray(hands, dtype=np.float32),
    }
    out.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **data)
    return data


__all__ = ["run"]
