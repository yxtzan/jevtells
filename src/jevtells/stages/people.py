"""Render a frame with all detector people numbered left-to-right.

The ``people`` command is intentionally a light-weight view over the cached
pose detections.  It never runs a second detector and therefore makes the
numbering used by ``--target`` reproducible for a given video and timestamp.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .track import person_box, sorted_person_indices


def _load(source: Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, (str, Path)):
        with np.load(source, allow_pickle=False) as loaded:
            return {key: loaded[key] for key in loaded.files}
    return dict(source)


def run(
    clip: Path,
    out: Path,
    at: float = 0.0,
    detections: Path | Mapping[str, Any] | None = None,
    force: bool = False,
    config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Write ``people_<seconds>s.png`` and return the numbered boxes.

    ``at`` is in seconds and is clamped to the available video frames.  Boxes
    are expressed in pixel coordinates in the returned metadata while the
    detector arrays remain normalised coordinates on disk.
    """

    stamp = f"{float(at):04.1f}s"
    destination = out / f"people_{stamp}.png"
    metadata = out / f"people_{stamp}.json"
    if destination.exists() and metadata.exists() and not force:
        import json
        cached = json.loads(metadata.read_text(encoding="utf-8"))
        # ``people --at`` changes the anchor frame; only reuse metadata when
        # it was generated for the same timestamp.
        if not cached or abs(float(cached[0].get("time", at)) - float(at)) <= 1.0 / 60.0:
            return cached

    if detections is None:
        detections = out / "detections.npz"
    data = _load(detections)
    poses = np.asarray(data.get("poses_all", data.get("poses", data.get("pose"))), dtype=float)
    if poses.ndim == 3:
        poses = poses[:, None, ...]
    if poses.ndim != 4 or not len(poses):
        raise ValueError("detections.npz is missing poses_all [T,K,landmark,channels]")
    fps = float(np.asarray(data.get("fps", 30.0)).reshape(-1)[0])
    width = int(float(np.asarray(data.get("width", 1.0)).reshape(-1)[0]))
    height = int(float(np.asarray(data.get("height", 1.0)).reshape(-1)[0]))
    threshold = 0.5
    if config:
        detection_cfg = config.get("detection", {})
        threshold = float(detection_cfg.get("pose_visibility_threshold", threshold))
    frame_index = min(max(int(round(float(at) * fps)), 0), len(poses) - 1)

    capture = cv2.VideoCapture(str(clip))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        frame = np.zeros((max(height, 1), max(width, 1), 3), dtype=np.uint8)
        height, width = frame.shape[:2]
    else:
        height, width = frame.shape[:2]

    frame_poses = poses[frame_index]
    ordered = sorted_person_indices(poses, frame_index, width, height, threshold)
    records: list[dict[str, Any]] = []
    for ordinal, person_index in enumerate(ordered, start=1):
        box = person_box(frame_poses[person_index], width, height, threshold)
        if not np.isfinite(box).all():
            continue
        x0, y0, x1, y1 = [int(round(value)) for value in box]
        # BGR: yellow boxes stay visible on both light and dark source frames.
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 220, 255), 2)
        cv2.putText(frame, str(ordinal), (x0 + 8, max(28, y0 + 34)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 255), 3, cv2.LINE_AA)
        records.append({"number": ordinal, "detector_index": int(person_index), "box": [x0, y0, x1, y1], "frame": frame_index, "time": frame_index / fps})
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), frame)
    import json

    metadata.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


__all__ = ["run"]
