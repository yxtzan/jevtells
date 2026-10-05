"""Fixed anchor clothing references and conservative identity decisions."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


def clothing_histograms(frame: np.ndarray, pose: np.ndarray, visibility: float = .5) -> np.ndarray | None:
    """Measure torso and upper shirt inside clipped shoulder/hip polygons."""
    if not np.isfinite(pose[[11, 12, 24, 23], :2]).all():
        return None
    if pose.shape[1] >= 4 and not (pose[[11, 12], 3] >= visibility).all():
        return None
    height, width = frame.shape[:2]
    corners = pose[[11, 12, 24, 23], :2] * [width, height]
    upper = corners.copy()
    upper[2] = corners[1] + .5 * (corners[2] - corners[1])
    upper[3] = corners[0] + .5 * (corners[3] - corners[0])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    result = []
    for polygon in (corners, upper):
        mask = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(mask, np.round(polygon).astype(np.int32), 255)
        if cv2.countNonZero(mask) < 16:
            return None
        hist = cv2.calcHist([hsv], [0, 1, 2], mask, [16, 4, 4], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist, alpha=1, norm_type=cv2.NORM_L1)
        result.append(hist.flatten())
    return np.asarray(result)


def read_appearances(clip: Path, poses: np.ndarray, visibility: float) -> np.ndarray:
    """Descriptors remain ephemeral; no unbounded full-resolution video cache."""
    result = np.full((*poses.shape[:2], 2, 256), np.nan, np.float32)
    capture = cv2.VideoCapture(str(clip))
    try:
        for index, candidates in enumerate(poses):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError('appearance decoding ended early')
            frame = cv2.resize(frame, (320, 180))
            for person, pose in enumerate(candidates):
                descriptor = clothing_histograms(frame, pose, visibility)
                if descriptor is not None:
                    result[index, person] = descriptor
    finally:
        capture.release()
    return result


def similarity(descriptor: np.ndarray, references: Sequence[np.ndarray]) -> float:
    """Bhattacharyya coefficient (histogram overlap), averaged across regions."""
    if not np.isfinite(descriptor).all():
        return 0.0
    return max((float(np.mean([1 - cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA) ** 2 for a, b in zip(descriptor, ref)])) for ref in references), default=0.0)


def recognize(poses: np.ndarray, descriptors: np.ndarray, start: int, end: int, references: Sequence[np.ndarray], width: float, height: float, rules: Any, settings: Mapping[str, Any]) -> tuple[dict[int, int], dict[str, Any]]:
    """Median evidence over candidate trajectories, independent of detector slots."""
    from .track import _candidate_match, sorted_person_indices
    count = int(settings.get('appearance_probe_frames', 10))
    stop = min(end, start + count)
    trajectories: list[dict[str, Any]] = []
    for frame in range(start, stop):
        unused = set(sorted_person_indices(poses, frame, width, height, rules.visibility_threshold))
        for candidate in trajectories:
            if candidate['last_frame'] != frame - 1:
                continue
            masked = poses[frame].copy()
            for index in range(len(masked)):
                if index not in unused:
                    masked[index] = np.nan
            match = _candidate_match(candidate['last_pose'], masked, width, height, rules)
            if match >= 0:
                unused.remove(match)
                candidate['frames'][frame] = match
                candidate['scores'].append(similarity(descriptors[frame, match], references))
                candidate.update(last_frame=frame, last_pose=poses[frame, match])
        for person in sorted(unused):
            trajectories.append({'frames': {frame: person}, 'scores': [similarity(descriptors[frame, person], references)], 'last_frame': frame, 'last_pose': poses[frame, person]})
    # Require at least half the available probe frames: a one-frame accidental
    # match must not identify a person for an entire shot.
    minimum = max(1, int(np.ceil((stop - start) / 2)))
    ranked = sorted([(float(np.median(c['scores'])), c) for c in trajectories if len(c['scores']) >= minimum], key=lambda pair: -pair[0])
    best = ranked[0][0] if ranked else 0.0
    second = ranked[1][0] if len(ranked) > 1 else 0.0
    accepted = bool(ranked and best >= float(settings.get('appearance_similarity', .7)) and best - second >= float(settings.get('appearance_margin', .1)))
    winner = ranked[0][1]['frames'] if accepted else {}
    first = min(winner) if winner else None
    ordered = sorted_person_indices(poses, first, width, height, rules.visibility_threshold) if first is not None else []
    meta = {'frame': start, 'selected_index': winner[first] if first is not None else None, 'person_number': ordered.index(winner[first]) + 1 if first is not None else None, 'similarity': best, 'second_similarity': second, 'result': 'recognized' if accepted else 'target_offscreen', 'probe_frames': stop - start, 'candidate_scores': [score for score, _ in ranked]}
    return winner, meta


def track_appearance(poses: np.ndarray, descriptors: np.ndarray, anchors: Sequence[tuple[int, float]], cuts: Sequence[int], fps: float, width: float, height: float, config: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    from .track import _config, _candidate_match, _track_segment, select_anchor, bbox_iou, person_box
    rules = _config(config)
    settings = config.get('tracking', {})
    total = len(poses)
    anchor_map = {min(total - 1, max(0, round(t * fps))): person for person, t in sorted(anchors, key=lambda pair: pair[1])}
    cut_frames = sorted({0, total, *(int(c) for c in cuts if 0 < c < total)})
    references, reference_meta = [], []
    for frame, person in sorted(anchor_map.items()):
        index = select_anchor(poses, frame, person, width, height, rules.visibility_threshold)
        stop = min(total, frame + max(1, round(float(settings.get('appearance_reference_seconds', 1)) * fps)), next(c for c in cut_frames if c > frame), next((a for a in sorted(anchor_map) if a > frame), total))
        samples = []
        if index >= 0:
            tracked = _track_segment(poses, frame, stop - 1, index, width, height, rules)
            stride = max(1, int(settings.get('appearance_reference_stride', 3)))
            for offset in range(0, len(tracked), stride):
                selected = tracked[offset]
                if selected >= 0 and np.isfinite(descriptors[frame + offset, selected]).all():
                    samples.append(descriptors[frame + offset, selected])
        if samples:
            reference = np.median(samples, axis=0)
            reference /= reference.sum(axis=1, keepdims=True)
            references.append(reference)
        reference_meta.append({'frame': frame, 't': frame / fps, 'person_number': person, 'selected_index': index, 'samples': len(samples), 'end': stop / fps})
    indices = np.full(total, -1, int)
    first_anchor = min(anchor_map)
    shot_meta, recoveries = [], []
    for shot_start, shot_end in zip(cut_frames, cut_frames[1:]):
        boundaries = sorted({shot_start, shot_end, *(a for a in anchor_map if shot_start <= a < shot_end)})
        decisions = []
        for start, end in zip(boundaries, boundaries[1:]):
            initial = anchor_map.get(start)
            if initial is not None:
                selected = select_anchor(poses, start, initial, width, height, rules.visibility_threshold)
                indices[start] = selected
                decision = {'frame': start, 'selected_index': selected if selected >= 0 else None, 'person_number': initial, 'similarity': similarity(descriptors[start, selected], references) if selected >= 0 else 0.0, 'second_similarity': None, 'result': 'anchor'}
                cursor = start + 1
                previous = poses[start, selected] if selected >= 0 else None
            elif end == first_anchor and shot_start <= first_anchor < shot_end:
                # The first explicit anchor also follows backwards, but never
                # carries positional identity across a hard cut.
                selected = select_anchor(poses, first_anchor, anchor_map[first_anchor], width, height, rules.visibility_threshold)
                back = _track_segment(poses, first_anchor, start, selected, width, height, rules, -1)
                indices[start:end] = back[::-1][:-1]
                decisions.append({'frame': start, 'selected_index': selected if selected >= 0 else None, 'person_number': anchor_map[first_anchor], 'similarity': None, 'second_similarity': None, 'result': 'anchor', 'backward_from': first_anchor})
                continue
            else:
                winners, decision = recognize(poses, descriptors, start, end, references, width, height, rules, settings)
                if not winners:
                    decisions.append(decision)
                    continue  # Rejected identity stays absent until cut/anchor.
                for frame, person in winners.items():
                    indices[frame] = person
                cursor = min(end, start + int(settings.get('appearance_probe_frames', 10)))
                last = cursor - 1
                previous = poses[last, indices[last]] if indices[last] >= 0 else None
            decisions.append(decision)
            while cursor < end:
                selected = _candidate_match(previous, poses[cursor], width, height, rules) if previous is not None else -1
                if selected >= 0 and bbox_iou(person_box(previous, width, height, rules.visibility_threshold), person_box(poses[cursor, selected], width, height, rules.visibility_threshold)) < rules.iou_threshold:
                    # A distance-only fallback may be an occluding neighbour.
                    # Confirm it through the same multi-frame identity rule.
                    selected = -1
                if selected >= 0:
                    indices[cursor] = selected
                    previous = poses[cursor, selected]
                    cursor += 1
                else:
                    winners, recovery = recognize(poses, descriptors, cursor, end, references, width, height, rules, settings)
                    recoveries.append({**recovery, 't': cursor / fps})
                    for frame, person in winners.items():
                        indices[frame] = person
                    cursor = min(end, cursor + int(settings.get('appearance_probe_frames', 10)))
                    last = cursor - 1
                    previous = poses[last, indices[last]] if indices[last] >= 0 else None
        shot_meta.append({'t0': shot_start / fps, 't1': shot_end / fps, 'cut_at_start': shot_start > 0, 'decisions': decisions, 'target_presence_ratio': float(np.mean(indices[shot_start:shot_end] >= 0))})
    return indices, {'references': reference_meta, 'shots': shot_meta, 'recoveries': recoveries, 'appearance_settings': dict(settings)}
