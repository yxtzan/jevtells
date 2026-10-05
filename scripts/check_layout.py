"""Audit every encoded frame against its actual label composition trace."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevtells.render.layout_audit import frame_violations, leader_violations


def check(directory: Path, kind: str) -> dict:
    video = directory / f"output_{kind}.mp4"
    trace = directory / f"layout_trace_{kind}.jsonl"
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode {video}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    result = {"layout": kind, "video": str(video), "frames": total, "checked_frames": 0, "visible_label_frames": 0, "violation_frames": 0, "nonfallback_violation_frames": 0, "fallback_frames": 0, "fallback_violation_frames": 0, "examples": [], "leader_face_frames": 0, "leader_torso_frames": 0, "leader_midline_frames": 0, "hidden_label_frames": 0, "hidden_leader_frames": 0, "label_instance_frames": 0, "displayed_labels": [], "leader_examples": [], "suppressed_label_frames": 0, "blocked_face_leader_frames": 0}
    displayed=set()
    placements=json.loads((directory/f"layout_{kind}.json").read_text())
    result["relaxed_count"]=sum(bool(p.get('selected',{}).get(slot,{}).get('relaxed')) for p in placements for slot,count in p.get('sample_counts',{}).items() if count)
    result["forced_count"]=sum(bool(p.get('selected',{}).get(slot,{}).get('forced')) for p in placements for slot,count in p.get('sample_counts',{}).items() if count)
    try:
        with trace.open() as handle:
            for index, line in enumerate(handle):
                frame = json.loads(line)
                ok, decoded = capture.read()
                if not ok or frame["frame"] != index or tuple(frame["output"]) != (decoded.shape[1], decoded.shape[0]):
                    raise RuntimeError("composition trace does not match encoded frames")
                failures = frame_violations(frame)
                crossing = leader_violations(frame)
                result["leader_face_frames"] += any(item["zone"] == "face" for item in crossing)
                result["leader_midline_frames"] += any(item["zone"] == "midline" for item in crossing)
                visible={(l['slot'],l['event']) for l in frame['labels']}
                expected={(l['slot'],l['event']) for l in frame.get('expected_labels',[])}
                result["hidden_label_frames"]+=bool(expected-visible)
                result["hidden_leader_frames"]+=bool(expected-visible) or any(len(l.get('leader',[]))<2 for l in frame['labels'])
                result["label_instance_frames"]+=len(frame['labels'])
                displayed.update(l['event'] for l in frame['labels'])
                if crossing and len(result["leader_examples"]) < 30:
                    result["leader_examples"].append({"frame":index,"seconds":frame["seconds"],"failures":crossing})
                result["suppressed_label_frames"] += bool(frame.get("suppressed_labels"))
                result["blocked_face_leader_frames"] += any(label.get("blocked_face_leader") for label in frame["labels"])
                result["checked_frames"] += 1
                result["visible_label_frames"] += bool(frame["labels"])
                result["fallback_frames"] += any(label.get("fallback") for label in frame["labels"])
                result["violation_frames"] += bool(failures)
                result["nonfallback_violation_frames"] += any(not failure["fallback"] for failure in failures)
                result["fallback_violation_frames"] += any(failure["fallback"] for failure in failures)
                if failures and len(result["examples"]) < 30:
                    result["examples"].append({"frame": index, "seconds": frame["seconds"], "failures": failures})
        if result["checked_frames"] != total:
            raise RuntimeError("missing composition trace frames")
    finally:
        capture.release()
    result["displayed_labels"] = sorted(displayed)
    result["face_crossing_ratio"] = result["leader_face_frames"]/result["visible_label_frames"] if result["visible_label_frames"] else 0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip_dir", type=Path)
    parser.add_argument("--layout", choices=("h", "v", "both"), default="both")
    args = parser.parse_args()
    results = [check(args.clip_dir, kind) for kind in (("h", "v") if args.layout == "both" else (args.layout,))]
    (args.clip_dir / "layout_check.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if any(result["violation_frames"] or result["hidden_label_frames"] or result["hidden_leader_frames"] or result["face_crossing_ratio"] > .02 for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
