"""One validated, deterministic UI → CLI mapping shared by server and tests.

Anchor and other-speaker times are clip relative, exactly as on the CLI.
In auto-start mode they are source relative, as required by --start auto.
"""
from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any

from ..stages.speakers import parse_intervals


def number(value: Any, minimum: float = 0) -> float:
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError("invalid_number")
    return result


def safe_path(root: Path, value: str, area: str | None = None) -> Path:
    """Resolve paths including symlinks; never allow escape or '..'."""
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or "\x00" in value:
        raise ValueError("unsafe_path")
    allowed = root / area if area else root
    result = (root / path).resolve()
    if not result.is_relative_to(allowed.resolve()) or result == allowed.resolve():
        raise ValueError("unsafe_path")
    if not area and (not path.parts or path.parts[0] not in {"samples", "work"}):
        raise ValueError("unsafe_path")
    return result


def intervals_from_drag(a: float, b: float, width: float, duration: float) -> list[list[float]]:
    width, duration = number(width, .001), number(duration, .001)
    start, end = sorted(max(0., min(width, number(v, -math.inf))) / width * duration for v in (a, b))
    return [[start, end]] if end > start else []


def source_rect(rect: list[float], display: list[float], source: list[int]) -> list[int]:
    """Map a drag in an image's content box to original pixels, reversed too."""
    x, y, w, h = map(float, rect)
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        raise ValueError("invalid_rectangle")
    dw, dh = (number(v, .001) for v in display)
    sw, sh = (number(v, 1) for v in source)
    x0, x1 = sorted(max(0., min(dw, v)) for v in (x, x+w))
    y0, y1 = sorted(max(0., min(dh, v)) for v in (y, y+h))
    left, top, right, bottom = round(x0/dw*sw), round(y0/dh*sh), round(x1/dw*sw), round(y1/dh*sh)
    if right <= left or bottom <= top:
        raise ValueError("invalid_rectangle")
    return [left, top, right-left, bottom-top]


def normalize(params: dict[str, Any]) -> dict[str, Any]:
    allowed = {"input", "speaker", "start", "duration", "target", "target_desc", "face_identity", "lang", "title", "blur", "others_speaking", "subtitles", "subtitle_source", "layout", "reframe", "output_dir"}
    if set(params) - allowed:
        raise ValueError("unknown_parameter")
    p = {"start": 0, "duration": None, "target": [], "face_identity": True, "lang": "zh", "blur": [], "others_speaking": [], "subtitles": "on", "subtitle_source": "auto", "layout": "both", "reframe": "auto", "output_dir": "work", **params}
    if not p.get("input") or not str(p.get("speaker", "")).strip():
        raise ValueError("input_and_speaker_required")
    for key in ("input", "speaker", "title", "target_desc", "output_dir"):
        if key in p and p[key] is not None:
            if not isinstance(p[key], str) or len(p[key]) > 1000 or any(c in p[key] for c in "\r\n\x00"):
                raise ValueError("invalid_text")
    for key, choices in {"lang": {"zh", "en"}, "subtitles": {"on", "off"}, "subtitle_source": {"auto", "ocr", "asr"}, "layout": {"h", "v", "both"}, "reframe": {"auto", "off"}}.items():
        if p[key] not in choices:
            raise ValueError("invalid_choice")
    if not isinstance(p["face_identity"], bool):
        raise ValueError("invalid_choice")
    p["start"] = "auto" if p["start"] == "auto" else number(p["start"])
    p["duration"] = None if p["duration"] is None else number(p["duration"], .001)
    anchors = {}
    for anchor in p["target"]:
        n, at = anchor
        if int(n) != float(n) or int(n) < 1:
            raise ValueError("invalid_target")
        anchors[number(at)] = int(n)
    p["target"] = [[n, at] for at, n in sorted(anchors.items())]
    if not p["target"] and not p.get("target_desc"):
        raise ValueError("select_target")
    p["others_speaking"] = [list(pair) for pair in parse_intervals([f"{number(a):g}-{number(b):g}" for a,b in p["others_speaking"]])]
    p["blur"] = [[round(number(v)) for v in rect] for rect in p["blur"]]
    if any(len(r) != 4 or r[2] <= 0 or r[3] <= 0 for r in p["blur"]):
        raise ValueError("invalid_rectangle")
    p["title"] = p.get("title") or (f"AI 读{p['speaker']}" if p["lang"] == "zh" else f"AI reads {p['speaker']}")
    return p


def output_path(params: dict[str, Any], root: Path) -> Path:
    p = normalize(params)
    base = safe_path(root, p["output_dir"] + "/.ui", "work").parent
    source = safe_path(root, p["input"], "samples")
    start = "auto" if p["start"] == "auto" else f"{p['start']:g}"
    duration = "auto" if p["duration"] is None else f"{p['duration']:g}"
    return safe_path(root, str((base / f"{source.stem}_s{start}_d{duration}").relative_to(root)), "work")


def cli_args(params: dict[str, Any], root: Path, mode: str = "run", render_only: bool = False) -> list[str]:
    p = normalize(params)
    source = safe_path(root, p["input"], "samples")
    if mode == "people":
        return ["people", str(source)]
    if mode not in {"run", "prepare"}:
        raise ValueError("invalid_mode")
    args = ["run", str(source), "-o", str(output_path(p, root) / "output.mp4")]
    if render_only and p['start'] == 'auto':
        # Bypass auto-start's source tracking/description selection on a visual
        # rerender, using the exact resolved CLI parameters of the cached run.
        meta = json.loads((output_path(p, root)/'run_meta.json').read_text())['parameters']
        p = {**p, 'start': meta['start'], 'target_desc': None,
             'target': [[int(v.split('@')[0]), float(v.split('@')[1])] for v in meta['target']],
             'others_speaking': [[float(t) for t in v.split('-')] for v in meta['others_speaking']]}
    for key in ("speaker", "start", "lang", "title", "subtitles", "subtitle_source", "layout", "reframe"):
        args += ["--"+key.replace("_", "-"), str(p[key])]
    if p["duration"] is not None:
        args += ["--duration", f"{p['duration']:g}"]
    if p.get("target_desc"):
        args += ["--target-desc", p["target_desc"]]
    else:
        for n, at in p["target"]:
            args += ["--target", f"{n}@{at:g}"]
    if p["face_identity"]:
        args += ["--face-identity"]
    for rect in p["blur"]:
        args += ["--blur", ",".join(map(str, rect))]
    for a, b in p["others_speaking"]:
        args += ["--others-speaking", f"{a:g}-{b:g}"]
    if mode == "prepare":
        args += ["--until", "segment"]
    elif render_only:
        args += ["--from", "render"]
    return args
