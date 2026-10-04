"""Optional visual scene description stage."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import cv2

from ..clients.openrouter import OpenRouterClient
from ..schemas import SceneDescription


def _content(raw: Mapping[str, Any]) -> str:
    choices = raw.get("choices", [])
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        message = choices[0].get("message", {})
        value = message.get("content", "") if isinstance(message, Mapping) else ""
        if isinstance(value, list):
            return " ".join(str(part.get("text", "")) for part in value if isinstance(part, Mapping)).strip()
        return str(value).strip()
    return str(raw.get("output", raw.get("text", ""))).strip()


def _clean(value: str) -> str:
    value = re.sub(r"[`\"']", "", value).strip()
    value = re.sub(r"\s+", " ", value)
    return " ".join(value.split()[:20]) or "unknown"


def _middle_frame(clip: Path) -> bytes:
    capture = cv2.VideoCapture(str(clip))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, count // 2))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError("无法读取场景中间帧")
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise RuntimeError("无法编码场景中间帧")
    return encoded.tobytes()


def _write(result: dict[str, Any], destination: Path) -> dict[str, Any]:
    usage = result.get("usage", {})
    usage = usage if isinstance(usage, Mapping) else {}
    result.setdefault("calls", 0 if result.get("source") in {"cli", "disabled"} else 1)
    result.setdefault("prompt_tokens", int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0))
    result.setdefault("completion_tokens", int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0))
    result.setdefault("failed", int("error" in result))
    SceneDescription.model_validate(result)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run(clip: Path, out: Path, force: bool = False, config: Mapping[str, Any] | None = None, explicit: str | None = None, client: OpenRouterClient | None = None) -> dict[str, Any]:
    out = Path(out)
    destination = out / "scene.json"
    if destination.exists() and not force:
        return _write(json.loads(destination.read_text(encoding="utf-8")), destination)
    if explicit:
        result = {"scene": str(explicit), "model": None, "cost": None, "source": "cli"}
        return _write(result, destination)
    settings = (config or {}).get("scene", {})
    if not isinstance(settings, Mapping) or not bool(settings.get("auto", True)):
        result = {"scene": "unknown", "model": None, "cost": None, "source": "disabled"}
        return _write(result, destination)
    model = str(settings.get("model", "google/gemini-3.1-flash-lite"))
    active_client = client or OpenRouterClient()
    try:
        image = _middle_frame(Path(clip))
        response = active_client.chat(
            [{"role": "user", "content": "Describe only the environment and camera setup in this frame in one English sentence of no more than 20 words. Do not evaluate people and do not name anyone."}],
            model,
            images=[image],
            max_tokens=48,
            temperature=0.0,
        )
        raw = response.get("raw", response.get("response", response))
        text = _clean(_content(raw if isinstance(raw, Mapping) else {}))
        result = {"scene": text, "model": model, "cost": response.get("cost"), "usage": response.get("usage", {})}
        (out / "raw").mkdir(parents=True, exist_ok=True)
        (out / "raw" / "scene.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as error:
        result = {"scene": "unknown", "model": model, "cost": None, "error": type(error).__name__}
    return _write(result, destination)


__all__ = ["run"]
