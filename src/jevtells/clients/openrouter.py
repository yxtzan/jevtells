"""The sole OpenRouter network client used by M3.

The module deliberately keeps authentication, retry policy, request shaping,
and response accounting in one place so stages remain deterministic and easy
to mock in tests.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from ..resources import asset_path

try:  # python-dotenv is a runtime dependency, with a tiny fallback for old envs.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised only in incomplete installs
    load_dotenv = None


DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
# Keep the singular spelling as a public alias used by stages and tests.
DECIDE_URL = DECISIONS_URL
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
MISSING_KEY_MESSAGE = "缺少 OPENROUTER_API_KEY：请把 .env.example 复制为 .env 并填写"
MISSING_KEY = MISSING_KEY_MESSAGE


class OpenRouterError(RuntimeError):
    """A safe, key-free error returned by the OpenRouter client."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        self.status = status
        super().__init__(message)


class MissingAPIKeyError(OpenRouterError):
    """Raised before any request when the configured key is absent."""


class _BodyReader:
    """Tiny file-like wrapper used to feed injected HTTP errors through one path."""

    def __init__(self, body: bytes | str) -> None:
        self.body = body.encode("utf-8") if isinstance(body, str) else bytes(body)

    def read(self) -> bytes:
        return self.body

    def close(self) -> None:
        return None


def _load_key(env_path: str | Path | None = None) -> str:
    env_path = Path(env_path) if env_path is not None else asset_path(".env")
    if load_dotenv is not None:
        # Do not let python-dotenv silently search unrelated parent folders;
        # the project key must come from this checkout's .env.
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
    else:  # fallback only prevents a cryptic import error in a partial install
        if env_path.exists():
            for raw in env_path.read_text(encoding="utf-8").splitlines():
                key, separator, value = raw.partition("=")
                if separator and key.strip() == "OPENROUTER_API_KEY" and "OPENROUTER_API_KEY" not in os.environ:
                    os.environ[key.strip()] = value.strip().strip("'\"")
    value = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not value:
        raise MissingAPIKeyError(MISSING_KEY_MESSAGE)
    return value


def _cost(payload: Mapping[str, Any]) -> float | None:
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return None
    value = usage.get("cost")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _redact(value: Any, key: str) -> str:
    """Redact the API key before an exception reaches a log or report."""

    return str(value).replace(key, "[REDACTED]") if key else str(value)


def _request_json(
    url: str,
    payload: Mapping[str, Any],
    timeout: float = 60.0,
    attempts: int = 4,
    *,
    api_key: str | None = None,
    env_path: str | Path | None = None,
    transport: Any | None = None,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    """POST JSON with an initial request plus up to three retries."""

    key = api_key or _load_key(env_path)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/yxtzan/jevtells",
        "X-Title": "JevTells",
    }
    last_error: Exception | None = None
    for attempt in range(max(1, int(attempts))):
        try:
            if transport is not None:
                status, response_body = transport(url, headers, body, float(timeout))
                if int(status) >= 400:
                    raise HTTPError(url, int(status), "", {}, _BodyReader(response_body))
                raw = response_body.decode("utf-8", errors="replace") if isinstance(response_body, bytes) else str(response_body)
            else:
                request = Request(url, data=body, headers=headers, method="POST")
                with urlopen(request, timeout=float(timeout)) as response:
                    raw = response.read().decode("utf-8")
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise OpenRouterError("OpenRouter 返回的 JSON 顶层不是对象")
            return parsed
        except HTTPError as error:
            response_body = error.read().decode("utf-8", errors="replace")
            if error.code == 429 or error.code >= 500:
                last_error = OpenRouterError(f"OpenRouter HTTP {error.code}: {_redact(response_body[:5000], key)}", status=error.code)
            else:
                raise OpenRouterError(f"OpenRouter HTTP {error.code}: {_redact(response_body[:5000], key)}", status=error.code) from error
        except (TimeoutError, socket.timeout, URLError, OSError, json.JSONDecodeError) as error:
            last_error = OpenRouterError(f"OpenRouter 请求失败：{_redact(error, key)}")
        if attempt + 1 < max(1, int(attempts)):
            sleep(0.5 * (2**attempt))
    if isinstance(last_error, OpenRouterError):
        raise last_error
    raise OpenRouterError(f"OpenRouter 请求失败：{_redact(type(last_error).__name__ if last_error else 'unknown', key)}") from last_error


def _result(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else {}
    return {"raw": response, "response": response, "usage": dict(usage), "cost": _cost(response)}


def _jev_question(value: Mapping[str, Any]) -> dict[str, Any]:
    """Convert our editable YAML question shape to Jev's alpha schema."""

    kind = str(value.get("type", "choice"))
    instructions = value.get("instructions", value.get("question", ""))
    criteria = value.get("criteria")
    options = value.get("options", [])
    if criteria is None:
        if kind == "score":
            criteria = list(options) if isinstance(options, Sequence) and not isinstance(options, (str, bytes)) else options
        elif kind == "noul" and isinstance(options, Sequence) and not isinstance(options, (str, bytes)):
            criteria = {"true": str(options[-1]) if options else "yes", "false": str(options[0]) if options else "no"}
        elif isinstance(options, Mapping):
            criteria = dict(options)
        elif isinstance(options, Sequence) and not isinstance(options, (str, bytes)):
            criteria = {str(item): str(item) for item in options}
        else:
            criteria = {}
    return {"type": kind, "instructions": str(instructions), "criteria": criteria}


def decide(
    state: Mapping[str, Any],
    questions: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    model: str = "typesafe/jev-1.13",
    *,
    timeout: float = 60.0,
    attempts: int = 4,
    api_key: str | None = None,
    env_path: str | Path | None = None,
    transport: Any | None = None,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    """Call Jev's alpha decisions endpoint once and retain raw accounting."""

    if isinstance(questions, Mapping):
        question_record = {str(key): _jev_question(value) for key, value in questions.items()}
    else:
        question_record = {
            str(item.get("id")): _jev_question(item)
            for item in questions
            if isinstance(item, Mapping) and item.get("id") is not None
        }
    payload = {"model": model, "state": dict(state), "questions": question_record}
    return _result(_request_json(DECISIONS_URL, payload, timeout=timeout, attempts=attempts, api_key=api_key, env_path=env_path, transport=transport, sleep=sleep))


def _image_content(image: str | bytes | Path) -> str:
    if isinstance(image, Path) or (isinstance(image, str) and Path(image).exists()):
        path = Path(image)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        suffix = path.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        return f"data:{mime};base64,{encoded}"
    if isinstance(image, bytes):
        return f"data:image/jpeg;base64,{base64.b64encode(image).decode('ascii')}"
    return str(image)


def chat(
    messages: Sequence[Mapping[str, Any]],
    model: str,
    *,
    images: Sequence[str | bytes | Path] | None = None,
    image: str | bytes | Path | None = None,
    response_format: Mapping[str, Any] | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.2,
    timeout: float = 90.0,
    attempts: int = 4,
    api_key: str | None = None,
    env_path: str | Path | None = None,
    transport: Any | None = None,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    """Call chat completions, optionally appending image parts to the last user message."""

    serialised = [dict(message) for message in messages]
    all_images = list(images or [])
    if image is not None:
        all_images.append(image)
    if all_images:
        found_user = False
        for index in range(len(serialised) - 1, -1, -1):
            if serialised[index].get("role") == "user":
                found_user = True
                content = serialised[index].get("content", "")
                parts = content if isinstance(content, list) else [{"type": "text", "text": str(content)}]
                parts.extend({"type": "image_url", "image_url": {"url": _image_content(item)}} for item in all_images)
                serialised[index]["content"] = parts
                break
        if not found_user:
            serialised.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": _image_content(item)}} for item in all_images]})
    payload: dict[str, Any] = {"model": model, "messages": serialised, "temperature": temperature}
    if response_format is not None:
        payload["response_format"] = dict(response_format)
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    return _result(_request_json(CHAT_URL, payload, timeout=timeout, attempts=attempts, api_key=api_key, env_path=env_path, transport=transport, sleep=sleep))


class OpenRouterClient:
    """Injectable facade useful for stages and unit tests."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        env_path: str | Path | None = None,
        timeout: float = 60.0,
        attempts: int = 4,
        sleep: Any = time.sleep,
        transport: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.env_path = env_path
        self.timeout = timeout
        self.attempts = attempts
        self.sleep = sleep
        self.transport = transport

    def decide(self, state: Mapping[str, Any], questions: Sequence[Mapping[str, Any]], model: str = "typesafe/jev-1.13", **kwargs: Any) -> dict[str, Any]:
        params = {"timeout": self.timeout, "attempts": self.attempts, "api_key": self.api_key, "env_path": self.env_path, "sleep": self.sleep, "transport": self.transport}
        params.update(kwargs)
        return decide(state, questions, model, **params)

    def chat(self, messages: Sequence[Mapping[str, Any]], model: str, **kwargs: Any) -> dict[str, Any]:
        params = {"timeout": self.timeout, "attempts": self.attempts, "api_key": self.api_key, "env_path": self.env_path, "sleep": self.sleep, "transport": self.transport}
        params.update(kwargs)
        return chat(messages, model, **params)


__all__ = ["CHAT_URL", "DECIDE_URL", "DECISIONS_URL", "MISSING_KEY", "MISSING_KEY_MESSAGE", "MissingAPIKeyError", "OpenRouterClient", "OpenRouterError", "chat", "decide"]
