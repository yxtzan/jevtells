"""Small YAML-backed translation loader with per-key fallback."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from .resources import data_path


def _path_for(lang: str) -> Path:
    return data_path(f"i18n/{lang}.yaml")


def load_translations(lang: str = "zh") -> dict[str, Any]:
    """Load the requested language and merge missing values from English."""

    requested = str(lang or "zh").lower()
    if requested not in {"zh", "en"}:
        requested = "zh"

    def read(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        return dict(value) if isinstance(value, Mapping) else {}

    english = read(_path_for("en"))
    chosen = read(_path_for(requested))

    def merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = dict(base)
        for key, value in override.items():
            if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
                result[key] = merge(result[key], value)
            else:
                result[key] = value
        return result

    return merge(english, chosen)


def translate(key: str, lang: str = "zh", translations: Mapping[str, Any] | None = None) -> str:
    """Return a dotted translation, falling back to English and then key."""

    values: Any = translations if translations is not None else load_translations(lang)
    for part in str(key).split("."):
        if not isinstance(values, Mapping) or part not in values:
            return str(key)
        values = values[part]
    return str(values) if values is not None else str(key)


__all__ = ["load_translations", "translate"]
