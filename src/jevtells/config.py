"""Configuration loading for the JevTells pipeline.

The command line accepts a small YAML file that is merged on top of
``config/default.yaml``.  Keeping the merge here gives every stage the same
defaults and makes it possible to tune a run without editing Python files.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from .resources import data_path


Config = dict[str, Any]


def default_path() -> Path:
    """Return the repository's default configuration path."""

    return data_path("config/default.yaml")


def _read(path: Path) -> Config:
    if not path.exists():
        raise FileNotFoundError(f"configuration file not found: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration root must be a YAML mapping: {path}")
    return dict(value)


def _merge(base: Config, override: Mapping[str, Any]) -> Config:
    """Recursively merge ``override`` into ``base`` without mutating either."""

    result: Config = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> Config:
    """Load defaults and optionally merge a user supplied YAML file.

    ``path`` is an override, rather than a replacement: a small file such as
    ``{voice: {f0_max_hz: 450}}`` inherits every other project default.
    Passing the default path explicitly is harmless and is useful for callers
    that expose ``--config`` in their own argument parser.
    """

    default = _read(default_path())
    if path is None:
        return default
    override_path = Path(path).expanduser()
    if not override_path.is_absolute():
        override_path = Path.cwd() / override_path
    if override_path.resolve() == default_path().resolve():
        return default
    return _merge(default, _read(override_path))


def config_value(config: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Read a dotted configuration key, returning ``default`` if absent."""

    if key in config:
        return config[key]
    current: Any = config
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


# A short alias keeps stage code readable while preserving one public helper.
get = config_value


# Backwards-friendly spelling for small stage modules.
load = load_config
