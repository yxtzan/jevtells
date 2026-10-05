"""Packaged defaults and explicit user asset locations."""
from __future__ import annotations

import os
from pathlib import Path


def data_path(relative: str) -> Path:
    repository=Path(__file__).resolve().parents[2]
    source=repository/relative
    if (repository/'pyproject.toml').exists() and source.exists():
        return source
    return Path(__file__).resolve().parent/'data'/relative


def asset_roots() -> list[Path]:
    configured=os.environ.get('JEVTELLS_HOME')
    roots=[Path(configured).expanduser()] if configured else []
    return [*roots,Path.cwd(),Path.home()/'.jevtells']


def asset_path(relative: str | Path) -> Path:
    relative=Path(relative).expanduser()
    if relative.is_absolute():
        return relative
    roots=asset_roots()
    return next((root/relative for root in roots if (root/relative).is_file()),roots[0]/relative)
