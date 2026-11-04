"""Subprocess adapter: stage notifications around the original CLI functions."""
from __future__ import annotations

import functools
from typing import Any, Callable


def main() -> None:
    from .. import cli
    from .. import render

    def progress(fn: Callable[..., Any], name: str) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            print(f"UI_STAGE {name}", flush=True)
            return fn(*args, **kwargs)
        return wrapped

    for name in ("prepare", "pose", "track", "asr", "voice", "shots", "segment", "scene_stage", "state", "debug", "judge", "narrate"):
        module = getattr(cli, name)
        module.run = progress(module.run, "scene" if name == "scene_stage" else name)
    cli._run_actions = progress(cli._run_actions, "actions")
    render.run = progress(render.run, "render")
    cli.main()


if __name__ == "__main__":
    main()
