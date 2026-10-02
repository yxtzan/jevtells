"""Pydantic schemas for stage boundary JSON files."""

from pydantic import BaseModel


class Window(BaseModel):
    """A speech or silence interval in the clip."""

    id: str
    index: int
    t0: float
    t1: float
    subtitle: str
    prev_subtitle: str = ""
    shot: str = "target"
    kind: str = "speech"


class State(BaseModel):
    """The exactly five fields sent to the later Jev stage."""

    scene: str
    speaker: str
    subtitle: dict[str, str]
    voice: str
    measured_actions: list[str]
