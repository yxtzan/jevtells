"""Pydantic schemas for stage boundary JSON files."""

from typing import Any

from pydantic import BaseModel, Field


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
    speaker_other: bool = False
    target_offscreen: bool = False
    target_presence_ratio: float | None = Field(default=None, ge=0.0, le=1.0)


class State(BaseModel):
    """The exactly five fields sent to the later Jev stage."""

    scene: str
    speaker: str
    subtitle: dict[str, str]
    voice: str
    measured_actions: list[str]


class JudgmentScore(BaseModel):
    """A normalized score alongside the unmodified service value."""
    value: float | None = Field(default=None, ge=0.0, le=1.0)
    raw: Any = None


class JudgmentChoice(BaseModel):
    """A choice with model confidence and its optional distribution."""
    label: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    probs: dict[str, float] = Field(default_factory=dict)


class Judgment(BaseModel):
    """One window's Jev judgment, including nullable failed results."""
    scores: dict[str, JudgmentScore | None]
    intent: JudgmentChoice | None = None
    emotion: JudgmentChoice | None = None
    actions: dict[str, float | None] = Field(default_factory=dict)
    error: str | None = None


class Narration(BaseModel):
    """A neutral annotation and a checked subtitle quotation."""
    line: str
    quote: str
    error: str | None = None


class SceneDescription(BaseModel):
    """A cached visual description with API accounting."""
    scene: str
    model: str | None = None
    cost: float | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    calls: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    failed: int = Field(ge=0)
    source: str | None = None
    error: str | None = None


class IdentityDecision(BaseModel):
    frame: int
    selected_index: int | None
    person_number: int | None
    similarity: float | None
    second_similarity: float | None
    result: str


class IdentityShot(BaseModel):
    t0: float
    t1: float
    cut_at_start: bool
    decisions: list[IdentityDecision]
    target_presence_ratio: float = Field(ge=0, le=1)


class TrackMetadata(BaseModel):
    anchors: list[tuple[int, float]]
    lost_frames: int
    identity_version: int
    config: dict[str, Any]
    references: list[dict[str, Any]] = Field(default_factory=list)
    shots: list[IdentityShot] = Field(default_factory=list)
    recoveries: list[IdentityDecision] = Field(default_factory=list)
