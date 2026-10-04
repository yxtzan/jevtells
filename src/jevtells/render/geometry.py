"""Scale design coordinates and map source pixels into the video rectangle."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def parse_blurs(values: Sequence[str]) -> list[tuple[float, float, float, float]]:
    result = []
    for value in values:
        try:
            rect = tuple(float(part) for part in value.split(","))
        except ValueError as error:
            raise ValueError("--blur requires X,Y,W,H") from error
        if len(rect) != 4 or not all(math.isfinite(n) for n in rect) or min(rect[:2]) < 0 or min(rect[2:]) <= 0:
            raise ValueError(f"invalid blur rectangle: {value!r}")
        result.append(rect)
    return result


@dataclass(frozen=True)
class Layout:
    kind: str
    design: tuple[int, int]
    output: tuple[int, int]
    video: tuple[float, float, float, float]
    source: tuple[int, int]
    crop: tuple[float, float, float, float] | None = None

    @classmethod
    def create(cls, kind: str, settings: Mapping[str, Any], source: tuple[int, int], crop: tuple[float, float, float, float] | None = None) -> Layout:
        values = settings[kind]
        design, output = tuple(values["design"]), tuple(values["output"])
        if not math.isclose(output[0] / design[0], output[1] / design[1]):
            raise ValueError("render output must preserve the design aspect ratio")
        x, y, width, height = values["video"]
        sw,sh=(crop[2],crop[3]) if crop else source
        fit = min(width / sw, height / sh)
        drawn_w, drawn_h = round(sw * fit), round(sh * fit)
        return cls(kind, design, output, (x + (width - drawn_w) / 2, y + (height - drawn_h) / 2, drawn_w, drawn_h), source,crop)

    @property
    def scale(self) -> float:
        return self.output[0] / self.design[0]

    def px(self, value: float) -> int:
        return round(value * self.scale)

    def point(self, x: float, y: float) -> tuple[int, int]:
        return self.px(x), self.px(y)

    def rect(self, values: Sequence[float]) -> tuple[int, int, int, int]:
        x, y, width, height = values
        return self.px(x), self.px(y), self.px(x + width), self.px(y + height)

    def source_point(self, x: float, y: float) -> tuple[int, int]:
        left, top, width, height = self.video
        if self.crop:
            cx,cy,cw,ch=self.crop
            return self.point(left+(x-cx)*width/cw,top+(y-cy)*height/ch)
        return self.point(left + x * width / self.source[0], top + y * height / self.source[1])

    def source_rect(self, values: Sequence[float]) -> tuple[int, int, int, int]:
        x, y, width, height = values
        return (*self.source_point(x, y), *self.source_point(x + width, y + height))
