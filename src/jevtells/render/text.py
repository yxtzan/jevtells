"""Cached fonts, shrink-before-wrap fitting, and highlighted quotation runs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class FittedText:
    font: ImageFont.FreeTypeFont
    size: int
    lines: tuple[str, ...]


def _quote_runs(text: str, quoted: bool = False) -> tuple[list[tuple[str, bool]], bool]:
    """Carry quotation highlighting across wrapped lines, including brackets."""
    runs: list[tuple[str, bool]] = []
    for char in text:
        if char == "「":
            quoted = True
        highlighted = quoted
        if runs and runs[-1][1] == highlighted:
            runs[-1] = (runs[-1][0] + char, highlighted)
        else:
            runs.append((char, highlighted))
        if char == "」":
            quoted = False
    return runs, quoted


def _line_width(text: str, font: ImageFont.FreeTypeFont, padding: float, quoted: bool = False) -> float:
    runs, _ = _quote_runs(text, quoted)
    return sum(font.getlength(run) + (2 * padding if highlighted else 0) for run, highlighted in runs)


class Fonts:
    def __init__(self, settings: Mapping[str, Any], scale: float) -> None:
        self.paths = settings["fonts"]
        self.scale = scale
        self.minimum = float(settings["min_font_ratio"])
        self.root = Path(__file__).resolve().parents[3]
        for relative in self.paths.values():
            if not (self.root / relative).exists():
                raise FileNotFoundError(f"render font missing: {relative}; run scripts/download_models.py")

    @lru_cache(maxsize=128)
    def get(self, role: str, size: int) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(str(self.root / self.paths[role]), max(1, size))

    def font(self, role: str, size: float, text: str = "") -> ImageFont.FreeTypeFont:
        if role == "mono" and re.search(r"[\u3400-\u9fff]", text):
            role = "sans"
        return self.get(role, round(size * self.scale))

    def fit(self, text: str, role: str, size: float, width: float, max_lines: int = 2, highlight_padding: float = 0) -> FittedText:
        # The OFL Latin mono font has no CJK glyphs. Use the downloaded CJK
        # face for mixed UI labels; pure numbers and window IDs remain mono.
        if role == "mono" and re.search(r"[\u3400-\u9fff]", text):
            role = "sans"
        original = round(size * self.scale)
        minimum = round(original * self.minimum)
        available = width * self.scale
        padding = highlight_padding * self.scale
        # Exhaust the allowed font reduction before attempting any wrapping.
        for candidate in range(original, minimum - 1, -1):
            font = self.get(role, candidate)
            if _line_width(text, font, padding) <= available:
                return FittedText(font, candidate, (text,))
        font = self.get(role, minimum)
        tokens = re.findall(r"「[^」]*」|[a-zA-Z0-9][a-zA-Z0-9'’\-]*\s*|.", text)
        rows: list[str] = []
        current = ""
        row_quoted = False
        for token in tokens:
            _, token_quoted = _quote_runs(current, row_quoted)
            fragments = list(token) if _line_width(token, font, padding, token_quoted) > available else [token]
            for fragment in fragments:
                candidate = current + fragment
                if current and _line_width(candidate, font, padding, row_quoted) > available:
                    rows.append(current.rstrip())
                    _, row_quoted = _quote_runs(current, row_quoted)
                    current = fragment.lstrip()
                else:
                    current = candidate
        if current:
            rows.append(current.rstrip())
        row_quoted = False
        oversized = False
        for row in rows:
            oversized |= _line_width(row, font, padding, row_quoted) > available
            _, row_quoted = _quote_runs(row, row_quoted)
        if len(rows) > max_lines or oversized:
            raise ValueError(f"text cannot fit {max_lines} lines at the minimum font size: {text!r}")
        return FittedText(font, minimum, tuple(rows or [""]))


def draw_fitted(image: Image.Image, position: tuple[int, int], fitted: FittedText, colors: Mapping[str, str], *, line_height: float, highlight_padding: int = 0, fill: str | None = None) -> int:
    draw = ImageDraw.Draw(image)
    x, y = position
    quoted = False
    for line in fitted.lines:
        cursor = float(x)
        runs, quoted = _quote_runs(line, quoted)
        for run, highlighted in runs:
            width = fitted.font.getlength(run)
            if highlighted:
                draw.rectangle((round(cursor), y, round(cursor + width + 2 * highlight_padding), y + round(fitted.size * line_height)), fill=colors["lime"])
                cursor += highlight_padding
            draw.text((round(cursor), y), run, font=fitted.font, fill=colors["ink"] if highlighted else (fill or colors["fg"]), anchor="lt")
            cursor += width + (highlight_padding if highlighted else 0)
        y += round(fitted.size * line_height)
    return y


def spaced_text(draw: ImageDraw.ImageDraw, position: tuple[int, int], text: str, font: ImageFont.FreeTypeFont, fill: str, spacing: int) -> None:
    x, y = position
    for char in text:
        draw.text((round(x), y), char, font=font, fill=fill, anchor="lt")
        x += font.getlength(char) + spacing
