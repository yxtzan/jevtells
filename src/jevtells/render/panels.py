"""Cached magazine-style panels and the few values animated at a cut."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping, Sequence

from PIL import Image, ImageColor, ImageDraw

from ..stages.narrate_facts import SCORE_IDS, score_value
from ..stages.certainty import uncertain
from .animation import interpolate, progress, panel_index
from .geometry import Layout
from .text import Fonts, draw_fitted, spaced_text


def opacity(image: Image.Image, amount: float) -> Image.Image:
    if amount >= 1:
        return image
    result = image.copy()
    result.putalpha(image.getchannel("A").point(lambda value: round(value * max(0.0, amount))))
    return result


class Panels:
    def __init__(self, settings: Mapping[str, Any], layout: Layout, fonts: Fonts, translations: Mapping[str, Any], windows: Sequence[Mapping[str, Any]], judgments: Mapping[str, Any], narration: Mapping[str, Any], title: str, sources: str) -> None:
        self.settings, self.layout, self.fonts = settings, layout, fonts
        self.tr, self.windows, self.judgments, self.narration = translations, windows, judgments, narration
        self.c, self.p, self.g = settings["colors"], dict(settings[layout.kind]), settings["components"]
        self.title, self.sources = title, sources
        self.s = layout.px
        if layout.kind == "v" and windows:
            # Measure all sentences once, then fix the slot and dependent
            # rows for the entire clip (including the other-speaker message).
            old_height = self.p["commentary"][3]
            bottoms = [self.commentary(i)[0].getbbox() for i in range(len(windows))]
            height = max(self.p["commentary_size"] * self.g["line_height"], max(box[3] / layout.scale for box in bottoms if box))
            delta = height - old_height
            self.p["commentary"] = [*self.p["commentary"][:3], height]
            for name in ("quote_y", "metrics_y", "lower_y"):
                self.p[name] += delta
            self.commentary.cache_clear()
            self.p["choice_top"] = max(6, self.p["choice_top"])
            self.p["choice_bottom"] = max(8, self.p["choice_bottom"])
            self.p["arc_top"] = max(8, self.p["arc_top"])
            # Render every content block once to measure glyphs, probability
            # rows and wrapped legends, rather than guessing their heights.
            content_bottom = 0.0
            for index in range(len(windows)):
                sample = self.canvas()
                self._choices(sample, index)
                self._metrics(sample, index, 1.0)
                box = sample.getbbox()
                if box:
                    content_bottom = max(content_bottom, box[3]/layout.scale)
            free = max(0.0, self.p["footer"][1] - content_bottom)
            extra = min(32.0, free/3)
            self.p["metrics_y"] += extra
            self.p["lower_y"] += 2*extra
            self.spacing = {"maximum_content_bottom": content_bottom, "free_height": free, "added_per_gap": extra, "remaining_bottom": free-3*extra}

        self.base = self._base()

    def canvas(self) -> Image.Image:
        return Image.new("RGBA", self.layout.output)

    def text(self, image: Image.Image, text: str, x: float, y: float, width: float, size: float, role: str = "sans", color: str | None = None, lines: int = 1) -> None:
        fit = self.fonts.fit(str(text), role, size, width, lines)
        draw_fitted(image, self.layout.point(x, y), fit, self.c, line_height=self.g["line_height"], fill=color)

    def hair(self, image: Image.Image, x: float, y: float, width: float) -> None:
        ImageDraw.Draw(image).line((*self.layout.point(x, y), *self.layout.point(x + width, y)), fill=self.c["hair"], width=max(1, self.s(self.g["hair_width"])))

    def _base(self) -> Image.Image:
        image = self.canvas()
        draw = ImageDraw.Draw(image)
        if self.layout.kind == "v":
            x, y, width, height = self.p["title"]
            pad_x, pad_y = self.p["title_padding"]
            fit = self.fonts.fit(self.title, "sans_black", self.p["title_size"], width - 2 * pad_x, 1)
            actual_width = min(width, fit.font.getlength(self.title) / self.layout.scale + 2 * pad_x)
            draw.rectangle(self.layout.rect((x, y, actual_width, height)), fill=self.c["lime"])
            draw_fitted(image, self.layout.point(x + pad_x, y + pad_y), fit, self.c, line_height=self.g["line_height"], fill=self.c["ink"])
            self.text(image, self.tr["ui"]["brand"], x, y + height + self.p["brand_gap"], width, self.p["brand_size"])
            fx, fy, fw, fh = self.p["footer"]
            draw.line((*self.layout.point(fx, fy), *self.layout.point(fx + fw, fy)), fill=self.c["baseline"], width=max(1, self.s(self.g["footer_width"])))
            self.text(image, self.sources, fx, fy + self.p["footer_top"], fw, self.p["footer_size"], color=self.c["muted"], lines=2)
        else:
            fx, fy, fw, fh = self.p["footer"]
            px, py = self.p["footer_padding"]
            fit = self.fonts.fit(self.sources, "sans", self.p["footer_size"], fw - 2 * px, 1)
            width = min(fw, fit.font.getlength(self.sources) / self.layout.scale + 2 * px)
            draw.rectangle(self.layout.rect((fx, fy, width, fh)), fill=(*ImageColor.getrgb(self.c["fg"]), round(255 * self.settings["footer_opacity"])))
            draw_fitted(image, self.layout.point(fx + px, fy + py), fit, self.c, line_height=self.g["line_height"], fill=self.c["ink"])
        return image

    @lru_cache(maxsize=32)
    def commentary(self, index: int | None) -> tuple[Image.Image, tuple[int, int]]:
        x, y, width, height = self.p["commentary"]
        layer = Image.new("RGBA", (self.s(width), self.s(height + (22 if index is not None and self.windows[index].get("two_person") and self.layout.kind == "v" else 0))))
        if index is None:
            self.text(layer, self.tr["ui"]["waiting"], 0, 0, width, self.p["commentary_size"], color=self.c["muted"])
            return layer, self.layout.point(x, y)
        window = self.windows[index]
        value = self.narration.get(str(window["id"])) or {}
        line = self.tr["ui"]["other_speaker"] if window.get("speaker_other") else self.tr["ui"]["target_offscreen"] if window.get("target_offscreen") else value.get("line", self.tr["ui"]["missing"])
        if window.get("two_person"):
            if window.get("speaker_unknown"): line = "（无法确定说话人）"
            elif window.get("target_offscreen"): line = f"（{window['speaker']}在画外，本句不做判定）"
        number = f"W{index + 1:02d}"
        tag_font = self.fonts.font("mono", self.p["index_size"])
        pad_x, pad_y = self.g["tag_padding"]
        tag_w = tag_font.getlength(number) / self.layout.scale + 2 * pad_x
        if self.layout.kind == "h":
            px, py = self.p["commentary_padding"]
        else:
            px, py = 0, 0
        offset = px + tag_w + self.p["commentary_gap"]
        speaker_tag = str(window.get("speaker") or "") if window.get("two_person") else ""
        if speaker_tag and self.layout.kind == "h":
            offset += self.fonts.font("sans", self.p["index_size"]+2).getlength(speaker_tag)/self.layout.scale + 2*pad_x + self.p["commentary_gap"]
        fitted = self.fonts.fit(str(line), "serif_black" if self.layout.kind == "v" else "sans_black", self.p["commentary_size"], width - offset - px, 2 if self.layout.kind == "v" else 1, self.g["highlight_padding"])
        draw = ImageDraw.Draw(layer)
        if self.layout.kind == "h":
            content_width = min(width, offset + max(fitted.font.getlength(row) / self.layout.scale for row in fitted.lines) + px + 2 * self.g["highlight_padding"])
            draw.rectangle(self.layout.rect((0, 0, content_width, height)), fill=self.c["ink"])
        tag_y = py + self.p["index_top"] + (22 if window.get("two_person") and self.layout.kind == "v" else 0)
        draw.rectangle(self.layout.rect((px, tag_y, tag_w, self.p["index_size"] * self.g["line_height"] + 2 * pad_y)), fill=self.c["lime"])
        draw.text(self.layout.point(px + pad_x, tag_y + pad_y), number, font=tag_font, fill=self.c["ink"], anchor="lt")
        if speaker_tag:
            if self.layout.kind == "h":
                sx = px + tag_w + self.p["commentary_gap"]
                self.text(layer,speaker_tag,sx,tag_y,width-sx,self.p["index_size"]+2,color=window.get("speaker_color",self.c["lime"]))
            else:
                self.text(layer,"正在说话 · "+speaker_tag,0,0,width,14,color=window.get("speaker_color",self.c["lime"]))
                py += 22
        draw_fitted(layer, self.layout.point(offset, py), fitted, self.c, line_height=self.g["line_height"], highlight_padding=self.s(self.g["highlight_padding"]), fill=self.c["muted"] if window.get("speaker_other") or window.get("target_offscreen") or window.get("speaker_unknown") else self.c["fg"])
        return layer, self.layout.point(x, y)

    @lru_cache(maxsize=32)
    def quote(self, index: int) -> Image.Image:
        image = self.canvas()
        window = self.windows[index]
        value = self.narration.get(str(window["id"])) or {}
        if window.get("speaker_other") or window.get("target_offscreen") or window.get("speaker_unknown") or not value.get("quote"):
            return image
        x, _y, width, _height = self.p["commentary"]
        y, size = self.p["quote_y"], self.p["quote_size"]
        label = self.tr["ui"]["quote"]
        tag_size = self.p.get("quote_tag_size", self.g["quote_tag_size"])
        font = self.fonts.font("sans_black", tag_size)
        px, py = self.p.get("quote_padding", [0, 0])
        tx, ty = self.g["tag_padding"]
        tag_width = font.getlength(label) / self.layout.scale + 2 * tx
        quote = '“' + str(value["quote"]) + '”'
        fit = self.fonts.fit(quote, "sans", size, width - tag_width - self.g["quote_gap"] - 2 * px, 1)
        draw = ImageDraw.Draw(image)
        if self.layout.kind == "h":
            card_width = tag_width + self.g["quote_gap"] + fit.font.getlength(quote) / self.layout.scale + 2 * px
            draw.rectangle(self.layout.rect((x, y, card_width, size * self.g["line_height"] + 2 * py)), fill=self.c["fg"])
        draw.rectangle(self.layout.rect((x + px, y + py, tag_width, tag_size * self.g["line_height"] + 2 * ty)), fill=self.c["lime"])
        draw.text(self.layout.point(x + px + tx, y + py + ty), label, font=font, fill=self.c["ink"], anchor="lt")
        draw_fitted(image, self.layout.point(x + px + tag_width + self.g["quote_gap"], y + py), fit, self.c, line_height=self.g["line_height"], fill=self.c["ink"] if self.layout.kind == "h" else self.c["fg"])
        return image

    def metric_boxes(self) -> list[tuple[float, float, float]]:
        if self.layout.kind == "v":
            x, _y, width, _h = self.p["commentary"]
        else:
            cx, _cy, cw, _ch = self.p["card"]
            x, width = cx + self.p["card_padding"], cw - 2 * self.p["card_padding"]
        gap = self.p["metrics_gap"]
        width = (width - 2 * gap) / 3
        return [(x + i * (width + gap), self.p["metrics_y"], width) for i in range(3)]

    @lru_cache(maxsize=32)
    def panel_static(self, index: int) -> Image.Image:
        image = self.canvas()
        draw = ImageDraw.Draw(image)
        if self.layout.kind == "h":
            x, y, width, height = self.p["card"]
            draw.rectangle(self.layout.rect((x, y, width, height)), fill=(*ImageColor.getrgb(self.c["ink"]), round(255 * self.settings["panel_opacity"])))
            draw.rectangle(self.layout.rect((x, y, width, self.p["card_border"])), fill=self.c["lime"])
            pad = self.p["card_padding"]
            card_title = self.tr["ui"]["card_title"] + (" · " + str(self.windows[index].get("speaker") or "未确定") if self.windows[index].get("two_person") else "")
            self.text(image, card_title, x + pad, y + pad, width - 2 * pad, self.p["card_title_size"], color=self.c["lime"])
            source = "" if self.windows[index].get("two_person") else self.tr["ui"]["card_source"]
            font = self.fonts.font("sans", self.p["card_source_size"])
            self.text(image, source, x + width - pad - font.getlength(source) / self.layout.scale, y + pad, width - 2 * pad, self.p["card_source_size"])
        for key, (x, y, width) in zip(SCORE_IDS, self.metric_boxes()):
            self.hair(image, x, y, width)
            spaced_text(draw, self.layout.point(x, y + self.g["heading_gap"]), self.tr["scores"][key], self.fonts.font("sans", self.p["metric_label_size"]), self.c["muted"], self.s(self.g["letter_spacing"]) if self.layout.kind == "v" else 0)
        self._choices(image, index)
        return image

    def _choices(self, image: Image.Image, index: int) -> None:
        if self.layout.kind == "v":
            x, _y, width, _h = self.p["commentary"]
            half = (width - self.p["lower_gap"]) / 2
            positions = [("intent", "intents", x, self.p["lower_y"], half), ("emotion", "emotions", x + half + self.p["lower_gap"], self.p["lower_y"], half)]
        else:
            x, _y, width, _h = self.p["card"]
            pad = self.p["card_padding"]
            positions = [("intent", "intents", x + pad, self.p["intent_y"], width - 2 * pad), ("emotion", "emotions", x + pad, self.p["emotion_y"], width - 2 * pad)]
        judgment = self.judgments.get(str(self.windows[index]["id"])) or {}
        for field, category, x, y, width in positions:
            self.hair(image, x, y, width)
            heading_y = y + self.g["heading_gap"]
            self.text(image, self.tr["ui"][field], x, heading_y, width, self.p["choice_heading_size"], color=self.c["muted"])
            value = judgment.get(field) or {}
            ambiguous = uncertain(value,float(self.settings.get('min_judgment_confidence',.4)))
            label = self.tr['ui']['unclear'] if ambiguous else self.tr[category].get(value.get("label"), self.tr["ui"]["missing"])
            label_color = self.c['muted'] if ambiguous else self.c['fg']
            if self.layout.kind == "v":
                certainty = value.get("confidence")
                certainty_text = self.tr["ui"]["certainty"].format(value=f"{certainty:.2f}" if isinstance(certainty, (float, int)) else self.tr["ui"]["missing"])
                font = self.fonts.font("mono", self.p["choice_heading_size"], certainty_text)
                self.text(image, certainty_text, x + width - font.getlength(certainty_text) / self.layout.scale, heading_y, width, self.p["choice_heading_size"], role="mono")
                label_y = heading_y + self.p["choice_heading_size"] + self.p["choice_top"]
                self.text(image, label, x, label_y, width, self.p["choice_size"], role="serif_black",color=label_color)
                content_y = label_y + self.p["choice_size"] + (self.p["choice_bottom"] if field == "intent" else self.p["arc_top"])
            else:
                font = self.fonts.font("sans_black", self.p["choice_size"])
                fit = self.fonts.fit(label, "sans_black", self.p["choice_size"], width * self.settings["choice_width_ratio"], 1)
                self.text(image, label, x + width - fit.font.getlength(label) / self.layout.scale, heading_y, width, self.p["choice_size"], role="sans_black",color=label_color)
                content_y = heading_y + self.p["choice_size"] + self.g["bar_gap"] if field == "intent" else self.p["arc_y"]
            if field == "intent":
                # Probability bars are drawn with the animated values below.
                pass
            else:
                self._arc(image, index, x, content_y, width)

    def _bars(self, image: Image.Image, probabilities: Sequence[tuple[str, float]], x: float, y: float, width: float, previous: Mapping[str, float] | None = None, amount: float = 1.0) -> None:
        draw = ImageDraw.Draw(image)
        size = self.p.get("bar_size", self.p["choice_heading_size"])
        label_width = self.p.get("bar_label_em", self.g["bar_label_em"]) * size
        value_width = self.p.get("bar_value_em", self.g["bar_value_em"]) * size
        gap = self.p.get("bar_inner_gap", self.g["bar_inner_gap"])
        bar_x = x + label_width + gap
        bar_width = width - label_width - value_width - 2 * gap
        row_height = size * self.g["line_height"] + self.g["bar_gap"]
        bar_height = self.p.get("bar_height", self.g["bar_height"])
        for row, (label, value) in enumerate(probabilities):
            if previous is not None:
                value = float(previous.get(label, 0)) + (value - float(previous.get(label, 0))) * amount
            top = y + row * row_height
            self.text(image, self.tr["intents"].get(label, label), x, top, label_width, size)
            center = top + (size - bar_height) / 2
            draw.rectangle(self.layout.rect((bar_x, center, bar_width, bar_height)), fill=self.c["track"])
            if value > 0:
                draw.rectangle(self.layout.rect((bar_x, center, bar_width * value, bar_height)), fill=self.c["lime"] if row == 0 else self.c["secondary_bar"])
            self.text(image, f"{value:.2f}", x + width - value_width, top, value_width, size, role="mono", color=self.c["fg"] if row == 0 else self.c["muted"])

    def _arc(self, image: Image.Image, index: int, x: float, y: float, width: float) -> None:
        draw = ImageDraw.Draw(image)
        gap, height = self.g["arc_gap"], self.p["arc_height"]
        cell = (width - (len(self.windows) - 1) * gap) / len(self.windows)
        seen: list[str] = []
        for i, window in enumerate(self.windows):
            rect = self.layout.rect((x + i * (cell + gap), y, cell, height))
            if i > index:
                draw.rectangle(rect, fill=self.c["track"])
                continue
            judgment = self.judgments.get(str(window["id"])) or {}
            emotion = (judgment.get("emotion") or {}).get("label")
            color = self.c["emotions"].get(emotion)
            if color and not (window.get("speaker_other") or window.get("target_offscreen") or window.get("speaker_unknown")):
                choice = judgment.get('emotion')
                fill = (*ImageColor.getrgb(color),round(255*float(self.settings.get('uncertain_arc_alpha',.4)))) if uncertain(choice,float(self.settings.get('min_judgment_confidence',.4))) else color
                draw.rectangle(rect, fill=fill)
                if emotion not in seen:
                    seen.append(emotion)
            else:
                stripe = Image.new("RGBA", (max(1, rect[2] - rect[0]), max(1, rect[3] - rect[1])), self.c["track"])
                sd = ImageDraw.Draw(stripe)
                step = max(1, self.s(self.g["stripe_step"]))
                for pos in range(-stripe.height, stripe.width + stripe.height, step):
                    sd.line((pos, 0, pos + stripe.height, stripe.height), fill=self.c["stripe"], width=max(1, self.s(self.g["stripe_width"])))
                image.alpha_composite(stripe, (rect[0], rect[1]))
                if "other" not in seen:
                    seen.append("other")
            if window.get("two_person"):
                draw.rectangle((rect[0],rect[1],rect[2],rect[1]+self.s(3)),fill=window.get("speaker_color",self.c["muted"]))
            if i == index:
                offset = self.s(self.g["arc_outline_offset"])
                draw.rectangle((rect[0] - offset, rect[1] - offset, rect[2] + offset, rect[3] + offset), outline=self.c["fg"], width=max(1, self.s(self.g["arc_outline"])))
        legend_y = self.p.get("legend_y", y + height + self.g["legend_top"])
        size, square = self.p["legend_size"], self.p.get("legend_square", self.g["legend_square"])
        cursor = x
        for emotion in seen:
            label = self.tr["ui"]["unjudged"] if emotion == "other" else self.tr["emotions"][emotion]
            font = self.fonts.font("sans", size)
            entry_width = square + self.g["legend_inner_gap"] + font.getlength(label) / self.layout.scale
            if cursor + entry_width > x + width:
                cursor = x
                legend_y += size * self.g["line_height"]
            draw.rectangle(self.layout.rect((cursor, legend_y, square, square)), fill=self.c["stripe"] if emotion == "other" else self.c["emotions"][emotion])
            self.text(image, label, cursor + square + self.g["legend_inner_gap"], legend_y, entry_width - square, size, color=self.c["muted"])
            cursor += entry_width + self.p.get("legend_gap", self.g["legend_gap"])

    def _metrics(self, image: Image.Image, index: int, elapsed: float) -> None:
        draw = ImageDraw.Draw(image)
        window = self.windows[index]
        identifier = str(window["id"])
        judgment = self.judgments.get(identifier)
        previous = self.judgments.get(str(self.windows[panel_index(self.windows,index - 1)]["id"])) if index else None
        if window.get("two_person"):
            prior = next((w for w in reversed(self.windows[:index]) if w.get("speaker") == window.get("speaker") and self.judgments.get(str(w["id"]))), None)
            previous = self.judgments.get(str(prior["id"])) if prior else None
        amount = progress(elapsed, self.settings["animation"]["number_seconds"])
        for key, (x, y, width) in zip(SCORE_IDS, self.metric_boxes()):
            new, old = score_value(judgment, key), score_value(previous, key)
            number = interpolate(old, new, amount)
            size = self.p["metric_label_size"]
            if new is not None and old is not None:
                delta = (new - old) * amount
                color = self.c["up"] if (delta >= 0) != (key == "tension") else self.c["red"]
                text = f'{"▲" if delta >= 0 else "▼"} {abs(delta):.2f}'
                font = self.fonts.font("sans", size)
                self.text(image, text, x + width - font.getlength(text) / self.layout.scale, y + self.g["heading_gap"], width, size, color=color)
            number_y = y + self.g["heading_gap"] + size + self.p["number_top"]
            self.text(image, f"{number:.2f}" if number is not None else self.tr["ui"]["missing"], x, number_y, width, self.p["number_size"], role="serif_black" if self.layout.kind == "v" else "sans_black", color=self.c["red"] if key == "tension" else self.c["fg"])
            spark_y = number_y + self.p["number_size"] + self.g["number_bottom"]
            self._spark(image, key, index, elapsed, x, spark_y, min(width, self.p["spark"][0]), self.p["spark"][1])
        intent = (judgment or {}).get("intent") or {}
        probabilities = sorted((intent.get("probs") or {}).items(), key=lambda item: item[1], reverse=True)[:3]
        old_probabilities = ((previous or {}).get("intent") or {}).get("probs")
        if self.layout.kind == "v":
            x, _y, width, _h = self.p["commentary"]
            width = (width - self.p["lower_gap"]) / 2
            bar_y = self.p["lower_y"] + self.g["heading_gap"] + self.p["choice_heading_size"] + self.p["choice_top"] + self.p["choice_size"] + self.p["choice_bottom"]
        else:
            cx, _cy, cw, _ch = self.p["card"]
            x, width = cx + self.p["card_padding"], cw - 2 * self.p["card_padding"]
            bar_y = self.p["intent_y"] + self.g["heading_gap"] + self.p["choice_size"] + self.g["bar_gap"]
        self._bars(image, probabilities, x, bar_y, width, old_probabilities, amount)

    def _spark(self, image: Image.Image, key: str, index: int, elapsed: float, x: float, y: float, width: float, height: float) -> None:
        draw = ImageDraw.Draw(image)
        pad = self.g["spark_padding"]
        values = [score_value(self.judgments.get(str(window["id"])), key) if not self.windows[index].get("two_person") or window.get("speaker")==self.windows[index].get("speaker") else None for window in self.windows]
        known = [value for value in values if value is not None]
        # A fixed per-video domain keeps small changes visible without
        # changing the axis at every window. Future points remain absent.
        low, high = (min(known), max(known)) if known else (0.0, 1.0)
        span = max(high - low, self.g["spark_min_span"]) * (1 + 2 * self.g["spark_domain_padding"])
        center = (low + high) / 2
        domain_low = max(0.0, min(center - span / 2, 1 - span))
        domain_high = min(1.0, domain_low + span)
        span = max(domain_high - domain_low, self.g["spark_min_span"])
        coordinates = [self.layout.point(x + pad + i * (width - 2 * pad) / max(1, len(values) - 1), y + pad + (1 - ((value or 0) - domain_low) / span) * (height - 2 * pad)) for i, value in enumerate(values)]
        draw.line((*self.layout.point(x, y + height), *self.layout.point(x + width, y + height)), fill=self.c["baseline"], width=max(1, self.s(self.g["spark_baseline"])))
        color = self.c["red"] if key == "tension" else self.c["blue"]
        amount = progress(elapsed, self.settings["animation"]["spark_seconds"])
        current = None
        for i in range(index + 1):
            if values[i] is None:
                continue
            point = coordinates[i]
            if i and values[i - 1] is not None:
                before = coordinates[i - 1]
                if i == index:
                    point = (round(before[0] + (point[0] - before[0]) * amount), round(before[1] + (point[1] - before[1]) * amount))
                draw.line((*before, *point), fill=color, width=max(1, self.s(self.g["spark_line"])))
            if i == index:
                current = point
        if current:
            radius = self.s(self.g["spark_dot"])
            draw.ellipse((current[0] - radius, current[1] - radius, current[0] + radius, current[1] + radius), fill=self.c["lime"], outline=self.c["ink"], width=max(1, self.s(self.g["spark_outline"])))

    @lru_cache(maxsize=32)
    def panel(self, index: int) -> Image.Image:
        image = self.panel_static(index).copy()
        self._metrics(image, index, float("inf"))
        return image

    def analysis(self, index: int, elapsed: float) -> Image.Image:
        if elapsed >= max(self.settings["animation"]["number_seconds"], self.settings["animation"]["spark_seconds"]):
            return self.panel(index)
        image = self.panel_static(index).copy()
        self._metrics(image, index, elapsed)
        return image

    @lru_cache(maxsize=512)
    def status(self, index: int | None, tenth: int, lost: bool) -> tuple[Image.Image, tuple[int, int]]:
        x, y, width, _height = self.p["status"]
        text = self.tr["ui"]["status"].format(time=tenth / 10, index=(index + 1) if index is not None else 0, total=len(self.windows), target=self.tr["ui"]["offscreen_short" if index is not None and self.windows[index].get("target_offscreen") else "lost" if lost else "locked"])
        px, py = self.p["status_padding"]
        fit = self.fonts.fit(text, "mono", self.p["status_size"], width - 2 * px, 1)
        w = round(fit.font.getlength(text) + 2 * self.s(px))
        h = round(fit.size * self.g["line_height"] + 2 * self.s(py))
        layer = Image.new("RGBA", (w, h), self.c["muted"] if lost else self.c["lime"])
        draw_fitted(layer, (self.s(px), self.s(py)), fit, self.c, line_height=self.g["line_height"], fill=self.c["ink"])
        return layer, (self.s(x + width) - w if self.layout.kind == "v" else self.s(x), self.s(y))
