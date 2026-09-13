#!/usr/bin/env python3
"""Build the editorial, open-composition REIM overview as a vector PDF.

The PDF is the publication source. Raster simulation frames remain raster, while
all typography, arrows, separators, annotations, and bars are vector objects.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from reportlab.lib.colors import Color, HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas


ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "results" / "figures" / "recovery_operation_sequence_frames"

# Design coordinates are deliberately larger than the PDF canvas. Geometry is
# scaled by 0.5 while font sizes remain in points, preserving 8--10 pt labels
# when the figure is placed at IEEE two-column width.
S = 0.5
DESIGN_W = 1200
DESIGN_H = 460
PAGE_W = DESIGN_W * S
PAGE_H = DESIGN_H * S


def register_fonts() -> dict[str, str]:
    fonts = {
        "regular": ("TNR", Path("C:/Windows/Fonts/times.ttf"), "Times-Roman"),
        "bold": ("TNR-Bold", Path("C:/Windows/Fonts/timesbd.ttf"), "Times-Bold"),
        "italic": ("TNR-Italic", Path("C:/Windows/Fonts/timesi.ttf"), "Times-Italic"),
        "bolditalic": (
            "TNR-BoldItalic",
            Path("C:/Windows/Fonts/timesbi.ttf"),
            "Times-BoldItalic",
        ),
    }
    resolved: dict[str, str] = {}
    for role, (name, path, fallback) in fonts.items():
        if path.exists():
            pdfmetrics.registerFont(TTFont(name, str(path)))
            resolved[role] = name
        else:
            resolved[role] = fallback
    return resolved


INK = HexColor("#292929")
MUTED = HexColor("#666B70")
LIGHT = HexColor("#D6D7D8")
BLUE = HexColor("#3C6E9D")
BLUE_LIGHT = HexColor("#9BB2C7")
AMBER = HexColor("#AE7024")
RED = HexColor("#94423D")
GREEN = HexColor("#3F7C5B")
PALE_GREEN = HexColor("#E8F0EB")


class Figure:
    def __init__(self, canvas: Canvas, fonts: dict[str, str]) -> None:
        self.c = canvas
        self.fonts = fonts

    @staticmethod
    def x(value: float) -> float:
        return value * S

    @staticmethod
    def y(value: float) -> float:
        return PAGE_H - value * S

    def text(
        self,
        value: str,
        x: float,
        y: float,
        size: float,
        color: Color = INK,
        role: str = "regular",
        max_width: float | None = None,
        align: str = "left",
    ) -> None:
        font = self.fonts[role]
        if max_width is not None:
            target = self.x(max_width)
            while size > 8.5 and pdfmetrics.stringWidth(value, font, size) > target:
                size -= 0.25
        self.c.setFillColor(color)
        self.c.setFont(font, size)
        px = self.x(x)
        py = self.y(y)
        if align == "center":
            self.c.drawCentredString(px, py, value)
        elif align == "right":
            self.c.drawRightString(px, py, value)
        else:
            self.c.drawString(px, py, value)

    def multiline(
        self,
        lines: list[str],
        x: float,
        y: float,
        size: float,
        leading: float,
        color: Color = INK,
        role: str = "regular",
        align: str = "left",
    ) -> None:
        for index, value in enumerate(lines):
            self.text(value, x, y + index * leading, size, color, role, align=align)

    def line(
        self,
        points: list[tuple[float, float]],
        color: Color,
        width: float = 0.8,
        dash: tuple[float, ...] | None = None,
    ) -> None:
        self.c.saveState()
        self.c.setStrokeColor(color)
        self.c.setLineWidth(width)
        self.c.setLineCap(1)
        self.c.setLineJoin(1)
        if dash:
            self.c.setDash(*dash)
        path = self.c.beginPath()
        path.moveTo(self.x(points[0][0]), self.y(points[0][1]))
        for point in points[1:]:
            path.lineTo(self.x(point[0]), self.y(point[1]))
        self.c.drawPath(path, stroke=1, fill=0)
        self.c.restoreState()

    def arrow(
        self,
        points: list[tuple[float, float]],
        color: Color,
        width: float = 0.9,
        dash: tuple[float, ...] | None = None,
        head: float = 7,
    ) -> None:
        self.line(points, color, width, dash)
        x2, y2 = points[-1]
        x1, y1 = points[-2]
        angle = math.atan2(y2 - y1, x2 - x1)
        head_scaled = head * S
        tip_x, tip_y = self.x(x2), self.y(y2)
        # Convert the design-space direction into PDF coordinates.
        pdf_angle = -angle
        back_x = tip_x - math.cos(pdf_angle) * head_scaled
        back_y = tip_y - math.sin(pdf_angle) * head_scaled
        side = head_scaled * 0.48
        p1 = (
            back_x + math.cos(pdf_angle + math.pi / 2) * side,
            back_y + math.sin(pdf_angle + math.pi / 2) * side,
        )
        p2 = (
            back_x + math.cos(pdf_angle - math.pi / 2) * side,
            back_y + math.sin(pdf_angle - math.pi / 2) * side,
        )
        self.c.saveState()
        self.c.setFillColor(color)
        path = self.c.beginPath()
        path.moveTo(tip_x, tip_y)
        path.lineTo(*p1)
        path.lineTo(*p2)
        path.close()
        self.c.drawPath(path, stroke=0, fill=1)
        self.c.restoreState()

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        fill: Color | None,
        stroke: Color | None = None,
        line_width: float = 0.7,
    ) -> None:
        self.c.saveState()
        if fill is not None:
            self.c.setFillColor(fill)
        if stroke is not None:
            self.c.setStrokeColor(stroke)
            self.c.setLineWidth(line_width)
        self.c.rect(
            self.x(x),
            self.y(y + height),
            self.x(width),
            self.x(height),
            stroke=int(stroke is not None),
            fill=int(fill is not None),
        )
        self.c.restoreState()

    def ellipse(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        fill: Color | None,
        stroke: Color | None = None,
        line_width: float = 0.7,
    ) -> None:
        self.c.saveState()
        if fill is not None:
            self.c.setFillColor(fill)
        if stroke is not None:
            self.c.setStrokeColor(stroke)
            self.c.setLineWidth(line_width)
        self.c.ellipse(
            self.x(x),
            self.y(y + height),
            self.x(x + width),
            self.y(y),
            stroke=int(stroke is not None),
            fill=int(fill is not None),
        )
        self.c.restoreState()

    def diamond(
        self,
        cx: float,
        cy: float,
        width: float,
        height: float,
        stroke: Color,
    ) -> None:
        self.c.saveState()
        self.c.setStrokeColor(stroke)
        self.c.setLineWidth(0.9)
        path = self.c.beginPath()
        path.moveTo(self.x(cx), self.y(cy - height / 2))
        path.lineTo(self.x(cx + width / 2), self.y(cy))
        path.lineTo(self.x(cx), self.y(cy + height / 2))
        path.lineTo(self.x(cx - width / 2), self.y(cy))
        path.close()
        self.c.drawPath(path, stroke=1, fill=0)
        self.c.restoreState()

    def image(
        self,
        path: Path,
        x: float,
        y: float,
        width: float,
        height: float,
        stroke: Color,
    ) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        self.c.drawImage(
            str(path),
            self.x(x),
            self.y(y + height),
            self.x(width),
            self.x(height),
            preserveAspectRatio=True,
            mask="auto",
        )
        self.rect(x, y, width, height, None, stroke, 0.65)


def draw(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fonts = register_fonts()
    canvas = Canvas(str(output), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    canvas.setTitle("REIM overview - editorial open composition")
    fig = Figure(canvas, fonts)

    # Two quiet separators replace the previous three enclosing cards.
    fig.line([(326, 18), (326, 442)], LIGHT, 0.55)
    fig.line([(884, 18), (884, 442)], LIGHT, 0.55)

    # (a) Motivation: show the failure sequence, then one concise interpretation.
    fig.text("(a)", 22, 31, 11, INK, "bold")
    fig.text("Motivation", 51, 31, 16, RED, "bold")
    fig.image(FRAMES / "02_act_disturbance_seed8300042_t003.png", 28, 70, 124, 124, AMBER)
    fig.image(FRAMES / "04_act_failure_seed8300042_t200.png", 181, 86, 124, 124, RED)
    fig.arrow([(156, 131), (168, 126), (176, 139)], RED, 0.75)
    fig.text("object shifted", 29, 212, 9.5, AMBER, "bold")
    fig.text("ACT timeout", 182, 228, 9.5, RED, "bold")

    fig.text("ACT keeps the stale action chunk", 29, 261, 10.5, INK, "bold", 270)
    for index in range(9):
        fig.rect(30 + index * 25, 276, 18, 14, BLUE if index < 4 else BLUE_LIGHT)
    fig.ellipse(119, 267, 18, 18, RED)
    fig.text("!", 128, 281, 9, white, "bold", align="center")
    fig.line([(128, 267), (128, 251)], RED, 0.7, (1.5, 1.5))
    fig.text("disturbance", 94, 310, 9.5, RED, "italic")

    fig.text("state drifts away", 28, 350, 11, INK, "bold")
    fig.text("outside demonstrated states", 28, 383, 14, RED, "italic", 270)
    fig.line([(28, 393), (205, 393)], RED, 0.65)
    fig.text("no recovery branch", 28, 420, 10.5, RED, "italic")
    fig.line([(164, 413), (202, 398), (231, 408), (269, 387)], RED, 0.65, (2.5, 2))

    # (b) Framework: one primary online loop and a subordinate training strip.
    fig.text("(b)", 346, 31, 11, INK, "bold")
    fig.text("REIM", 376, 31, 16, BLUE, "bold")
    fig.text("risk-gated selective recovery", 470, 31, 9.5, MUTED, "italic")

    fig.rect(347, 77, 142, 142, None, LIGHT, 0.5)
    fig.rect(353, 71, 142, 142, None, LIGHT, 0.5)
    fig.image(FRAMES / "01_act_initial_seed8300042_t000.png", 359, 65, 142, 142, MUTED)
    fig.text("state history", 359, 228, 10.5, MUTED, "bold")
    fig.text("sₜ₋₉ ... sₜ", 444, 228, 10, MUTED, "italic")

    fig.text("ACT", 543, 84, 14, BLUE, "bold")
    fig.text("nominal chunk", 543, 108, 10, INK)
    for index in range(6):
        fig.rect(544 + index * 24, 118, 17, 9, BLUE if index < 3 else BLUE_LIGHT)
    fig.arrow([(504, 116), (532, 116)], BLUE, 0.85)

    fig.text("causal risk", 543, 162, 11, AMBER, "bold")
    fig.text("last 10 states", 543, 183, 9.5, MUTED)
    fig.line(
        [(544, 199), (557, 191), (569, 205), (581, 184), (595, 200), (610, 193), (625, 202)],
        AMBER,
        0.8,
    )
    fig.arrow([(504, 177), (532, 177)], AMBER, 0.85)

    fig.diamond(704, 147, 76, 64, RED)
    fig.text("pₜ ≥ 0.20?", 704, 151, 9.5, RED, "bold", align="center")
    fig.arrow([(645, 122), (666, 122), (666, 137)], BLUE, 0.8)
    fig.arrow([(628, 199), (650, 199), (666, 160)], AMBER, 0.8)
    fig.arrow([(742, 137), (778, 111), (812, 111)], BLUE, 0.85)
    fig.text("continue ACT", 770, 92, 10.5, BLUE, "bold")
    fig.arrow([(742, 158), (775, 188), (812, 188)], GREEN, 0.85)
    fig.text("recovery actor", 766, 211, 10.5, GREEN, "bold")

    fig.ellipse(834, 138, 34, 34, INK)
    fig.text("aₜ", 851, 161, 11, white, "bolditalic", align="center")
    fig.arrow([(812, 111), (851, 111), (851, 136)], BLUE, 0.8)
    fig.arrow([(812, 188), (851, 188), (851, 174)], GREEN, 0.8)
    fig.arrow([(833, 155), (804, 240), (512, 240), (512, 200)], MUTED, 0.65, (2.5, 2))
    fig.text("next state", 646, 235, 9.5, MUTED, "italic")

    fig.line([(347, 268), (863, 268)], LIGHT, 0.5)
    fig.text("Recovery learns at trigger states", 347, 292, 11, INK, "bold")
    fig.text("42,386 train / 8,212 validation pairs", 649, 310, 9.5, MUTED, max_width=210)
    fig.image(FRAMES / "03_trigger_seed5100042_t009.png", 360, 314, 88, 88, AMBER)
    fig.image(FRAMES / "04_relift_seed5100042_t040.png", 518, 327, 88, 88, GREEN)
    fig.text("trigger state", 360, 422, 9.5, AMBER, "bold")
    fig.text("expert continuation", 518, 435, 9.5, GREEN, "bold")
    fig.arrow([(452, 357), (487, 349), (511, 365)], MUTED, 0.7)
    fig.arrow([(615, 372), (655, 357)], GREEN, 0.75)
    fig.text("Smooth-L1", 666, 348, 11, GREEN, "bold")
    fig.text("expert actions", 666, 368, 9.5, MUTED)
    fig.line([(664, 375), (746, 375)], GREEN, 0.65)
    fig.arrow([(748, 357), (790, 357)], GREEN, 0.75)
    fig.ellipse(797, 334, 60, 60, PALE_GREEN, GREEN, 0.7)
    fig.text("πrec", 827, 368, 12, GREEN, "bolditalic", align="center")
    fig.text("same visited states", 696, 420, 9.5, MUTED, "italic")

    # (c) Result: direct labels, no chart box, and the paired interval next to the gain.
    fig.text("(c)", 903, 31, 11, INK, "bold")
    fig.text("Outcome", 933, 31, 16, GREEN, "bold")
    fig.image(FRAMES / "05_reim_trigger_seed8300042_t009.png", 906, 68, 82, 82, AMBER)
    fig.image(FRAMES / "06_reim_relift_seed8300042_t040.png", 1003, 80, 82, 82, GREEN)
    fig.image(FRAMES / "08_reim_success_seed8300042_t062.png", 1100, 65, 82, 82, GREEN)
    fig.arrow([(991, 110), (999, 118)], MUTED, 0.6)
    fig.arrow([(1088, 120), (1096, 110)], MUTED, 0.6)
    fig.text("risk", 932, 171, 9.5, AMBER, "bold")
    fig.text("re-grasp", 1018, 183, 9.5, GREEN, "bold")
    fig.text("success", 1118, 168, 9.5, GREEN, "bold")

    fig.text("Task success (%)", 907, 220, 11, INK, "bold")
    fig.text("paired Δ  +17.0 pp", 907, 243, 11, GREEN, "bold")
    fig.text("95% CI  +14.7 to +19.3", 907, 262, 9.5, MUTED)
    fig.line([(924, 407), (1171, 407)], MUTED, 0.55)
    fig.rect(955, 315, 48, 92, BLUE)
    fig.rect(1072, 294, 48, 113, GREEN)
    fig.text("73.4%", 979, 307, 11, INK, "bold", align="center")
    fig.text("90.4%", 1096, 286, 11, INK, "bold", align="center")
    fig.text("ACT", 979, 431, 10, INK, "bold", align="center")
    fig.text("REIM", 1096, 431, 10, INK, "bold", align="center")
    fig.text("n = 1,000 paired episodes", 930, 445, 9.5, MUTED)

    canvas.showPage()
    canvas.save()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tmp" / "pdfs" / "Figure1_overview_candidate.pdf",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    draw(args.output.resolve())
    print(args.output.resolve())
