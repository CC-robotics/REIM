#!/usr/bin/env python3
"""Create rounded-corner copies of the qualitative keyframes (originals kept)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "figures" / "recovery_operation_sequence_frames"
DST = ROOT / "results" / "figures" / "recovery_operation_sequence_frames_rounded"
RADIUS = 28


def rounded(path: Path, out: Path) -> None:
    img = Image.open(path).convert("RGBA")
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, img.size[0], img.size[1]), radius=RADIUS, fill=255)
    img.putalpha(mask)
    img.save(out)


def main() -> None:
    DST.mkdir(exist_ok=True)
    count = 0
    for path in sorted(SRC.glob("*.png")):
        rounded(path, DST / path.name)
        count += 1
    print(f"rounded {count} frames -> {DST}")


if __name__ == "__main__":
    main()
