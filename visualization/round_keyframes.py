#!/usr/bin/env python3
"""Create rounded-corner copies of the qualitative keyframes (originals kept)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "figures" / "recovery_operation_sequence_frames"
DST = ROOT / "results" / "figures" / "recovery_operation_sequence_frames_rounded"
RADIUS = 28
USED_FRAME_NAMES = (
    "02_act_disturbance_seed8300042_t003.png",
    "04_act_failure_seed8300042_t200.png",
    "05_reim_trigger_seed8300042_t009.png",
    "07_reim_transport_seed8300042_t051.png",
)


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
    for name in USED_FRAME_NAMES:
        path = SRC / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing source keyframe: {path}")
        rounded(path, DST / path.name)
        count += 1
    print(f"rounded {count} frames -> {DST}")


if __name__ == "__main__":
    main()
