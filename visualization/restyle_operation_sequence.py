#!/usr/bin/env python3
"""Re-render the Figure 5 operation-sequence montage in the flat paper style.

Reads the frozen provenance metadata
(``results/figures/recovery_operation_sequence.json``) and the audited raw
simulation frames, and re-typesets the 2x4 panel montage to match the
Figure 1/2 design language. No simulation is re-run; frame bytes are verified
against the sha256 recorded in the metadata before use.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "results" / "figures" / "recovery_operation_sequence.json"
OUT_FIGURES = ROOT / "results" / "figures" / "recovery_operation_sequence.png"
OUT_PAPER = ROOT / "paper_assets" / "Figure5_operation_sequence.png"

# Flat paper palette (aligned with visualization/generate_paper_figure1.py).
C_INK = "#0F172A"
C_GRAY = "#64748B"
C_BLUE = "#2563EB"
C_AMBER = "#D97706"
C_GREEN = "#059669"

# Semantic border colors: paired start (blue), ACT failure path + risk trigger
# (amber), recovery path (green).
BORDER_BY_KEY = {
    "act_initial": C_BLUE,
    "act_disturbance": C_AMBER,
    "act_unrecovered": C_AMBER,
    "act_failure": C_AMBER,
    "reim_trigger": C_AMBER,
    "reim_relift": C_GREEN,
    "reim_transport": C_GREEN,
    "reim_success": C_GREEN,
}


def _verified_frame(path: Path, expected: dict[str, object]) -> np.ndarray:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected["sha256"] or len(data) != expected["bytes"]:
        raise ValueError(f"frame failed provenance check: {path}")
    return np.asarray(Image.open(path).convert("RGB"))


def main() -> None:
    meta = json.loads(METADATA.read_text(encoding="utf-8"))
    keyframes = meta["keyframes"]
    artifacts = meta["artifacts"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
        }
    )

    fig, axes = plt.subplots(2, 4, figsize=(7.6, 4.15), facecolor="white")
    for axis, keyframe in zip(axes.flat, keyframes, strict=True):
        frame_path = ROOT / keyframe["raw_frame"]
        frame = _verified_frame(frame_path, artifacts[keyframe["raw_frame"]])
        axis.imshow(frame)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(
            keyframe["title"],
            loc="left",
            color=C_INK,
            fontsize=8.6,
            fontweight="bold",
            pad=3.5,
        )
        axis.text(
            0.0,
            -0.065,
            keyframe["subtitle"],
            transform=axis.transAxes,
            ha="left",
            va="top",
            color=C_GRAY,
            fontsize=7.2,
        )
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.6)
            spine.set_edgecolor(BORDER_BY_KEY[keyframe["key"]])

    fig.suptitle(
        "Paired Sawyer PickPlace rollout: ACT failure vs. REIM recovery",
        x=0.035,
        y=0.985,
        ha="left",
        color=C_INK,
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.035,
        0.945,
        (
            "Meta-World/MuJoCo simulation frames"
            f"   ·   separate qualitative seed {meta['seed']}"
            f"   ·   noise {meta['noise_level']:.1f}"
            rf"   ·   $\tau_{{\mathrm{{on}}}}={meta['failure_threshold']:.2f}$"
        ),
        ha="left",
        va="top",
        color=C_GRAY,
        fontsize=8,
    )
    fig.text(
        0.035,
        0.022,
        (
            "Recovery control: until success or 150-step budget"
            "   ·   amber: risk   ·   green: recovery"
            "   ·   simulated Sawyer frames"
        ),
        ha="left",
        va="bottom",
        color=C_GRAY,
        fontsize=7.4,
    )
    fig.subplots_adjust(
        left=0.035,
        right=0.99,
        top=0.875,
        bottom=0.105,
        wspace=0.08,
        hspace=0.30,
    )

    for out in (OUT_FIGURES, OUT_PAPER):
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.stem}.tmp{out.suffix}")
        fig.savefig(tmp, dpi=320)
        tmp.replace(out)
        pdf = out.with_suffix(".pdf")
        pdf_tmp = pdf.with_name(f".{pdf.stem}.tmp.pdf")
        fig.savefig(pdf_tmp)
        pdf_tmp.replace(pdf)
        print(f"wrote {out} (+pdf)")
    plt.close(fig)


if __name__ == "__main__":
    main()
