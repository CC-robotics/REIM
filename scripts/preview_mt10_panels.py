"""MT10 preview panels mirroring the MT1 paper figures.

Figure A (method comparison): task-macro success with 95% CI, mean episode
steps, and intervention episode rate for the four official methods under the
official-clean condition — the MT10 counterpart of MT1's
``success_comparison`` (Success Rate / Recovery Rate / Average Steps).

Figure B (threshold sensitivity): detector precision/recall/F1 against the
operating threshold from the tuner grid — the MT10 counterpart of MT1's
``gate_sensitivity``.

Numbers come from the publication-validated statistics CSVs produced by
``scripts/preview_mt10_robustness.py`` plus the tuner grid. Preview outputs
go to ``results/preview/``; ``paper_assets/`` is never touched.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import matplotlib.pyplot as plt
import numpy as np

TABLES = PROJECT_ROOT / "results" / "tables"
PREVIEW = PROJECT_ROOT / "results" / "preview"

METHODS = ["mlp_bc", "act", "heuristic_recovery", "reim"]
LABELS = {
    "mlp_bc": "MT-MLP BC",
    "act": "MT-ACT",
    "heuristic_recovery": "MT-ACT + Heuristic-Gated Learned Recovery",
    "reim": "MT-REIM",
}
SHORT = {
    "mlp_bc": "MT-MLP BC",
    "act": "MT-ACT",
    "heuristic_recovery": "Heuristic gate",
    "reim": "MT-REIM",
}
COLORS = {
    "mlp_bc": "#737B85",
    "act": "#2457A6",
    "heuristic_recovery": "#D48A1F",
    "reim": "#17805C",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _style_ax(ax) -> None:
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.55, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def _method_comparison() -> Path:
    summary = json.loads((TABLES / "mt10_clean_summary.json").read_text())
    stats: dict[str, dict[str, str]] = {}
    with (PREVIEW / "mt10_clean_statistics_preview.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            stats[row["method"]] = row

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.6))
    x = np.arange(len(METHODS))
    colors = [COLORS[m] for m in METHODS]
    labels = [SHORT[m] for m in METHODS]

    success = [float(stats[LABELS[m]]["task_macro_success"]) * 100.0 for m in METHODS]
    low = [float(stats[LABELS[m]]["success_ci_lower"]) * 100.0 for m in METHODS]
    high = [float(stats[LABELS[m]]["success_ci_upper"]) * 100.0 for m in METHODS]
    axes[0].bar(x, success, color=colors, width=0.62, zorder=3)
    axes[0].errorbar(
        x,
        success,
        yerr=np.vstack((np.asarray(success) - low, np.asarray(high) - success)),
        fmt="none",
        ecolor="#3F4650",
        elinewidth=0.9,
        capsize=2.5,
        zorder=4,
    )
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Task-macro success (%)")
    axes[0].set_title("Success (official clean)")

    steps = [
        float(summary["aggregates"][m]["summary"]["mean_steps_task_macro"])
        for m in METHODS
    ]
    axes[1].bar(x, steps, color=colors, width=0.62, zorder=3)
    axes[1].set_ylim(0, 500)
    axes[1].set_ylabel("Mean steps per episode")
    axes[1].set_title("Average steps")

    intervention = [
        float(summary["aggregates"][m]["summary"]["intervention_episode_rate_task_macro"])
        * 100.0
        for m in METHODS
    ]
    axes[2].bar(x, intervention, color=colors, width=0.62, zorder=3)
    axes[2].set_ylim(0, 105)
    axes[2].set_ylabel("Episodes with intervention (%)")
    axes[2].set_title("Intervention rate")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        _style_ax(ax)
    fig.tight_layout()
    out = PREVIEW / "mt10_method_comparison_preview.png"
    fig.savefig(out, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _threshold_sensitivity() -> Path:
    with (TABLES / "mt10_detector_threshold_grid.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    threshold = np.asarray([float(r["threshold"]) for r in rows])
    selected = next(float(r["threshold"]) for r in rows if r["selected"] == "True")

    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    series = [
        ("task_macro_precision", "Precision (macro)", "#2457A6", "-"),
        ("task_macro_recall", "Recall (macro)", "#D48A1F", "-."),
        ("task_macro_f1", "F1 (macro)", "#17805C", "-"),
    ]
    for key, label, color, ls in series:
        ax.plot(
            threshold,
            [float(r[key]) for r in rows],
            color=color,
            linestyle=ls,
            linewidth=1.3,
            label=label,
        )
    ax.axhline(0.6, color="#737B85", linewidth=0.8, linestyle=":")
    ax.text(0.02, 0.605, "precision floor 0.60", fontsize=6.5, color="#3F4650")
    ax.axvline(selected, color="#B03A2E", linewidth=0.9, linestyle="--")
    ax.text(
        selected + 0.02,
        0.10,
        f"selected τ = {selected:.2f}",
        fontsize=7.0,
        color="#B03A2E",
    )
    ax.set_xlabel("Detector operating threshold τ")
    ax.set_ylabel("Score on held-out validation bank")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.03)
    ax.set_title("MT10 detector threshold selection (validation only)")
    ax.legend(frameon=False, loc="lower left")
    _style_ax(ax)
    fig.tight_layout()
    out = PREVIEW / "mt10_threshold_sensitivity_preview.png"
    fig.savefig(out, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    a = _method_comparison()
    b = _threshold_sensitivity()
    print(f"wrote {a}")
    print(f"wrote {b}")


if __name__ == "__main__":
    sys.exit(main())
