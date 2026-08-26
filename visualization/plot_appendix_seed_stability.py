#!/usr/bin/env python3
"""Appendix A assets: MT10 multi-seed stability (supervised REIM vs heuristic).

Reads the audited aggregate ``results/tables/mt10_multiseed_summary.csv``
(three recovery-training seeds 42/43/44, bank 20265010, 50 episodes per task)
and renders a flat-style figure plus a LaTeX table into ``paper_assets/``.
No simulation is re-run; the CSV is the single source of truth.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "results" / "tables" / "mt10_multiseed_summary.csv"
OUT_FIGURE = ROOT / "paper_assets" / "FigureA1_mt10_seed_stability.png"
OUT_TABLE = ROOT / "paper_assets" / "TableA1_mt10_seed_stability.tex"

C_INK = "#0F172A"
C_GRAY = "#64748B"
C_GRID = "#E9EEF3"
C_GREEN = "#059669"
C_AMBER = "#D97706"

METHOD_STYLE = {
    "reim": ("MT-REIM", C_AMBER, "o", "-"),
    "heuristic_recovery": ("Heuristic gate", C_GREEN, "s", "--"),
}
NOISE_ORDER = (0.0, 0.1, 0.4)
SEEDS = (42, 43, 44)


def _load() -> dict[tuple[float, str], dict[str, str]]:
    rows: dict[tuple[float, str], dict[str, str]] = {}
    with SOURCE_CSV.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[(float(row["noise_level"]), row["method"])] = row
    expected = {(n, m) for n in NOISE_ORDER for m in METHOD_STYLE}
    missing = expected.difference(rows)
    if missing:
        raise ValueError(f"multiseed summary is incomplete: {sorted(missing)}")
    return rows


def _plot(rows: dict[tuple[float, str], dict[str, str]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.edgecolor": "#CBD5E1",
            "text.color": C_INK,
            "axes.labelcolor": C_INK,
            "xtick.color": C_GRAY,
            "ytick.color": C_GRAY,
        }
    )
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    x = np.asarray([n * 100.0 for n in NOISE_ORDER])
    for method, (label, color, marker, linestyle) in METHOD_STYLE.items():
        means = np.asarray(
            [float(rows[(n, method)]["mean"]) * 100.0 for n in NOISE_ORDER]
        )
        stds = np.asarray(
            [float(rows[(n, method)]["sample_std"]) * 100.0 for n in NOISE_ORDER]
        )
        ax.fill_between(
            x, means - stds, means + stds, color=color, alpha=0.14, zorder=2
        )
        ax.plot(
            x,
            means,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=2.0 if method == "reim" else 1.5,
            label=f"{label} (mean ± s.d.)",
            zorder=4 if method == "reim" else 3,
        )
        # Per-seed points show the actual spread behind each band.
        for n in NOISE_ORDER:
            per_seed = [
                float(rows[(n, method)][f"seed{seed}"]) * 100.0 for seed in SEEDS
            ]
            ax.scatter(
                np.full(len(per_seed), n * 100.0),
                per_seed,
                s=14,
                color=color,
                alpha=0.55,
                zorder=3,
            )
    ax.set(
        xlabel="Injected noise level",
        ylabel="Task-macro success (%)",
        xticks=x,
        xticklabels=["clean", "0.1", "0.4"],
        xlim=(-4, 44),
        ylim=(30, 104),
    )
    ax.set_title(
        "MT10 success across three recovery-training seeds",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )
    ax.grid(axis="y", color=C_GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.text(
        0.5,
        0.005,
        "Bank 20265010 • seeds 42/43/44 • 50 episodes per task per seed",
        ha="center",
        va="bottom",
        fontsize=8,
        color=C_GRAY,
    )
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 1.0))
    for out in (OUT_FIGURE, OUT_FIGURE.with_suffix(".pdf")):
        tmp = out.with_name(f".{out.stem}.tmp{out.suffix}")
        fig.savefig(tmp, dpi=320)
        tmp.replace(out)
    plt.close(fig)
    print(f"wrote {OUT_FIGURE} (+pdf)")


def _tex_table(rows: dict[tuple[float, str], dict[str, str]]) -> None:
    def pct(value: str) -> str:
        return f"{100.0 * float(value):.1f}"

    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{MT10 multi-seed stability of the supervised recovery gate",
        "  (bank 20265010, 50 episodes per task per seed). Rescued/Harmed count",
        "  episodes where the method flips the paired MT-ACT outcome.",
        "  Seed-to-seed variation is small relative to the noise-0.4 gap over",
        "  the heuristic gate.}",
        "  \\label{tab:mt10-seed-stability}",
        "  \\small",
        "  \\setlength{\\tabcolsep}{3.4pt}",
        "  \\begin{tabular}{llcccc}",
        "    \\toprule",
        "    Noise & Method & Seed 42 & Seed 43 & Seed 44 & Mean $\\pm$ s.d. \\\\",
        "    \\midrule",
    ]
    for noise in NOISE_ORDER:
        noise_label = "clean" if noise == 0.0 else f"{noise:.1f}"
        for method in ("reim", "heuristic_recovery"):
            row = rows[(noise, method)]
            label = "MT-REIM" if method == "reim" else "Heuristic gate"
            cells = [
                pct(row[f"seed{seed}"]) + "\\%" for seed in SEEDS
            ]
            mean_sd = (
                f"{pct(row['mean'])} $\\pm$ {pct(row['sample_std'])}\\%"
            )
            lines.append(
                f"    {noise_label} & {label} & "
                + " & ".join(cells)
                + f" & {mean_sd} \\\\"
            )
        if noise != NOISE_ORDER[-1]:
            lines.append("    \\midrule")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    tmp = OUT_TABLE.with_name(f".{OUT_TABLE.stem}.tmp.tex")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(OUT_TABLE)
    print(f"wrote {OUT_TABLE}")


def main() -> None:
    rows = _load()
    _plot(rows)
    _tex_table(rows)
    # Machine-readable record of what was rendered, for provenance.
    record = {
        "source": str(SOURCE_CSV.relative_to(ROOT)).replace("\\", "/"),
        "outputs": [
            str(OUT_FIGURE.relative_to(ROOT)).replace("\\", "/"),
            str(OUT_TABLE.relative_to(ROOT)).replace("\\", "/"),
        ],
    }
    print(json.dumps(record))


if __name__ == "__main__":
    main()
