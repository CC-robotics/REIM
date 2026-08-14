"""MT10-only preview of the multi-task robustness figure.

Reuses the publication pipeline's validation and statistics code
(``validate_suite`` / ``_summary_rows``) so every number is computed by the
same fail-closed logic that will later gate the paper assets. Only the
single-panel plotting is new; colors, markers, and line styles are copied
from ``_plot_robustness``. Writes preview files under ``results/preview/``
and never touches ``paper_assets/`` or the TeX gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from visualization.generate_multitask_paper_assets import (
    CANONICAL_METHODS,
    SHORT_LABELS,
    _summary_rows,
    _write_csv,
    validate_suite,
)

TABLES_DIR = PROJECT_ROOT / "results" / "tables"
AUDIT = PROJECT_ROOT / "results" / "audits" / "mt10_bank_separation.json"
OUT_DIR = PROJECT_ROOT / "results" / "preview"


def main() -> None:
    suite = validate_suite(
        benchmark="MT10",
        tables_dir=TABLES_DIR,
        audit_path=AUDIT,
        n_bootstrap=5_000,
        bootstrap_seed=20260806,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clean_rows, robustness_rows = _summary_rows([suite])
    _write_csv(OUT_DIR / "mt10_clean_statistics_preview.csv", clean_rows)
    _write_csv(OUT_DIR / "mt10_robustness_statistics_preview.csv", robustness_rows)

    colors = {
        "mlp_bc": "#737B85",
        "act": "#2457A6",
        "heuristic_recovery": "#D48A1F",
        "reim": "#17805C",
    }
    markers = {"mlp_bc": "o", "act": "s", "heuristic_recovery": "D", "reim": "^"}
    linestyles = {"mlp_bc": ":", "act": "--", "heuristic_recovery": "-.", "reim": "-"}
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
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    x = np.asarray([item.noise_level * 100.0 for item in suite.robustness])
    for method in CANONICAL_METHODS:
        values = np.asarray(
            [float(item.statistics[method]["success"]) * 100.0 for item in suite.robustness]
        )
        low = np.asarray(
            [float(item.statistics[method]["success_ci_lower"]) * 100.0 for item in suite.robustness]
        )
        high = np.asarray(
            [float(item.statistics[method]["success_ci_upper"]) * 100.0 for item in suite.robustness]
        )
        ax.errorbar(
            x,
            values,
            yerr=np.vstack((values - low, high - values)),
            color=colors[method],
            linestyle=linestyles[method],
            marker=markers[method],
            markersize=4.0,
            linewidth=1.45 if method == "reim" else 1.05,
            capsize=2.0,
            capthick=0.75,
            label=SHORT_LABELS[method],
            zorder=4 if method == "reim" else 3,
        )
    ax.set_title(f"{suite.benchmark} (preview, MT10 only)")
    ax.set_xlabel("Injected noise level (%)")
    ax.set_ylabel("Task-macro success (%)")
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xlim(-2, 42)
    ax.set_ylim(-2, 102)
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.55, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(
        *ax.get_legend_handles_labels(),
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        columnspacing=1.15,
        handlelength=2.3,
    )
    fig.text(
        0.5,
        0.005,
        "REIM robustness extension — task-universal action/observation noise; "
        "not official Meta-World scores",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#3F4650",
    )
    fig.tight_layout(rect=(0.0, 0.075, 1.0, 0.90))
    png = OUT_DIR / "mt10_robustness_preview.png"
    fig.savefig(png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"validated {suite.benchmark}: {len(suite.robustness)} robustness conditions")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
