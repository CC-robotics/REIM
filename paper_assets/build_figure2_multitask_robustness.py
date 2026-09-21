"""Build the MT10/MT50 robustness figure from frozen confirmation artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "paper_assets"
CONFIRM = ROOT / "results" / "tables" / "confirmation_202660xx"
STATS = ASSETS / "multitask_robustness_statistics.csv"
OUT = ASSETS / "Figure2_multitask_robustness"

SUITES = ("MT10", "MT50")
NOISE_LEVELS = (0.0, 0.1, 0.2, 0.3, 0.4)
METHODS = {
    "MT-MLP BC": {
        "label": "MLP BC",
        "color": "#B9C1C8",
        "marker": "o",
        "linestyle": ":",
        "linewidth": 1.15,
    },
    "MT-ACT": {
        "label": "ACT",
        "color": "#91A8BC",
        "marker": "s",
        "linestyle": "--",
        "linewidth": 1.2,
    },
    "MT-ACT + Heuristic-Gated Learned Recovery": {
        "label": "Heuristic + recovery",
        "color": "#4777A7",
        "marker": "^",
        "linestyle": "-.",
        "linewidth": 1.35,
    },
    "MT-REIM": {
        "label": "REIM",
        "color": "#7651A1",
        "marker": "D",
        "linestyle": "-",
        "linewidth": 1.8,
    },
}

INK = "#20252B"
MID = "#6E747B"
GRID = "#D9DDDF"


def load_and_verify() -> pd.DataFrame:
    stats = pd.read_csv(STATS)
    stats = stats.loc[
        stats["benchmark"].isin(SUITES)
        & stats["noise_level"].isin(NOISE_LEVELS)
        & stats["method"].isin(METHODS)
    ].copy()
    expected = len(SUITES) * len(NOISE_LEVELS) * len(METHODS)
    if len(stats) != expected:
        raise RuntimeError(f"Expected {expected} rows, found {len(stats)}.")

    for suite in SUITES:
        for noise in NOISE_LEVELS:
            tag = int(round(noise * 100))
            csv_path = CONFIRM / f"{suite.lower()}_confirm_robustness_noise_{tag:02d}_episodes.csv"
            run_path = Path(f"{csv_path}.run.json")
            frame = pd.read_csv(csv_path)
            protocol = json.loads(run_path.read_text(encoding="utf-8"))["protocol"]

            assert protocol["benchmark"] == suite
            assert np.isclose(protocol["noise_level"], noise)
            assert protocol["action_std_scale"] == 0.4
            assert protocol["observation_std_scale"] == 0.025
            assert protocol["object_position_noise"] is False
            assert protocol["episodes_per_task"] == 50
            expected_n = 50 * len(protocol["task_vocabulary"])

            for method in METHODS:
                raw = frame.loc[frame["method"] == method, "success"].astype(float)
                if len(raw) != expected_n:
                    raise RuntimeError(f"Unexpected n for {suite}, lambda={noise}, {method}.")
                reported = stats.loc[
                    (stats["benchmark"] == suite)
                    & np.isclose(stats["noise_level"], noise)
                    & (stats["method"] == method),
                    "task_macro_success",
                ].iloc[0]
                if not np.isclose(raw.mean(), reported):
                    raise RuntimeError(f"Mean mismatch for {suite}, lambda={noise}, {method}.")
    return stats


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.3,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def main() -> None:
    stats = load_and_verify()
    configure_style()

    fig, axes = plt.subplots(2, 1, figsize=(3.50, 4.25), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.11, top=0.83, hspace=0.30)

    for ax, suite, task_count in zip(axes, SUITES, (10, 50)):
        for method, visual in METHODS.items():
            rows = stats.loc[
                (stats["benchmark"] == suite) & (stats["method"] == method)
            ].sort_values("noise_level")
            x = rows["noise_level"].to_numpy(float)
            y = 100.0 * rows["task_macro_success"].to_numpy(float)
            lower = 100.0 * rows["success_ci_lower"].to_numpy(float)
            upper = 100.0 * rows["success_ci_upper"].to_numpy(float)
            ax.errorbar(
                x,
                y,
                yerr=np.vstack((y - lower, upper - y)),
                color=visual["color"],
                marker=visual["marker"],
                markerfacecolor="white" if method != "MT-REIM" else visual["color"],
                markeredgecolor=visual["color"],
                markeredgewidth=1.0,
                markersize=4.4,
                linestyle=visual["linestyle"],
                linewidth=visual["linewidth"],
                elinewidth=0.7,
                capsize=1.7,
                zorder=4 if method == "MT-REIM" else 3,
                label=visual["label"],
            )

        panel = "(a)" if suite == "MT10" else "(b)"
        ax.set_title(f"{panel}  {suite} ({task_count} tasks)", loc="left", fontweight="bold")
        ax.set_xlim(-0.012, 0.412)
        ax.set_ylim(-3, 103)
        ax.set_xticks(NOISE_LEVELS, ["0", ".10", ".20", ".30", ".40"])
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid(axis="y", color=GRID, linewidth=0.55, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MID)

    axes[0].tick_params(labelbottom=False)
    fig.supxlabel(r"Noise level $\lambda$", y=0.025, fontsize=9.0)
    fig.supylabel("Task-macro success (%)", x=0.025, fontsize=9.0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.0,
        handletextpad=0.45,
    )
    fig.savefig(OUT.with_suffix(".png"), dpi=400, facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    print("Figure 2 generated from confirmation_202660xx.")


if __name__ == "__main__":
    main()
