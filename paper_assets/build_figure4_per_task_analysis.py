"""Build per-task MT10/MT50 REIM-versus-heuristic comparison at lambda=0.40."""

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
OUT = ASSETS / "Figure4_per_task_analysis"

HEURISTIC = "MT-ACT + Heuristic-Gated Learned Recovery"
REIM = "MT-REIM"
INK = "#20252B"
MID = "#6E747B"
GRID = "#D9DDDF"
MT10_COLOR = "#7651A1"
MT50_COLOR = "#4777A7"
BETTER = "#527F61"
WORSE = "#C9693E"

DISPLAY_NAMES = {
    "button-press-topdown-v3": "button top",
    "coffee-pull-v3": "coffee pull",
    "coffee-push-v3": "coffee push",
    "door-open-v3": "door open",
    "drawer-close-v3": "drawer close",
    "pick-place-v3": "pick-place",
    "push-v3": "push",
    "stick-push-v3": "stick push",
    "window-close-v3": "window close",
}


def load_suite(suite: str) -> pd.DataFrame:
    csv_path = CONFIRM / f"{suite.lower()}_confirm_robustness_noise_40_episodes.csv"
    run_path = Path(f"{csv_path}.run.json")
    protocol = json.loads(run_path.read_text(encoding="utf-8"))["protocol"]
    assert protocol["benchmark"] == suite
    assert protocol["noise_level"] == 0.4
    assert protocol["episodes_per_task"] == 50
    assert protocol["object_position_noise"] is False

    frame = pd.read_csv(csv_path)
    frame = frame.loc[frame["method"].isin((HEURISTIC, REIM))].copy()
    counts = frame.groupby(["task_name", "method"]).size()
    if not (counts == 50).all():
        raise RuntimeError(f"Every {suite} task/method must contain 50 episodes.")

    table = (
        frame.groupby(["task_name", "method"], as_index=False)["success"]
        .mean()
        .pivot(index="task_name", columns="method", values="success")
        .reset_index()
    )
    table["heuristic"] = 100.0 * table[HEURISTIC]
    table["reim"] = 100.0 * table[REIM]
    table["delta"] = table["reim"] - table["heuristic"]
    expected_tasks = 10 if suite == "MT10" else 50
    if len(table) != expected_tasks:
        raise RuntimeError(f"Expected {expected_tasks} tasks for {suite}.")
    return table


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
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def annotate_at(
    ax: plt.Axes,
    row: pd.Series,
    text_x: float,
    text_y: float,
    ha: str = "left",
) -> None:
    name = DISPLAY_NAMES.get(str(row["task_name"]), str(row["task_name"]).replace("-v3", ""))
    ax.annotate(
        name,
        (float(row["heuristic"]), float(row["reim"])),
        xytext=(text_x, text_y),
        textcoords="data",
        ha=ha,
        va="center",
        fontsize=7.6,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": MID, "lw": 0.45, "shrinkA": 1.5, "shrinkB": 2.0},
    )


def main() -> None:
    mt10 = load_suite("MT10")
    mt50 = load_suite("MT50")
    shared = set(mt10["task_name"])
    configure_style()

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.05), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.16, top=0.80, wspace=0.22)

    # Light regions communicate direction while the 1:1 line remains decisive.
    x = np.linspace(0, 100, 201)
    for ax in axes:
        ax.fill_between(x, x, 100, color=BETTER, alpha=0.055, zorder=0)
        ax.fill_between(x, 0, x, color=WORSE, alpha=0.045, zorder=0)
        ax.plot([0, 100], [0, 100], color=MID, lw=0.8, ls="--", zorder=1)
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid(color=GRID, lw=0.45, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MID)

    axes[0].scatter(
        mt10["heuristic"],
        mt10["reim"],
        s=33,
        marker="s",
        facecolor="white",
        edgecolor=MT10_COLOR,
        linewidth=1.25,
        zorder=3,
    )
    axes[0].set_title("(a)  MT10 tasks", loc="left", fontweight="bold")

    mt50_shared = mt50["task_name"].isin(shared)
    axes[1].scatter(
        mt50.loc[~mt50_shared, "heuristic"],
        mt50.loc[~mt50_shared, "reim"],
        s=27,
        marker="o",
        facecolor=MT50_COLOR,
        edgecolor=MT50_COLOR,
        alpha=0.72,
        linewidth=0.6,
        label="MT50-specific",
        zorder=3,
    )
    axes[1].scatter(
        mt50.loc[mt50_shared, "heuristic"],
        mt50.loc[mt50_shared, "reim"],
        s=35,
        marker="o",
        facecolor="white",
        edgecolor=MT50_COLOR,
        linewidth=1.25,
        label="Shared with MT10",
        zorder=4,
    )
    axes[1].set_title("(b)  MT50 tasks", loc="left", fontweight="bold")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.745, 0.945),
        ncol=2,
        frameon=False,
        fontsize=8.0,
        handletextpad=0.3,
        columnspacing=0.9,
    )

    # Labels identify the clearest negative and positive task-level effects.
    label_positions_10 = {
        "drawer-close-v3": (92, 92, "right"),
        "push-v3": (18, 8, "left"),
        "pick-place-v3": (3, 13, "left"),
        "door-open-v3": (14, 84, "left"),
    }
    for name, (tx, ty, ha) in label_positions_10.items():
        annotate_at(axes[0], mt10.loc[mt10["task_name"] == name].iloc[0], tx, ty, ha)

    label_positions_50 = {
        "stick-push-v3": (58, 2, "left"),
        "coffee-pull-v3": (27, 15, "left"),
        "door-open-v3": (19, 84, "left"),
        "button-press-topdown-v3": (38, 98, "left"),
    }
    for name, (tx, ty, ha) in label_positions_50.items():
        annotate_at(axes[1], mt50.loc[mt50["task_name"] == name].iloc[0], tx, ty, ha)

    fig.supxlabel("Heuristic-gated recovery success (%)", y=0.055, fontsize=9.0)
    fig.supylabel("REIM success (%)", x=0.025, fontsize=9.0)

    fig.savefig(OUT.with_suffix(".png"), dpi=400, facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    print("Figure 4 generated from confirmation_202660xx.")
    print("MT10 improved/tied/worse:", int((mt10.delta > 0).sum()), int((mt10.delta == 0).sum()), int((mt10.delta < 0).sum()))
    print("MT50 improved/tied/worse:", int((mt50.delta > 0).sum()), int((mt50.delta == 0).sum()), int((mt50.delta < 0).sum()))


if __name__ == "__main__":
    main()
