"""Build the single-column paired-outcome Figure 3.

The aggregate success comparison already appears in Figure 1(c). This figure
therefore focuses on the non-redundant paired rescued/harmed decomposition.
All values are recomputed from the frozen MT10/MT50 confirmation banks.
"""

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
OUT = ASSETS / "Figure3_paired_effects"

SUITES = ("MT10", "MT50")
METHODS = (
    ("MT-MLP BC", "MLP BC"),
    ("MT-ACT", "ACT"),
    ("MT-ACT + Heuristic-Gated Learned Recovery", "Heuristic + recovery"),
    ("MT-REIM", "REIM"),
)

COLORS = {
    "ink": "#20252B",
    "mid": "#6E747B",
    "grid": "#D9DDDF",
    "mt10": "#7651A1",
    "mt50": "#4777A7",
    "rescued": "#6F93B4",
    "harmed": "#C6CCD1",
}


def load_and_verify() -> tuple[pd.DataFrame, dict[str, dict[str, float | int]]]:
    stats = pd.read_csv(STATS)
    stats = stats.loc[
        (stats["condition"] == "robustness_noise_40")
        & (np.isclose(stats["noise_level"], 0.4))
        & (stats["benchmark"].isin(SUITES))
        & (stats["method"].isin([name for name, _ in METHODS]))
    ].copy()
    if len(stats) != len(SUITES) * len(METHODS):
        raise RuntimeError("Expected four methods for each of MT10 and MT50.")

    paired: dict[str, dict[str, float | int]] = {}
    for suite in SUITES:
        lower = suite.lower()
        csv_path = CONFIRM / f"{lower}_confirm_robustness_noise_40_episodes.csv"
        run_path = Path(f"{csv_path}.run.json")
        frame = pd.read_csv(csv_path)
        protocol = json.loads(run_path.read_text(encoding="utf-8"))["protocol"]

        assert protocol["benchmark"] == suite
        assert protocol["noise_level"] == 0.4
        assert protocol["action_std_scale"] == 0.4
        assert protocol["observation_std_scale"] == 0.025
        assert protocol["object_position_noise"] is False
        assert protocol["episodes_per_task"] == 50
        expected_n = protocol["episodes_per_task"] * len(protocol["task_vocabulary"])

        for method, _ in METHODS:
            raw = frame.loc[frame["method"] == method, "success"].astype(float)
            if len(raw) != expected_n:
                raise RuntimeError(f"Unexpected row count for {suite} / {method}.")
            reported = stats.loc[
                (stats["benchmark"] == suite) & (stats["method"] == method),
                "task_macro_success",
            ].iloc[0]
            if not np.isclose(raw.mean(), reported):
                raise RuntimeError(f"Mean mismatch for {suite} / {method}.")

        wide = frame.loc[
            frame["method"].isin(
                ["MT-ACT + Heuristic-Gated Learned Recovery", "MT-REIM"]
            ),
            ["paired_episode_id", "method", "success"],
        ].pivot(index="paired_episode_id", columns="method", values="success")
        if len(wide) != expected_n or wide.isna().any().any():
            raise RuntimeError(f"Incomplete paired outcomes for {suite}.")

        heuristic = wide["MT-ACT + Heuristic-Gated Learned Recovery"].astype(int)
        reim = wide["MT-REIM"].astype(int)
        rescued = int(((reim == 1) & (heuristic == 0)).sum())
        harmed = int(((reim == 0) & (heuristic == 1)).sum())
        paired[suite] = {
            "n": expected_n,
            "rescued": rescued,
            "harmed": harmed,
            "rescued_pct": 100.0 * rescued / expected_n,
            "harmed_pct": 100.0 * harmed / expected_n,
            "net_pp": 100.0 * (rescued - harmed) / expected_n,
        }

    return stats, paired


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def main() -> None:
    stats, paired = load_and_verify()
    style()

    fig = plt.figure(figsize=(3.50, 3.50), facecolor="white")

    # Panel (a): formal success comparison with uncertainty. Figure 1(c)
    # remains a deliberately simplified headline view with only two methods.
    ax = fig.add_axes([0.34, 0.57, 0.63, 0.31])
    y_base = np.arange(len(METHODS))[::-1]
    offsets = {"MT10": 0.17, "MT50": -0.17}
    suite_style = {
        "MT10": (COLORS["mt10"], "o"),
        "MT50": (COLORS["mt50"], "D"),
    }
    for suite in SUITES:
        color, marker = suite_style[suite]
        xs: list[float] = []
        xerr_lo: list[float] = []
        xerr_hi: list[float] = []
        ys: list[float] = []
        for y_value, (method, _) in zip(y_base, METHODS):
            row = stats.loc[
                (stats["benchmark"] == suite) & (stats["method"] == method)
            ].iloc[0]
            value = 100.0 * float(row["task_macro_success"])
            lower = 100.0 * float(row["success_ci_lower"])
            upper = 100.0 * float(row["success_ci_upper"])
            xs.append(value)
            xerr_lo.append(value - lower)
            xerr_hi.append(upper - value)
            ys.append(y_value + offsets[suite])
        ax.errorbar(
            xs,
            ys,
            xerr=[xerr_lo, xerr_hi],
            fmt=marker,
            markersize=4.6,
            markerfacecolor="white" if suite == "MT10" else color,
            markeredgecolor=color,
            markeredgewidth=1.2,
            ecolor=color,
            elinewidth=1.0,
            capsize=1.8,
            linestyle="none",
            label=suite,
            zorder=3,
        )
        for x_value, y_value, upper_error in zip(xs, ys, xerr_hi):
            label_x = min(x_value + upper_error + 1.4, 68.4)
            ax.text(
                label_x,
                y_value,
                f"{x_value:.1f}",
                va="center",
                ha="left",
                color=color,
                fontsize=7.8,
            )

    ax.set_yticks(y_base, [label for _, label in METHODS])
    ax.set_xlim(0, 70)
    ax.set_xticks(np.arange(0, 71, 10))
    ax.set_xlabel("Task-macro success (%)")
    ax.set_title(r"(a) Success at $\lambda=0.40$", loc="left", fontweight="bold")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", colors=COLORS["mid"])
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.01, 1.03),
        ncol=2,
        frameon=False,
        borderpad=0.1,
        handletextpad=0.3,
        columnspacing=0.7,
        fontsize=8.0,
    )

    # Panel (b): paired outcomes normalized because suite sizes differ.
    ax = fig.add_axes([0.25, 0.11, 0.72, 0.25])
    y = np.array([1.0, 0.0])
    rescued = np.array([float(paired[s]["rescued_pct"]) for s in SUITES])
    harmed = -np.array([float(paired[s]["harmed_pct"]) for s in SUITES])
    ax.barh(
        y,
        harmed,
        height=0.34,
        color=COLORS["harmed"],
        edgecolor=COLORS["mid"],
        linewidth=0.55,
        hatch="///",
        label="Harmed",
        zorder=2,
    )
    ax.barh(
        y,
        rescued,
        height=0.34,
        color=COLORS["rescued"],
        edgecolor=COLORS["mt50"],
        linewidth=0.55,
        label="Rescued",
        zorder=2,
    )
    ax.axvline(0, color=COLORS["ink"], linewidth=0.75)

    for yy, suite, hp, rp in zip(y, SUITES, -harmed, rescued):
        info = paired[suite]
        ax.text(-hp * 0.5, yy, f"{hp:.1f}%", ha="center", va="center", fontsize=8.3, fontweight="bold", color=COLORS["ink"])
        ax.text(rp - 0.65, yy, f"{rp:.1f}%", ha="right", va="center", fontsize=8.3, fontweight="bold", color="white")
        ax.text(rp * 0.52, yy + 0.235, f"net +{info['net_pp']:.1f} pp", ha="center", va="center", fontsize=8.3, fontweight="bold", color=COLORS["mt50"])

    ax.set_yticks(y, [f"MT10\n(n={paired['MT10']['n']})", f"MT50\n(n={paired['MT50']['n']})"])
    ax.set_xlim(-13, 27)
    ax.set_xticks([-10, 0, 10, 20])
    ax.set_xticklabels(["10", "0", "10", "20"])
    ax.set_xlabel("Harmed  ←  paired episodes (%)  →  Rescued")
    ax.set_title("(b) Paired changes vs. heuristic", loc="left", fontweight="bold")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.tick_params(axis="x", colors=COLORS["mid"])

    fig.savefig(OUT.with_suffix(".png"), dpi=400, facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
    plt.close(fig)

    print("Figure 3 generated from confirmation_202660xx.")
    for suite in SUITES:
        print(suite, paired[suite])


if __name__ == "__main__":
    main()
