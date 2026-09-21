"""Build the publication-ready REIM overview (Figure 1, v12).

The vector shell (text, arrows, boxes, and chart) is retained in SVG/PDF.
Simulator frames remain raster images embedded in those outputs.
All numerical results and switching parameters are read from the confirmed
MT10/MT50 experiment artifacts rather than copied by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, PathPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "paper_assets"
RESULTS = ROOT / "results"
CONFIRM = RESULTS / "tables" / "confirmation_202660xx"
FRAMES = RESULTS / "figures" / "recovery_operation_sequence_frames_rounded"
TASK_PREVIEWS = ASSETS / "task_candidate_previews"

OUT_STEM = ASSETS / "Figure1_v12_overview"

COLORS = {
    "ink": "#202020",
    "muted": "#62666B",
    "line": "#B7BAB8",
    "grid": "#E2E3E2",
    "panel": "#FFFFFF",
    "purple": "#7028A8",
    "lavender_fill": "#F1E7F5",
    "blue": "#355E9C",
    "blue_fill": "#DDE9F4",
    "yellow": "#B88B24",
    "yellow_fill": "#FFF0B8",
    "orange": "#C9693E",
    "orange_fill": "#F7DDCE",
    "green": "#527F61",
    "green_fill": "#DDEBD6",
    "red": "#B84A45",
    "red_fill": "#F3DADA",
    "gray_fill": "#F0F1F1",
    "bar_base": "#9AA9B8",
    "bar_reim": "#7651A1",
}


def load_protocol_and_results() -> tuple[dict[str, dict], dict[str, dict[str, float]]]:
    protocols: dict[str, dict] = {}
    values: dict[str, dict[str, float]] = {}
    method_map = {
        "Heuristic": "MT-ACT + Heuristic-Gated Learned Recovery",
        "REIM": "MT-REIM",
    }

    for suite in ("mt10", "mt50"):
        csv_path = CONFIRM / f"{suite}_confirm_robustness_noise_40_episodes.csv"
        run_path = Path(str(csv_path) + ".run.json")
        if not csv_path.exists() or not run_path.exists():
            raise FileNotFoundError(f"Missing confirmed artifact for {suite.upper()}")

        frame = pd.read_csv(csv_path)
        run_record = json.loads(run_path.read_text(encoding="utf-8"))
        protocols[suite] = run_record["protocol"]
        values[suite] = {}
        expected_rows = protocols[suite]["episodes_per_task"] * len(
            protocols[suite]["task_vocabulary"]
        )
        for label, method in method_map.items():
            selected = frame.loc[frame["method"] == method, "success"]
            assert len(selected) == expected_rows
            values[suite][label] = 100.0 * float(selected.mean())

    # Fail loudly if the figure text would contradict the recorded runs.
    assert protocols["mt10"]["detector_threshold"] == 0.65
    assert protocols["mt50"]["detector_threshold"] == 0.64
    for suite in ("mt10", "mt50"):
        cfg = protocols[suite]
        assert cfg["min_recovery_steps"] == 5
        assert cfg["release_threshold"] == 0.05
        assert cfg["release_patience"] == 10
        assert cfg["intervention_cooldown"] == 10
        assert cfg["noise_level"] == 0.4
        assert cfg["action_std_scale"] == 0.4
        assert cfg["observation_std_scale"] == 0.025
        assert cfg["episodes_per_task"] == 50

    # MT10 is a subset of MT50 in the official task vocabulary used here.
    assert "door-open-v3" in protocols["mt10"]["task_vocabulary"]
    assert "door-open-v3" in protocols["mt50"]["task_vocabulary"]
    assert "peg-insert-side-v3" in protocols["mt10"]["task_vocabulary"]
    assert "peg-insert-side-v3" in protocols["mt50"]["task_vocabulary"]
    assert "shelf-place-v3" not in protocols["mt10"]["task_vocabulary"]
    assert "shelf-place-v3" in protocols["mt50"]["task_vocabulary"]
    for key in (
        "noise_level",
        "action_std_scale",
        "observation_std_scale",
        "object_position_noise",
        "episodes_per_task",
        "max_episode_steps",
    ):
        assert protocols["mt10"][key] == protocols["mt50"][key]
    for component in ("act", "detector", "recovery"):
        assert (
            protocols["mt10"]["checkpoint_sha256"][component]
            != protocols["mt50"]["checkpoint_sha256"][component]
        )

    return protocols, values


def rounded_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str = "",
    *,
    fc: str = "white",
    ec: str = COLORS["line"],
    lw: float = 0.8,
    radius: float = 0.008,
    fontsize: float = 7.0,
    color: str = COLORS["ink"],
    weight: str = "normal",
    linestyle: str = "solid",
    zorder: int = 3,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=linestyle,
        transform=ax.transAxes,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if text:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            fontweight=weight,
            linespacing=1.14,
            zorder=zorder + 1,
        )
    return patch


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["muted"],
    lw: float = 0.9,
    style: str = "-|>",
    connectionstyle: str = "arc3",
    zorder: int = 4,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=7.5,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
        transform=ax.transAxes,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def add_image(fig, bounds, path: Path, *, crop_top: float = 0.0):
    if not path.exists():
        raise FileNotFoundError(path)
    image = mpimg.imread(path)
    if crop_top:
        image = image[int(image.shape[0] * crop_top) :, ...]
    image_ax = fig.add_axes(bounds, zorder=2)
    image_ax.imshow(image, interpolation="lanczos", aspect="auto")
    image_ax.set_xticks([])
    image_ax.set_yticks([])
    for spine in image_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_edgecolor(COLORS["line"])
    return image_ax


def rounded_top_bar(ax, x: float, height: float, width: float, color: str) -> PathPatch:
    """Draw a flat-bottom bar with restrained rounded top corners."""
    left = x - width / 2
    right = x + width / 2
    radius_x = width * 0.18
    radius_y = min(2.4, height * 0.12)
    vertices = [
        (left, 0.0),
        (left, height - radius_y),
        (left, height),
        (left + radius_x, height),
        (right - radius_x, height),
        (right, height),
        (right, height - radius_y),
        (right, 0.0),
        (left, 0.0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ]
    patch = PathPatch(
        MplPath(vertices, codes),
        facecolor=color,
        edgecolor="none",
        linewidth=0,
        zorder=3,
    )
    ax.add_patch(patch)
    return patch


def main() -> None:
    protocols, values = load_protocol_and_results()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig = plt.figure(figsize=(7.16, 3.52), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    panels = {
        "a": (0.012, 0.035, 0.268, 0.93),
        "b": (0.292, 0.035, 0.402, 0.93),
        "c": (0.706, 0.035, 0.282, 0.93),
    }
    for x, y, w, h in panels.values():
        rounded_box(ax, x, y, w, h, fc=COLORS["panel"], ec=COLORS["line"],
                    lw=0.8, radius=0.012, zorder=0)

    # ------------------------------------------------------------------
    # (a) PickPlace case study: one disturbance, two outcomes.
    # ------------------------------------------------------------------
    ax.text(0.025, 0.916, "(a) Problem (Motivation)", transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", color=COLORS["purple"], va="center")
    ax.text(0.025, 0.858, "PickPlace example · separate protocol", transform=ax.transAxes,
            fontsize=7.7, color=COLORS["muted"], va="center")

    rounded_box(ax, 0.023, 0.263, 0.246, 0.570, fc="none", ec=COLORS["line"],
                lw=0.75, radius=0.010, linestyle=(0, (4, 3)), zorder=1)

    # A paired close-up camera keeps the gripper, displaced object, and goal
    # visible in all three frames; do not mix viewpoints across the branches.
    common_bounds = [0.027, 0.450, 0.108, 0.220]
    fail_bounds = [0.162, 0.615, 0.103, 0.208]
    recover_bounds = [0.162, 0.320, 0.103, 0.208]
    add_image(
        fig,
        common_bounds,
        FRAMES / "02_act_disturbance_seed8300042_t003.png",
    )
    add_image(
        fig,
        fail_bounds,
        FRAMES / "04_act_failure_seed8300042_t200.png",
    )
    add_image(
        fig,
        recover_bounds,
        FRAMES / "07_reim_transport_seed8300042_t051.png",
    )

    ax.text(0.081, 0.427, "4.8 cm object shift", transform=ax.transAxes, ha="center",
            fontsize=7.4, color=COLORS["muted"])
    ax.text(0.146, 0.776, "ACT", transform=ax.transAxes, ha="center",
            fontsize=7.8, fontweight="bold", color=COLORS["blue"])
    ax.text(0.214, 0.590, "ACT timeout", transform=ax.transAxes, ha="center",
            fontsize=7.5, fontweight="bold", color=COLORS["red"])
    ax.text(0.214, 0.546, "REIM", transform=ax.transAxes, ha="center",
            fontsize=7.8, fontweight="bold", color=COLORS["green"])
    ax.text(0.214, 0.292, "re-grasp + transport", transform=ax.transAxes, ha="center",
            fontsize=7.5, fontweight="bold", color=COLORS["green"])

    arrow(ax, (0.136, 0.620), (0.158, 0.735), color=COLORS["red"], lw=1.0)
    arrow(ax, (0.136, 0.555), (0.158, 0.420), color=COLORS["green"], lw=1.0)

    # Small trajectory motif: a nominal path and post-disturbance drift.
    ax.plot([0.035, 0.083, 0.132, 0.178, 0.246],
            [0.205, 0.205, 0.205, 0.205, 0.205],
            transform=ax.transAxes, color=COLORS["blue"], lw=1.2, zorder=3)
    ax.plot([0.035, 0.083, 0.132, 0.178, 0.246],
            [0.205, 0.205, 0.205, 0.155, 0.130],
            transform=ax.transAxes, color=COLORS["red"], lw=1.1,
            linestyle=(0, (3, 2)), zorder=4)
    ax.plot([0.132, 0.132], [0.175, 0.238], transform=ax.transAxes,
            color=COLORS["red"], lw=0.8, zorder=4)
    ax.text(0.132, 0.244, "shift", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7.2, color=COLORS["red"])
    ax.text(0.025, 0.047, "A displaced object can invalidate\nthe nominal action chunk.",
            transform=ax.transAxes, fontsize=7.8, color=COLORS["ink"],
            ha="left", va="bottom", linespacing=1.25)

    # ------------------------------------------------------------------
    # (b) Runtime framework and training provenance.
    # ------------------------------------------------------------------
    ax.text(0.305, 0.916, "(b) REIM Framework", transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", color=COLORS["purple"], va="center")
    ax.text(0.305, 0.863, "MT10/MT50 arbitration and trigger-aligned supervision",
            transform=ax.transAxes, fontsize=7.6, color=COLORS["muted"], va="center")

    rounded_box(ax, 0.302, 0.358, 0.383, 0.475, fc="none", ec=COLORS["line"],
                lw=0.75, radius=0.010, linestyle=(0, (4, 3)), zorder=1)
    rounded_box(ax, 0.302, 0.061, 0.383, 0.270, fc="none", ec=COLORS["line"],
                lw=0.75, radius=0.010, linestyle=(0, (4, 3)), zorder=1)

    add_image(fig, [0.310, 0.594, 0.068, 0.137],
              FRAMES / "05_reim_trigger_seed8300042_t009.png")
    ax.text(0.344, 0.574, "$x_t=[o_t;e_k]$", transform=ax.transAxes, ha="center", va="center",
            fontsize=7.4, color=COLORS["ink"], fontweight="bold")
    rounded_box(ax, 0.405, 0.714, 0.104, 0.092, "ACT\n$\\pi_{ACT}$",
                fc=COLORS["blue_fill"], ec=COLORS["blue"], fontsize=8.0, weight="bold")
    rounded_box(ax, 0.405, 0.580, 0.104, 0.092, "Risk detector\n$p_t$",
                fc=COLORS["yellow_fill"], ec=COLORS["yellow"], fontsize=7.8, weight="bold")
    rounded_box(ax, 0.405, 0.446, 0.104, 0.092, "Recovery BC\n$\\pi_R$",
                fc=COLORS["orange_fill"], ec=COLORS["orange"], fontsize=7.8, weight="bold")

    # The conditioned state fans out cleanly to all three runtime modules.
    arrow(ax, (0.378, 0.675), (0.402, 0.758), connectionstyle="arc3,rad=-0.15")
    arrow(ax, (0.378, 0.652), (0.402, 0.626))
    arrow(ax, (0.378, 0.625), (0.402, 0.492), connectionstyle="arc3,rad=0.15")

    rounded_box(ax, 0.535, 0.476, 0.118, 0.330, "", fc=COLORS["lavender_fill"],
                ec=COLORS["purple"], lw=0.9, radius=0.010)
    ax.text(0.594, 0.768, "Hysteretic switch", transform=ax.transAxes,
            ha="center", va="center", fontsize=8.0, fontweight="bold", color=COLORS["ink"])
    ax.add_patch(Rectangle((0.551, 0.716), 0.086, 0.028, transform=ax.transAxes,
                           facecolor=COLORS["blue_fill"], edgecolor="none", zorder=4))
    ax.text(0.594, 0.730, "ACT state", transform=ax.transAxes, ha="center", va="center",
            fontsize=7.2, color=COLORS["blue"], fontweight="bold", zorder=5)
    ax.text(0.594, 0.684, "trigger:  $p_t \\geq \\tau_{on}$", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.5, color=COLORS["ink"])
    ax.text(0.594, 0.646, "hold:  ≥ 5 steps", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.5, color=COLORS["ink"])
    ax.text(0.594, 0.608, "release:  $p_t \\leq .05$", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.3, color=COLORS["ink"])
    ax.text(0.594, 0.575, "for 10 steps", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.1, color=COLORS["muted"])
    ax.text(0.594, 0.541, "cooldown: 10 steps", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.5, color=COLORS["ink"])
    ax.add_patch(Rectangle((0.551, 0.499), 0.086, 0.028, transform=ax.transAxes,
                           facecolor=COLORS["orange_fill"], edgecolor="none", zorder=4))
    ax.text(0.594, 0.513, "Recovery state", transform=ax.transAxes, ha="center", va="center",
            fontsize=7.2, color=COLORS["orange"], fontweight="bold", zorder=5)

    arrow(ax, (0.509, 0.760), (0.532, 0.760), color=COLORS["blue"])
    arrow(ax, (0.509, 0.626), (0.532, 0.650), color=COLORS["yellow"])
    arrow(ax, (0.509, 0.492), (0.532, 0.515), color=COLORS["orange"])

    rounded_box(ax, 0.666, 0.594, 0.022, 0.120, "$a_t$", fc=COLORS["green_fill"],
                ec=COLORS["green"], fontsize=8.0, weight="bold")
    arrow(ax, (0.656, 0.654), (0.663, 0.654), color=COLORS["green"])
    # Executed action enters the environment; the next observation returns to
    # the state junction along a separate, unobstructed feedback path.
    rounded_box(ax, 0.580, 0.382, 0.097, 0.055, "Environment",
                fc="#F1F2F2", ec=COLORS["muted"], fontsize=7.2,
                weight="bold", lw=0.7, radius=0.008)
    arrow(ax, (0.677, 0.591), (0.668, 0.440), color=COLORS["green"], lw=0.85)
    ax.plot([0.580, 0.304, 0.304], [0.409, 0.409, 0.652],
            transform=ax.transAxes, color=COLORS["muted"], lw=0.75, zorder=2)
    arrow(ax, (0.304, 0.652), (0.309, 0.652), color=COLORS["muted"], lw=0.75)
    ax.text(0.475, 0.419, "$o_{t+1}$", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=7.2, color=COLORS["muted"])

    ax.text(0.307, 0.308, "Offline recovery training (illustrative pair)", transform=ax.transAxes,
            fontsize=7.8, fontweight="bold", color=COLORS["purple"], va="center")
    add_image(fig, [0.312, 0.118, 0.082, 0.165],
              FRAMES / "05_reim_trigger_seed8300042_t009.png")
    add_image(fig, [0.430, 0.118, 0.082, 0.165],
              FRAMES / "07_reim_transport_seed8300042_t051.png")
    ax.text(0.353, 0.096, "trigger state", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.4, color=COLORS["muted"])
    ax.text(0.471, 0.096, "expert continuation", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.4, color=COLORS["muted"])
    ax.text(0.411, 0.200, "+", transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color=COLORS["muted"])
    arrow(ax, (0.516, 0.200), (0.553, 0.200), color=COLORS["muted"])
    rounded_box(ax, 0.557, 0.135, 0.116, 0.118, "Task-conditioned\nRecovery BC",
                fc=COLORS["orange_fill"], ec=COLORS["orange"], fontsize=7.6, weight="bold")

    # ------------------------------------------------------------------
    # (c) Task breadth and linked MT10/MT50 aggregate evidence.
    # ------------------------------------------------------------------
    ax.text(0.719, 0.916, "(c) Multi-task Results", transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", color=COLORS["purple"], va="center")
    ax.text(0.719, 0.864, "Representative tasks and aggregate robustness", transform=ax.transAxes,
            fontsize=7.6, color=COLORS["muted"], va="center")

    rounded_box(ax, 0.716, 0.598, 0.262, 0.235, fc="none", ec=COLORS["line"],
                lw=0.75, radius=0.010, linestyle=(0, (4, 3)), zorder=1)

    task_specs = [
        ("Door open", "shared", TASK_PREVIEWS / "mt10_door-open-v3_success.png"),
        ("Peg insert", "shared", TASK_PREVIEWS / "mt10_peg-insert-side-v3_interaction.png"),
        ("Shelf place", "MT50 only", TASK_PREVIEWS / "mt50_shelf-place-v3_success.png"),
    ]
    x_positions = [0.719, 0.809, 0.899]
    for x, (name, suite, path) in zip(x_positions, task_specs):
        add_image(fig, [x, 0.665, 0.078, 0.158], path)
        ax.text(x + 0.039, 0.641, name, transform=ax.transAxes, ha="center", va="center",
                fontsize=7.2, color=COLORS["ink"])
        ax.text(x + 0.039, 0.617, suite, transform=ax.transAxes, ha="center", va="center",
                fontsize=7.0, color=COLORS["muted"])

    noise_level = protocols["mt10"]["noise_level"]
    action_std = noise_level * protocols["mt10"]["action_std_scale"]
    observation_std = noise_level * protocols["mt10"]["observation_std_scale"]
    ax.text(0.847, 0.572, f"Suite success at noise level $\\lambda={noise_level:.2f}$",
            transform=ax.transAxes,
            ha="center", va="center", fontsize=7.8, fontweight="bold", color=COLORS["purple"])
    chart_ax = fig.add_axes([0.738, 0.180, 0.228, 0.285], zorder=3)
    suites = ["MT10", "MT50"]
    heuristic = [values["mt10"]["Heuristic"], values["mt50"]["Heuristic"]]
    reim = [values["mt10"]["REIM"], values["mt50"]["REIM"]]
    xpos = np.arange(len(suites))
    width = 0.26
    offset = 0.16
    bar_records = []
    for index, suite_x in enumerate(xpos):
        rounded_top_bar(chart_ax, suite_x - offset, heuristic[index], width, COLORS["bar_base"])
        rounded_top_bar(chart_ax, suite_x + offset, reim[index], width, COLORS["bar_reim"])
        bar_records.extend(
            [
                (suite_x - offset, heuristic[index], COLORS["bar_base"]),
                (suite_x + offset, reim[index], COLORS["bar_reim"]),
            ]
        )
        gain = reim[index] - heuristic[index]
        chart_ax.text(
            suite_x,
            reim[index] + 9.0,
            f"+{gain:.1f} pp",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=COLORS["purple"],
            fontweight="bold",
        )
    chart_ax.set_ylim(0, 78)
    chart_ax.set_xlim(-0.55, 1.55)
    chart_ax.set_yticks([0, 20, 40, 60])
    chart_ax.set_ylabel("Task-macro success (%)", labelpad=1)
    chart_ax.set_xticks(xpos, suites)
    chart_ax.grid(axis="y", color=COLORS["grid"], lw=0.55, zorder=0)
    chart_ax.spines[["top", "right"]].set_visible(False)
    chart_ax.spines[["left", "bottom"]].set_color(COLORS["line"])
    chart_ax.tick_params(axis="both", length=2.2, color=COLORS["line"], pad=1.5)
    chart_ax.legend(
        handles=[
            Patch(facecolor=COLORS["bar_base"], edgecolor="none", label="Heuristic gate"),
            Patch(facecolor=COLORS["bar_reim"], edgecolor="none", label="REIM"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        frameon=False,
        fontsize=7.2,
        handlelength=1.5,
        columnspacing=1.0,
    )
    for bar_x, height, color in bar_records:
        chart_ax.text(
            bar_x,
            height + 1.5,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=color,
            fontweight="bold",
        )

    ax.text(0.847, 0.112, f"$\\sigma_a={action_std:.2f}$, $\\sigma_o={observation_std:.2f}$ · 50 episodes/task",
            transform=ax.transAxes, ha="center", va="center", fontsize=7.2, color=COLORS["ink"])

    # Output in a vector-first set plus a high-resolution preview.
    fig.savefig(OUT_STEM.with_suffix(".pdf"), bbox_inches=None, facecolor="white")
    fig.savefig(OUT_STEM.with_suffix(".svg"), bbox_inches=None, facecolor="white")
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=450, bbox_inches=None, facecolor="white")
    plt.close(fig)

    print("Figure 1 v12 written to:")
    for suffix in (".png", ".pdf", ".svg"):
        print(f"  {OUT_STEM.with_suffix(suffix)}")
    print("Confirmed noise-level 0.40 aggregate success:")
    for suite in ("mt10", "mt50"):
        print(
            f"  {suite.upper()}: Heuristic={values[suite]['Heuristic']:.2f}%, "
            f"REIM={values[suite]['REIM']:.2f}%"
        )


if __name__ == "__main__":
    main()
