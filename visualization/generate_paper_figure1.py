#!/usr/bin/env python3
"""
Generate a top-tier conference/journal framework figure for REIM.
Publication-quality layout, modern typography, embedded simulation visuals,
modular neural architectures, and clear closed-loop flow.
"""

from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch
import matplotlib.path as mpath
from PIL import Image
import numpy as np

def create_top_tier_framework_figure(
    output_png: Path,
    output_pdf: Path | None = None,
    dpi: int = 300,
):
    # Set up publication-grade styling
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["axes.unicode_minus"] = False

    # Figure dimensions: standard double-column top figure (16.2 x 9.2 inches)
    fig, ax = plt.subplots(figsize=(16.2, 9.2), dpi=dpi)
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 9.2)
    ax.axis("off")

    # -------------------------------------------------------------
    # COLOR PALETTE (Modern Top-Tier Conference Aesthetic)
    # -------------------------------------------------------------
    C_BG_PAGE = "#FFFFFF"
    C_SECTION_BG_A = "#F8FAFC"       # Slate ultra-light
    C_SECTION_BG_B = "#F8FAFC"       # Slate ultra-light
    C_SECTION_BORDER = "#CBD5E1"     # Slate border
    
    # Module Themes
    C_BLUE_CARD = "#F0F7FF"
    C_BLUE_BORDER = "#2563EB"
    C_BLUE_HEADER = "#1E40AF"
    C_BLUE_TEXT = "#1E3A8A"
    
    C_AMBER_CARD = "#FFFBEB"
    C_AMBER_BORDER = "#D97706"
    C_AMBER_HEADER = "#B45309"
    C_AMBER_TEXT = "#92400E"

    C_GREEN_CARD = "#ECFDF5"
    C_GREEN_BORDER = "#059669"
    C_GREEN_HEADER = "#047857"
    C_GREEN_TEXT = "#065F46"

    C_PURPLE_CARD = "#F5F3FF"
    C_PURPLE_BORDER = "#7C3AED"
    C_PURPLE_HEADER = "#6D28D9"
    C_PURPLE_TEXT = "#4C1D95"

    C_SLATE_CARD = "#F8FAFC"
    C_SLATE_BORDER = "#475569"
    C_SLATE_HEADER = "#1E293B"
    C_SLATE_TEXT = "#0F172A"

    def draw_card(
        x, y, w, h,
        title, badge_text="",
        bg_color="#FFFFFF", border_color="#64748B",
        header_color="#334155", title_color="#FFFFFF",
        badge_bg="#3B82F6", badge_fg="#FFFFFF",
        radius=0.10, linewidth=1.3, shadow=True,
        header_h=0.40
    ):
        if shadow:
            shadow_patch = FancyBboxPatch(
                (x + 0.035, y - 0.035), w, h,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                facecolor="#000000", edgecolor="none",
                alpha=0.05, zorder=2
            )
            ax.add_patch(shadow_patch)

        card_patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=bg_color, edgecolor=border_color,
            linewidth=linewidth, zorder=3
        )
        ax.add_patch(card_patch)

        # Header Bar
        header_patch = FancyBboxPatch(
            (x, y + h - header_h), w, header_h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=header_color, edgecolor=border_color,
            linewidth=linewidth, zorder=4
        )
        ax.add_patch(header_patch)

        # Flat lower half of header
        header_rect = patches.Rectangle(
            (x, y + h - header_h), w, header_h / 2,
            facecolor=header_color, edgecolor=header_color,
            linewidth=0, zorder=4
        )
        ax.add_patch(header_rect)

        # Badge calculation
        badge_w = 0
        if badge_text:
            badge_w = len(badge_text) * 0.062 + 0.20
            badge_x = x + w - badge_w - 0.08
            badge_y = y + h - header_h + 0.08
            badge_bh = header_h - 0.16
            b_patch = FancyBboxPatch(
                (badge_x, badge_y), badge_w, badge_bh,
                boxstyle="round,pad=0,rounding_size=0.05",
                facecolor=badge_bg, edgecolor="none",
                zorder=5
            )
            ax.add_patch(b_patch)
            ax.text(
                badge_x + badge_w / 2, badge_y + badge_bh / 2,
                badge_text,
                fontsize=6.0, weight="bold", color=badge_fg,
                va="center", ha="center", zorder=6
            )

        ax.text(
            x + 0.10, y + h - header_h / 2,
            title,
            fontsize=7.6, weight="bold", color=title_color,
            va="center", ha="left", zorder=5
        )

        return card_patch

    def draw_arrow(
        start, end,
        color="#3B82F6", width=1.5,
        style="-|>", rad=0.0,
        linestyle="-", label="",
        label_pos=(0.5, 0.5), label_color=None,
        label_bg=None, fontsize=6.8
    ):
        conn = f"arc3,rad={rad}" if rad != 0 else "arc3,rad=0"
        arrow = FancyArrowPatch(
            start, end,
            arrowstyle=style,
            mutation_scale=11,
            linewidth=width,
            color=color,
            linestyle=linestyle,
            connectionstyle=conn,
            zorder=8
        )
        ax.add_patch(arrow)

        if label:
            lx = start[0] + (end[0] - start[0]) * label_pos[0]
            ly = start[1] + (end[1] - start[1]) * label_pos[1]
            if rad != 0:
                ly += rad * 0.35
            
            tb = dict(boxstyle="round,pad=0.2", facecolor=label_bg if label_bg else "#FFFFFF",
                      edgecolor=color, linewidth=0.75, alpha=0.95) if label_bg else None
            ax.text(
                lx, ly, label,
                fontsize=fontsize, color=label_color if label_color else color,
                weight="bold", ha="center", va="center",
                bbox=tb, zorder=9
            )
        return arrow

    # Load simulation thumbnail images
    img_dir = Path("results/figures/recovery_operation_sequence_frames")
    img_disturb = None
    img_trigger = None
    img_expert = None
    img_success = None
    img_env = None

    if img_dir.exists():
        p_disturb = img_dir / "02_act_disturbance_seed8300042_t003.png"
        p_trigger = img_dir / "05_reim_trigger_seed8300042_t009.png"
        p_expert = img_dir / "06_reim_relift_seed8300042_t040.png"
        p_success = img_dir / "08_reim_success_seed8300042_t062.png"
        p_env = img_dir / "01_act_initial_seed8300042_t000.png"

        if p_disturb.exists(): img_disturb = Image.open(p_disturb)
        if p_trigger.exists(): img_trigger = Image.open(p_trigger)
        if p_expert.exists(): img_expert = Image.open(p_expert)
        if p_success.exists(): img_success = Image.open(p_success)
        if p_env.exists(): img_env = Image.open(p_env)

    # -------------------------------------------------------------
    # 0. MAIN TITLE / BANNER
    # -------------------------------------------------------------
    title_box = FancyBboxPatch(
        (0.35, 8.62), 15.5, 0.44,
        boxstyle="round,pad=0,rounding_size=0.08",
        facecolor="#0F172A", edgecolor="none", zorder=3
    )
    ax.add_patch(title_box)
    ax.text(
        0.55, 8.84,
        "REIM: Recovery-Enhanced Imitation Learning for Robust Embodied Manipulation",
        fontsize=10.0, weight="bold", color="#FFFFFF", va="center", ha="left", zorder=5
    )
    ax.text(
        15.65, 8.84,
        "System Architecture & Execution Pipeline",
        fontsize=8.0, weight="bold", color="#94A3B8", va="center", ha="right", zorder=5
    )

    # -------------------------------------------------------------
    # SECTION 1: (a) TRAINING PHASE (Trigger-Aligned Curriculum)
    # -------------------------------------------------------------
    sec_a = FancyBboxPatch(
        (0.35, 5.10), 15.5, 3.38,
        boxstyle="round,pad=0,rounding_size=0.12",
        facecolor=C_SECTION_BG_A, edgecolor=C_SECTION_BORDER,
        linewidth=1.1, zorder=1
    )
    ax.add_patch(sec_a)

    sec_a_pill = FancyBboxPatch(
        (0.55, 8.16), 5.4, 0.26,
        boxstyle="round,pad=0,rounding_size=0.05",
        facecolor="#334155", edgecolor="none", zorder=4
    )
    ax.add_patch(sec_a_pill)
    ax.text(
        0.65, 8.29,
        "PHASE 1 : TRIGGER-ALIGNED RECOVERY CURRICULUM (TRAINING STAGE)",
        fontsize=6.6, weight="bold", color="#FFFFFF", va="center", ha="left", zorder=5
    )

    # Card 1.1: Disturbed ACT Rollout
    draw_card(
        0.55, 5.25, 2.65, 2.80,
        "1. Disturbed Rollout",
        badge_text="Meta-World",
        bg_color=C_BLUE_CARD, border_color=C_BLUE_BORDER,
        header_color=C_BLUE_HEADER, badge_bg="#3B82F6"
    )
    ax.text(
        1.875, 7.35,
        "Nominal ACT Rollout\nwith Stochastic Perturbations",
        fontsize=6.7, weight="bold", color=C_BLUE_TEXT, ha="center", va="center", zorder=5
    )
    ax.text(
        1.875, 6.88,
        "• Action noise $\\sigma_a \\in [0, 0.16]$\n• Obs. jitter $\\sigma_o \\in [0, 0.01]$\n• Object displacement $\\Delta p$",
        fontsize=6.0, color="#334155", ha="center", va="center", zorder=5
    )
    if img_disturb:
        ax.imshow(img_disturb, extent=[1.05, 2.70, 5.40, 6.50], zorder=6)
        thumb_border = patches.Rectangle(
            (1.05, 5.40), 1.65, 1.10,
            linewidth=0.8, edgecolor=C_BLUE_BORDER, facecolor="none", zorder=7
        )
        ax.add_patch(thumb_border)
        ax.text(
            1.875, 5.32, "Perturbed state $s_t$",
            fontsize=5.6, color="#64748B", weight="bold", ha="center", zorder=7
        )

    draw_arrow((3.20, 6.65), (3.65, 6.65), color=C_BLUE_BORDER, width=1.5)

    # Card 1.2: Causal LSTM Risk Trigger
    draw_card(
        3.65, 5.25, 2.75, 2.80,
        "2. Risk Monitor",
        badge_text="Causal LSTM",
        bg_color=C_AMBER_CARD, border_color=C_AMBER_BORDER,
        header_color=C_AMBER_HEADER, badge_bg="#D97706"
    )
    ax.text(
        5.025, 7.35,
        "Recurrent Failure-Risk Monitor\n$p_t = P(\\text{fail} \\in [t, t+10] \\mid s_{t-9:t})$",
        fontsize=6.5, weight="bold", color=C_AMBER_TEXT, ha="center", va="center", zorder=5
    )
    trig_box = FancyBboxPatch(
        (3.85, 6.35), 2.35, 0.60,
        boxstyle="round,pad=0,rounding_size=0.06",
        facecolor="#FEF3C7", edgecolor=C_AMBER_BORDER,
        linewidth=0.9, zorder=5
    )
    ax.add_patch(trig_box)
    ax.text(
        5.025, 6.72, "Collection Threshold Gate",
        fontsize=6.2, weight="bold", color=C_AMBER_HEADER, ha="center", zorder=6
    )
    ax.text(
        5.025, 6.50, "$p_t \\geq \\tau_{\\mathrm{collect}} = 0.10$",
        fontsize=7.4, weight="bold", color="#B45309", ha="center", zorder=6
    )
    ax.text(
        5.025, 5.85,
        "• Pre-grasp risk triggers\n• 98.9% triggered before object lift\n• Identifies recoverable boundary",
        fontsize=6.1, color="#475569", ha="center", va="center", zorder=5
    )
    ax.text(
        5.025, 5.36,
        "[Disjoint seed offsets: +3M / +4M]",
        fontsize=5.6, color="#94A3B8", weight="bold", ha="center", zorder=5
    )

    draw_arrow((6.40, 6.65), (6.85, 6.65), color=C_AMBER_BORDER, width=1.5,
               label="Trigger", label_pos=(0.5, 0.5), label_color=C_AMBER_HEADER, label_bg="#FFFBEB", fontsize=6.5)

    # Card 1.3: Exact MuJoCo State Serialization
    draw_card(
        6.85, 5.25, 2.75, 2.80,
        "3. Physics Snapshot",
        badge_text="MuJoCo State",
        bg_color=C_PURPLE_CARD, border_color=C_PURPLE_BORDER,
        header_color=C_PURPLE_HEADER, badge_bg="#7C3AED"
    )
    ax.text(
        8.225, 7.35,
        "Exact Simulator State Capture\n$\\mathcal{S}_{\\mathrm{trig}} = (q, \\dot{q}, \\text{mocap}, \\text{goal})$",
        fontsize=6.5, weight="bold", color=C_PURPLE_TEXT, ha="center", va="center", zorder=5
    )
    snap_box = FancyBboxPatch(
        (7.05, 6.00), 2.35, 0.95,
        boxstyle="round,pad=0,rounding_size=0.06",
        facecolor="#EDE9FE", edgecolor=C_PURPLE_BORDER,
        linewidth=0.8, zorder=5
    )
    ax.add_patch(snap_box)
    ax.text(
        8.225, 6.72, "Serialized Physics State:",
        fontsize=6.2, weight="bold", color=C_PURPLE_HEADER, ha="center", zorder=6
    )
    ax.text(
        8.225, 6.38,
        "• Joint Pos / Vel ($q, \\dot{q}$)\n• Mocap target position\n• Gripper dynamics & Goal pos",
        fontsize=5.8, color="#4C1D95", ha="center", va="center", zorder=6
    )
    ax.text(
        8.225, 5.55,
        "Eliminates hand-crafted reset bias\nand train-to-deploy distribution gap",
        fontsize=6.0, color="#475569", ha="center", va="center", zorder=5
    )

    draw_arrow((9.60, 6.65), (10.05, 6.65), color=C_PURPLE_BORDER, width=1.5)

    # Card 1.4: Scripted Continuation & Disjoint Bank
    draw_card(
        10.05, 5.25, 2.80, 2.80,
        "4. Expert Rollout",
        badge_text="Scripted Expert",
        bg_color=C_GREEN_CARD, border_color=C_GREEN_BORDER,
        header_color=C_GREEN_HEADER, badge_bg="#059669"
    )
    ax.text(
        11.45, 7.35,
        "Corrective Demonstrations\nfrom Trigger Snapshot $\\mathcal{S}_{\\mathrm{trig}}$",
        fontsize=6.5, weight="bold", color=C_GREEN_TEXT, ha="center", va="center", zorder=5
    )
    ds_box = FancyBboxPatch(
        (10.25, 6.05), 2.40, 0.95,
        boxstyle="round,pad=0,rounding_size=0.06",
        facecolor="#D1FAE5", edgecolor=C_GREEN_BORDER,
        linewidth=0.8, zorder=5
    )
    ax.add_patch(ds_box)
    ax.text(
        11.45, 6.78, "Successful Corrective Buffer:",
        fontsize=6.2, weight="bold", color=C_GREEN_HEADER, ha="center", zorder=6
    )
    ax.text(
        11.45, 6.42,
        "• Train: 42,386 pairs\n• Val: 8,212 pairs (disjoint)",
        fontsize=6.2, weight="bold", color="#065F46", ha="center", va="center", zorder=6
    )
    ax.text(
        11.45, 5.55,
        "• Discard failed continuations\n• Zero RL environment rollouts\n• Standalone supervised buffer",
        fontsize=5.9, color="#334155", ha="center", va="center", zorder=5
    )

    draw_arrow((12.85, 6.65), (13.30, 6.65), color=C_GREEN_BORDER, width=1.5)

    # Card 1.5: Supervised Policy Optimization
    draw_card(
        13.30, 5.25, 2.40, 2.80,
        "5. Recovery Policy",
        badge_text="MLP Actor",
        bg_color=C_GREEN_CARD, border_color=C_GREEN_BORDER,
        header_color=C_GREEN_HEADER, badge_bg="#047857"
    )
    ax.text(
        14.50, 7.35,
        "Deterministic Actor $\\pi_{\\mathrm{rec}}$\n$21 \\to 256 \\to 256 \\to 4$",
        fontsize=6.5, weight="bold", color=C_GREEN_TEXT, ha="center", va="center", zorder=5
    )
    loss_box = FancyBboxPatch(
        (13.48, 6.20), 2.04, 0.75,
        boxstyle="round,pad=0,rounding_size=0.06",
        facecolor="#D1FAE5", edgecolor=C_GREEN_BORDER,
        linewidth=0.8, zorder=5
    )
    ax.add_patch(loss_box)
    ax.text(
        14.50, 6.72, "Smooth L1 Loss",
        fontsize=6.2, weight="bold", color=C_GREEN_HEADER, ha="center", zorder=6
    )
    ax.text(
        14.50, 6.45, "$\\mathcal{L} = \\text{SmoothL1}(a_t, \\pi_{\\mathrm{rec}}(s_t))$",
        fontsize=5.7, weight="bold", color="#065F46", ha="center", zorder=6
    )
    ax.text(
        14.50, 5.65,
        "• Standalone actor (72K params)\n• Disjoint validation selection\n• Immutable audit verified",
        fontsize=5.8, color="#334155", ha="center", va="center", zorder=5
    )

    # -------------------------------------------------------------
    # SECTION 2: (b) DEPLOYMENT CLOSED-LOOP EXECUTION
    # -------------------------------------------------------------
    sec_b = FancyBboxPatch(
        (0.35, 0.50), 15.5, 4.35,
        boxstyle="round,pad=0,rounding_size=0.12",
        facecolor=C_SECTION_BG_B, edgecolor=C_SECTION_BORDER,
        linewidth=1.1, zorder=1
    )
    ax.add_patch(sec_b)

    sec_b_pill = FancyBboxPatch(
        (0.55, 4.52), 5.4, 0.26,
        boxstyle="round,pad=0,rounding_size=0.05",
        facecolor="#1E293B", edgecolor="none", zorder=4
    )
    ax.add_patch(sec_b_pill)
    ax.text(
        0.65, 4.65,
        "PHASE 2 : CLOSED-LOOP ARBITRATION & DEPLOYMENT (ONLINE EXECUTION)",
        fontsize=6.6, weight="bold", color="#FFFFFF", va="center", ha="left", zorder=5
    )

    # Box 2.1: Embodied Environment (Meta-World Sawyer)
    draw_card(
        0.55, 0.70, 2.65, 3.65,
        "Embodied Environment",
        badge_text="Sawyer",
        bg_color=C_SLATE_CARD, border_color=C_SLATE_BORDER,
        header_color=C_SLATE_HEADER, badge_bg="#475569"
    )
    ax.text(
        1.875, 3.75,
        "Meta-World PickPlace (MuJoCo)",
        fontsize=6.6, weight="bold", color=C_SLATE_TEXT, ha="center", va="center", zorder=5
    )
    if img_env:
        ax.imshow(img_env, extent=[0.80, 2.95, 2.12, 3.48], zorder=6)
        t_b = patches.Rectangle(
            (0.80, 2.12), 2.15, 1.36,
            linewidth=0.8, edgecolor=C_SLATE_BORDER, facecolor="none", zorder=7
        )
        ax.add_patch(t_b)

    s_box = FancyBboxPatch(
        (0.70, 0.85), 2.35, 1.15,
        boxstyle="round,pad=0,rounding_size=0.06",
        facecolor="#FFFFFF", edgecolor=C_SLATE_BORDER,
        linewidth=0.8, zorder=5
    )
    ax.add_patch(s_box)
    ax.text(
        1.875, 1.82, "Robot State $s_t \\in \\mathbb{R}^{21}$:",
        fontsize=6.2, weight="bold", color=C_SLATE_HEADER, ha="center", zorder=6
    )
    ax.text(
        1.875, 1.38,
        "• Sawyer joints $q$ (7D)\n• EE Position & Quat (7D)\n• Object pose (3D) & Goal (3D)\n• Gripper status $g$ (1D)",
        fontsize=5.7, color="#334155", ha="center", va="center", zorder=6
    )

    # Arrows from Environment State -> Policy Branches
    draw_arrow((3.20, 1.60), (3.90, 3.20), color=C_BLUE_BORDER, width=1.6, rad=-0.12,
               label="State $s_t$", label_pos=(0.4, 0.45), label_color=C_BLUE_HEADER, label_bg="#EFF6FF", fontsize=6.8)
    draw_arrow((3.20, 1.35), (3.90, 1.35), color=C_AMBER_BORDER, width=1.6, rad=0.0,
               label="History $s_{t-9:t}$", label_pos=(0.4, 0.45), label_color=C_AMBER_HEADER, label_bg="#FFFBEB", fontsize=6.8)

    # Box 2.2: Nominal Policy (ACT)
    draw_card(
        3.90, 2.70, 3.10, 1.65,
        "Nominal Task Policy (ACT)",
        badge_text="20-Chunk CVAE",
        bg_color=C_BLUE_CARD, border_color=C_BLUE_BORDER,
        header_color=C_BLUE_HEADER, badge_bg="#2563EB"
    )
    ax.text(
        5.45, 3.75,
        "Action Chunking with Transformers",
        fontsize=6.5, weight="bold", color=C_BLUE_TEXT, ha="center", zorder=5
    )
    ax.text(
        5.45, 3.38,
        "• Predicts action chunk $\\hat{a}_{t:t+k} = \\pi_{\\mathrm{ACT}}(s_t, z=0)$\n• Exponential temporal ensembling\n• High fidelity on demonstrated trajectories",
        fontsize=5.8, color="#1E3A8A", ha="center", va="center", zorder=5
    )
    ax.text(
        5.45, 2.88,
        "Nominal Action Candidate $a_t^{\\mathrm{ACT}}$",
        fontsize=6.4, weight="bold", color="#1D4ED8", ha="center", zorder=5
    )

    # Box 2.3: Runtime Causal Risk Monitor (LSTM)
    draw_card(
        3.90, 0.70, 3.10, 1.70,
        "Causal Risk Monitor (LSTM)",
        badge_text="10-Step History",
        bg_color=C_AMBER_CARD, border_color=C_AMBER_BORDER,
        header_color=C_AMBER_HEADER, badge_bg="#D97706"
    )
    ax.text(
        5.45, 1.82,
        "Causal Recurrent Risk Monitor",
        fontsize=6.5, weight="bold", color=C_AMBER_TEXT, ha="center", zorder=5
    )
    ax.text(
        5.45, 1.45,
        "• Sliding window $[s_{t-9}, \\dots, s_t]$\n• Failure risk within the next 10 steps\n• Strictly causal (no future observations)",
        fontsize=5.8, color="#92400E", ha="center", va="center", zorder=5
    )
    ax.text(
        5.45, 0.90,
        "Failure Probability $p_t \\in [0, 1]$",
        fontsize=6.4, weight="bold", color="#B45309", ha="center", zorder=5
    )

    # Arrows to Risk Gate Arbiter
    draw_arrow((7.00, 3.35), (7.65, 2.65), color=C_BLUE_BORDER, width=1.5, rad=-0.06)
    draw_arrow((7.00, 1.55), (7.65, 1.95), color=C_AMBER_BORDER, width=1.5, rad=0.06,
               label="$p_t$", label_pos=(0.4, 0.4), label_color=C_AMBER_HEADER, label_bg="#FFFBEB", fontsize=6.8)

    # Box 2.4: Central Dynamic Risk Gate / Arbiter
    draw_card(
        7.65, 1.30, 2.65, 2.15,
        "Risk Gate & Arbiter",
        badge_text="$\\tau=0.20$",
        bg_color="#FFFFFF", border_color="#E11D48",
        header_color="#BE123C", badge_bg="#E11D48"
    )
    gate_box = FancyBboxPatch(
        (7.80, 1.85), 2.35, 1.05,
        boxstyle="round,pad=0,rounding_size=0.08",
        facecolor="#FFF1F2", edgecolor="#E11D48",
        linewidth=1.1, zorder=5
    )
    ax.add_patch(gate_box)
    ax.text(
        8.975, 2.62, "Threshold Comparison",
        fontsize=6.2, weight="bold", color="#9F1239", ha="center", zorder=6
    )
    ax.text(
        8.975, 2.34, "Is $p_t \\geq \\tau_{\\mathrm{deploy}} = 0.20$ ?",
        fontsize=7.2, weight="bold", color="#BE123C", ha="center", zorder=6
    )
    ax.text(
        8.975, 2.05, "[Frozen Gate Calibration]",
        fontsize=5.6, color="#881337", weight="bold", ha="center", zorder=6
    )
    ax.text(
        8.975, 1.52,
        "• Low Risk ($p_t < 0.20$) $\\to$ Nominal\n• High Risk ($p_t \\geq 0.20$) $\\to$ Recovery",
        fontsize=5.8, color="#334155", ha="center", va="center", zorder=5
    )

    # Routing Paths from Gate:
    draw_arrow((10.30, 2.85), (10.95, 3.40), color=C_BLUE_BORDER, width=1.6, rad=0.08,
               label="Nominal Mode ($p_t < 0.20$)", label_pos=(0.45, 0.6), label_color=C_BLUE_HEADER, label_bg="#EFF6FF", fontsize=6.5)

    draw_arrow((10.30, 1.90), (10.95, 1.50), color=C_AMBER_BORDER, width=1.6, rad=-0.08,
               label="Intervention ($p_t \\geq 0.20$)", label_pos=(0.45, 0.4), label_color="#BE123C", label_bg="#FFF1F2", fontsize=6.5)

    # Box 2.5: Nominal Action Buffer / Bypass
    draw_card(
        10.95, 2.70, 2.55, 1.55,
        "Nominal ACT Control",
        badge_text="ACT Mode",
        bg_color=C_BLUE_CARD, border_color=C_BLUE_BORDER,
        header_color=C_BLUE_HEADER, badge_bg="#2563EB"
    )
    ax.text(
        12.225, 3.65, "Execute ACT Chunk",
        fontsize=6.4, weight="bold", color=C_BLUE_TEXT, ha="center", zorder=5
    )
    ax.text(
        12.225, 3.28,
        "$a_t = a_t^{\\mathrm{ACT}}$\n(Temporal Ensemble Output)",
        fontsize=6.4, weight="bold", color="#1D4ED8", ha="center", va="center", zorder=5
    )
    ax.text(
        12.225, 2.88,
        "Advances nominal task execution",
        fontsize=5.7, color="#475569", ha="center", zorder=5
    )

    # Box 2.6: Persistent Supervised Recovery Policy
    draw_card(
        10.95, 0.70, 2.55, 1.80,
        "Recovery Policy ($\\pi_{\\mathrm{rec}}$)",
        badge_text="Persistent",
        bg_color=C_GREEN_CARD, border_color=C_GREEN_BORDER,
        header_color=C_GREEN_HEADER, badge_bg="#059669"
    )
    ax.text(
        12.225, 1.95, "Supervised Corrective Actor",
        fontsize=6.4, weight="bold", color=C_GREEN_TEXT, ha="center", zorder=5
    )
    ax.text(
        12.225, 1.60,
        "$a_t = \\pi_{\\mathrm{rec}}(s_t)$\n$[\\Delta x, \\Delta y, \\Delta z, g]^T$",
        fontsize=6.4, weight="bold", color="#047857", ha="center", zorder=5
    )
    sticky_box = FancyBboxPatch(
        (11.10, 0.80), 2.25, 0.58,
        boxstyle="round,pad=0,rounding_size=0.05",
        facecolor="#D1FAE5", edgecolor=C_GREEN_BORDER,
        linewidth=0.7, zorder=5
    )
    ax.add_patch(sticky_box)
    ax.text(
        12.225, 1.20, "Persistent Mode Active",
        fontsize=5.8, weight="bold", color="#065F46", ha="center", zorder=6
    )
    ax.text(
        12.225, 0.98, "Holds control until success or 150 steps\n(Flushes stale ACT chunk; no jittery release)",
        fontsize=5.0, color="#047857", ha="center", zorder=6
    )

    # Action Consolidation & Execution (Rightmost Box)
    draw_card(
        13.80, 1.30, 1.95, 2.30,
        "Action Execution",
        badge_text="$\\mathbb{R}^4$",
        bg_color="#FFFFFF", border_color="#1E40AF",
        header_color="#1E40AF", badge_bg="#3B82F6"
    )
    ax.text(
        14.775, 3.02, "Low-Level Command",
        fontsize=6.5, weight="bold", color="#1E3A8A", ha="center", zorder=5
    )
    act_vec_box = FancyBboxPatch(
        (13.95, 2.12), 1.65, 0.76,
        boxstyle="round,pad=0,rounding_size=0.06",
        facecolor="#EFF6FF", edgecolor="#3B82F6",
        linewidth=0.8, zorder=5
    )
    ax.add_patch(act_vec_box)
    ax.text(
        14.775, 2.62, "Robot Cartesian Action:",
        fontsize=5.6, weight="bold", color="#1D4ED8", ha="center", zorder=6
    )
    ax.text(
        14.775, 2.32, "$a_t = [\\Delta x, \\Delta y, \\Delta z, g]^T$\n$a_t \\in [-1, 1]^4$",
        fontsize=5.9, weight="bold", color="#1E3A8A", ha="center", zorder=6
    )
    ax.text(
        14.775, 1.68,
        "• $\\Delta x, \\Delta y, \\Delta z$ EE velocity\n• $g$ gripper grasp command\n• 20 Hz MuJoCo simulation",
        fontsize=5.6, color="#475569", ha="center", va="center", zorder=5
    )

    # Arrows from paths into Action Execution
    draw_arrow((13.50, 3.40), (13.80, 2.90), color=C_BLUE_BORDER, width=1.5, rad=-0.08)
    draw_arrow((13.50, 1.50), (13.80, 1.95), color=C_GREEN_BORDER, width=1.5, rad=0.08)

    # Big Feedback Arrow from Action Execution back to Embodied Environment
    arrow_pts = [
        (14.775, 1.30),
        (14.775, 0.28),
        (1.875, 0.28),
        (1.875, 0.70)
    ]
    path_data = [
        (mpath.Path.MOVETO, arrow_pts[0]),
        (mpath.Path.LINETO, arrow_pts[1]),
        (mpath.Path.LINETO, arrow_pts[2]),
        (mpath.Path.LINETO, arrow_pts[3])
    ]
    codes, verts = zip(*path_data)
    curved_path = mpath.Path(verts, codes)
    patch_feedback = PathPatch(
        curved_path, facecolor="none", edgecolor="#0284C7",
        linewidth=1.8, linestyle="-", zorder=8
    )
    ax.add_patch(patch_feedback)
    
    # Arrow head for feedback
    fb_head = FancyArrowPatch(
        (1.875, 0.38), (1.875, 0.70),
        arrowstyle="-|>", mutation_scale=12,
        linewidth=1.8, color="#0284C7", zorder=9
    )
    ax.add_patch(fb_head)

    # Feedback Label Pill
    fb_pill = FancyBboxPatch(
        (6.0, 0.16), 4.6, 0.24,
        boxstyle="round,pad=0,rounding_size=0.05",
        facecolor="#E0F2FE", edgecolor="#0284C7",
        linewidth=0.8, zorder=10
    )
    ax.add_patch(fb_pill)
    ax.text(
        8.3, 0.28,
        "Closed-Loop Physical Transition : $s_{t+1} \\sim P(s_{t+1} \\mid s_t, a_t)$ + Task Outcome",
        fontsize=6.5, weight="bold", color="#0369A1", ha="center", va="center", zorder=11
    )

    # Save outputs
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=dpi, bbox_inches="tight", facecolor=C_BG_PAGE)
    print(f"Saved PNG to {output_png}")

    if output_pdf:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_pdf, bbox_inches="tight", facecolor=C_BG_PAGE)
        print(f"Saved PDF to {output_pdf}")

    plt.close(fig)

    # Replicate across paper_assets and results/figures
    targets = [
        Path("paper_assets/Figure1_final_framework.png"),
        Path("paper_assets/Figure1_framework.png"),
        Path("results/figures/framework_architecture.png"),
    ]
    for target in targets:
        if target.resolve() != output_png.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_png, target)
            print(f"Synced PNG -> {target}")

    if output_pdf:
        pdf_targets = [
            Path("paper_assets/Figure1_final_framework.pdf"),
            Path("paper_assets/Figure1_framework.pdf"),
            Path("results/figures/framework_architecture.pdf"),
        ]
        for target in pdf_targets:
            if target.resolve() != output_pdf.resolve():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(output_pdf, target)
                print(f"Synced PDF -> {target}")

if __name__ == "__main__":
    out_png = Path("paper_assets/Figure1_final_framework.png")
    out_pdf = Path("paper_assets/Figure1_final_framework.pdf")
    create_top_tier_framework_figure(out_png, out_pdf)
