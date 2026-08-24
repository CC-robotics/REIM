# -*- coding: utf-8 -*-
"""MT10 success-occupancy 曲线：网格散点 + heuristic 参考点 + 0.05/10 高亮。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

setup_plot()

BASE = Path("results/diagnostics/release_patience_search")
grid = pd.read_csv(BASE / "mt10_reim_grid.csv")
grid = grid.rename(columns={"micro_success": "success"})
grid["noise_level"] = grid["noise_level"].astype(float)

# heuristic 参考点（同库重跑，见 mt10_references.csv）
heur = pd.DataFrame({
    "noise_level": [0.0, 0.1, 0.2, 0.4],
    "success": [0.98, 0.47, 0.48, 0.47],
    "recovery_occupancy_mean": [0.132313, 0.481940, 0.487594, 0.475755],
})

noise_levels = sorted(grid["noise_level"].unique())
palette = dict(zip(noise_levels, sns.color_palette("viridis", len(noise_levels))))

fig, ax = plt.subplots(figsize=(8.5, 5.5))
for nl in noise_levels:
    sub = grid[grid["noise_level"] == nl]
    ax.scatter(sub["recovery_occupancy_mean"], sub["success"],
               color=palette[nl], alpha=0.55, s=45, edgecolors="none",
               label=f"REIM grid, noise={nl:g}")

# robustness-first 工作点 0.05/10 高亮
op = grid[(grid["release_threshold"] == 0.05) & (grid["release_patience"] == 10)]
ax.scatter(op["recovery_occupancy_mean"], op["success"],
           facecolors="none", edgecolors="black", s=160, linewidths=1.8,
           label="REIM 0.05/10 (robustness-first)")

# heuristic 参考点（noise 0.1/0.2/0.4 几乎重合，合并标注）
ax.scatter(heur["recovery_occupancy_mean"], heur["success"],
           marker="*", s=260, color="crimson", edgecolors="black",
           linewidths=0.8, zorder=5, label="Heuristic-gated recovery")
clean = heur[heur["noise_level"] == 0.0].iloc[0]
ax.annotate("noise=0", (clean["recovery_occupancy_mean"], clean["success"]),
            textcoords="offset points", xytext=(8, -14), fontsize=8,
            color="crimson")
noisy = heur[heur["noise_level"] > 0.0]
cx, cy = noisy["recovery_occupancy_mean"].mean(), noisy["success"].mean()
ax.annotate("noise=0.1 / 0.2 / 0.4\n(三点近似重合)", (cx, cy),
            textcoords="offset points", xytext=(12, -26), fontsize=8,
            color="crimson")

# noise 0.4 matched 点 0.3/5
m = grid[(grid["noise_level"] == 0.4) & (grid["release_threshold"] == 0.3)
         & (grid["release_patience"] == 5)]
if not m.empty:
    ax.scatter(m["recovery_occupancy_mean"], m["success"],
               facecolors="none", edgecolors="crimson", s=200, linewidths=1.6,
               linestyle="--", label="Matched-occupancy 0.3/5 (noise=0.4)")

ax.set_xlabel("Recovery occupancy（recovery 步数占比）")
ax.set_ylabel("Micro success rate")
ax.set_title("MT10 success–occupancy（200 回合搜索验证库, seed 20264010, 20 ep/task）")
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.86))
ax.set_xlim(left=0)
fig.savefig("results/figures/mt10_success_occupancy.png", dpi=220, bbox_inches="tight")
print("saved results/figures/mt10_success_occupancy.png")
