"""Render clean expert-rollout previews for candidate Figure 1 tasks.

These images are for visual task selection only. They are not REIM evaluation
evidence and must not be reported as experimental outcomes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import imageio.v3 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from env.metaworld_multitask import REIMMetaWorldMultiTaskEnv


CANDIDATES = (
    ("MT10", "door-open-v3", "Door open", 20266010),
    ("MT10", "peg-insert-side-v3", "Peg insert", 20266010),
    ("MT50", "shelf-place-v3", "Shelf place", 20266050),
    ("MT50", "faucet-close-v3", "Faucet close", 20266050),
)


def render_rgb(env: REIMMetaWorldMultiTaskEnv) -> np.ndarray:
    frame = np.asarray(env.render())
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise RuntimeError(f"Unexpected render shape: {frame.shape}")
    return np.ascontiguousarray(frame[..., :3].astype(np.uint8, copy=False))


def configure_camera(env: REIMMetaWorldMultiTaskEnv) -> None:
    renderer = getattr(env.backend_env, "mujoco_renderer", None)
    if renderer is None:
        return
    config = {
        "lookat": np.asarray((0.0, 0.72, 0.12), dtype=np.float64),
        "distance": 1.15,
        "azimuth": 150.0,
        "elevation": -28.0,
    }
    renderer.default_cam_config = config
    viewer = getattr(renderer, "viewer", None)
    if viewer is not None:
        viewer.cam.lookat[:] = config["lookat"]
        viewer.cam.distance = config["distance"]
        viewer.cam.azimuth = config["azimuth"]
        viewer.cam.elevation = config["elevation"]


def successful_preview(
    suite: str, task: str, benchmark_seed: int
) -> tuple[int, list[np.ndarray]]:
    env = REIMMetaWorldMultiTaskEnv(
        benchmark=suite,
        task_id=task,
        variant_id=0,
        seed=benchmark_seed,
        render_mode="rgb_array",
    )
    try:
        for variant in range(12):
            env.select_task(task, variant)
            env.reset(seed=benchmark_seed + variant)
            configure_camera(env)
            frames = [render_rgb(env)]
            succeeded = False
            for _ in range(500):
                action = env.get_expert_action()
                _, _, terminated, truncated, info = env.step(action)
                frames.append(render_rgb(env))
                if bool(info.get("success", False)):
                    succeeded = True
                    break
                if terminated or truncated:
                    break
            if succeeded:
                return variant, [frames[0], frames[len(frames) // 2], frames[-1]]
        raise RuntimeError(f"No successful expert preview for {suite} {task}")
    finally:
        env.close()


def main() -> None:
    output = ROOT / "paper_assets" / "task_candidate_previews"
    output.mkdir(exist_ok=True)
    rows: list[tuple[str, str, int, list[np.ndarray]]] = []
    for suite, task, label, seed in CANDIDATES:
        variant, frames = successful_preview(suite, task, seed)
        rows.append((suite, label, variant, frames))
        for stage, frame in zip(("start", "interaction", "success"), frames):
            iio.imwrite(output / f"{suite.lower()}_{task}_{stage}.png", frame)

    fig, axes = plt.subplots(len(rows), 3, figsize=(8.0, 8.4), constrained_layout=True)
    column_titles = ("Start", "Interaction", "Success")
    for row_index, (suite, label, variant, frames) in enumerate(rows):
        for column_index, frame in enumerate(frames):
            ax = axes[row_index, column_index]
            ax.imshow(frame)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#777777")
                spine.set_linewidth(0.7)
            if row_index == 0:
                ax.set_title(column_titles[column_index], fontsize=11, weight="bold")
            if column_index == 0:
                ax.set_ylabel(
                    f"{label}\n{suite}, variant {variant}",
                    fontsize=10,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=12,
                )
    fig.suptitle(
        "Candidate task previews (clean scripted expert; visual selection only)",
        fontsize=12,
        weight="bold",
    )
    fig.savefig(output / "candidate_contact_sheet.png", dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
