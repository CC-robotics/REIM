"""Capture a real Meta-World/MuJoCo REIM recovery rollout for publication.

The script searches a separate qualitative seed range for a paired case in which
(1) ACT fails, while (2) REIM triggers the learned failure detector,
(3) executes the trigger-aligned imitation recovery option, and (4) completes
PickPlace.
The selected same-seed ACT/REIM pair is validated on fresh environments with
``render_mode="rgb_array"`` before exporting raw keyframes and an annotated
publication figure.

This is a *simulation* visualization.  It never synthesizes or claims physical
robot hardware imagery.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:  # Support direct script execution.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate_reim import (
    ControllerConfig,
    REIMController,
    _is_success,
    _space_dim,
    _step_env,
    load_bc_policy,
    load_failure_detector,
    load_recovery_policy,
    make_env,
    seed_everything,
)

LOGGER = logging.getLogger("reim.visualization.recovery_rollout")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class Rollout:
    """One closed-loop REIM episode and its optional rendered frames."""

    method: str
    seed: int
    success: bool
    steps: int
    recovery_attempts: int
    recovery_successes: int
    detector_triggers: int
    recovery_steps: int
    max_failure_probability: float
    sources: list[str]
    risks: list[float]
    actions: list[list[float]]
    infos: list[dict[str, Any]]
    frames: list[np.ndarray]
    disturbance_indices: list[int]
    trigger_indices: list[int]


@dataclass(frozen=True, slots=True)
class Keyframe:
    """A semantically selected frame in a successful recovery."""

    rollout: str
    key: str
    title: str
    subtitle: str
    index: int
    color: str


def _json_value(value: Any) -> Any:
    """Convert common NumPy values to a compact JSON-compatible structure."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _render_rgb(env: Any) -> np.ndarray:
    """Return one validated RGB frame from the actual simulator renderer."""

    frame = env.render()
    if frame is None:
        raise RuntimeError(
            "Meta-World render() returned None. Construct the environment with "
            "render_mode='rgb_array' and use a working MuJoCo GL backend."
        )
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise RuntimeError(f"Expected an RGB/RGBA render, received {image.shape}")
    image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(image)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0.0, 255.0).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _configure_camera(
    env: Any,
    *,
    lookat: Sequence[float],
    distance: float,
    azimuth: float,
    elevation: float,
) -> None:
    """Configure a close oblique free camera for interpretable keyframes."""

    base_getter = getattr(env, "_base_metaworld_env", None)
    base = base_getter() if callable(base_getter) else None
    renderer = getattr(base, "mujoco_renderer", None)
    if renderer is None:
        raise RuntimeError("Meta-World environment does not expose a MuJoCo renderer")
    camera_config = {
        "lookat": np.asarray(lookat, dtype=np.float64).reshape(3),
        "distance": float(distance),
        "azimuth": float(azimuth),
        "elevation": float(elevation),
    }
    renderer.default_cam_config = camera_config
    viewer = getattr(renderer, "viewer", None)
    if viewer is not None:
        viewer.cam.lookat[:] = camera_config["lookat"]
        viewer.cam.distance = camera_config["distance"]
        viewer.cam.azimuth = camera_config["azimuth"]
        viewer.cam.elevation = camera_config["elevation"]


def _compact_info(info: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only state/event fields needed for keyframe selection."""

    keys = (
        "object_position",
        "goal_position",
        "ee_position",
        "tcp_position",
        "distance_to_goal",
        "distance",
        "success",
        "is_success",
        "failure",
        "failure_reason",
        "object_dropped",
        "object_disturbance_applied",
        "object_disturbance_delta",
        "step",
        "backend",
        "env_name",
    )
    return {key: _json_value(info[key]) for key in keys if key in info}


def _run_rollout(
    env: Any,
    *,
    method: str,
    seed: int,
    bc_policy: Any,
    detector: Any | None,
    recovery_policy: Any | None,
    controller_config: ControllerConfig,
    max_steps: int,
    capture_frames: bool,
) -> Rollout:
    """Execute one ACT or REIM episode using the deployment controller."""

    result = env.reset(seed=seed)
    if isinstance(result, tuple) and len(result) == 2:
        observation, info = result
    else:
        observation, info = result, {}
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    info = dict(info or {})
    controller = REIMController(
        method,
        bc_policy,
        detector=detector if method == "reim" else None,
        recovery_policy=recovery_policy if method == "reim" else None,
        config=controller_config,
    )
    controller.reset_observation(observation)

    frames = [_render_rgb(env)] if capture_frames else []
    infos = [_compact_info(info)]
    sources: list[str] = []
    risks: list[float] = []
    actions: list[list[float]] = []
    disturbance_indices: list[int] = []
    trigger_indices: list[int] = []
    success = False
    action_dim = _space_dim(env.action_space, "action")
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(action_dim)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(action_dim)

    for step_index in range(max_steps):
        decision = controller.act(observation, info)
        action = np.asarray(decision.action, dtype=np.float32).reshape(-1)
        if action.size != action_dim:
            raise RuntimeError(
                f"Controller emitted {action.size} values, expected {action_dim}"
            )
        action = np.clip(action, low, high)
        if decision.source == "recovery" and (
            not sources or sources[-1] != "recovery"
        ):
            # The detector observes the current state and switches before this
            # action.  Therefore the pre-action render index is ``step_index``.
            trigger_indices.append(step_index)

        next_observation, reward, terminated, truncated, info = _step_env(env, action)
        success = _is_success(info, reward)
        controller.observe_transition(
            next_observation,
            reward,
            info,
            success=success,
        )
        sources.append(decision.source)
        risks.append(float(decision.failure_probability))
        actions.append(action.astype(float).tolist())
        infos.append(_compact_info(info))
        if bool(info.get("object_disturbance_applied", False)):
            # Post-action state/render index.
            disturbance_indices.append(step_index + 1)
        if capture_frames:
            frames.append(_render_rgb(env))
        observation = next_observation
        if success or terminated or truncated:
            break

    controller.finalize(success)
    return Rollout(
        method=method,
        seed=seed,
        success=success,
        steps=len(sources),
        recovery_attempts=controller.recovery_attempts,
        recovery_successes=controller.recovery_successes,
        detector_triggers=controller.detector_triggers,
        recovery_steps=controller.recovery_steps_total,
        max_failure_probability=controller.failure_probability_max,
        sources=sources,
        risks=risks,
        actions=actions,
        infos=infos,
        frames=frames,
        disturbance_indices=disturbance_indices,
        trigger_indices=trigger_indices,
    )


def _excluded_seeds(paths: Sequence[Path]) -> set[int]:
    """Load episode seeds from recovery-start datasets to enforce held-out use."""

    excluded: set[int] = set()
    for path in paths:
        if not path.is_file():
            LOGGER.warning("Exclusion dataset not found; skipping %s", path)
            continue
        with np.load(path, allow_pickle=False) as archive:
            if "episode_seed" not in archive:
                raise KeyError(f"{path} does not contain episode_seed")
            excluded.update(
                int(value)
                for value in np.asarray(archive["episode_seed"]).reshape(-1)
            )
        LOGGER.info("Excluded recovery-start seeds from %s", path)
    return excluded


def _distance(info: Mapping[str, Any]) -> float:
    value = info.get("distance_to_goal", info.get("distance", np.nan))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _position(info: Mapping[str, Any], key: str) -> np.ndarray:
    try:
        value = np.asarray(info[key], dtype=np.float64).reshape(-1)
    except (KeyError, TypeError, ValueError):
        return np.full(3, np.nan, dtype=np.float64)
    if value.size < 3:
        return np.full(3, np.nan, dtype=np.float64)
    return value[:3]


def _select_keyframes(act: Rollout, reim: Rollout) -> list[Keyframe]:
    """Select an eight-panel paired ACT-failure/REIM-success sequence."""

    if act.success:
        raise ValueError("The paired qualitative case requires ACT to fail")
    if not reim.success or not reim.trigger_indices:
        raise ValueError("REIM must trigger recovery and succeed in the paired case")
    for rollout in (act, reim):
        if len(rollout.frames) != len(rollout.infos):
            raise ValueError("Rendered frames and state info are not aligned")

    act_final = len(act.frames) - 1
    reim_final = len(reim.frames) - 1
    trigger = int(reim.trigger_indices[0])
    disturbance = int(
        reim.disturbance_indices[0]
        if reim.disturbance_indices
        else max(1, trigger - 1)
    )
    disturbance = min(disturbance, max(trigger - 1, 1))

    object_positions = np.stack(
        [_position(info, "object_position") for info in reim.infos]
    )
    distances = np.asarray([_distance(info) for info in reim.infos])
    # The object displacement can contain a positive z component.  Using that
    # instantaneous disturbed height as the table baseline can make the lift
    # condition unreachable and collapse all recovery keyframes onto the final
    # three timesteps.  Estimate the settled table height from the first ten
    # post-trigger states instead.
    settle_end = min(trigger + 10, reim_final + 1)
    settled_z = object_positions[trigger:settle_end, 2]
    finite_settled_z = settled_z[np.isfinite(settled_z)]
    reference_z = float(
        np.median(finite_settled_z)
        if finite_settled_z.size
        else object_positions[min(max(trigger, 0), reim_final), 2]
    )
    recovery_candidates = np.arange(
        min(trigger + 1, reim_final), reim_final + 1
    )
    lifted = recovery_candidates[
        np.flatnonzero(
            object_positions[recovery_candidates, 2] >= reference_z + 0.015
        )
    ]
    if lifted.size:
        lift = int(lifted[0])
    elif recovery_candidates.size:
        lift = int(
            recovery_candidates[
                np.nanargmax(object_positions[recovery_candidates, 2])
            ]
        )
    else:
        lift = trigger
    lift = min(max(lift, trigger + 1), max(reim_final - 2, trigger + 1))

    transport_candidates = np.arange(min(lift + 1, reim_final), reim_final)
    if transport_candidates.size and np.isfinite(
        distances[transport_candidates]
    ).any():
        start_distance = _distance(reim.infos[lift])
        final_distance = _distance(reim.infos[reim_final])
        target_distance = 0.5 * (start_distance + final_distance)
        finite = transport_candidates[np.isfinite(distances[transport_candidates])]
        transport = int(
            finite[np.argmin(np.abs(distances[finite] - target_distance))]
        )
    else:
        transport = int(round(0.5 * (lift + reim_final)))
    transport = min(
        max(transport, lift + 1), max(reim_final - 1, lift + 1)
    )

    # On the ACT branch, prefer an explicit physical failure event but do not
    # treat the terminal timeout itself as trajectory drift.  If no physical
    # event exists, show ACT's best task progress before its final timeout; this
    # gives a semantically meaningful and temporally separated comparison.
    failure_events = [
        index
        for index, info in enumerate(act.infos)
        if index > disturbance
        and bool(info.get("failure", False))
        and not any(
            token in str(info.get("failure_reason", "")).lower()
            for token in ("timeout", "time limit", "time_limit", "truncated")
        )
    ]
    if failure_events:
        act_drift = int(failure_events[0])
    else:
        act_candidates = np.arange(
            min(disturbance + 1, act_final - 1), max(act_final, disturbance + 2)
        )
        act_distances = np.asarray([_distance(info) for info in act.infos])
        finite = act_candidates[np.isfinite(act_distances[act_candidates])]
        act_drift = int(
            finite[np.argmin(act_distances[finite])]
            if finite.size
            else max(disturbance + 1, round(0.55 * act_final))
        )
    act_drift = min(act_drift, max(act_final - 1, disturbance + 1))

    risk = (
        reim.risks[trigger]
        if trigger < len(reim.risks)
        else reim.max_failure_probability
    )
    disturbance_delta = reim.infos[disturbance].get(
        "object_disturbance_delta", [0.0, 0.0, 0.0]
    )
    delta_norm = float(
        np.linalg.norm(np.asarray(disturbance_delta, dtype=np.float64))
    )
    act_reason = str(act.infos[act_final].get("failure_reason", "")).strip()
    if not act_reason:
        act_reason = "time limit"

    return [
        Keyframe(
            "act",
            "act_initial",
            "(a) Paired start",
            f"ACT | t=0 | d={_distance(act.infos[0]):.3f} m",
            0,
            "#356A9A",
        ),
        Keyframe(
            "act",
            "act_disturbance",
            "(b) Object displacement",
            f"ACT | t={disturbance} | Δ={delta_norm:.3f} m",
            disturbance,
            "#D97828",
        ),
        Keyframe(
            "act",
            "act_unrecovered",
            "(c) ACT drift",
            (
                f"ACT | t={act_drift} | "
                f"d={_distance(act.infos[act_drift]):.3f} m"
            ),
            act_drift,
            "#356A9A",
        ),
        Keyframe(
            "act",
            "act_failure",
            "(d) ACT failure",
            (
                f"ACT | t={act_final} | S=0 | "
                f"{act_reason.replace('_', ' ')}"
            ),
            act_final,
            "#D97828",
        ),
        Keyframe(
            "reim",
            "reim_trigger",
            "(e) Risk trigger",
            f"REIM | t={trigger} | p={risk:.3f}",
            trigger,
            "#D97828",
        ),
        Keyframe(
            "reim",
            "reim_relift",
            "(f) Recovery grasp / lift",
            (
                f"REIM | t={lift} | z={object_positions[lift, 2]:.3f} m"
            ),
            lift,
            "#2E8B74",
        ),
        Keyframe(
            "reim",
            "reim_transport",
            "(g) Recovery transport",
            f"REIM | t={transport} | d={distances[transport]:.3f} m",
            transport,
            "#2E8B74",
        ),
        Keyframe(
            "reim",
            "reim_success",
            "(h) REIM success",
            (
                f"REIM | t={reim_final} | S=1 | "
                f"d={distances[reim_final]:.3f} m"
            ),
            reim_final,
            "#2E8B74",
        ),
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_outputs(
    act: Rollout,
    reim: Rollout,
    keyframes: Sequence[Keyframe],
    *,
    output_dir: Path,
    paper_assets_dir: Path,
    stem: str,
    noise_level: float,
    failure_threshold: float,
    recovery_budget: int,
    camera: Mapping[str, Any],
) -> dict[str, Any]:
    """Save paired raw frames, trajectories, figures, and provenance metadata."""

    import imageio.v3 as iio
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = output_dir / f"{stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    # These are reproducible derived artifacts. Remove only stale keyframes for
    # the selected qualitative seed before writing the new semantic selection.
    for stale_frame in frame_dir.glob(f"*_seed{reim.seed}_t*.png"):
        stale_frame.unlink()
    raw_frame_paths: list[Path] = []
    for order, keyframe in enumerate(keyframes, start=1):
        source = act if keyframe.rollout == "act" else reim
        path = frame_dir / (
            f"{order:02d}_{keyframe.key}_seed{reim.seed}"
            f"_t{keyframe.index:03d}.png"
        )
        iio.imwrite(path, source.frames[keyframe.index], extension=".png")
        raw_frame_paths.append(path)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 7.4,
            "figure.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.8), facecolor="white")
    for axis, keyframe in zip(axes.flat, keyframes, strict=True):
        source = act if keyframe.rollout == "act" else reim
        axis.imshow(source.frames[keyframe.index])
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(
            keyframe.title,
            loc="left",
            color="#15202B",
            fontweight="semibold",
            pad=3,
        )
        axis.text(
            0.0,
            -0.07,
            keyframe.subtitle,
            transform=axis.transAxes,
            ha="left",
            va="top",
            color="#34495E",
            fontsize=6.2,
        )
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.25)
            spine.set_edgecolor(keyframe.color)

    fig.suptitle(
        "Paired Sawyer PickPlace rollout: ACT failure vs. REIM recovery",
        x=0.04,
        y=0.982,
        ha="left",
        color="#15202B",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.text(
        0.04,
        0.942,
        (
            "Meta-World/MuJoCo simulation frames"
            f"  ·  separate qualitative seed {reim.seed}"
            f"  ·  noise {noise_level:.1f}"
            rf"  ·  $\tau_{{\mathrm{{on}}}}={failure_threshold:.2f}$"
        ),
        ha="left",
        va="top",
        color="#52616B",
        fontsize=6.8,
    )
    fig.text(
        0.04,
        0.025,
        (
            f"Recovery control: until success or {recovery_budget}-step budget"
            "  ·  orange: risk  ·  green: recovery"
            "  ·  simulated Sawyer frames"
        ),
        ha="left",
        va="bottom",
        color="#52616B",
        fontsize=6.1,
    )
    fig.subplots_adjust(
        left=0.04,
        right=0.99,
        top=0.875,
        bottom=0.11,
        wspace=0.08,
        hspace=0.27,
    )

    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    png_tmp = png_path.with_name(f"{png_path.stem}.tmp.png")
    pdf_tmp = pdf_path.with_name(f"{pdf_path.stem}.tmp.pdf")
    fig.savefig(
        png_tmp,
        dpi=320,
        facecolor="white",
        metadata={
            "Title": "REIM Meta-World/MuJoCo recovery rollout",
            "Author": "REIM experiment pipeline",
            "Description": (
                "Actual simulator-rendered frames; not a physical-robot photograph."
            ),
        },
    )
    fig.savefig(
        pdf_tmp,
        facecolor="white",
        metadata={
            "Title": "REIM Meta-World/MuJoCo recovery rollout",
            "Author": "REIM experiment pipeline",
            "Subject": "Simulation recovery sequence",
        },
    )
    plt.close(fig)
    png_tmp.replace(png_path)
    pdf_tmp.replace(pdf_path)

    def trace_payload(rollout: Rollout) -> dict[str, Any]:
        return {
            "schema_version": "reim-simulation-trajectory-v1",
            "method": rollout.method,
            "seed": rollout.seed,
            "success": rollout.success,
            "steps": rollout.steps,
            "detector_triggers": rollout.detector_triggers,
            "recovery_attempts": rollout.recovery_attempts,
            "recovery_successes": rollout.recovery_successes,
            "recovery_steps": rollout.recovery_steps,
            "max_failure_probability": rollout.max_failure_probability,
            "sources": rollout.sources,
            "failure_probabilities": rollout.risks,
            "actions": rollout.actions,
            "states": rollout.infos,
        }

    act_trace_path = output_dir / f"{stem}_act_failure_trace.json"
    reim_trace_path = output_dir / f"{stem}_reim_success_trace.json"
    for trace_path, rollout in (
        (act_trace_path, act),
        (reim_trace_path, reim),
    ):
        temporary_trace = trace_path.with_suffix(".json.tmp")
        with temporary_trace.open("w", encoding="utf-8") as handle:
            json.dump(
                trace_payload(rollout),
                handle,
                indent=2,
                ensure_ascii=False,
            )
            handle.write("\n")
        temporary_trace.replace(trace_path)

    paper_assets_dir.mkdir(parents=True, exist_ok=True)
    paper_png = paper_assets_dir / "Figure5_operation_sequence.png"
    paper_pdf = paper_assets_dir / "Figure5_operation_sequence.pdf"
    shutil.copy2(png_path, paper_png)
    shutil.copy2(pdf_path, paper_pdf)

    def display_path(path: Path) -> str:
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    metadata_path = output_dir / f"{stem}.json"
    output_files = [
        *raw_frame_paths,
        png_path,
        pdf_path,
        act_trace_path,
        reim_trace_path,
        paper_png,
        paper_pdf,
    ]
    initial_state_delta = float(
        np.nanmax(
            np.abs(
                np.concatenate(
                    [
                        _position(act.infos[0], "object_position"),
                        _position(act.infos[0], "goal_position"),
                    ]
                )
                - np.concatenate(
                    [
                        _position(reim.infos[0], "object_position"),
                        _position(reim.infos[0], "goal_position"),
                    ]
                )
            )
        )
    )
    act_displacement = np.asarray(
        act.infos[act.disturbance_indices[0]].get(
            "object_disturbance_delta", np.full(3, np.nan)
        ),
        dtype=np.float64,
    )
    reim_displacement = np.asarray(
        reim.infos[reim.disturbance_indices[0]].get(
            "object_disturbance_delta", np.full(3, np.nan)
        ),
        dtype=np.float64,
    )
    displacement_delta = float(
        np.nanmax(np.abs(act_displacement - reim_displacement))
    )
    metadata = {
        "schema_version": "reim-paired-simulation-rollout-figure-v3",
        "provenance_statement": (
            "All robot-operation panels are RGB frames rendered directly from "
            "paired Meta-World/MuJoCo simulation rollouts from a separate "
            "qualitative seed. They are not "
            "physical-robot photographs and were not generated by an image model. "
            "This single qualitative example does not estimate aggregate success."
        ),
        "seed": reim.seed,
        "paired_protocol": {
            "same_seed": act.seed == reim.seed,
            "initial_object_goal_max_abs_delta": initial_state_delta,
            "object_displacement_max_abs_delta": displacement_delta,
            "act_must_fail": True,
            "reim_must_trigger_and_succeed": True,
        },
        "noise_level": noise_level,
        "failure_threshold": failure_threshold,
        "camera": _json_value(camera),
        "recovery_control_rule": (
            "Imitation recovery holds control until task success or "
            f"{recovery_budget}-step budget"
        ),
        "act": {
            "success": act.success,
            "steps": act.steps,
            "trajectory": display_path(act_trace_path),
        },
        "reim": {
            "success": reim.success,
            "steps": reim.steps,
            "detector_triggers": reim.detector_triggers,
            "recovery_attempts": reim.recovery_attempts,
            "recovery_successes": reim.recovery_successes,
            "recovery_steps": reim.recovery_steps,
            "max_failure_probability": reim.max_failure_probability,
            "disturbance_indices": reim.disturbance_indices,
            "trigger_indices": reim.trigger_indices,
            "trajectory": display_path(reim_trace_path),
        },
        "keyframes": [
            {
                "rollout": item.rollout,
                "key": item.key,
                "title": item.title,
                "subtitle": item.subtitle,
                "frame_index": item.index,
                "raw_frame": display_path(path),
            }
            for item, path in zip(keyframes, raw_frame_paths, strict=True)
        ],
        "artifacts": {
            display_path(path): {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in output_files
        },
    }
    temporary = metadata_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(metadata_path)
    return {
        "png": png_path,
        "pdf": pdf_path,
        "metadata": metadata_path,
        "raw_frames": raw_frame_paths,
        "act_trace": act_trace_path,
        "reim_trace": reim_trace_path,
        "paper_png": paper_png,
        "paper_pdf": paper_pdf,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        "--seed-start",
        dest="seed_start",
        type=int,
        default=5_100_042,
        help="First separate qualitative seed to search.",
    )
    parser.add_argument(
        "--max-search",
        type=int,
        default=100,
        help="Maximum consecutive qualitative seeds to test.",
    )
    parser.add_argument("--noise-level", type=float, default=0.2)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--failure-threshold", type=float, default=0.2)
    parser.add_argument("--recovery-exit-threshold", type=float, default=0.0)
    parser.add_argument("--recovery-budget", type=int, default=150)
    parser.add_argument("--recovery-min-steps", type=int, default=150)
    parser.add_argument("--recovery-clear-steps", type=int, default=200)
    parser.add_argument(
        "--env-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "environment.yaml",
    )
    parser.add_argument(
        "--act-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "bc_policy.pt",
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "failure_detector.pt",
    )
    parser.add_argument(
        "--recovery-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "imitation_recovery.pt",
    )
    parser.add_argument(
        "--exclude-dataset",
        type=Path,
        nargs="*",
        default=[
            PROJECT_ROOT / "datasets" / "recovery_starts" / "train.npz",
            PROJECT_ROOT / "datasets" / "recovery_starts" / "validation.npz",
        ],
        help="NPZ datasets whose episode_seed values must not be used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "figures",
    )
    parser.add_argument(
        "--paper-assets-dir",
        type=Path,
        default=PROJECT_ROOT / "paper_assets",
        help="Directory receiving Figure5_operation_sequence.{png,pdf}.",
    )
    parser.add_argument("--stem", default="recovery_operation_sequence")
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto' selects CUDA for ACT/LSTM when available.",
    )
    parser.add_argument(
        "--mujoco-gl",
        choices=("egl", "osmesa", "glfw"),
        default="egl",
        help="Headless MuJoCo rendering backend.",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--camera-lookat",
        type=float,
        nargs=3,
        default=(0.0, 0.70, 0.12),
        metavar=("X", "Y", "Z"),
        help="Free-camera look-at point for the qualitative sequence.",
    )
    parser.add_argument("--camera-distance", type=float, default=1.0)
    parser.add_argument("--camera-azimuth", type=float, default=150.0)
    parser.add_argument("--camera-elevation", type=float, default=-22.0)
    return parser


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if args.max_search <= 0:
        raise ValueError("--max-search must be positive")
    if args.noise_level < 0.0:
        raise ValueError("--noise-level must be non-negative")
    os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)
    device = _resolve_device(args.device)
    seed_everything(args.seed_start)
    excluded = _excluded_seeds(args.exclude_dataset)
    LOGGER.info(
        "Searching seeds [%d, %d] on device=%s; excluded=%d training/validation seeds",
        args.seed_start,
        args.seed_start + args.max_search - 1,
        device,
        len(excluded),
    )

    act_env = make_env(
        backend="metaworld",
        seed=args.seed_start,
        env_config=args.env_config,
        noise_level=args.noise_level,
        render_mode="rgb_array",
    )
    reim_env = make_env(
        backend="metaworld",
        seed=args.seed_start,
        env_config=args.env_config,
        noise_level=args.noise_level,
        render_mode="rgb_array",
    )
    controller_config = ControllerConfig(
        failure_threshold=args.failure_threshold,
        recovery_exit_threshold=args.recovery_exit_threshold,
        recovery_budget=args.recovery_budget,
        recovery_min_steps=args.recovery_min_steps,
        recovery_clear_steps=args.recovery_clear_steps,
    )
    state_dim = _space_dim(act_env.observation_space, "observation")
    action_dim = _space_dim(act_env.action_space, "action")
    bc_policy = load_bc_policy(
        args.act_checkpoint,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
    )
    detector = load_failure_detector(
        args.detector_checkpoint,
        state_dim=state_dim,
        device=device,
    )
    recovery_policy = load_recovery_policy(
        args.recovery_checkpoint,
        env=None,
        device=device,
    )

    selected_act: Rollout | None = None
    selected_reim: Rollout | None = None
    try:
        for offset in range(args.max_search):
            candidate_seed = args.seed_start + offset
            if candidate_seed in excluded:
                LOGGER.info("Skipping excluded seed %d", candidate_seed)
                continue
            act_probe = _run_rollout(
                act_env,
                method="bc",
                seed=candidate_seed,
                bc_policy=bc_policy,
                detector=None,
                recovery_policy=None,
                controller_config=controller_config,
                max_steps=args.max_steps,
                capture_frames=False,
            )
            reim_probe = _run_rollout(
                reim_env,
                method="reim",
                seed=candidate_seed,
                bc_policy=bc_policy,
                detector=detector,
                recovery_policy=recovery_policy,
                controller_config=controller_config,
                max_steps=args.max_steps,
                capture_frames=False,
            )
            LOGGER.info(
                (
                    "seed=%d ACT(success=%s,steps=%d) "
                    "REIM(success=%s,trigger=%d,recovery_steps=%d,steps=%d,max_risk=%.3f)"
                ),
                candidate_seed,
                act_probe.success,
                act_probe.steps,
                reim_probe.success,
                reim_probe.detector_triggers,
                reim_probe.recovery_steps,
                reim_probe.steps,
                reim_probe.max_failure_probability,
            )
            if (
                not act_probe.success
                and reim_probe.success
                and reim_probe.detector_triggers > 0
                and reim_probe.recovery_attempts > 0
                and "recovery" in reim_probe.sources
            ):
                # Validate and render on two fresh same-seed environments.  This
                # prevents task-state carry-over from the search loop and keeps
                # the ACT/REIM initial conditions paired.
                rendered_act_env = make_env(
                    backend="metaworld",
                    seed=candidate_seed,
                    env_config=args.env_config,
                    noise_level=args.noise_level,
                    render_mode="rgb_array",
                )
                rendered_reim_env = make_env(
                    backend="metaworld",
                    seed=candidate_seed,
                    env_config=args.env_config,
                    noise_level=args.noise_level,
                    render_mode="rgb_array",
                )
                try:
                    for rendered_env in (rendered_act_env, rendered_reim_env):
                        _configure_camera(
                            rendered_env,
                            lookat=args.camera_lookat,
                            distance=args.camera_distance,
                            azimuth=args.camera_azimuth,
                            elevation=args.camera_elevation,
                        )
                    rendered_act = _run_rollout(
                        rendered_act_env,
                        method="bc",
                        seed=candidate_seed,
                        bc_policy=bc_policy,
                        detector=None,
                        recovery_policy=None,
                        controller_config=controller_config,
                        max_steps=args.max_steps,
                        capture_frames=True,
                    )
                    rendered_reim = _run_rollout(
                        rendered_reim_env,
                        method="reim",
                        seed=candidate_seed,
                        bc_policy=bc_policy,
                        detector=detector,
                        recovery_policy=recovery_policy,
                        controller_config=controller_config,
                        max_steps=args.max_steps,
                        capture_frames=True,
                    )
                finally:
                    rendered_act_env.close()
                    rendered_reim_env.close()
                initial_delta = float(
                    np.nanmax(
                        np.abs(
                            np.concatenate(
                                [
                                    _position(
                                        rendered_act.infos[0], "object_position"
                                    ),
                                    _position(rendered_act.infos[0], "goal_position"),
                                ]
                            )
                            - np.concatenate(
                                [
                                    _position(
                                        rendered_reim.infos[0], "object_position"
                                    ),
                                    _position(
                                        rendered_reim.infos[0], "goal_position"
                                    ),
                                ]
                            )
                        )
                    )
                )
                act_displacement = np.asarray(
                    rendered_act.infos[
                        rendered_act.disturbance_indices[0]
                    ].get("object_disturbance_delta", np.full(3, np.nan)),
                    dtype=np.float64,
                )
                reim_displacement = np.asarray(
                    rendered_reim.infos[
                        rendered_reim.disturbance_indices[0]
                    ].get("object_disturbance_delta", np.full(3, np.nan)),
                    dtype=np.float64,
                )
                displacement_delta = float(
                    np.nanmax(
                        np.abs(act_displacement - reim_displacement)
                    )
                )
                if (
                    not rendered_act.success
                    and rendered_reim.success
                    and rendered_reim.detector_triggers > 0
                    and "recovery" in rendered_reim.sources
                    and initial_delta <= 1e-6
                    and displacement_delta <= 1e-6
                ):
                    selected_act = rendered_act
                    selected_reim = rendered_reim
                    break
                LOGGER.warning(
                    (
                        "Seed %d did not validate as a paired rendered case "
                        "(ACT=%s REIM=%s triggers=%d initial_delta=%.3g "
                        "displacement_delta=%.3g); continuing."
                    ),
                    candidate_seed,
                    rendered_act.success,
                    rendered_reim.success,
                    rendered_reim.detector_triggers,
                    initial_delta,
                    displacement_delta,
                )
    finally:
        act_env.close()
        reim_env.close()

    if selected_act is None or selected_reim is None:
        raise RuntimeError(
            "No paired ACT-failure / detector-triggered REIM-success rollout "
            f"was found in {args.max_search} qualitative seeds from {args.seed_start}."
        )

    keyframes = _select_keyframes(selected_act, selected_reim)
    outputs = _save_outputs(
        selected_act,
        selected_reim,
        keyframes,
        output_dir=args.output_dir.resolve(),
        paper_assets_dir=args.paper_assets_dir.resolve(),
        stem=args.stem,
        noise_level=args.noise_level,
        failure_threshold=args.failure_threshold,
        recovery_budget=args.recovery_budget,
        camera={
            "lookat": args.camera_lookat,
            "distance": args.camera_distance,
            "azimuth": args.camera_azimuth,
            "elevation": args.camera_elevation,
        },
    )
    LOGGER.info(
        (
            "Captured paired qualitative seed=%d: ACT failed in %d steps; "
            "REIM succeeded in %d steps with %d recovery steps (max risk=%.3f)"
        ),
        selected_reim.seed,
        selected_act.steps,
        selected_reim.steps,
        selected_reim.recovery_steps,
        selected_reim.max_failure_probability,
    )
    LOGGER.info("Saved composite PNG to %s", outputs["png"])
    LOGGER.info("Saved composite PDF to %s", outputs["pdf"])
    for keyframe in keyframes:
        LOGGER.info(
            "keyframe %-11s frame=%03d %s",
            keyframe.key,
            keyframe.index,
            keyframe.subtitle,
        )
    return {
        "seed": selected_reim.seed,
        "act_success": selected_act.success,
        "reim_success": selected_reim.success,
        "keyframes": [item.index for item in keyframes],
        **outputs,
    }


if __name__ == "__main__":
    main()
