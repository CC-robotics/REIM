#!/usr/bin/env python3
"""Collect exact online ACT trigger states and successful recovery demonstrations.

The resulting NPZ is a trigger-aligned corrective-imitation bank. Each entry
contains a complete numeric Meta-World/MuJoCo snapshot captured *before* the
recovery controller acts, plus state/action pairs from the PickPlace scripted
controller. The scripted actions supervise the standalone recovery actor;
benchmark evaluation loads that learned actor and never calls the expert.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import trange

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.io import atomic_save_npz, atomic_write_json, file_sha256
from env.metaworld_pickplace import make_scripted_expert
from evaluation.evaluate_reim import (
    ROBUSTNESS_ACTION_STD_SCALE,
    ROBUSTNESS_OBJECT_STD_SCALE,
    ROBUSTNESS_OBSERVATION_STD_SCALE,
    load_bc_policy,
    load_failure_detector,
    make_env,
    seed_everything,
)
from utils.common import select_device


LOGGER = logging.getLogger("reim.collect_recovery_starts")
SCHEMA_VERSION = "reim-recovery-starts-v1"


def _resume_existing(
    args: argparse.Namespace,
    output: Path,
    manifest_path: Path,
) -> dict[str, Any] | None:
    """Return a verified existing split, or ``None`` for fresh collection."""

    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    exists = output.exists() or manifest_path.exists()
    if not exists:
        return None
    if args.overwrite:
        return None
    if not args.resume:
        raise FileExistsError(
            f"{output} or {manifest_path} exists; pass --resume to verify and "
            "reuse it, or --overwrite to replace it."
        )
    if not output.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "Recovery-start resume requires both the NPZ and JSON manifest: "
            f"{output}, {manifest_path}"
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    mismatches: list[str] = []
    expected_exact = {
        "schema_version": SCHEMA_VERSION,
        "backend": "metaworld",
        "episodes_attempted": int(args.episodes),
        "act_checkpoint_sha256": file_sha256(args.act_checkpoint),
        "detector_checkpoint_sha256": file_sha256(args.detector_checkpoint),
        "npz_sha256": file_sha256(output),
    }
    for key, expected in expected_exact.items():
        if manifest.get(key) != expected:
            mismatches.append(
                f"{key}: stored={manifest.get(key)!r}, requested={expected!r}"
            )
    for key, expected in (
        ("failure_threshold", float(args.failure_threshold)),
        ("noise_level", float(args.noise_level)),
    ):
        try:
            matches = bool(
                np.isclose(float(manifest.get(key)), expected, atol=1e-8, rtol=0.0)
            )
        except (TypeError, ValueError):
            matches = False
        if not matches:
            mismatches.append(
                f"{key}: stored={manifest.get(key)!r}, requested={expected!r}"
            )
    stored_seed = manifest.get("collection_seed", manifest.get("seed_min"))
    if stored_seed is None or int(stored_seed) != int(args.seed):
        mismatches.append(
            f"collection_seed: stored={stored_seed!r}, requested={args.seed!r}"
        )
    if mismatches:
        detail = "; ".join(mismatches)
        raise ValueError(
            "Existing recovery-start split is incompatible with this resume: "
            f"{detail}"
        )
    LOGGER.info("Verified and retained recovery-start split %s", output)
    return manifest


def _detector_input(
    history: list[np.ndarray], sequence_length: int
) -> tuple[np.ndarray, int]:
    valid = min(len(history), sequence_length)
    window = np.zeros(
        (sequence_length, int(np.asarray(history[-1]).size)), dtype=np.float32
    )
    if valid:
        window[:valid] = np.stack(history[-valid:]).astype(np.float32)
    return window, valid


def _stack_snapshots(
    snapshots: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if not snapshots:
        raise RuntimeError("No recovery trigger states were captured.")
    keys = set(snapshots[0])
    for index, snapshot in enumerate(snapshots[1:], start=1):
        if set(snapshot) != keys:
            raise RuntimeError(
                f"Snapshot {index} has a different schema from snapshot 0."
            )
    return {
        f"snapshot_{key}": np.stack(
            [np.asarray(snapshot[key]) for snapshot in snapshots], axis=0
        )
        for key in sorted(keys)
    }


def collect(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    manifest_path = output.with_suffix(".json")
    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError("--episodes and --max-steps must be positive.")
    if not 0.0 < args.failure_threshold < 1.0:
        raise ValueError("--failure-threshold must lie in (0, 1).")
    existing = _resume_existing(args, output, manifest_path)
    if existing is not None:
        return existing

    args.device = select_device(args.device)
    seed_everything(args.seed)
    env = make_env(
        backend=args.backend,
        seed=args.seed,
        env_config=args.env_config,
        noise_level=args.noise_level,
    )
    if getattr(env, "backend", None) != "metaworld":
        env.close()
        raise RuntimeError(
            "Recovery-start snapshots are scientific Meta-World artifacts; "
            "the toy backend is not supported."
        )
    bc = load_bc_policy(
        args.act_checkpoint,
        state_dim=int(np.prod(env.observation_space.shape)),
        action_dim=int(np.prod(env.action_space.shape)),
        device=args.device,
    )
    detector = load_failure_detector(
        args.detector_checkpoint,
        state_dim=int(np.prod(env.observation_space.shape)),
        device=args.device,
    )
    expert = make_scripted_expert(env)

    snapshots: list[dict[str, np.ndarray]] = []
    episode_seeds: list[int] = []
    trigger_steps: list[int] = []
    trigger_probabilities: list[float] = []
    expert_successes: list[bool] = []
    expert_lengths: list[int] = []
    demo_states: list[np.ndarray] = []
    demo_actions: list[np.ndarray] = []
    demo_start_indices: list[int] = []
    triggered_episodes = 0

    try:
        progress = trange(
            args.episodes, desc="Collecting recovery starts", unit="episode"
        )
        for episode in progress:
            episode_seed = args.seed + episode
            observation, info = env.reset(seed=episode_seed)
            bc.reset()
            expert.reset()
            history: list[np.ndarray] = []
            snapshot: dict[str, np.ndarray] | None = None
            probability = 0.0
            trigger_step = -1
            terminated = truncated = False

            for step in range(args.max_steps):
                if history and step >= args.min_trigger_step:
                    detector_window, valid = _detector_input(
                        history, args.sequence_length
                    )
                    probability = detector(detector_window, valid)
                    if probability >= args.failure_threshold:
                        snapshot = env.capture_sim_state()
                        trigger_step = step
                        break
                action = np.asarray(bc(observation), dtype=np.float32)
                observation, _, terminated, truncated, info = env.step(action)
                history.append(np.asarray(observation, dtype=np.float32).copy())
                if bool(info.get("success", False)) or terminated or truncated:
                    break

            if snapshot is None:
                progress.set_postfix(
                    starts=len(snapshots),
                    successful=sum(expert_successes),
                )
                continue
            triggered_episodes += 1

            local_states: list[np.ndarray] = []
            local_actions: list[np.ndarray] = []
            expert_success = False
            recovery_steps = 0
            for _ in range(max(args.max_steps - trigger_step, 0)):
                clean_state = np.asarray(env.get_state(), dtype=np.float32).copy()
                action = np.clip(
                    np.asarray(expert.act(clean_state), dtype=np.float32),
                    -1.0,
                    1.0,
                )
                local_states.append(clean_state)
                local_actions.append(action.copy())
                observation, _, terminated, truncated, info = env.step(action)
                recovery_steps += 1
                expert_success = bool(info.get("success", False))
                if expert_success or terminated or truncated:
                    break

            start_index = len(snapshots)
            snapshots.append(snapshot)
            episode_seeds.append(episode_seed)
            trigger_steps.append(trigger_step)
            trigger_probabilities.append(probability)
            expert_successes.append(expert_success)
            expert_lengths.append(recovery_steps)
            # Failed scripted rollouts are useful for start-state coverage but
            # are not valid behavior-cloning targets.
            if expert_success and local_states:
                demo_states.extend(local_states)
                demo_actions.extend(local_actions)
                demo_start_indices.extend([start_index] * len(local_states))
            progress.set_postfix(
                starts=len(snapshots),
                successful=sum(expert_successes),
            )
    finally:
        env.close()

    if not snapshots:
        raise RuntimeError(
            "No detector triggers were captured. Lower --failure-threshold or "
            "increase --episodes."
        )
    if not demo_states:
        raise RuntimeError(
            "No successful scripted recoveries were collected; PPO cannot be "
            "warm-started from this protocol."
        )

    arrays = _stack_snapshots(snapshots)
    arrays.update(
        {
            "episode_seed": np.asarray(episode_seeds, dtype=np.int64),
            "trigger_step": np.asarray(trigger_steps, dtype=np.int16),
            "trigger_probability": np.asarray(
                trigger_probabilities, dtype=np.float32
            ),
            "expert_success": np.asarray(expert_successes, dtype=np.bool_),
            "expert_steps": np.asarray(expert_lengths, dtype=np.int16),
            "demo_states": np.stack(demo_states).astype(np.float32),
            "demo_actions": np.stack(demo_actions).astype(np.float32),
            "demo_start_index": np.asarray(demo_start_indices, dtype=np.int32),
            "schema_version": np.asarray(SCHEMA_VERSION),
            "backend": np.asarray("metaworld"),
            "state_mode": np.asarray(getattr(env, "state_mode", "semantic")),
            "failure_threshold": np.asarray(
                args.failure_threshold, dtype=np.float32
            ),
            "noise_level": np.asarray(args.noise_level, dtype=np.float32),
        }
    )
    atomic_save_npz(output, **arrays)

    successes = int(np.sum(expert_successes))
    statistics = {
        "schema_version": SCHEMA_VERSION,
        "backend": "metaworld",
        "episodes_attempted": args.episodes,
        "collection_seed": args.seed,
        "episodes_triggered": triggered_episodes,
        "trigger_rate": triggered_episodes / args.episodes,
        "recovery_starts": len(snapshots),
        "expert_successes": successes,
        "expert_success_rate": successes / len(snapshots),
        "expert_demonstration_steps": len(demo_states),
        "mean_trigger_step": float(np.mean(trigger_steps)),
        "mean_expert_steps": float(np.mean(expert_lengths)),
        "seed_min": int(min(episode_seeds)),
        "seed_max": int(max(episode_seeds)),
        "failure_threshold": args.failure_threshold,
        "noise_level": args.noise_level,
        "disturbance": {
            "action_noise_std": (
                args.noise_level * ROBUSTNESS_ACTION_STD_SCALE
            ),
            "observation_noise_std": (
                args.noise_level * ROBUSTNESS_OBSERVATION_STD_SCALE
            ),
            "object_impulse_std_m": (
                args.noise_level * ROBUSTNESS_OBJECT_STD_SCALE
            ),
        },
        "act_checkpoint": str(Path(args.act_checkpoint)),
        "act_checkpoint_sha256": file_sha256(args.act_checkpoint),
        "detector_checkpoint": str(Path(args.detector_checkpoint)),
        "detector_checkpoint_sha256": file_sha256(args.detector_checkpoint),
        "npz": str(output),
        "npz_sha256": file_sha256(output),
    }
    atomic_write_json(manifest_path, statistics)
    LOGGER.info(
        "Saved %d trigger states (%d expert-successful) and %d demo steps to %s",
        len(snapshots),
        successes,
        len(demo_states),
        output,
    )
    return statistics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=3_000_042)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--min-trigger-step", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--failure-threshold", type=float, default=0.1)
    parser.add_argument("--noise-level", type=float, default=0.2)
    parser.add_argument("--backend", choices=("metaworld",), default="metaworld")
    parser.add_argument(
        "--env-config", type=Path, default=PROJECT_ROOT / "configs/environment.yaml"
    )
    parser.add_argument(
        "--act-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/bc_policy.pt",
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/failure_detector.pt",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "datasets/recovery_starts/train.npz",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    statistics = collect(args)
    print(
        "recovery_start_expert_success_rate="
        f"{statistics['expert_success_rate']:.6f}"
    )


if __name__ == "__main__":
    main()
