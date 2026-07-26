#!/usr/bin/env python3
"""Collect reproducible sequential PickPlace expert demonstrations.

Each trajectory is stored as an independent compressed NPZ so interrupted
collections can resume without rewriting prior episodes. The state/action
sequence remains in temporal order for ACT sequence/chunk training.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.io import atomic_save_npz, atomic_write_json, file_sha256
from env.metaworld_pickplace import (
    REIMPickPlaceEnv,
    load_project_config,
    make_scripted_expert,
)


LOGGER = logging.getLogger("reim.collect_demonstrations")
SCHEMA_VERSION = "reim-demonstrations-v1"


def _section(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _trajectory_summary(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        rewards = np.asarray(archive["rewards"], dtype=np.float32)
        success = bool(np.asarray(archive["success"]).reshape(-1)[0])
        episode_seed = int(np.asarray(archive["episode_seed"]).reshape(-1)[0])
        state_dim = int(np.asarray(archive["states"]).shape[-1])
        action_dim = int(np.asarray(archive["actions"]).shape[-1])
    return {
        "file": path.name,
        "length": int(len(rewards)),
        "return": float(rewards.sum(dtype=np.float64)),
        "success": success,
        "episode_seed": episode_seed,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "sha256": file_sha256(path),
    }


def _rollout(
    env: REIMPickPlaceEnv,
    *,
    episode_seed: int,
) -> dict[str, np.ndarray]:
    state, info = env.reset(seed=episode_seed)
    expert = make_scripted_expert(env)
    expert.reset()

    states: list[np.ndarray] = []
    raw_observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    rewards: list[float] = []
    next_states: list[np.ndarray] = []
    next_raw_observations: list[np.ndarray] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    success_flags: list[bool] = []
    distances: list[float] = []

    for _ in range(env.max_episode_steps):
        pre_raw = env.raw_observation
        action = np.asarray(expert.act(state), dtype=np.float32).reshape(-1)
        if action.shape != env.action_space.shape:
            raise RuntimeError(
                f"Scripted expert produced action {action.shape}; "
                f"expected {env.action_space.shape}."
            )
        next_state, reward, terminated, truncated, info = env.step(action)
        states.append(np.asarray(state, dtype=np.float32).copy())
        raw_observations.append(pre_raw)
        actions.append(action.copy())
        executed_actions.append(
            np.asarray(info["executed_action"], dtype=np.float32).copy()
        )
        rewards.append(float(reward))
        next_states.append(np.asarray(next_state, dtype=np.float32).copy())
        next_raw_observations.append(env.raw_observation)
        terminated_flags.append(bool(terminated))
        truncated_flags.append(bool(truncated))
        success_flags.append(bool(info["success"]))
        distances.append(float(info["distance_to_goal"]))
        state = next_state
        if terminated or truncated:
            break

    if not states:
        raise RuntimeError("Environment produced an empty expert trajectory.")
    return {
        "states": np.stack(states).astype(np.float32),
        "raw_observations": np.stack(raw_observations).astype(np.float32),
        "actions": np.stack(actions).astype(np.float32),
        "executed_actions": np.stack(executed_actions).astype(np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "next_states": np.stack(next_states).astype(np.float32),
        "next_raw_observations": np.stack(next_raw_observations).astype(np.float32),
        "terminated": np.asarray(terminated_flags, dtype=np.bool_),
        "truncated": np.asarray(truncated_flags, dtype=np.bool_),
        "step_success": np.asarray(success_flags, dtype=np.bool_),
        "distance_to_goal": np.asarray(distances, dtype=np.float32),
        "success": np.asarray(any(success_flags), dtype=np.bool_),
        "episode_seed": np.asarray(episode_seed, dtype=np.int64),
        "backend": np.asarray(env.backend),
        "env_name": np.asarray(env.env_name),
        "state_mode": np.asarray(env.state_mode),
        "expert_algorithm": np.asarray("Sawyer scripted"),
        "imitation_target": np.asarray("ACT"),
        "schema_version": np.asarray(SCHEMA_VERSION),
    }


def collect(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _resolve_path(args.config) if args.config else None
    config = load_project_config(config_path)
    env_config = _section(config, "environment")
    demo_config = _section(config, "demonstrations")
    project_config = _section(config, "project")

    episodes = int(
        args.episodes
        if args.episodes is not None
        else demo_config.get("episodes", 500)
    )
    if episodes <= 0:
        raise ValueError("--episodes must be positive.")
    seed = int(args.seed if args.seed is not None else project_config.get("seed", 42))
    backend = str(
        args.backend
        if args.backend is not None
        else env_config.get("backend", "metaworld")
    )
    state_mode = str(
        args.state_mode
        if args.state_mode is not None
        else env_config.get("state_mode", "semantic")
    )
    max_steps = int(
        args.max_steps
        if args.max_steps is not None
        else env_config.get("max_episode_steps", 200)
    )
    max_attempts = int(
        args.max_attempts
        if args.max_attempts is not None
        else demo_config.get("max_attempts_per_trajectory", 3)
    )
    if max_attempts <= 0:
        raise ValueError("--max-attempts must be positive.")
    output_dir = _resolve_path(
        args.output_dir
        if args.output_dir is not None
        else demo_config.get("save_dir", "datasets/demonstrations")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(output_dir.glob("trajectory_*.npz"))
    if existing and args.overwrite:
        for path in existing:
            path.unlink()
        for name in ("manifest.json", "statistics.json"):
            metadata_path = output_dir / name
            if metadata_path.exists():
                metadata_path.unlink()
        existing = []
    elif existing and not args.resume:
        raise FileExistsError(
            f"{output_dir} already has {len(existing)} trajectories. "
            "Use --resume to keep them or --overwrite to replace them."
        )
    if existing and args.resume:
        extra = [
            path
            for path in existing
            if int(path.stem.rsplit("_", 1)[1]) >= episodes
        ]
        if extra:
            raise ValueError(
                f"Cannot resume with --episodes={episodes}: {len(extra)} existing "
                "trajectories lie outside the requested range and would still be "
                "loaded by training. Keep the original episode count or use a new "
                "output directory."
            )

    env = REIMPickPlaceEnv(
        config=config,
        seed=seed,
        backend=backend,
        render_mode=args.render_mode,
        state_mode=state_mode,
        max_episode_steps=max_steps,
        action_noise_std=0.0,
        observation_noise_std=0.0,
        object_noise_probability=0.0,
    )
    if existing and args.resume:
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(
                "Cannot resume demonstrations without manifest.json provenance. "
                "Use --overwrite to start a new protocol."
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            previous_manifest = json.load(handle)
        previous_statistics = previous_manifest.get("statistics", {})
        expected = {
            "schema_version": (previous_manifest.get("schema_version"), SCHEMA_VERSION),
            "seed": (previous_manifest.get("seed"), seed),
            "backend": (previous_manifest.get("backend"), backend),
            "env_name": (previous_manifest.get("env_name"), env.env_name),
            "state_mode": (previous_statistics.get("state_mode"), state_mode),
        }
        mismatches = [
            f"{key}: stored={stored!r}, requested={requested!r}"
            for key, (stored, requested) in expected.items()
            if stored != requested
        ]
        previous_protocol = previous_manifest.get("protocol")
        if isinstance(previous_protocol, Mapping):
            for key, requested in (
                ("max_episode_steps", max_steps),
                ("max_attempts_per_trajectory", max_attempts),
            ):
                if previous_protocol.get(key) != requested:
                    mismatches.append(
                        f"{key}: stored={previous_protocol.get(key)!r}, "
                        f"requested={requested!r}"
                    )
        if mismatches:
            raise ValueError(
                "Cannot resume demonstrations with a different protocol: "
                + "; ".join(mismatches)
                + ". Use --overwrite."
            )
    entries: dict[int, dict[str, Any]] = {}
    for path in existing:
        try:
            index = int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if index < episodes:
            summary = _trajectory_summary(path)
            if summary["state_dim"] != env.state_dim:
                raise ValueError(
                    f"Cannot resume: {path.name} has state_dim="
                    f"{summary['state_dim']}, current environment has {env.state_dim}."
                )
            entries[index] = summary

    try:
        progress = tqdm(
            range(episodes),
            desc="Collecting demonstrations",
            unit="trajectory",
        )
        for trajectory_index in progress:
            if trajectory_index in entries:
                continue
            selected: dict[str, np.ndarray] | None = None
            for attempt in range(max_attempts):
                episode_seed = (
                    seed + trajectory_index * max_attempts + attempt
                )
                candidate = _rollout(env, episode_seed=episode_seed)
                selected = candidate
                if bool(candidate["success"]):
                    break
            assert selected is not None
            if not bool(selected["success"]):
                LOGGER.warning(
                    "Trajectory %d remained unsuccessful after %d scripted attempts.",
                    trajectory_index,
                    max_attempts,
                )
            destination = output_dir / f"trajectory_{trajectory_index:05d}.npz"
            atomic_save_npz(destination, **selected)
            entries[trajectory_index] = _trajectory_summary(destination)
            successes = sum(bool(item["success"]) for item in entries.values())
            progress.set_postfix(
                success_rate=f"{successes / max(1, len(entries)):.3f}"
            )
    finally:
        env.close()

    ordered_entries = [entries[index] for index in sorted(entries) if index < episodes]
    if len(ordered_entries) != episodes:
        raise RuntimeError(
            f"Expected {episodes} trajectory files, found {len(ordered_entries)}."
        )
    lengths = np.asarray([entry["length"] for entry in ordered_entries], dtype=float)
    returns = np.asarray([entry["return"] for entry in ordered_entries], dtype=float)
    successes = np.asarray(
        [entry["success"] for entry in ordered_entries], dtype=bool
    )
    statistics = {
        "schema_version": SCHEMA_VERSION,
        "episodes_requested": episodes,
        "episodes_collected": len(ordered_entries),
        "successful_trajectories": int(successes.sum()),
        "failed_trajectories": int((~successes).sum()),
        "demo_success_rate": float(successes.mean()),
        "mean_trajectory_length": float(lengths.mean()),
        "std_trajectory_length": float(lengths.std()),
        "min_trajectory_length": int(lengths.min()),
        "max_trajectory_length": int(lengths.max()),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "total_transitions": int(lengths.sum()),
        "backend": backend,
        "env_name": env.env_name,
        "state_mode": state_mode,
        "state_dim": env.state_dim,
        "action_dim": env.action_dim,
        "seed": seed,
        "expert_algorithm": "Sawyer scripted",
        "imitation_target": "ACT",
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_type": "sequential_expert_demonstrations",
        "expert_algorithm": "Sawyer scripted",
        "imitation_target": "ACT",
        "config": str(config_path) if config_path else None,
        "seed": seed,
        "backend": backend,
        "env_name": env.env_name,
        "protocol": {
            "max_episode_steps": max_steps,
            "max_attempts_per_trajectory": max_attempts,
        },
        "state_metadata": env.state_metadata,
        "trajectory_schema": {
            "states": ["T", env.state_dim],
            "raw_observations": ["T", 39],
            "actions": ["T", env.action_dim],
            "executed_actions": ["T", env.action_dim],
            "rewards": ["T"],
            "next_states": ["T", env.state_dim],
            "success": [],
        },
        "statistics": statistics,
        "trajectories": ordered_entries,
    }
    atomic_write_json(output_dir / "statistics.json", statistics)
    atomic_write_json(output_dir / "manifest.json", manifest)
    LOGGER.info(
        "Collected %d trajectories (%d transitions); demo_success_rate=%.4f",
        episodes,
        statistics["total_transitions"],
        statistics["demo_success_rate"],
    )
    print(f"demo_success_rate={statistics['demo_success_rate']:.6f}")
    return statistics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect sequential Sawyer scripted demonstrations for REIM/ACT."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "environment.yaml"),
        help="Environment YAML (default: configs/environment.yaml).",
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--backend",
        choices=("metaworld", "toy", "auto"),
        default=None,
        help="Use toy only for explicit CI smoke tests.",
    )
    parser.add_argument("--state-mode", choices=("semantic", "raw"), default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--render-mode", choices=("human", "rgb_array"), default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep valid existing trajectory_NNNNN.npz files and collect missing ones.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing trajectory files in the selected output directory.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive.")
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    collect(args)


if __name__ == "__main__":
    main()
