#!/usr/bin/env python3
"""Generate noisy rollouts with causal histories and prospective failure targets."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.failure_labels import (
    CausalFailureLabeler,
    build_causal_windows,
    build_prospective_failure_targets,
)
from data.io import atomic_save_npz, atomic_write_json, file_sha256
from env.metaworld_pickplace import (
    REIMPickPlaceEnv,
    load_project_config,
    make_scripted_expert,
)


LOGGER = logging.getLogger("reim.generate_failures")
SCHEMA_VERSION = "reim-failures-v2"


class Policy(Protocol):
    def reset(self) -> None: ...

    def act(self, state: np.ndarray) -> np.ndarray: ...


class _ScriptedPolicy:
    """Adapter used only for dependency-light pipeline smoke tests."""

    def __init__(self, env: REIMPickPlaceEnv) -> None:
        self._expert = make_scripted_expert(env)
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.policy_type = "Sawyer scripted"

    def reset(self) -> None:
        self._expert.reset()

    def act(self, state: np.ndarray) -> np.ndarray:
        return self._expert.act(state)


class _LearnedPolicy:
    """Adapter preserving ACT's per-episode temporal ensemble lifecycle."""

    def __init__(self, checkpoint: Path, device: str) -> None:
        try:
            from models.bc_policy import load_bc_policy
        except ImportError as exc:
            raise RuntimeError(
                "Loading an imitation checkpoint requires PyTorch and models.bc_policy."
            ) from exc
        self._model = load_bc_policy(checkpoint, map_location=device)
        self.state_dim = int(getattr(self._model, "state_dim"))
        self.action_dim = int(getattr(self._model, "action_dim"))
        self.policy_type = str(getattr(self._model, "policy_type", "ACT"))

    def reset(self) -> None:
        reset = getattr(self._model, "reset", None)
        if callable(reset):
            reset()

    def act(self, state: np.ndarray) -> np.ndarray:
        act = getattr(self._model, "act", None)
        if not callable(act):
            act = getattr(self._model, "predict", None)
        if not callable(act):
            raise TypeError("Loaded imitation policy exposes neither act() nor predict().")
        try:
            result = act(state, deterministic=True)
        except TypeError:
            result = act(state)
        if isinstance(result, tuple):
            result = result[0]
        action = np.asarray(result, dtype=np.float32).squeeze()
        return action


def _section(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _resolve_device(value: str) -> str:
    if value.lower() != "auto":
        return value
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _trajectory_summary(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        labels = np.asarray(archive["labels"], dtype=np.uint8)
        event_labels = np.asarray(
            archive["event_labels"] if "event_labels" in archive else labels,
            dtype=np.uint8,
        )
        rewards = np.asarray(archive["rewards"], dtype=np.float32)
        success = bool(np.asarray(archive["success"]).reshape(-1)[0])
        episode_seed = int(np.asarray(archive["episode_seed"]).reshape(-1)[0])
        state_dim = int(np.asarray(archive["states"]).shape[-1])
        action_dim = int(np.asarray(archive["actions"]).shape[-1])
        disturbed = bool(
            np.asarray(archive["object_disturbance_applied"], dtype=bool).any()
        )
        schema_version = str(
            np.asarray(
                archive["schema_version"] if "schema_version" in archive else ""
            ).reshape(-1)[0]
        )
        forecast_horizon = int(
            np.asarray(
                archive["forecast_horizon"]
                if "forecast_horizon" in archive
                else -1
            ).reshape(-1)[0]
        )
    return {
        "file": path.name,
        "length": int(len(labels)),
        "positive_labels": int(labels.sum()),
        "positive_event_labels": int(event_labels.sum()),
        "prospective_label_rate": float(labels.mean()) if len(labels) else 0.0,
        "failure_event_rate": (
            float(event_labels.mean()) if len(event_labels) else 0.0
        ),
        "return": float(rewards.sum(dtype=np.float64)),
        "success": success,
        "object_disturbed": disturbed,
        "episode_seed": episode_seed,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "schema_version": schema_version,
        "forecast_horizon": forecast_horizon,
        "sha256": file_sha256(path),
    }


def _rollout(
    env: REIMPickPlaceEnv,
    policy: Policy,
    *,
    episode_seed: int,
    sequence_length: int,
    horizon: int,
) -> dict[str, np.ndarray]:
    state, reset_info = env.reset(seed=episode_seed)
    policy.reset()  # Required for ACT temporal-ensemble chunk history.
    labeler = CausalFailureLabeler(progress_window=sequence_length)
    labeler.reset(reset_info)

    policy_states: list[np.ndarray] = []
    states: list[np.ndarray] = []
    raw_observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    rewards: list[float] = []
    event_labels: list[int] = []
    event_reasons: list[str] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    success_flags: list[bool] = []
    distances: list[float] = []
    disturbance_flags: list[bool] = []
    disturbance_deltas: list[np.ndarray] = []

    for _ in range(env.max_episode_steps):
        policy_state = np.asarray(state, dtype=np.float32)
        action = np.asarray(policy.act(policy_state), dtype=np.float32).reshape(-1)
        if action.shape != env.action_space.shape:
            raise RuntimeError(
                f"Policy produced action {action.shape}; expected {env.action_space.shape}."
            )
        next_state, reward, terminated, truncated, info = env.step(action)
        label, reason = labeler.update(
            info, terminated=terminated, truncated=truncated
        )

        policy_states.append(policy_state.copy())
        # Detector input is the newly observed state. Online event annotation
        # uses only this and prior transitions; prospective targets are built
        # after the rollout without adding future states to the input window.
        states.append(np.asarray(next_state, dtype=np.float32).copy())
        raw_observations.append(env.raw_observation)
        actions.append(action.copy())
        executed_actions.append(
            np.asarray(info["executed_action"], dtype=np.float32).copy()
        )
        rewards.append(float(reward))
        event_labels.append(int(label))
        event_reasons.append(reason)
        terminated_flags.append(bool(terminated))
        truncated_flags.append(bool(truncated))
        success_flags.append(bool(info["success"]))
        distances.append(float(info["distance_to_goal"]))
        disturbance_flags.append(bool(info["object_disturbance_applied"]))
        disturbance_deltas.append(
            np.asarray(info["object_disturbance_delta"], dtype=np.float32)
        )
        state = next_state
        if terminated or truncated:
            break

    if not states:
        raise RuntimeError("Environment produced an empty failure rollout.")
    state_array = np.stack(states).astype(np.float32)
    windows, valid_lengths = build_causal_windows(state_array, sequence_length)
    labels, event_offsets, target_reasons = build_prospective_failure_targets(
        np.asarray(event_labels, dtype=np.uint8),
        event_reasons,
        horizon=horizon,
    )
    return {
        "states": state_array,
        "policy_states": np.stack(policy_states).astype(np.float32),
        "raw_observations": np.stack(raw_observations).astype(np.float32),
        "windows": windows,
        "valid_lengths": valid_lengths,
        "actions": np.stack(actions).astype(np.float32),
        "executed_actions": np.stack(executed_actions).astype(np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "labels": labels,
        "failure_labels": labels,
        "failure_reasons": target_reasons,
        "steps_to_failure_event": event_offsets,
        "event_labels": np.asarray(event_labels, dtype=np.uint8),
        "event_reasons": np.asarray(event_reasons, dtype="<U32"),
        "forecast_horizon": np.asarray(horizon, dtype=np.int16),
        "terminated": np.asarray(terminated_flags, dtype=np.bool_),
        "truncated": np.asarray(truncated_flags, dtype=np.bool_),
        "step_success": np.asarray(success_flags, dtype=np.bool_),
        "distance_to_goal": np.asarray(distances, dtype=np.float32),
        "object_disturbance_applied": np.asarray(
            disturbance_flags, dtype=np.bool_
        ),
        "object_disturbance_delta": np.stack(disturbance_deltas).astype(np.float32),
        "success": np.asarray(any(success_flags), dtype=np.bool_),
        "failure": np.asarray(any(event_labels), dtype=np.bool_),
        "episode_seed": np.asarray(episode_seed, dtype=np.int64),
        "backend": np.asarray(env.backend),
        "env_name": np.asarray(env.env_name),
        "state_mode": np.asarray(env.state_mode),
        "label_policy": np.asarray("failure_event_within_horizon"),
        "schema_version": np.asarray(SCHEMA_VERSION),
    }


def generate(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _resolve_path(args.config) if args.config else None
    config = load_project_config(config_path)
    env_config = _section(config, "environment")
    failure_config = _section(config, "failure_dataset")
    project_config = _section(config, "project")

    episodes = int(
        args.episodes
        if args.episodes is not None
        else failure_config.get("episodes", 2000)
    )
    if episodes <= 0:
        raise ValueError("--episodes must be positive.")
    sequence_length = int(
        args.sequence_length
        if args.sequence_length is not None
        else failure_config.get("sequence_length", 10)
    )
    if sequence_length <= 0:
        raise ValueError("--sequence-length must be positive.")
    horizon = int(
        args.horizon
        if args.horizon is not None
        else failure_config.get("horizon", 10)
    )
    if horizon < 0:
        raise ValueError("--horizon must be non-negative.")
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
    action_noise_std = float(
        args.action_noise_std
        if args.action_noise_std is not None
        else failure_config.get("action_noise_std", 0.15)
    )
    observation_noise_std = float(
        args.observation_noise_std
        if args.observation_noise_std is not None
        else failure_config.get("observation_noise_std", 0.01)
    )
    object_noise_probability = float(
        args.object_noise_probability
        if args.object_noise_probability is not None
        else failure_config.get("object_noise_probability", 0.02)
    )
    object_noise_magnitude = float(
        args.object_noise_magnitude
        if args.object_noise_magnitude is not None
        else failure_config.get("object_noise_magnitude", 0.04)
    )
    object_noise_std = float(
        args.object_noise_std if args.object_noise_std is not None else 0.0
    )
    output_dir = _resolve_path(
        args.output_dir
        if args.output_dir is not None
        else failure_config.get("save_dir", "datasets/failures")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(output_dir.glob("failure_*.npz"))
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
            f"{output_dir} already has {len(existing)} failure rollouts. "
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
                "rollouts lie outside the requested range and would still be loaded "
                "by training. Keep the original episode count or use a new output "
                "directory."
            )

    env = REIMPickPlaceEnv(
        config=config,
        seed=seed,
        backend=backend,
        render_mode=args.render_mode,
        state_mode=state_mode,
        max_episode_steps=max_steps,
        action_noise_std=action_noise_std,
        observation_noise_std=observation_noise_std,
        object_noise_probability=object_noise_probability,
        object_noise_magnitude=object_noise_magnitude,
        object_noise_std=object_noise_std,
    )

    checkpoint: Path | None = None
    if args.policy == "scripted":
        policy: Policy = _ScriptedPolicy(env)
    else:
        checkpoint = _resolve_path(args.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Imitation checkpoint does not exist: {checkpoint}. "
                "Train ACT first or use --policy scripted for an explicit CI smoke test."
            )
        policy = _LearnedPolicy(checkpoint, _resolve_device(args.device))
    if int(getattr(policy, "state_dim")) != env.state_dim:
        raise ValueError(
            f"Policy expects state_dim={getattr(policy, 'state_dim')}, "
            f"but environment state_mode={state_mode!r} emits {env.state_dim}."
        )
    if int(getattr(policy, "action_dim")) != env.action_dim:
        raise ValueError(
            f"Policy expects action_dim={getattr(policy, 'action_dim')}, "
            f"but environment emits {env.action_dim}."
        )

    if existing and args.resume:
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(
                "Cannot resume failure rollouts without manifest.json provenance. "
                "Use --overwrite to start a new protocol."
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            previous_manifest = json.load(handle)
        previous_statistics = previous_manifest.get("statistics", {})
        previous_policy = previous_manifest.get("policy", {})
        expected = {
            "schema_version": (previous_manifest.get("schema_version"), SCHEMA_VERSION),
            "seed": (previous_manifest.get("seed"), seed),
            "backend": (previous_manifest.get("backend"), backend),
            "env_name": (previous_manifest.get("env_name"), env.env_name),
            "state_mode": (previous_statistics.get("state_mode"), state_mode),
            "sequence_length": (
                previous_statistics.get("sequence_length"),
                sequence_length,
            ),
            "forecast_horizon": (
                previous_statistics.get("forecast_horizon"),
                horizon,
            ),
            "policy_checkpoint_sha256": (
                previous_policy.get("checkpoint_sha256"),
                file_sha256(checkpoint) if checkpoint else None,
            ),
        }
        mismatches = [
            f"{key}: stored={stored!r}, requested={requested!r}"
            for key, (stored, requested) in expected.items()
            if stored != requested
        ]
        previous_noise = previous_statistics.get("noise", {})
        requested_noise = {
            "action_noise_std": action_noise_std,
            "observation_noise_std": observation_noise_std,
            "object_noise_probability": object_noise_probability,
            "object_noise_magnitude": object_noise_magnitude,
            "object_noise_std": object_noise_std,
        }
        for key, requested in requested_noise.items():
            stored = previous_noise.get(key)
            if stored is None or not np.isclose(float(stored), requested):
                mismatches.append(
                    f"{key}: stored={stored!r}, requested={requested!r}"
                )
        previous_protocol = previous_manifest.get("protocol")
        if isinstance(previous_protocol, Mapping) and (
            previous_protocol.get("max_episode_steps") != max_steps
        ):
            mismatches.append(
                "max_episode_steps: "
                f"stored={previous_protocol.get('max_episode_steps')!r}, "
                f"requested={max_steps!r}"
            )
        if mismatches:
            raise ValueError(
                "Cannot resume failure rollouts with a different protocol: "
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
            if (
                summary["schema_version"] != SCHEMA_VERSION
                or summary["forecast_horizon"] != horizon
            ):
                raise ValueError(
                    f"Cannot resume: {path.name} uses schema="
                    f"{summary['schema_version']!r}, horizon="
                    f"{summary['forecast_horizon']}; expected schema="
                    f"{SCHEMA_VERSION!r}, horizon={horizon}. Use --overwrite."
                )
            entries[index] = summary

    try:
        progress = tqdm(
            range(episodes), desc="Generating failure data", unit="episode"
        )
        for episode_index in progress:
            if episode_index in entries:
                continue
            episode_seed = seed + episode_index
            trajectory = _rollout(
                env,
                policy,
                episode_seed=episode_seed,
                sequence_length=sequence_length,
                horizon=horizon,
            )
            destination = output_dir / f"failure_{episode_index:05d}.npz"
            atomic_save_npz(destination, **trajectory)
            entries[episode_index] = _trajectory_summary(destination)
            positives = sum(item["positive_labels"] for item in entries.values())
            samples = sum(item["length"] for item in entries.values())
            progress.set_postfix(
                positive_rate=f"{positives / max(1, samples):.3f}"
            )
    finally:
        env.close()

    ordered_entries = [entries[index] for index in sorted(entries) if index < episodes]
    if len(ordered_entries) != episodes:
        raise RuntimeError(
            f"Expected {episodes} failure rollouts, found {len(ordered_entries)}."
        )
    lengths = np.asarray([entry["length"] for entry in ordered_entries], dtype=float)
    returns = np.asarray([entry["return"] for entry in ordered_entries], dtype=float)
    positives = int(sum(entry["positive_labels"] for entry in ordered_entries))
    positive_events = int(
        sum(entry["positive_event_labels"] for entry in ordered_entries)
    )
    total_samples = int(lengths.sum())
    successes = int(sum(bool(entry["success"]) for entry in ordered_entries))
    disturbed = int(
        sum(bool(entry["object_disturbed"]) for entry in ordered_entries)
    )
    statistics = {
        "schema_version": SCHEMA_VERSION,
        "episodes_requested": episodes,
        "episodes_collected": len(ordered_entries),
        "total_samples": total_samples,
        "positive_labels": positives,
        "negative_labels": total_samples - positives,
        "positive_label_rate": positives / max(1, total_samples),
        "positive_event_labels": positive_events,
        "failure_event_rate": positive_events / max(1, total_samples),
        "episode_success_rate": successes / episodes,
        "episodes_with_object_disturbance": disturbed,
        "object_disturbance_episode_rate": disturbed / episodes,
        "mean_episode_length": float(lengths.mean()),
        "std_episode_length": float(lengths.std()),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "backend": backend,
        "env_name": env.env_name,
        "state_mode": state_mode,
        "state_dim": env.state_dim,
        "action_dim": env.action_dim,
        "sequence_length": sequence_length,
        "forecast_horizon": horizon,
        "seed": seed,
        "policy_type": str(getattr(policy, "policy_type", type(policy).__name__)),
        "causal_features": True,
        "target": "failure_event_within_horizon",
        "noise": {
            "action_noise_std": action_noise_std,
            "observation_noise_std": observation_noise_std,
            "object_noise_probability": object_noise_probability,
            "object_noise_magnitude": object_noise_magnitude,
            "object_noise_std": object_noise_std,
        },
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_type": "prospective_failure_prediction",
        "config": str(config_path) if config_path else None,
        "seed": seed,
        "backend": backend,
        "env_name": env.env_name,
        "protocol": {"max_episode_steps": max_steps},
        "state_metadata": env.state_metadata,
        "policy": {
            "type": statistics["policy_type"],
            "checkpoint": str(checkpoint) if checkpoint else None,
            "checkpoint_sha256": file_sha256(checkpoint) if checkpoint else None,
            "episode_reset_called": True,
            "act_interface": (
                "temporal_ensemble" if checkpoint else "scripted_state_machine"
            ),
        },
        "labeling": {
            "causal_features": True,
            "target": "failure_event_within_horizon",
            "forecast_horizon": horizon,
            "target_interval": "[t, t + forecast_horizon]",
            "feature_interval": "[max(0, t - sequence_length + 1), t]",
            "sequence_length": sequence_length,
            "padding": "right_zero",
            "valid_frames": "windows[i, :valid_lengths[i]]",
            "rules": [
                "observed_object_drop",
                "workspace_violation",
                "failed_grasp_after_contact",
                "post_contact_no_progress",
                "post_contact_trajectory_deviation",
                "unsuccessful_terminal_or_timeout",
            ],
        },
        "trajectory_schema": {
            "states": ["T", env.state_dim],
            "policy_states": ["T", env.state_dim],
            "windows": ["T", sequence_length, env.state_dim],
            "valid_lengths": ["T"],
            "actions": ["T", env.action_dim],
            "labels": ["T"],
            "failure_reasons": ["T"],
            "event_labels": ["T"],
            "event_reasons": ["T"],
            "steps_to_failure_event": ["T"],
            "forecast_horizon": [],
        },
        "statistics": statistics,
        "trajectories": ordered_entries,
    }
    atomic_write_json(output_dir / "statistics.json", statistics)
    atomic_write_json(output_dir / "manifest.json", manifest)
    LOGGER.info(
        "Generated %d prospective detector samples; positive_label_rate=%.4f",
        total_samples,
        statistics["positive_label_rate"],
    )
    print(f"failure_positive_rate={statistics['positive_label_rate']:.6f}")
    return statistics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate noisy ACT rollouts with causal history windows and "
            "prospective failure-event targets."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "environment.yaml"),
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--backend", choices=("metaworld", "toy", "auto"), default=None
    )
    parser.add_argument("--state-mode", choices=("semantic", "raw"), default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Inclusive future failure-event forecast horizon (config default: 10).",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--policy",
        choices=("checkpoint", "scripted"),
        default="checkpoint",
        help="scripted is intended only for a lightweight data-pipeline smoke test.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/bc_policy.pt",
        help="ACT/BC-compatible imitation checkpoint.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--action-noise-std", type=float, default=None)
    parser.add_argument("--observation-noise-std", type=float, default=None)
    parser.add_argument("--object-noise-probability", type=float, default=None)
    parser.add_argument("--object-noise-magnitude", type=float, default=None)
    parser.add_argument("--object-noise-std", type=float, default=None)
    parser.add_argument("--render-mode", choices=("human", "rgb_array"), default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
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
    generate(args)


if __name__ == "__main__":
    main()
