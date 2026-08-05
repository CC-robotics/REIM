#!/usr/bin/env python3
"""Collect balanced MT10/MT50 scripted-expert demonstrations.

This collector intentionally does not depend on the PickPlace wrapper.  It
uses Meta-World's benchmark classes and ``ENV_POLICY_MAP`` directly so every
task is paired with its official scripted expert.  Expert policies consume the
native 39-D observation, while the saved policy state appends an ordered task
one-hot vector (49-D for MT10 and 89-D for MT50).

Only successful trajectories are committed.  Failed attempts are recorded in
the manifest and collection continues until every task has the requested,
equal number of demonstrations or the per-trajectory attempt budget is
exhausted.  Files and metadata are written atomically for safe resumption.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.io import atomic_save_npz, atomic_write_json, file_sha256


LOGGER = logging.getLogger("reim.collect_multitask_demonstrations")
SCHEMA_VERSION = "reim-multitask-demonstrations-v1"
DATASET_TYPE = "balanced_multitask_scripted_expert_demonstrations"
RAW_OBSERVATION_DIM = 39
ACTION_DIM = 4
MAX_EPISODE_STEPS = 500
MAX_EPISODES_PER_TASK = 500
OFFICIAL_GOALS_PER_TASK = 50
SUPPORTED_BENCHMARKS = {"MT10": 10, "MT50": 50}
REQUIRED_TRAJECTORY_KEYS = frozenset(
    {
        "states",
        "raw_observations",
        "actions",
        "rewards",
        "success",
        "task_id",
        "task_name",
        "task_variant",
        "seed",
    }
)


@dataclass(frozen=True)
class MetaWorldComponents:
    """Lazy-loaded Meta-World objects, also providing a unit-test seam."""

    module: Any
    policy_map: Mapping[str, type]
    version: str


def _load_metaworld_components() -> MetaWorldComponents:
    try:
        import metaworld
        from metaworld.policies import ENV_POLICY_MAP
    except ImportError as exc:  # pragma: no cover - exercised on deployment hosts
        raise RuntimeError(
            "Meta-World is required for MT10/MT50 collection. Run setup.sh in "
            "the project environment before launching this script."
        ) from exc

    try:
        version = importlib.metadata.version("metaworld")
    except importlib.metadata.PackageNotFoundError:
        version = str(getattr(metaworld, "__version__", "unknown"))
    return MetaWorldComponents(
        module=metaworld,
        policy_map=dict(ENV_POLICY_MAP),
        version=version,
    )


def task_vocabulary_sha256(task_vocabulary: Sequence[str]) -> str:
    """Hash an ordered vocabulary using a stable, unambiguous encoding."""

    serialized = json.dumps(
        list(task_vocabulary),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _episode_seed(
    base_seed: int,
    task_id: int,
    trajectory_index: int,
    attempt_index: int,
) -> int:
    """Derive a deterministic 31-bit seed independent of Python hash state."""

    material = (
        f"{base_seed}:{task_id}:{trajectory_index}:{attempt_index}"
    ).encode("ascii")
    value = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return value % (2**31 - 1)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not cleaned:
        raise ValueError(f"Task name {value!r} has no filesystem-safe characters.")
    return cleaned


def _scalar_bool(value: Any) -> bool:
    array = np.asarray(value)
    return bool(array.reshape(-1)[0]) if array.size else False


def _reset_observation(env: Any, episode_seed: int) -> np.ndarray:
    seed_method = getattr(env, "seed", None)
    if callable(seed_method):
        seed_method(episode_seed)
    result = env.reset(seed=episode_seed)
    observation = result[0] if isinstance(result, tuple) else result
    return _validate_raw_observation(observation)


def _validate_raw_observation(observation: Any) -> np.ndarray:
    raw = np.asarray(observation, dtype=np.float32).reshape(-1)
    if raw.shape != (RAW_OBSERVATION_DIM,):
        raise RuntimeError(
            "Meta-World observation contract changed: expected raw shape "
            f"({RAW_OBSERVATION_DIM},), got {raw.shape}."
        )
    if not np.isfinite(raw).all():
        raise RuntimeError("Meta-World returned a non-finite observation.")
    return raw


def _rollout(
    env: Any,
    expert: Any,
    task: Any,
    *,
    task_id: int,
    task_count: int,
    task_name: str,
    task_variant: int,
    episode_seed: int,
    attempt_index: int,
    trajectory_index: int,
    benchmark_name: str,
    max_steps: int,
) -> dict[str, np.ndarray]:
    """Run one task variant and return a pickle-free trajectory payload."""

    env.set_task(task)
    raw = _reset_observation(env, episode_seed)
    one_hot = np.zeros(task_count, dtype=np.float32)
    one_hot[task_id] = 1.0

    states: list[np.ndarray] = []
    raw_observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    success_flags: list[bool] = []

    for _ in range(max_steps):
        # Official scripted experts are defined on the unwrapped 39-D state.
        proposed = np.asarray(
            expert.get_action(raw.astype(np.float64, copy=False)),
            dtype=np.float32,
        ).reshape(-1)
        if proposed.shape != (ACTION_DIM,):
            raise RuntimeError(
                f"Expert for {task_name} returned {proposed.shape}; expected "
                f"({ACTION_DIM},)."
            )
        if not np.isfinite(proposed).all():
            raise RuntimeError(f"Expert for {task_name} returned a non-finite action.")
        # ACT supervision and the environment receive exactly the same command.
        action = np.clip(proposed, -1.0, 1.0).astype(np.float32, copy=False)
        transition = env.step(action)
        if not isinstance(transition, tuple) or len(transition) != 5:
            raise RuntimeError(
                "Meta-World must implement the Gymnasium five-value step contract."
            )
        next_observation, reward, terminated, truncated, info = transition
        success = _scalar_bool(dict(info).get("success", False))

        states.append(np.concatenate((raw, one_hot)).astype(np.float32))
        raw_observations.append(raw.copy())
        actions.append(action.copy())
        rewards.append(float(reward))
        terminated_flags.append(bool(terminated))
        truncated_flags.append(bool(truncated))
        success_flags.append(success)

        if success or bool(terminated) or bool(truncated):
            break
        raw = _validate_raw_observation(next_observation)

    if not states:
        raise RuntimeError(f"Environment {task_name} produced an empty trajectory.")
    successful = any(success_flags)
    return {
        "states": np.stack(states).astype(np.float32),
        "raw_observations": np.stack(raw_observations).astype(np.float32),
        "actions": np.stack(actions).astype(np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminated": np.asarray(terminated_flags, dtype=np.bool_),
        "truncated": np.asarray(truncated_flags, dtype=np.bool_),
        "step_success": np.asarray(success_flags, dtype=np.bool_),
        "success": np.asarray(successful, dtype=np.bool_),
        "task_id": np.asarray(task_id, dtype=np.int64),
        "task_name": np.asarray(task_name),
        "task_variant": np.asarray(task_variant, dtype=np.int64),
        "seed": np.asarray(episode_seed, dtype=np.int64),
        "trajectory_index": np.asarray(trajectory_index, dtype=np.int64),
        "attempt_index": np.asarray(attempt_index, dtype=np.int64),
        "benchmark": np.asarray(benchmark_name),
        "schema_version": np.asarray(SCHEMA_VERSION),
    }


def _trajectory_summary(
    path: Path,
    *,
    task_vocabulary: Sequence[str],
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_TRAJECTORY_KEYS.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing keys: {sorted(missing)}")
        states = np.asarray(archive["states"], dtype=np.float32)
        raw = np.asarray(archive["raw_observations"], dtype=np.float32)
        actions = np.asarray(archive["actions"], dtype=np.float32)
        rewards = np.asarray(archive["rewards"], dtype=np.float32).reshape(-1)
        success = _scalar_bool(archive["success"])
        task_id = int(np.asarray(archive["task_id"]).reshape(-1)[0])
        task_name = str(np.asarray(archive["task_name"]).reshape(-1)[0])
        task_variant = int(np.asarray(archive["task_variant"]).reshape(-1)[0])
        episode_seed = int(np.asarray(archive["seed"]).reshape(-1)[0])
        if "trajectory_index" not in archive or "attempt_index" not in archive:
            raise ValueError(f"{path} lacks resume indices.")
        trajectory_index = int(
            np.asarray(archive["trajectory_index"]).reshape(-1)[0]
        )
        attempt_index = int(np.asarray(archive["attempt_index"]).reshape(-1)[0])

    task_count = len(task_vocabulary)
    if not 0 <= task_id < task_count or task_vocabulary[task_id] != task_name:
        raise ValueError(
            f"{path} has inconsistent task_id/task_name metadata: "
            f"{task_id}/{task_name!r}."
        )
    expected_state_dim = RAW_OBSERVATION_DIM + task_count
    if states.ndim != 2 or states.shape[1] != expected_state_dim:
        raise ValueError(
            f"{path} states must be [T,{expected_state_dim}], got {states.shape}."
        )
    if raw.shape != (states.shape[0], RAW_OBSERVATION_DIM):
        raise ValueError(f"{path} raw observations have invalid shape {raw.shape}.")
    if actions.shape != (states.shape[0], ACTION_DIM):
        raise ValueError(f"{path} actions have invalid shape {actions.shape}.")
    if rewards.shape != (states.shape[0],):
        raise ValueError(f"{path} rewards have invalid shape {rewards.shape}.")
    if not success:
        raise ValueError(f"{path} is not a successful collect-until-success sample.")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError(f"{path} contains non-finite state/action values.")
    if np.any(actions < -1.0) or np.any(actions > 1.0):
        raise ValueError(f"{path} contains action targets outside [-1,1].")
    expected_one_hot = np.zeros(task_count, dtype=np.float32)
    expected_one_hot[task_id] = 1.0
    expected_task_block = np.broadcast_to(
        expected_one_hot, (states.shape[0], task_count)
    )
    if not np.array_equal(states[:, RAW_OBSERVATION_DIM:], expected_task_block):
        raise ValueError(f"{path} has an invalid task one-hot block.")
    if not np.array_equal(states[:, :RAW_OBSERVATION_DIM], raw):
        raise ValueError(f"{path} states do not preserve raw_observations exactly.")

    return {
        "file": path.name,
        "sha256": file_sha256(path),
        "task_id": task_id,
        "task_name": task_name,
        "task_variant": task_variant,
        "trajectory_index": trajectory_index,
        "attempt_index": attempt_index,
        "seed": episode_seed,
        "length": int(states.shape[0]),
        "return": float(rewards.sum(dtype=np.float64)),
        "success": True,
    }


def _manifest_payload(
    *,
    benchmark_name: str,
    seed: int,
    episodes_per_task: int,
    max_attempts: int,
    max_steps: int,
    metaworld_version: str,
    task_vocabulary: Sequence[str],
    entries: Mapping[tuple[int, int], Mapping[str, Any]],
    progress: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    task_count = len(task_vocabulary)
    ordered_entries = [
        dict(entry)
        for _, entry in sorted(
            entries.items(), key=lambda item: (item[0][0], item[0][1])
        )
    ]
    per_task_yield: dict[str, dict[str, Any]] = {}
    for task_id, task_name in enumerate(task_vocabulary):
        task_entries = [
            item for item in ordered_entries if int(item["task_id"]) == task_id
        ]
        task_progress = dict(progress.get(task_name, {}))
        attempts = int(task_progress.get("attempts", len(task_entries)))
        successes = len(task_entries)
        per_task_yield[task_name] = {
            "task_id": task_id,
            "requested_successful_trajectories": episodes_per_task,
            "successful_trajectories": successes,
            "attempts": attempts,
            "discarded_failed_attempts": int(
                task_progress.get("discarded_failed_attempts", max(0, attempts - successes))
            ),
            "success_yield": float(successes / attempts) if attempts else 0.0,
            "total_transitions": int(sum(int(item["length"]) for item in task_entries)),
        }
    complete = all(
        item["successful_trajectories"] == episodes_per_task
        for item in per_task_yield.values()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_type": DATASET_TYPE,
        "benchmark": benchmark_name,
        "seed": seed,
        "metaworld_version": metaworld_version,
        "task_vocabulary": list(task_vocabulary),
        "task_vocabulary_sha256": task_vocabulary_sha256(task_vocabulary),
        "task_count": task_count,
        "complete": complete,
        "protocol": {
            "episodes_per_task": episodes_per_task,
            "collect_until_success": True,
            "max_attempts_per_successful_trajectory": max_attempts,
            "max_episode_steps": max_steps,
            "expert_source": "metaworld.policies.ENV_POLICY_MAP",
            "action_target_clip": [-1.0, 1.0],
        },
        "observation_schema": {
            "raw_observations": {
                "shape": ["T", RAW_OBSERVATION_DIM],
                "description": "Native fully-observable Meta-World observation",
            },
            "states": {
                "shape": ["T", RAW_OBSERVATION_DIM + task_count],
                "description": "raw_observations concatenated with ordered task one-hot",
            },
            "actions": {"shape": ["T", ACTION_DIM], "range": [-1.0, 1.0]},
        },
        "trajectory_schema": {
            "required_keys": sorted(REQUIRED_TRAJECTORY_KEYS),
            "scalar_metadata": [
                "success",
                "task_id",
                "task_name",
                "task_variant",
                "seed",
            ],
        },
        "per_task_yield": per_task_yield,
        "collection_progress": {
            name: dict(values) for name, values in progress.items()
        },
        "statistics": {
            "successful_trajectories": len(ordered_entries),
            "requested_successful_trajectories": task_count * episodes_per_task,
            "total_attempts": int(
                sum(int(item["attempts"]) for item in per_task_yield.values())
            ),
            "discarded_failed_attempts": int(
                sum(
                    int(item["discarded_failed_attempts"])
                    for item in per_task_yield.values()
                )
            ),
            "total_transitions": int(
                sum(int(item["length"]) for item in ordered_entries)
            ),
        },
        "trajectories": ordered_entries,
    }


def _validate_arguments(args: argparse.Namespace) -> tuple[str, int, int, int, int]:
    benchmark_name = str(args.benchmark).upper()
    if benchmark_name not in SUPPORTED_BENCHMARKS:
        raise ValueError("--benchmark must be MT10 or MT50.")
    episodes_per_task = int(args.episodes_per_task)
    if not 1 <= episodes_per_task <= MAX_EPISODES_PER_TASK:
        raise ValueError(
            f"--episodes-per-task must be in [1,{MAX_EPISODES_PER_TASK}]."
        )
    seed = int(args.seed)
    max_attempts = int(args.max_attempts)
    if max_attempts <= 0:
        raise ValueError("--max-attempts must be positive.")
    max_steps = int(getattr(args, "max_steps", MAX_EPISODE_STEPS))
    if max_steps != MAX_EPISODE_STEPS:
        raise ValueError(
            f"The official MT10/MT50 episode horizon is fixed at "
            f"{MAX_EPISODE_STEPS} steps."
        )
    return benchmark_name, episodes_per_task, seed, max_attempts, max_steps


def _resolve_output(args: argparse.Namespace, benchmark_name: str) -> Path:
    requested = getattr(args, "output", None)
    if requested is None:
        path = (
            PROJECT_ROOT
            / "datasets"
            / "demonstrations"
            / "multitask"
            / benchmark_name.lower()
        )
    else:
        path = Path(requested).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    return path.resolve()


def _guard_against_pickplace_overwrite(output_dir: Path) -> None:
    legacy_trajectories = sorted(output_dir.glob("trajectory_*.npz"))
    if legacy_trajectories:
        raise FileExistsError(
            f"Refusing to use {output_dir}: it contains legacy PickPlace "
            "trajectory_*.npz files. Select a dedicated MT10/MT50 output directory."
        )
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot validate existing {manifest_path}.") from exc
        if payload.get("dataset_type") != DATASET_TYPE:
            raise FileExistsError(
                f"Refusing to overwrite non-multitask manifest {manifest_path}."
            )


def _validate_resume_manifest(
    manifest: Mapping[str, Any],
    *,
    benchmark_name: str,
    seed: int,
    episodes_per_task: int,
    max_attempts: int,
    max_steps: int,
    metaworld_version: str,
    task_vocabulary: Sequence[str],
) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "dataset_type": DATASET_TYPE,
        "benchmark": benchmark_name,
        "seed": seed,
        "metaworld_version": metaworld_version,
        "task_vocabulary": list(task_vocabulary),
        "task_vocabulary_sha256": task_vocabulary_sha256(task_vocabulary),
    }
    mismatches = [
        f"{key}: stored={manifest.get(key)!r}, requested={value!r}"
        for key, value in expected.items()
        if manifest.get(key) != value
    ]
    protocol = manifest.get("protocol", {})
    if not isinstance(protocol, Mapping):
        mismatches.append("protocol is missing or invalid")
    else:
        stored_episodes = int(protocol.get("episodes_per_task", -1))
        stored_attempts = int(
            protocol.get("max_attempts_per_successful_trajectory", -1)
        )
        if episodes_per_task < stored_episodes:
            mismatches.append(
                f"episodes_per_task cannot decrease ({stored_episodes} -> "
                f"{episodes_per_task})"
            )
        if max_attempts < stored_attempts:
            mismatches.append(
                f"max_attempts cannot decrease ({stored_attempts} -> {max_attempts})"
            )
        if int(protocol.get("max_episode_steps", -1)) != max_steps:
            mismatches.append(
                "max_episode_steps differs from the stored collection protocol"
            )
    if mismatches:
        raise ValueError(
            "Cannot resume with incompatible provenance: " + "; ".join(mismatches)
        )


def collect(
    args: argparse.Namespace,
    *,
    components: MetaWorldComponents | None = None,
) -> dict[str, Any]:
    """Collect a balanced multi-task dataset and return its final manifest."""

    benchmark_name, episodes_per_task, seed, max_attempts, max_steps = (
        _validate_arguments(args)
    )
    components = components or _load_metaworld_components()
    benchmark_type = getattr(components.module, benchmark_name, None)
    if benchmark_type is None:
        raise RuntimeError(f"Installed Meta-World does not provide {benchmark_name}.")
    benchmark = benchmark_type(seed=seed)
    task_vocabulary = tuple(str(name) for name in benchmark.train_classes.keys())
    expected_task_count = SUPPORTED_BENCHMARKS[benchmark_name]
    if len(task_vocabulary) != expected_task_count:
        raise RuntimeError(
            f"{benchmark_name} must contain {expected_task_count} ordered tasks; "
            f"installed Meta-World exposed {len(task_vocabulary)}."
        )
    if len(set(task_vocabulary)) != len(task_vocabulary):
        raise RuntimeError("Meta-World returned duplicate task names.")
    missing_experts = [
        name for name in task_vocabulary if name not in components.policy_map
    ]
    if missing_experts:
        raise RuntimeError(
            "Meta-World scripted expert coverage is incomplete: "
            + ", ".join(missing_experts)
        )

    tasks_by_name: dict[str, list[Any]] = {name: [] for name in task_vocabulary}
    for task in benchmark.train_tasks:
        name = str(task.env_name)
        if name not in tasks_by_name:
            raise RuntimeError(f"Benchmark returned unknown train task {name!r}.")
        tasks_by_name[name].append(task)
    invalid_variant_counts = {
        name: len(tasks)
        for name, tasks in tasks_by_name.items()
        if len(tasks) != OFFICIAL_GOALS_PER_TASK
    }
    if invalid_variant_counts:
        raise RuntimeError(
            "Official Meta-World MT benchmarks must expose exactly "
            f"{OFFICIAL_GOALS_PER_TASK} goal variants per task; got "
            f"{invalid_variant_counts}."
        )

    output_dir = _resolve_output(args, benchmark_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    _guard_against_pickplace_overwrite(output_dir)
    manifest_path = output_dir / "manifest.json"
    prefix = benchmark_name.lower()
    existing_paths = sorted(output_dir.glob(f"{prefix}_task*_trajectory_*.npz"))
    resume = bool(getattr(args, "resume", False))
    previous_manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (existing_paths or previous_manifest is not None) and not resume:
        raise FileExistsError(
            f"{output_dir} already contains a multi-task collection. Use --resume."
        )
    if existing_paths and previous_manifest is None:
        raise ValueError(
            "Cannot safely resume trajectory files without manifest.json provenance."
        )
    if previous_manifest is not None:
        _validate_resume_manifest(
            previous_manifest,
            benchmark_name=benchmark_name,
            seed=seed,
            episodes_per_task=episodes_per_task,
            max_attempts=max_attempts,
            max_steps=max_steps,
            metaworld_version=components.version,
            task_vocabulary=task_vocabulary,
        )

    entries: dict[tuple[int, int], dict[str, Any]] = {}
    for path in existing_paths:
        summary = _trajectory_summary(path, task_vocabulary=task_vocabulary)
        key = (int(summary["task_id"]), int(summary["trajectory_index"]))
        if key in entries:
            raise ValueError(f"Duplicate task/trajectory slot {key} in {output_dir}.")
        if key[1] >= episodes_per_task:
            raise ValueError(
                f"Cannot resume with --episodes-per-task={episodes_per_task}: "
                f"{path.name} lies outside the requested range."
            )
        entries[key] = summary
    for task_id, task_name in enumerate(task_vocabulary):
        indices = sorted(index for tid, index in entries if tid == task_id)
        if indices != list(range(len(indices))):
            raise ValueError(
                f"Existing {task_name} trajectory indices are not contiguous: {indices}."
            )

    previous_progress = (
        previous_manifest.get("collection_progress", {})
        if previous_manifest is not None
        else {}
    )
    progress_state: dict[str, dict[str, int]] = {}
    for task_id, task_name in enumerate(task_vocabulary):
        accepted = sum(1 for tid, _ in entries if tid == task_id)
        previous = previous_progress.get(task_name, {})
        if not isinstance(previous, Mapping):
            previous = {}
        max_saved_attempt = max(
            (
                int(entry["attempt_index"])
                for (tid, _), entry in entries.items()
                if tid == task_id
            ),
            default=-1,
        )
        previous_slot = int(previous.get("current_trajectory_index", accepted))
        progress_state[task_name] = {
            # ``max_saved_attempt + 1`` repairs the narrow interruption window
            # where the NPZ rename completed but the following manifest write
            # did not.
            "attempts": max(
                int(previous.get("attempts", 0)),
                accepted,
                max_saved_attempt + 1,
            ),
            "discarded_failed_attempts": max(
                int(previous.get("discarded_failed_attempts", 0)), 0
            ),
            "next_attempt_index": max(
                int(previous.get("next_attempt_index", 0)), max_saved_attempt + 1
            ),
            "current_trajectory_index": accepted,
            "attempts_for_current_trajectory": (
                int(previous.get("attempts_for_current_trajectory", 0))
                if previous_slot == accepted
                else 0
            ),
        }

    def write_manifest() -> dict[str, Any]:
        payload = _manifest_payload(
            benchmark_name=benchmark_name,
            seed=seed,
            episodes_per_task=episodes_per_task,
            max_attempts=max_attempts,
            max_steps=max_steps,
            metaworld_version=components.version,
            task_vocabulary=task_vocabulary,
            entries=entries,
            progress=progress_state,
        )
        atomic_write_json(manifest_path, payload)
        return payload

    # Establish provenance before the first expensive rollout.
    final_manifest = write_manifest()
    target_total = len(task_vocabulary) * episodes_per_task
    progress_bar = tqdm(
        total=target_total,
        initial=len(entries),
        desc=f"Collecting {benchmark_name} expert demonstrations",
        unit="trajectory",
    )
    try:
        for task_id, task_name in enumerate(task_vocabulary):
            env_type = benchmark.train_classes[task_name]
            env = env_type()
            try:
                variants = tasks_by_name[task_name]
                while sum(1 for tid, _ in entries if tid == task_id) < episodes_per_task:
                    trajectory_index = sum(
                        1 for tid, _ in entries if tid == task_id
                    )
                    state = progress_state[task_name]
                    state["current_trajectory_index"] = trajectory_index
                    used_for_slot = int(state["attempts_for_current_trajectory"])
                    if used_for_slot >= max_attempts:
                        final_manifest = write_manifest()
                        raise RuntimeError(
                            f"{task_name} trajectory {trajectory_index} did not "
                            f"succeed within {max_attempts} attempts. Resume with a "
                            "larger --max-attempts after inspecting the expert rollout."
                        )
                    attempt_index = int(state["next_attempt_index"])
                    # Walk the official 50-goal bank in its benchmark order.
                    # The monotonically increasing attempt index keeps this
                    # deterministic even when collection is resumed.
                    variant_index = attempt_index % len(variants)
                    episode_seed = _episode_seed(
                        seed, task_id, trajectory_index, attempt_index
                    )
                    expert = components.policy_map[task_name]()
                    candidate = _rollout(
                        env,
                        expert,
                        variants[variant_index],
                        task_id=task_id,
                        task_count=len(task_vocabulary),
                        task_name=task_name,
                        task_variant=variant_index,
                        episode_seed=episode_seed,
                        attempt_index=attempt_index,
                        trajectory_index=trajectory_index,
                        benchmark_name=benchmark_name,
                        max_steps=max_steps,
                    )
                    state["attempts"] += 1
                    state["next_attempt_index"] += 1
                    state["attempts_for_current_trajectory"] += 1
                    if not _scalar_bool(candidate["success"]):
                        state["discarded_failed_attempts"] += 1
                        final_manifest = write_manifest()
                        LOGGER.info(
                            "%s trajectory %d attempt %d failed; trying another "
                            "task variant.",
                            task_name,
                            trajectory_index,
                            state["attempts_for_current_trajectory"],
                        )
                        continue

                    destination = output_dir / (
                        f"{prefix}_task{task_id:02d}_{_safe_name(task_name)}_"
                        f"trajectory_{trajectory_index:05d}.npz"
                    )
                    if destination.exists():
                        raise FileExistsError(
                            f"Refusing to overwrite existing trajectory {destination}."
                        )
                    atomic_save_npz(destination, **candidate)
                    summary = _trajectory_summary(
                        destination, task_vocabulary=task_vocabulary
                    )
                    entries[(task_id, trajectory_index)] = summary
                    state["current_trajectory_index"] = trajectory_index + 1
                    state["attempts_for_current_trajectory"] = 0
                    final_manifest = write_manifest()
                    progress_bar.update(1)
                    progress_bar.set_postfix(task=task_name)
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
    finally:
        progress_bar.close()

    final_manifest = write_manifest()
    if not final_manifest["complete"]:
        raise RuntimeError("Collection ended without equal per-task coverage.")
    LOGGER.info(
        "Collected %d successful %s trajectories in %s (%d failed attempts).",
        final_manifest["statistics"]["successful_trajectories"],
        benchmark_name,
        output_dir,
        final_manifest["statistics"]["discarded_failed_attempts"],
    )
    return final_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect balanced official-expert demonstrations for Meta-World "
            "MT10/MT50 using raw39 + task one-hot observations."
        )
    )
    parser.add_argument(
        "--benchmark",
        choices=tuple(SUPPORTED_BENCHMARKS),
        default="MT10",
    )
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output",
        default=None,
        help=(
            "Dedicated output directory. Defaults to "
            "datasets/demonstrations/multitask/<benchmark>."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=10,
        help="Maximum rollout attempts for each required successful trajectory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate provenance and continue an interrupted/increased collection.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    manifest = collect(args)
    print(
        json.dumps(
            {
                "benchmark": manifest["benchmark"],
                "successful_trajectories": manifest["statistics"][
                    "successful_trajectories"
                ],
                "discarded_failed_attempts": manifest["statistics"][
                    "discarded_failed_attempts"
                ],
                "task_vocabulary_sha256": manifest["task_vocabulary_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
