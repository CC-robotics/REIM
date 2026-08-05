#!/usr/bin/env python3
"""Collect deterministic trigger-aligned MT10/MT50 recovery continuations.

The ACT policy and frozen causal detector run online under task-universal
action and observation noise.  At the first detector trigger, the matching
Meta-World scripted expert takes over.  Only successful continuations become
supervised recovery data; the expert is never used by the deployed controller.

Resume is deliberately strict.  A manifest fingerprints the exact task bank,
checkpoints, task vocabulary, disturbance parameters, detector threshold and
collection budget.  Every completed attempt advances a persisted per-task
cursor, including attempts that do not trigger or do not succeed.  Existing
shards are accepted only when their paths, hashes, deterministic seeds,
metadata and task one-hot blocks exactly match the manifest.
"""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.io import atomic_save_npz, atomic_write_json, file_sha256
from models.bc_policy import ACTPolicy
from models.failure_detector import FailureDetector
from utils.common import configure_logging, seed_everything, select_device


LOGGER = logging.getLogger("reim.collect_multitask_recovery")
SCHEMA_VERSION = "reim-multitask-trigger-aligned-recovery-v2"
DATASET_TYPE = "online_detector_triggered_expert_continuations"
RAW_OBSERVATION_DIM = 39
ACTION_DIM = 4
MAX_EPISODE_STEPS = 500
OFFICIAL_GOALS_PER_TASK = 50
SUPPORTED_BENCHMARKS = {"MT10": 10, "MT50": 50}
ATTEMPT_SEED_STRIDE = 1_000_000
RUN_SEED_STRIDE = 100_000_000
REQUIRED_SHARD_KEYS = frozenset(
    {
        "states",
        "raw_observations",
        "actions",
        "rewards",
        "failure_probabilities",
        "success",
        "task_id",
        "task_name",
        "task_variant",
        "task_payload_sha256",
        "trigger_step",
        "trigger_probability",
        "episode_seed",
        "attempt_index",
        "shard_index",
        "protocol_fingerprint_sha256",
        "schema_version",
    }
)


@dataclass(frozen=True)
class MetaWorldComponents:
    """Lazy Meta-World dependencies and a lightweight unit-test seam."""

    module: Any
    policy_map: Mapping[str, type]
    version: str


def _load_metaworld_components() -> MetaWorldComponents:
    try:
        import metaworld
        from metaworld.policies import ENV_POLICY_MAP
    except ImportError as exc:  # pragma: no cover - deployment-only failure path
        raise RuntimeError(
            "Meta-World is required for MT10/MT50 recovery collection."
        ) from exc
    try:
        version = importlib.metadata.version("metaworld")
    except importlib.metadata.PackageNotFoundError:
        version = str(getattr(metaworld, "__version__", "unknown"))
    return MetaWorldComponents(metaworld, dict(ENV_POLICY_MAP), version)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _task_vocabulary_sha256(task_vocabulary: Sequence[str]) -> str:
    return _canonical_sha256(list(task_vocabulary))


def _episode_seed(base_seed: int, task_id: int, attempt_index: int) -> int:
    """Map a task/attempt cursor to a collision-free seed within one run."""

    if not 0 <= attempt_index < ATTEMPT_SEED_STRIDE:
        raise ValueError("attempt_index exceeds the deterministic seed namespace.")
    if not 0 <= task_id < SUPPORTED_BENCHMARKS["MT50"]:
        raise ValueError("task_id exceeds the MT50 seed namespace.")
    return (
        (int(base_seed) % 1_000_000_000) * RUN_SEED_STRIDE
        + task_id * ATTEMPT_SEED_STRIDE
        + attempt_index
    )


def _stream_seed(episode_seed: int, namespace: str) -> int:
    material = f"{episode_seed}:{namespace}".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little") % (
        2**63 - 1
    )


def _condition(raw: np.ndarray, task_id: int, task_count: int) -> np.ndarray:
    raw_array = np.asarray(raw, dtype=np.float32).reshape(-1)
    if raw_array.shape != (RAW_OBSERVATION_DIM,):
        raise RuntimeError(
            f"Expected raw Meta-World observation ({RAW_OBSERVATION_DIM},), "
            f"got {raw_array.shape}."
        )
    if not np.isfinite(raw_array).all():
        raise RuntimeError("Meta-World returned a non-finite observation.")
    task = np.zeros(task_count, dtype=np.float32)
    task[task_id] = 1.0
    return np.concatenate([raw_array, task]).astype(np.float32)


def _risk(
    detector: FailureDetector,
    history: deque[np.ndarray],
    sequence_length: int,
) -> float:
    window = np.zeros((sequence_length, detector.state_dim), dtype=np.float32)
    values = list(history)[-sequence_length:]
    window[: len(values)] = np.stack(values)
    probability = detector.predict_proba(
        window[None, ...], np.asarray([len(values)], dtype=np.int64)
    )
    value = float(probability.detach().cpu().numpy().reshape(-1)[0])
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise RuntimeError(f"Failure detector returned invalid probability {value}.")
    return value


def _array_content_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash array values independently of NPZ zip timestamps/compression."""

    digest = hashlib.sha256()
    for key in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[key]))
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _task_bank(
    benchmark: Any,
    task_vocabulary: Sequence[str],
) -> tuple[dict[str, list[Any]], dict[str, list[str]], str]:
    tasks_by_name: dict[str, list[Any]] = {
        name: [] for name in task_vocabulary
    }
    for task in benchmark.train_tasks:
        task_name = str(task.env_name)
        if task_name not in tasks_by_name:
            raise RuntimeError(f"Benchmark returned unknown task {task_name!r}.")
        tasks_by_name[task_name].append(task)
    invalid_counts = {
        name: len(tasks)
        for name, tasks in tasks_by_name.items()
        if len(tasks) != OFFICIAL_GOALS_PER_TASK
    }
    if invalid_counts:
        raise RuntimeError(
            "Official MT10/MT50 banks require exactly 50 variants per task; "
            f"got {invalid_counts}."
        )
    hashes_by_name = {
        name: [hashlib.sha256(bytes(task.data)).hexdigest() for task in tasks]
        for name, tasks in tasks_by_name.items()
    }
    duplicate_banks = {
        name: len(hashes) - len(set(hashes))
        for name, hashes in hashes_by_name.items()
        if len(set(hashes)) != len(hashes)
    }
    if duplicate_banks:
        raise RuntimeError(f"Task bank contains duplicate variants: {duplicate_banks}.")
    descriptor = [
        {
            "task_id": task_id,
            "task_name": task_name,
            "variant_sha256s": hashes_by_name[task_name],
        }
        for task_id, task_name in enumerate(task_vocabulary)
    ]
    return tasks_by_name, hashes_by_name, _canonical_sha256(descriptor)


def _validate_arguments(args: argparse.Namespace) -> str:
    benchmark_name = str(args.benchmark).upper()
    if benchmark_name not in SUPPORTED_BENCHMARKS:
        raise ValueError("--benchmark must be MT10 or MT50.")
    if int(args.target_per_task) <= 0:
        raise ValueError("--target-per-task must be positive.")
    if int(args.max_attempts_multiplier) <= 0:
        raise ValueError("--max-attempts-multiplier must be positive.")
    if int(args.target_per_task) * int(args.max_attempts_multiplier) >= ATTEMPT_SEED_STRIDE:
        raise ValueError(
            "target_per_task * max_attempts_multiplier must be below 1,000,000."
        )
    if not 1 <= int(args.max_steps) <= MAX_EPISODE_STEPS:
        raise ValueError(f"--max-steps must be in [1,{MAX_EPISODE_STEPS}].")
    for name in ("threshold", "noise_level", "action_std_scale", "observation_std_scale"):
        value = float(getattr(args, name))
        if not np.isfinite(value):
            raise ValueError(f"--{name.replace('_', '-')} must be finite.")
    if not 0.0 <= float(args.threshold) <= 1.0:
        raise ValueError("--threshold must be in [0,1].")
    if min(
        float(args.noise_level),
        float(args.action_std_scale),
        float(args.observation_std_scale),
    ) < 0.0:
        raise ValueError("Noise levels/scales must be non-negative.")
    if bool(args.resume) and bool(args.overwrite):
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    return benchmark_name


def _build_protocol(
    *,
    args: argparse.Namespace,
    benchmark_name: str,
    metaworld_version: str,
    device: str,
    task_vocabulary: Sequence[str],
    task_bank_sha256: str,
    act_checkpoint_sha256: str,
    detector_checkpoint_sha256: str,
) -> dict[str, Any]:
    action_std = float(args.action_std_scale) * float(args.noise_level)
    observation_std = float(args.observation_std_scale) * float(args.noise_level)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_type": DATASET_TYPE,
        "benchmark": benchmark_name,
        "metaworld_version": metaworld_version,
        "benchmark_seed": int(args.benchmark_seed),
        "rollout_seed": int(args.seed),
        "task_vocabulary": list(task_vocabulary),
        "task_vocabulary_sha256": _task_vocabulary_sha256(task_vocabulary),
        "task_bank_sha256": task_bank_sha256,
        "official_goal_variants_per_task": OFFICIAL_GOALS_PER_TASK,
        "observation_schema": "raw39_plus_official_task_one_hot",
        "state_dim": RAW_OBSERVATION_DIM + len(task_vocabulary),
        "action_dim": ACTION_DIM,
        "target_per_task": int(args.target_per_task),
        "max_attempts_multiplier": int(args.max_attempts_multiplier),
        "max_attempts_per_task": int(args.target_per_task)
        * int(args.max_attempts_multiplier),
        "max_episode_steps": int(args.max_steps),
        "detector_threshold": float(args.threshold),
        "disturbance": {
            "noise_level": float(args.noise_level),
            "action_std_scale": float(args.action_std_scale),
            "observation_std_scale": float(args.observation_std_scale),
            "action_noise_std": action_std,
            "observation_noise_std": observation_std,
            "object_position_noise": False,
            "task_one_hot_noise": False,
        },
        "act_checkpoint_sha256": act_checkpoint_sha256,
        "detector_checkpoint_sha256": detector_checkpoint_sha256,
        "expert_source": "metaworld.policies.ENV_POLICY_MAP",
        "inference_device": str(device),
    }


def _initial_progress(
    task_vocabulary: Sequence[str],
) -> dict[str, dict[str, int]]:
    return {
        task_name: {
            "task_id": task_id,
            "next_attempt_index": 0,
            "attempts": 0,
            "detector_triggers": 0,
            "successful_continuations": 0,
        }
        for task_id, task_name in enumerate(task_vocabulary)
    }


def _build_manifest(
    *,
    protocol: Mapping[str, Any],
    protocol_fingerprint: str,
    act_checkpoint: Path,
    detector_checkpoint: Path,
    entries: Sequence[Mapping[str, Any]],
    progress: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    task_vocabulary = list(protocol["task_vocabulary"])
    ordered_entries = sorted(
        (dict(entry) for entry in entries),
        key=lambda item: (int(item["task_id"]), int(item["shard_index"])),
    )
    per_task: dict[str, dict[str, Any]] = {}
    for task_id, task_name in enumerate(task_vocabulary):
        selected = [
            entry for entry in ordered_entries if int(entry["task_id"]) == task_id
        ]
        task_progress = dict(progress[task_name])
        per_task[task_name] = {
            "task_id": task_id,
            "successful_continuations": len(selected),
            "rows": int(sum(int(entry["length"]) for entry in selected)),
            "attempts": int(task_progress["attempts"]),
            "detector_triggers": int(task_progress["detector_triggers"]),
            "next_attempt_index": int(task_progress["next_attempt_index"]),
            "attempt_yield": (
                len(selected) / int(task_progress["attempts"])
                if int(task_progress["attempts"])
                else 0.0
            ),
        }
    target = int(protocol["target_per_task"])
    complete = all(
        value["successful_continuations"] == target for value in per_task.values()
    )
    disturbance = dict(protocol["disturbance"])
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_type": DATASET_TYPE,
        "protocol": dict(protocol),
        "protocol_fingerprint_sha256": protocol_fingerprint,
        "benchmark": protocol["benchmark"],
        "metaworld_version": protocol["metaworld_version"],
        "benchmark_seed": protocol["benchmark_seed"],
        "seed": protocol["rollout_seed"],
        "task_vocabulary": task_vocabulary,
        "task_vocabulary_sha256": protocol["task_vocabulary_sha256"],
        "task_bank_sha256": protocol["task_bank_sha256"],
        "observation_schema": protocol["observation_schema"],
        "state_dim": protocol["state_dim"],
        "action_dim": protocol["action_dim"],
        "target_per_task": target,
        "max_episode_steps": protocol["max_episode_steps"],
        "threshold": protocol["detector_threshold"],
        "noise_level": disturbance["noise_level"],
        "action_noise_std": disturbance["action_noise_std"],
        "observation_noise_std": disturbance["observation_noise_std"],
        "object_position_noise": False,
        "act_checkpoint": str(act_checkpoint),
        "act_checkpoint_sha256": protocol["act_checkpoint_sha256"],
        "detector_checkpoint": str(detector_checkpoint),
        "detector_checkpoint_sha256": protocol["detector_checkpoint_sha256"],
        "complete": complete,
        "successful_continuations": len(ordered_entries),
        "rows": int(sum(int(entry["length"]) for entry in ordered_entries)),
        "attempts": int(sum(value["attempts"] for value in per_task.values())),
        "detector_triggers": int(
            sum(value["detector_triggers"] for value in per_task.values())
        ),
        "collection_progress": {
            name: dict(progress[name]) for name in task_vocabulary
        },
        "per_task": per_task,
        "files": ordered_entries,
    }


def _run_attempt(
    *,
    env: Any,
    task: Any,
    expert: Any,
    act: ACTPolicy,
    detector: FailureDetector,
    task_id: int,
    task_count: int,
    episode_seed: int,
    max_steps: int,
    threshold: float,
    action_std: float,
    observation_std: float,
) -> dict[str, Any]:
    env.set_task(task)
    seed_method = getattr(env, "seed", None)
    if callable(seed_method):
        seed_method(episode_seed)
    reset_result = env.reset(seed=episode_seed)
    raw = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    raw = np.asarray(raw, dtype=np.float32).reshape(-1)
    if raw.shape != (RAW_OBSERVATION_DIM,):
        raise RuntimeError(f"Expected raw39 reset observation, got {raw.shape}.")
    act.reset()
    history: deque[np.ndarray] = deque(maxlen=detector.sequence_length)
    action_rng = np.random.default_rng(_stream_seed(episode_seed, "action"))
    observation_rng = np.random.default_rng(
        _stream_seed(episode_seed, "observation")
    )
    triggered = False
    trigger_step = -1
    trigger_probability = float("nan")
    continuation_states: list[np.ndarray] = []
    continuation_raw: list[np.ndarray] = []
    continuation_actions: list[np.ndarray] = []
    continuation_rewards: list[float] = []
    continuation_risks: list[float] = []
    success = False

    for step in range(max_steps):
        clean_raw = np.asarray(raw, dtype=np.float32).reshape(-1)
        if clean_raw.shape != (RAW_OBSERVATION_DIM,):
            raise RuntimeError(f"Expected raw39 step observation, got {clean_raw.shape}.")
        observed_raw = clean_raw.copy()
        if observation_std > 0.0:
            observed_raw += observation_rng.normal(
                0.0, observation_std, size=RAW_OBSERVATION_DIM
            ).astype(np.float32)
        state = _condition(observed_raw, task_id, task_count)
        history.append(state)
        probability = _risk(detector, history, detector.sequence_length)
        if not triggered and probability >= threshold:
            triggered = True
            trigger_step = step
            trigger_probability = probability
        if triggered:
            target_action = np.clip(
                np.asarray(expert.get_action(clean_raw), dtype=np.float32).reshape(-1),
                -1.0,
                1.0,
            )
            if target_action.shape != (ACTION_DIM,):
                raise RuntimeError(
                    f"Expert returned {target_action.shape}; expected ({ACTION_DIM},)."
                )
            intended = target_action
            continuation_states.append(state.copy())
            continuation_raw.append(clean_raw.copy())
            continuation_actions.append(target_action.copy())
            continuation_risks.append(probability)
        else:
            intended = np.asarray(act.act(state), dtype=np.float32).reshape(-1)
            if intended.shape != (ACTION_DIM,):
                raise RuntimeError(
                    f"ACT returned {intended.shape}; expected ({ACTION_DIM},)."
                )
        if not np.isfinite(intended).all():
            raise RuntimeError("Policy returned a non-finite action.")
        executed = intended.copy()
        if action_std > 0.0:
            executed += action_rng.normal(0.0, action_std, size=ACTION_DIM).astype(
                np.float32
            )
        executed = np.clip(executed, -1.0, 1.0).astype(np.float32)
        next_raw, reward, terminated, truncated, info = env.step(executed)
        if triggered:
            continuation_rewards.append(float(reward))
        success = bool(dict(info).get("success", False))
        raw = next_raw
        if success or bool(terminated) or bool(truncated):
            break

    payload: dict[str, np.ndarray] | None = None
    if triggered and success and continuation_states:
        payload = {
            "states": np.stack(continuation_states).astype(np.float32),
            "raw_observations": np.stack(continuation_raw).astype(np.float32),
            "actions": np.stack(continuation_actions).astype(np.float32),
            "rewards": np.asarray(continuation_rewards, dtype=np.float32),
            "failure_probabilities": np.asarray(
                continuation_risks, dtype=np.float32
            ),
        }
    return {
        "triggered": triggered,
        "trigger_step": trigger_step,
        "trigger_probability": trigger_probability,
        "success": success,
        "payload": payload,
    }


def _shard_content(archive: Any) -> dict[str, np.ndarray]:
    return {key: np.asarray(archive[key]) for key in archive.files}


def _summarize_shard(
    path: Path,
    *,
    output_dir: Path,
    protocol: Mapping[str, Any],
    protocol_fingerprint: str,
    expected_task_id: int,
    expected_task_name: str,
    expected_shard_index: int,
    variant_hashes: Sequence[str],
) -> dict[str, Any]:
    file_digest = file_sha256(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_SHARD_KEYS.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing shard keys: {sorted(missing)}")
        arrays = _shard_content(archive)

    def scalar(name: str) -> Any:
        array = np.asarray(arrays[name])
        if array.size != 1:
            raise ValueError(f"{path}: {name} must be scalar.")
        return array.reshape(-1)[0]

    schema_version = str(scalar("schema_version"))
    fingerprint = str(scalar("protocol_fingerprint_sha256"))
    task_id = int(scalar("task_id"))
    task_name = str(scalar("task_name"))
    task_variant = int(scalar("task_variant"))
    task_payload_sha256 = str(scalar("task_payload_sha256"))
    episode_seed = int(scalar("episode_seed"))
    attempt_index = int(scalar("attempt_index"))
    shard_index = int(scalar("shard_index"))
    trigger_step = int(scalar("trigger_step"))
    trigger_probability = float(scalar("trigger_probability"))
    success = bool(scalar("success"))

    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"{path}: stale shard schema {schema_version!r}.")
    if fingerprint != protocol_fingerprint:
        raise ValueError(f"{path}: protocol fingerprint mismatch.")
    if (task_id, task_name) != (expected_task_id, expected_task_name):
        raise ValueError(f"{path}: task identity does not match its manifest slot.")
    if shard_index != expected_shard_index:
        raise ValueError(f"{path}: non-contiguous shard index {shard_index}.")
    expected_path = output_dir / f"task_{task_id:02d}" / f"recovery_{shard_index:04d}.npz"
    if path.resolve() != expected_path.resolve():
        raise ValueError(f"{path}: shard filename/path is not canonical.")
    expected_variant = attempt_index % len(variant_hashes)
    if task_variant != expected_variant:
        raise ValueError(f"{path}: task variant is inconsistent with attempt index.")
    if task_payload_sha256 != variant_hashes[task_variant]:
        raise ValueError(f"{path}: task payload hash does not match the frozen bank.")
    expected_seed = _episode_seed(int(protocol["rollout_seed"]), task_id, attempt_index)
    if episode_seed != expected_seed:
        raise ValueError(f"{path}: episode seed does not match its attempt index.")
    if not success:
        raise ValueError(f"{path}: recovery shards must be successful continuations.")
    if not np.isfinite(trigger_probability) or trigger_step < 0:
        raise ValueError(f"{path}: invalid detector trigger metadata.")

    states = np.asarray(arrays["states"], dtype=np.float32)
    raw = np.asarray(arrays["raw_observations"], dtype=np.float32)
    actions = np.asarray(arrays["actions"], dtype=np.float32)
    rewards = np.asarray(arrays["rewards"], dtype=np.float32).reshape(-1)
    risks = np.asarray(arrays["failure_probabilities"], dtype=np.float32).reshape(-1)
    state_dim = int(protocol["state_dim"])
    if states.ndim != 2 or states.shape[1] != state_dim or len(states) == 0:
        raise ValueError(f"{path}: invalid states shape {states.shape}.")
    length = len(states)
    if raw.shape != (length, RAW_OBSERVATION_DIM):
        raise ValueError(f"{path}: invalid raw_observations shape {raw.shape}.")
    if actions.shape != (length, ACTION_DIM):
        raise ValueError(f"{path}: invalid actions shape {actions.shape}.")
    if rewards.shape != (length,) or risks.shape != (length,):
        raise ValueError(f"{path}: continuation arrays have inconsistent lengths.")
    if not all(np.isfinite(array).all() for array in (states, raw, actions, rewards, risks)):
        raise ValueError(f"{path}: shard contains non-finite arrays.")
    if np.any(actions < -1.0) or np.any(actions > 1.0):
        raise ValueError(f"{path}: expert targets lie outside [-1,1].")
    task_count = len(protocol["task_vocabulary"])
    expected_one_hot = np.zeros(task_count, dtype=np.float32)
    expected_one_hot[task_id] = 1.0
    expected_task_block = np.broadcast_to(expected_one_hot, (length, task_count))
    if not np.array_equal(states[:, RAW_OBSERVATION_DIM:], expected_task_block):
        raise ValueError(f"{path}: invalid task one-hot block.")

    relative = str(path.resolve().relative_to(output_dir.resolve()))
    return {
        "file": relative,
        "task_id": task_id,
        "task_name": task_name,
        "shard_index": shard_index,
        "attempt_index": attempt_index,
        "task_variant": task_variant,
        "task_payload_sha256": task_payload_sha256,
        "episode_seed": episode_seed,
        "length": length,
        "trigger_step": trigger_step,
        "trigger_probability": trigger_probability,
        "content_sha256": _array_content_sha256(arrays),
        "sha256": file_digest,
    }


def _protocol_changes(
    stored: Mapping[str, Any], requested: Mapping[str, Any]
) -> list[str]:
    return sorted(
        key
        for key in set(stored).union(requested)
        if stored.get(key) != requested.get(key)
    )


def _validate_model_task_provenance(
    model: Any,
    *,
    label: str,
    benchmark_name: str,
    task_vocabulary: Sequence[str],
) -> None:
    """Reject contradictory checkpoint metadata while allowing legacy absence."""

    provenance = getattr(model, "provenance", None)
    if not isinstance(provenance, Mapping):
        return
    stored_benchmark = provenance.get("benchmark")
    if stored_benchmark is not None and str(stored_benchmark).upper() != benchmark_name:
        raise ValueError(f"{label} checkpoint benchmark provenance is incompatible.")
    stored_vocabulary = provenance.get("task_vocabulary")
    if stored_vocabulary is not None and list(stored_vocabulary) != list(
        task_vocabulary
    ):
        raise ValueError(f"{label} checkpoint task vocabulary is incompatible.")
    stored_hash = provenance.get("task_vocabulary_sha256")
    expected_hash = _task_vocabulary_sha256(task_vocabulary)
    if stored_hash is not None and str(stored_hash) != expected_hash:
        raise ValueError(f"{label} checkpoint task vocabulary hash is incompatible.")


def _validate_resume(
    *,
    manifest: Mapping[str, Any],
    output_dir: Path,
    protocol: Mapping[str, Any],
    protocol_fingerprint: str,
    hashes_by_name: Mapping[str, Sequence[str]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    stored_protocol = manifest.get("protocol")
    if not isinstance(stored_protocol, Mapping):
        raise ValueError("Resume manifest has no strict protocol block (stale v1 data).")
    stored_target = int(stored_protocol.get("target_per_task", -1))
    requested_target = int(protocol["target_per_task"])
    if requested_target < stored_target:
        raise ValueError(
            f"Refusing to shrink target_per_task from {stored_target} to "
            f"{requested_target}."
        )
    if requested_target != stored_target:
        raise ValueError(
            "target_per_task is fingerprinted and cannot change during resume."
        )
    stored_fingerprint = str(manifest.get("protocol_fingerprint_sha256", ""))
    if stored_fingerprint != _canonical_sha256(dict(stored_protocol)):
        raise ValueError("Stored manifest protocol fingerprint is invalid or tampered.")
    if stored_fingerprint != protocol_fingerprint:
        changes = _protocol_changes(stored_protocol, protocol)
        raise ValueError(
            "Resume protocol fingerprint mismatch; changed fields: "
            + ", ".join(changes)
        )

    task_vocabulary = list(protocol["task_vocabulary"])
    raw_progress = manifest.get("collection_progress")
    if not isinstance(raw_progress, Mapping) or set(raw_progress) != set(task_vocabulary):
        raise ValueError("Resume manifest has incomplete or stale per-task progress.")
    progress: dict[str, dict[str, int]] = {}
    for task_id, task_name in enumerate(task_vocabulary):
        value = raw_progress[task_name]
        if not isinstance(value, Mapping):
            raise ValueError(f"Invalid progress record for {task_name}.")
        record = {
            "task_id": int(value.get("task_id", -1)),
            "next_attempt_index": int(value.get("next_attempt_index", -1)),
            "attempts": int(value.get("attempts", -1)),
            "detector_triggers": int(value.get("detector_triggers", -1)),
            "successful_continuations": int(
                value.get("successful_continuations", -1)
            ),
        }
        if record["task_id"] != task_id:
            raise ValueError(f"Stale task ID in progress for {task_name}.")
        if record["next_attempt_index"] != record["attempts"]:
            raise ValueError(f"Non-contiguous attempt cursor for {task_name}.")
        if min(record.values()) < 0:
            raise ValueError(f"Negative progress counter for {task_name}.")
        if record["detector_triggers"] > record["attempts"]:
            raise ValueError(f"Trigger count exceeds attempts for {task_name}.")
        if record["successful_continuations"] > record["detector_triggers"]:
            raise ValueError(f"Success count exceeds triggers for {task_name}.")
        if record["successful_continuations"] > requested_target:
            raise ValueError(f"Stored {task_name} data exceeds target_per_task.")
        progress[task_name] = record

    raw_entries = manifest.get("files")
    if not isinstance(raw_entries, list):
        raise ValueError("Resume manifest files list is missing or invalid.")
    expected_paths: set[Path] = set()
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping) or "file" not in raw_entry:
            raise ValueError("Resume manifest contains an invalid file entry.")
        path = (output_dir / str(raw_entry["file"])).resolve()
        try:
            path.relative_to(output_dir.resolve())
        except ValueError as exc:
            raise ValueError("Manifest shard path escapes the output directory.") from exc
        if path in expected_paths:
            raise ValueError(f"Duplicate manifest shard path {path}.")
        expected_paths.add(path)
    discovered_paths = {path.resolve() for path in output_dir.rglob("recovery_*.npz")}
    stale = sorted(str(path) for path in discovered_paths - expected_paths)
    missing = sorted(str(path) for path in expected_paths - discovered_paths)
    if stale or missing:
        raise ValueError(
            "Stale/missing recovery shards detected; "
            f"stale={stale}, missing={missing}."
        )

    stored_by_path = {
        str(entry["file"]): dict(entry) for entry in raw_entries
    }
    verified_entries: list[dict[str, Any]] = []
    for task_id, task_name in enumerate(task_vocabulary):
        task_files = sorted(
            (
                entry
                for entry in raw_entries
                if int(entry.get("task_id", -1)) == task_id
            ),
            key=lambda entry: int(entry.get("shard_index", -1)),
        )
        expected_successes = progress[task_name]["successful_continuations"]
        if len(task_files) != expected_successes:
            raise ValueError(f"Manifest shard count disagrees with {task_name} progress.")
        attempt_indices: list[int] = []
        for shard_index, stored_entry in enumerate(task_files):
            path = (output_dir / str(stored_entry["file"])).resolve()
            if file_sha256(path) != str(stored_entry.get("sha256", "")):
                raise ValueError(f"Shard file hash mismatch: {path}.")
            summary = _summarize_shard(
                path,
                output_dir=output_dir,
                protocol=protocol,
                protocol_fingerprint=protocol_fingerprint,
                expected_task_id=task_id,
                expected_task_name=task_name,
                expected_shard_index=shard_index,
                variant_hashes=hashes_by_name[task_name],
            )
            if summary != stored_by_path[str(stored_entry["file"])]:
                raise ValueError(f"Manifest metadata/hash mismatch for {path}.")
            attempt_indices.append(int(summary["attempt_index"]))
            verified_entries.append(summary)
        if attempt_indices != sorted(set(attempt_indices)):
            raise ValueError(f"Non-monotonic successful attempt indices for {task_name}.")
        if attempt_indices and attempt_indices[-1] >= progress[task_name]["next_attempt_index"]:
            raise ValueError(f"Shard attempt lies beyond the {task_name} cursor.")
    return verified_entries, progress


def collect(
    args: argparse.Namespace,
    *,
    components: MetaWorldComponents | None = None,
    attempt_hook: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Collect recovery shards, optionally invoking a post-commit test hook."""

    benchmark_name = _validate_arguments(args)
    seed_everything(int(args.seed))
    device = select_device(args.device)
    logger = configure_logging(
        "collect_multitask_recovery",
        args.log_file
        or f"results/logs/{benchmark_name.lower()}_recovery_collection.log",
    )
    components = components or _load_metaworld_components()
    benchmark_type = getattr(components.module, benchmark_name, None)
    if benchmark_type is None:
        raise RuntimeError(f"Installed Meta-World does not provide {benchmark_name}.")
    benchmark = benchmark_type(seed=int(args.benchmark_seed))
    task_vocabulary = tuple(str(name) for name in benchmark.train_classes.keys())
    expected_tasks = SUPPORTED_BENCHMARKS[benchmark_name]
    if len(task_vocabulary) != expected_tasks:
        raise RuntimeError(
            f"Expected {expected_tasks} ordered tasks, got {len(task_vocabulary)}."
        )
    if len(set(task_vocabulary)) != len(task_vocabulary):
        raise RuntimeError("Benchmark task vocabulary contains duplicates.")
    missing_experts = [
        name for name in task_vocabulary if name not in components.policy_map
    ]
    if missing_experts:
        raise RuntimeError("Missing scripted experts: " + ", ".join(missing_experts))
    tasks_by_name, hashes_by_name, task_bank_sha256 = _task_bank(
        benchmark, task_vocabulary
    )

    act_checkpoint = Path(args.act_checkpoint).expanduser().resolve()
    detector_checkpoint = Path(args.detector_checkpoint).expanduser().resolve()
    if not act_checkpoint.is_file() or not detector_checkpoint.is_file():
        raise FileNotFoundError("ACT and detector checkpoint files must both exist.")
    act_checkpoint_sha256 = file_sha256(act_checkpoint)
    detector_checkpoint_sha256 = file_sha256(detector_checkpoint)
    act = ACTPolicy.from_checkpoint(act_checkpoint, map_location=device)
    detector = FailureDetector.from_checkpoint(
        detector_checkpoint,
        map_location=device,
        state_dim=RAW_OBSERVATION_DIM + len(task_vocabulary),
    )
    expected_state_dim = RAW_OBSERVATION_DIM + len(task_vocabulary)
    if act.state_dim != detector.state_dim or act.state_dim != expected_state_dim:
        raise ValueError("ACT, detector, and benchmark observation dimensions disagree.")
    if int(act.action_dim) != ACTION_DIM:
        raise ValueError("ACT action dimension must be 4.")
    _validate_model_task_provenance(
        act,
        label="ACT",
        benchmark_name=benchmark_name,
        task_vocabulary=task_vocabulary,
    )
    _validate_model_task_provenance(
        detector,
        label="Detector",
        benchmark_name=benchmark_name,
        task_vocabulary=task_vocabulary,
    )

    protocol = _build_protocol(
        args=args,
        benchmark_name=benchmark_name,
        metaworld_version=components.version,
        device=device,
        task_vocabulary=task_vocabulary,
        task_bank_sha256=task_bank_sha256,
        act_checkpoint_sha256=act_checkpoint_sha256,
        detector_checkpoint_sha256=detector_checkpoint_sha256,
    )
    protocol_fingerprint = _canonical_sha256(protocol)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    if args.overwrite:
        for path in output_dir.rglob("recovery_*.npz"):
            path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()
    discovered = sorted(output_dir.rglob("recovery_*.npz"))
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(
            f"{output_dir} already contains recovery data; use --resume or --overwrite."
        )
    if discovered and not manifest_path.exists():
        raise ValueError("Recovery shards exist without a manifest; refusing stale data.")
    if args.resume and not manifest_path.exists():
        raise FileNotFoundError("--resume requires an existing manifest.json.")

    if args.resume:
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries, progress = _validate_resume(
            manifest=previous_manifest,
            output_dir=output_dir,
            protocol=protocol,
            protocol_fingerprint=protocol_fingerprint,
            hashes_by_name=hashes_by_name,
        )
        reconstructed_manifest = _build_manifest(
            protocol=protocol,
            protocol_fingerprint=protocol_fingerprint,
            act_checkpoint=act_checkpoint,
            detector_checkpoint=detector_checkpoint,
            entries=entries,
            progress=progress,
        )
        if previous_manifest != reconstructed_manifest:
            raise ValueError(
                "Resume manifest contains stale or tampered derived statistics."
            )
    else:
        entries = []
        progress = _initial_progress(task_vocabulary)

    def write_manifest() -> dict[str, Any]:
        manifest = _build_manifest(
            protocol=protocol,
            protocol_fingerprint=protocol_fingerprint,
            act_checkpoint=act_checkpoint,
            detector_checkpoint=detector_checkpoint,
            entries=entries,
            progress=progress,
        )
        atomic_write_json(manifest_path, manifest)
        return manifest

    manifest = write_manifest()
    action_std = float(protocol["disturbance"]["action_noise_std"])
    observation_std = float(protocol["disturbance"]["observation_noise_std"])
    target = int(protocol["target_per_task"])
    progress_bar = tqdm(
        total=target * len(task_vocabulary),
        initial=len(entries),
        desc=f"Collecting {benchmark_name} recovery continuations",
        unit="shard",
    )
    try:
        for task_id, task_name in enumerate(task_vocabulary):
            state = progress[task_name]
            if state["successful_continuations"] >= target:
                continue
            env = benchmark.train_classes[task_name](render_mode=None)
            expert = components.policy_map[task_name]()
            try:
                while state["successful_continuations"] < target:
                    if state["next_attempt_index"] >= int(
                        protocol["max_attempts_per_task"]
                    ):
                        manifest = write_manifest()
                        raise RuntimeError(
                            f"{task_name}: collected "
                            f"{state['successful_continuations']}/{target} successful "
                            f"continuations after {state['attempts']} attempts."
                        )
                    attempt_index = int(state["next_attempt_index"])
                    task_variant = attempt_index % OFFICIAL_GOALS_PER_TASK
                    task = tasks_by_name[task_name][task_variant]
                    episode_seed = _episode_seed(int(args.seed), task_id, attempt_index)
                    result = _run_attempt(
                        env=env,
                        task=task,
                        expert=expert,
                        act=act,
                        detector=detector,
                        task_id=task_id,
                        task_count=len(task_vocabulary),
                        episode_seed=episode_seed,
                        max_steps=int(args.max_steps),
                        threshold=float(args.threshold),
                        action_std=action_std,
                        observation_std=observation_std,
                    )
                    state["attempts"] += 1
                    state["next_attempt_index"] += 1
                    if bool(result["triggered"]):
                        state["detector_triggers"] += 1
                    payload = result["payload"]
                    if payload is not None:
                        shard_index = int(state["successful_continuations"])
                        task_dir = output_dir / f"task_{task_id:02d}"
                        task_dir.mkdir(parents=True, exist_ok=True)
                        destination = task_dir / f"recovery_{shard_index:04d}.npz"
                        if destination.exists():
                            raise FileExistsError(
                                f"Refusing to overwrite recovery shard {destination}."
                            )
                        task_payload_sha256 = hashes_by_name[task_name][task_variant]
                        shard_arrays = {
                            **payload,
                            "success": np.asarray(True, dtype=np.bool_),
                            "task_id": np.asarray(task_id, dtype=np.int16),
                            "task_name": np.asarray(task_name),
                            "task_variant": np.asarray(task_variant, dtype=np.int16),
                            "task_payload_sha256": np.asarray(task_payload_sha256),
                            "trigger_step": np.asarray(
                                result["trigger_step"], dtype=np.int16
                            ),
                            "trigger_probability": np.asarray(
                                result["trigger_probability"], dtype=np.float32
                            ),
                            "episode_seed": np.asarray(episode_seed, dtype=np.int64),
                            "attempt_index": np.asarray(
                                attempt_index, dtype=np.int64
                            ),
                            "shard_index": np.asarray(shard_index, dtype=np.int32),
                            "protocol_fingerprint_sha256": np.asarray(
                                protocol_fingerprint
                            ),
                            "schema_version": np.asarray(SCHEMA_VERSION),
                        }
                        atomic_save_npz(destination, **shard_arrays)
                        summary = _summarize_shard(
                            destination,
                            output_dir=output_dir,
                            protocol=protocol,
                            protocol_fingerprint=protocol_fingerprint,
                            expected_task_id=task_id,
                            expected_task_name=task_name,
                            expected_shard_index=shard_index,
                            variant_hashes=hashes_by_name[task_name],
                        )
                        entries.append(summary)
                        state["successful_continuations"] += 1
                        progress_bar.update(1)
                    manifest = write_manifest()
                    if attempt_hook is not None:
                        attempt_hook(
                            {
                                "task_id": task_id,
                                "task_name": task_name,
                                "attempt_index": attempt_index,
                                "triggered": bool(result["triggered"]),
                                "saved": payload is not None,
                                "manifest": manifest,
                            }
                        )
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
    finally:
        progress_bar.close()

    manifest = write_manifest()
    if not manifest["complete"]:
        raise RuntimeError("Recovery collection ended without equal task coverage.")
    logger.info(
        "%s recovery data: continuations=%d rows=%d attempts=%d triggers=%d",
        benchmark_name,
        manifest["successful_continuations"],
        manifest["rows"],
        manifest["attempts"],
        manifest["detector_triggers"],
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=("MT10", "MT50"))
    parser.add_argument("--benchmark-seed", type=int, required=True)
    parser.add_argument("--act-checkpoint", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-per-task", type=int, default=50)
    parser.add_argument("--max-attempts-multiplier", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--noise-level", type=float, default=0.2)
    parser.add_argument("--action-std-scale", type=float, default=0.40)
    parser.add_argument("--observation-std-scale", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-file")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    result = collect(build_parser().parse_args())
    print(
        json.dumps(
            {
                "benchmark": result["benchmark"],
                "successful_continuations": result["successful_continuations"],
                "rows": result["rows"],
                "protocol_fingerprint_sha256": result[
                    "protocol_fingerprint_sha256"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
