#!/usr/bin/env python3
"""Train a task-conditioned recovery actor from successful expert continuations.

The dataset is expected to contain one NPZ shard per recovery continuation. A
shared MLP is trained for the whole MT10 or MT50 suite, while hierarchical
task/episode balancing prevents long or easy continuations from dominating the
update stream.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
from pathlib import Path
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.io import file_sha256
from models.imitation_recovery_policy import ImitationRecoveryPolicy
from utils.common import (
    atomic_json_dump,
    capture_rng_state,
    configure_logging,
    restore_rng_state,
    seed_everything,
    select_device,
)


LOGGER = logging.getLogger("reim.train_multitask_recovery")
SCHEMA_VERSION = "reim-multitask-recovery-training-v1"
RECOVERY_DATA_SCHEMA_VERSION = "reim-multitask-trigger-aligned-recovery-v2"
RECOVERY_DATASET_TYPE = "online_detector_triggered_expert_continuations"
RAW_OBSERVATION_DIM = 39
ACTION_DIM = 4
OFFICIAL_TARGET_PER_TASK = 50
OFFICIAL_MAX_EPISODE_STEPS = 500
SUPPORTED_BENCHMARKS = {"MT10": 10, "MT50": 50}
SOURCE_ALGORITHM = "task_balanced_smooth_l1"


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _array_content_sha256(arrays: Mapping[str, np.ndarray]) -> str:
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


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _validate_training_manifest(
    manifest: Mapping[str, Any], benchmark: str
) -> tuple[list[str], Mapping[str, Any]]:
    """Fail closed unless this is a complete publication-scale recovery bank."""

    benchmark_name = str(benchmark).upper()
    if benchmark_name not in SUPPORTED_BENCHMARKS:
        raise ValueError("--benchmark must be MT10 or MT50")
    if manifest.get("schema_version") != RECOVERY_DATA_SCHEMA_VERSION:
        raise ValueError("Unsupported recovery dataset schema")
    if manifest.get("dataset_type") != RECOVERY_DATASET_TYPE:
        raise ValueError("Unsupported recovery dataset type")
    if manifest.get("benchmark") != benchmark_name:
        raise ValueError("Dataset benchmark does not match --benchmark")
    if manifest.get("complete") is not True:
        raise ValueError(
            "Recovery dataset is incomplete; refusing partial training data"
        )

    task_vocabulary = manifest.get("task_vocabulary")
    expected_tasks = SUPPORTED_BENCHMARKS[benchmark_name]
    if (
        not isinstance(task_vocabulary, list)
        or len(task_vocabulary) != expected_tasks
        or any(not isinstance(name, str) or not name for name in task_vocabulary)
        or len(set(task_vocabulary)) != expected_tasks
    ):
        raise ValueError(
            f"{benchmark_name} requires {expected_tasks} unique ordered tasks"
        )
    vocabulary_sha256 = _canonical_sha256(task_vocabulary)
    if manifest.get("task_vocabulary_sha256") != vocabulary_sha256:
        raise ValueError("Recovery manifest task vocabulary hash is invalid")
    for field in (
        "task_bank_sha256",
        "protocol_fingerprint_sha256",
        "act_checkpoint_sha256",
        "detector_checkpoint_sha256",
    ):
        if not _is_sha256(manifest.get(field)):
            raise ValueError(f"Recovery manifest has invalid {field}")
    if int(manifest.get("state_dim", -1)) != RAW_OBSERVATION_DIM + expected_tasks:
        raise ValueError("Recovery manifest state_dim is invalid")
    if int(manifest.get("action_dim", -1)) != ACTION_DIM:
        raise ValueError("Recovery manifest action_dim is invalid")
    if int(manifest.get("target_per_task", -1)) != OFFICIAL_TARGET_PER_TASK:
        raise ValueError(
            f"Recovery training requires exactly {OFFICIAL_TARGET_PER_TASK} "
            "successful continuations per task"
        )
    if int(manifest.get("max_episode_steps", -1)) != OFFICIAL_MAX_EPISODE_STEPS:
        raise ValueError("Recovery training requires the 500-step benchmark horizon")

    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("Recovery manifest lacks the frozen collection protocol")
    if _canonical_sha256(dict(protocol)) != manifest["protocol_fingerprint_sha256"]:
        raise ValueError("Recovery collection protocol fingerprint is invalid")
    mirrored_protocol_fields = {
        "schema_version": RECOVERY_DATA_SCHEMA_VERSION,
        "dataset_type": RECOVERY_DATASET_TYPE,
        "benchmark": benchmark_name,
        "task_vocabulary": task_vocabulary,
        "task_vocabulary_sha256": vocabulary_sha256,
        "task_bank_sha256": manifest["task_bank_sha256"],
        "state_dim": manifest["state_dim"],
        "action_dim": manifest["action_dim"],
        "target_per_task": manifest["target_per_task"],
        "max_episode_steps": manifest["max_episode_steps"],
        "act_checkpoint_sha256": manifest["act_checkpoint_sha256"],
        "detector_checkpoint_sha256": manifest["detector_checkpoint_sha256"],
    }
    inconsistent = [
        field
        for field, expected in mirrored_protocol_fields.items()
        if protocol.get(field) != expected
    ]
    if inconsistent:
        raise ValueError(
            "Recovery manifest/protocol fields disagree: " + ", ".join(inconsistent)
        )
    max_attempts = int(protocol.get("max_attempts_per_task", -1))
    multiplier = int(protocol.get("max_attempts_multiplier", -1))
    if max_attempts != OFFICIAL_TARGET_PER_TASK * multiplier or multiplier <= 0:
        raise ValueError("Recovery collection attempt budget is inconsistent")
    threshold = float(protocol.get("detector_threshold", math.nan))
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Recovery collection threshold is invalid")

    entries = manifest.get("files")
    expected_files = expected_tasks * OFFICIAL_TARGET_PER_TASK
    if not isinstance(entries, list) or len(entries) != expected_files:
        raise ValueError(
            f"Recovery manifest must list exactly {expected_files} continuation shards"
        )
    if any(not isinstance(entry, Mapping) for entry in entries):
        raise ValueError("Recovery manifest contains an invalid file entry")
    per_task = manifest.get("per_task")
    progress = manifest.get("collection_progress")
    if (
        not isinstance(per_task, Mapping)
        or set(per_task) != set(task_vocabulary)
        or not isinstance(progress, Mapping)
        or set(progress) != set(task_vocabulary)
    ):
        raise ValueError("Recovery manifest has incomplete per-task accounting")
    total_rows = 0
    total_attempts = 0
    total_triggers = 0
    for task_id, task_name in enumerate(task_vocabulary):
        task_entries = [entry for entry in entries if entry.get("task_id") == task_id]
        shard_indices = sorted(
            int(entry.get("shard_index", -1)) for entry in task_entries
        )
        if len(task_entries) != OFFICIAL_TARGET_PER_TASK or shard_indices != list(
            range(OFFICIAL_TARGET_PER_TASK)
        ):
            raise ValueError(f"{task_name}: recovery coverage is not exactly 50 shards")
        record = per_task[task_name]
        cursor = progress[task_name]
        if not isinstance(record, Mapping) or not isinstance(cursor, Mapping):
            raise ValueError(f"{task_name}: invalid per-task accounting record")
        rows = sum(int(entry.get("length", -1)) for entry in task_entries)
        attempts = int(cursor.get("attempts", -1))
        triggers = int(cursor.get("detector_triggers", -1))
        successes = int(cursor.get("successful_continuations", -1))
        if (
            int(record.get("task_id", -1)) != task_id
            or int(cursor.get("task_id", -1)) != task_id
            or int(record.get("successful_continuations", -1))
            != OFFICIAL_TARGET_PER_TASK
            or successes != OFFICIAL_TARGET_PER_TASK
            or int(record.get("rows", -1)) != rows
            or int(record.get("attempts", -1)) != attempts
            or int(record.get("detector_triggers", -1)) != triggers
            or int(record.get("next_attempt_index", -1)) != attempts
            or int(cursor.get("next_attempt_index", -1)) != attempts
            or attempts < successes
            or triggers < successes
            or triggers > attempts
            or attempts > max_attempts
        ):
            raise ValueError(f"{task_name}: inconsistent per-task recovery accounting")
        total_rows += rows
        total_attempts += attempts
        total_triggers += triggers
    if (
        int(manifest.get("successful_continuations", -1)) != expected_files
        or int(manifest.get("rows", -1)) != total_rows
        or int(manifest.get("attempts", -1)) != total_attempts
        or int(manifest.get("detector_triggers", -1)) != total_triggers
    ):
        raise ValueError("Recovery manifest global accounting is inconsistent")
    return list(task_vocabulary), protocol


def _load_dataset(
    data_dir: Path,
    manifest: dict[str, Any],
    task_vocabulary: list[str],
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[Path]]:
    if manifest.get("schema_version") != RECOVERY_DATA_SCHEMA_VERSION:
        raise ValueError("Recovery loader requires the trigger-aligned v2 schema")
    if manifest.get("dataset_type") != RECOVERY_DATASET_TYPE:
        raise ValueError("Recovery loader received an unsupported dataset type")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Recovery manifest must contain a non-empty files list")
    expected_relative_paths: list[str] = []
    entry_by_path: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
            raise ValueError("Every recovery manifest file entry must be a mapping")
        relative = Path(entry["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe recovery manifest path: {relative}")
        relative_name = relative.as_posix()
        if relative_name in entry_by_path:
            raise ValueError(f"Duplicate recovery manifest file: {relative_name}")
        expected_relative_paths.append(relative_name)
        entry_by_path[relative_name] = entry
    actual_relative_paths = sorted(
        path.relative_to(data_dir).as_posix() for path in data_dir.rglob("*.npz")
    )
    if sorted(expected_relative_paths) != actual_relative_paths:
        missing = sorted(set(expected_relative_paths) - set(actual_relative_paths))
        stale = sorted(set(actual_relative_paths) - set(expected_relative_paths))
        raise ValueError(
            "Recovery shards do not match the manifest whitelist: "
            f"missing={missing}, stale={stale}"
        )

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    task_ids: list[int] = []
    files: list[Path] = []
    seen_episode_seeds: set[int] = set()
    seen_attempts: set[tuple[int, int]] = set()
    expected_state_dim = RAW_OBSERVATION_DIM + len(task_vocabulary)
    protocol_fingerprint = str(manifest.get("protocol_fingerprint_sha256", ""))
    protocol = manifest.get("protocol")
    threshold = (
        float(protocol.get("detector_threshold", math.nan))
        if isinstance(protocol, Mapping)
        else math.nan
    )
    max_steps = (
        int(protocol.get("max_episode_steps", -1))
        if isinstance(protocol, Mapping)
        else -1
    )
    rollout_seed = (
        int(protocol.get("rollout_seed", -1)) if isinstance(protocol, Mapping) else -1
    )
    for relative_name in sorted(expected_relative_paths):
        entry = entry_by_path[relative_name]
        path = data_dir / relative_name
        if file_sha256(path) != entry.get("sha256"):
            raise ValueError(f"Recovery shard hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            required = {
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
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"{path}: missing required keys {sorted(missing)}")
            arrays = {name: np.asarray(archive[name]) for name in archive.files}

            def scalar(name: str) -> Any:
                value = np.asarray(arrays[name])
                if value.size != 1:
                    raise ValueError(f"{path}: {name} must be scalar")
                return value.reshape(-1)[0]

            success = bool(scalar("success"))
            if not success:
                raise ValueError(f"{path}: manifest lists an unsuccessful continuation")
            state = np.asarray(arrays["states"], dtype=np.float32)
            raw = np.asarray(arrays["raw_observations"], dtype=np.float32)
            action = np.asarray(arrays["actions"], dtype=np.float32)
            rewards = np.asarray(arrays["rewards"], dtype=np.float32).reshape(-1)
            risks = np.asarray(
                arrays["failure_probabilities"], dtype=np.float32
            ).reshape(-1)
            task_id = int(scalar("task_id"))
            task_name = str(scalar("task_name"))
            task_variant = int(scalar("task_variant"))
            task_payload_sha256 = str(scalar("task_payload_sha256"))
            trigger_step = int(scalar("trigger_step"))
            trigger_probability = float(scalar("trigger_probability"))
            episode_seed = int(scalar("episode_seed"))
            attempt_index = int(scalar("attempt_index"))
            shard_index = int(scalar("shard_index"))
            shard_fingerprint = str(scalar("protocol_fingerprint_sha256"))
            schema_version = str(scalar("schema_version"))
            if (
                state.ndim != 2
                or action.ndim != 2
                or state.shape != (len(action), expected_state_dim)
                or action.shape[1:] != (ACTION_DIM,)
            ):
                raise ValueError(
                    f"{path}: expected states [T,{expected_state_dim}] and "
                    f"actions [T,{ACTION_DIM}], "
                    f"got {state.shape}/{action.shape}"
                )
            if len(state) == 0:
                raise ValueError(f"{path}: empty recovery continuation")
            if (
                raw.shape != (len(state), RAW_OBSERVATION_DIM)
                or rewards.shape != (len(state),)
                or risks.shape != (len(state),)
            ):
                raise ValueError(
                    f"{path}: continuation arrays have inconsistent lengths"
                )
            if not all(
                np.isfinite(value).all()
                for value in (state, raw, action, rewards, risks)
            ):
                raise ValueError(f"{path}: non-finite continuation values")
            if np.any(action < -1.0) or np.any(action > 1.0):
                raise ValueError(f"{path}: actions leave the Meta-World action bounds")
            if np.any(risks < 0.0) or np.any(risks > 1.0):
                raise ValueError(f"{path}: risks leave the probability bounds")
            if not 0 <= task_id < len(task_vocabulary):
                raise ValueError(f"{path}: invalid task_id {task_id}")
            if task_vocabulary[task_id] != task_name:
                raise ValueError(
                    f"{path}: task_id/task_name disagree with ordered vocabulary"
                )
            one_hot = np.zeros(len(task_vocabulary), dtype=np.float32)
            one_hot[task_id] = 1.0
            if not np.array_equal(
                state[:, RAW_OBSERVATION_DIM:],
                np.broadcast_to(one_hot, (len(state), len(one_hot))),
            ):
                raise ValueError(f"{path}: invalid task one-hot block")
            expected_path = (
                Path(f"task_{task_id:02d}") / f"recovery_{shard_index:04d}.npz"
            ).as_posix()
            if relative_name != expected_path:
                raise ValueError(f"{path}: shard path is not canonical")
            expected_entry = {
                "task_id": task_id,
                "task_name": task_name,
                "shard_index": shard_index,
                "attempt_index": attempt_index,
                "task_variant": task_variant,
                "task_payload_sha256": task_payload_sha256,
                "episode_seed": episode_seed,
                "length": len(state),
                "trigger_step": trigger_step,
            }
            if any(entry.get(key) != value for key, value in expected_entry.items()):
                raise ValueError(f"{path}: shard metadata disagrees with manifest")
            if not math.isclose(
                float(entry.get("trigger_probability", math.nan)),
                trigger_probability,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise ValueError(f"{path}: trigger probability disagrees with manifest")
            if schema_version != str(manifest.get("schema_version")):
                raise ValueError(f"{path}: schema version disagrees with manifest")
            if shard_fingerprint != protocol_fingerprint:
                raise ValueError(
                    f"{path}: protocol fingerprint disagrees with manifest"
                )
            if not _is_sha256(task_payload_sha256):
                raise ValueError(f"{path}: invalid task payload hash")
            if not math.isfinite(threshold) or trigger_probability < threshold:
                raise ValueError(f"{path}: trigger probability is below threshold")
            if not np.isclose(
                trigger_probability, float(risks[0]), rtol=0.0, atol=1e-7
            ):
                raise ValueError(f"{path}: first risk is not the trigger probability")
            if trigger_step < 0 or trigger_step + len(state) > max_steps:
                raise ValueError(f"{path}: trigger/continuation exceeds the horizon")
            expected_seed = (
                (rollout_seed % 1_000_000_000) * 100_000_000
                + task_id * 1_000_000
                + attempt_index
            )
            if episode_seed != expected_seed:
                raise ValueError(f"{path}: episode seed is inconsistent with attempt")
            if entry.get("content_sha256") != _array_content_sha256(arrays):
                raise ValueError(f"{path}: array content hash disagrees with manifest")
            if episode_seed in seen_episode_seeds:
                raise ValueError(f"{path}: duplicate episode_seed {episode_seed}")
            seen_episode_seeds.add(episode_seed)
            attempt_identity = (task_id, attempt_index)
            if attempt_identity in seen_attempts:
                raise ValueError(f"{path}: duplicate task/attempt index")
            seen_attempts.add(attempt_identity)
            states.append(state)
            actions.append(action)
            task_ids.append(task_id)
            files.append(path)
    if not states:
        raise ValueError(f"No successful recovery continuations found in {data_dir}")
    state_dims = {array.shape[1] for array in states}
    action_dims = {array.shape[1] for array in actions}
    if len(state_dims) != 1 or len(action_dims) != 1:
        raise ValueError(
            f"Inconsistent recovery dimensions: states={state_dims}, actions={action_dims}"
        )
    return states, actions, task_ids, files


def _stratified_group_split(
    task_ids: list[int], validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    for task_id in sorted(set(task_ids)):
        indices = np.asarray(
            [index for index, value in enumerate(task_ids) if value == task_id],
            dtype=np.int64,
        )
        indices = rng.permutation(indices)
        if len(indices) == 1:
            train.extend(indices.tolist())
            continue
        count = min(
            max(int(round(len(indices) * validation_fraction)), 1), len(indices) - 1
        )
        validation.extend(indices[:count].tolist())
        train.extend(indices[count:].tolist())
    if not validation:
        raise ValueError("At least two continuation shards are needed for validation")
    return sorted(train), sorted(validation)


def _training_config(args: argparse.Namespace, *, device: str) -> dict[str, Any]:
    """Fingerprint every option that can change an optimizer trajectory."""

    config = {
        "algorithm": SOURCE_ALGORITHM,
        "loss": "smooth_l1_beta_1",
        "sampling": "equal_task_equal_episode_with_replacement",
        "validation_selection": "episode_grouped_task_macro_smooth_l1",
        "seed": int(args.seed),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "validation_fraction": float(args.validation_fraction),
        "hidden_dims": [int(value) for value in args.hidden_dims],
        "state_noise_std": float(args.state_noise_std),
        "grad_clip": float(args.grad_clip),
        "patience": int(args.patience),
        "min_delta": float(args.min_delta),
        "num_workers": int(args.num_workers),
        "torch_version": str(torch.__version__),
    }
    numeric_positive = (
        "batch_size",
        "learning_rate",
        "grad_clip",
        "patience",
    )
    for field in numeric_positive:
        value = float(config[field])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive and finite")
    for field in ("weight_decay", "state_noise_std", "min_delta"):
        value = float(config[field])
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"--{field.replace('_', '-')} must be non-negative and finite"
            )
    if not 0.0 < config["validation_fraction"] < 1.0:
        raise ValueError("--validation-fraction must lie in (0,1)")
    if config["num_workers"] < 0:
        raise ValueError("--num-workers must be non-negative")
    if not config["hidden_dims"] or any(value <= 0 for value in config["hidden_dims"]):
        raise ValueError("--hidden-dims must contain positive values")
    if int(args.epochs) <= 0:
        raise ValueError("--epochs must be positive")
    return config


def _flatten(
    states: list[np.ndarray],
    actions: list[np.ndarray],
    task_ids: list[int],
    groups: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.concatenate([states[index] for index in groups]).astype(np.float32),
        np.concatenate([actions[index] for index in groups]).astype(np.float32),
        np.concatenate(
            [
                np.full(len(states[index]), task_ids[index], dtype=np.int64)
                for index in groups
            ]
        ),
    )


def _task_episode_balanced_weights(
    states: Sequence[np.ndarray], task_ids: Sequence[int], groups: Sequence[int]
) -> np.ndarray:
    """Give every task equal mass and every episode equal mass within a task."""

    if not groups:
        raise ValueError("Task-balanced recovery sampling requires episode groups")
    selected_task_ids = np.asarray(
        [task_ids[index] for index in groups], dtype=np.int64
    )
    if np.any(selected_task_ids < 0):
        raise ValueError("Recovery task IDs must be non-negative")
    episode_counts = np.bincount(selected_task_ids)
    weights: list[np.ndarray] = []
    for index, task_id in zip(groups, selected_task_ids.tolist()):
        length = len(states[index])
        if length <= 0 or episode_counts[task_id] <= 0:
            raise ValueError("Recovery sampling received an empty episode")
        weights.append(
            np.full(
                length,
                1.0 / (float(episode_counts[task_id]) * float(length)),
                dtype=np.float64,
            )
        )
    return np.concatenate(weights)


def _save_history(
    history: list[dict[str, float]], csv_path: Path, curve_path: Path
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "epoch",
                "train_loss",
                "validation_loss",
                "validation_loss_micro",
            ),
        )
        writer.writeheader()
        writer.writerows(history)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    if history:
        epochs = [row["epoch"] for row in history]
        axis.plot(epochs, [row["train_loss"] for row in history], label="Train")
        axis.plot(
            epochs,
            [row["validation_loss"] for row in history],
            label="Validation (task macro)",
        )
    axis.set(xlabel="Epoch", ylabel="Smooth-L1", title="Multi-task recovery training")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(curve_path, dpi=220)
    plt.close(figure)


def train(args: argparse.Namespace) -> dict[str, Any]:
    device = select_device(args.device)
    training_config = _training_config(args, device=device)
    training_config_sha256 = _canonical_sha256(training_config)
    seed_everything(args.seed, deterministic_torch=True)
    logger = configure_logging(
        "train_multitask_recovery",
        args.log_file or f"results/logs/{args.benchmark.lower()}_recovery.log",
    )
    data_dir = Path(args.data_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Recovery dataset manifest is missing: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("Recovery dataset manifest must be a JSON object")
    manifest_sha256 = file_sha256(manifest_path)
    task_vocabulary, protocol = _validate_training_manifest(manifest, args.benchmark)
    expected_tasks = SUPPORTED_BENCHMARKS[args.benchmark.upper()]
    vocabulary_sha256 = _canonical_sha256(task_vocabulary)

    states, actions, group_task_ids, files = _load_dataset(
        data_dir, manifest, task_vocabulary
    )
    train_groups, validation_groups = _stratified_group_split(
        group_task_ids, args.validation_fraction, args.seed
    )
    if set(train_groups).intersection(validation_groups):
        raise RuntimeError("Recovery train/validation episode groups overlap")
    if sorted(train_groups + validation_groups) != list(range(len(files))):
        raise RuntimeError("Recovery split does not partition all episode groups")
    train_states, train_actions, train_task_ids = _flatten(
        states, actions, group_task_ids, train_groups
    )
    validation_states, validation_actions, validation_task_ids = _flatten(
        states, actions, group_task_ids, validation_groups
    )
    state_dim, action_dim = train_states.shape[1], train_actions.shape[1]
    if state_dim != 39 + expected_tasks or action_dim != 4:
        raise ValueError(
            f"Unexpected {args.benchmark} data dimensions: state={state_dim}, action={action_dim}"
        )

    mean = train_states.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_states.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)
    # Preserve exact binary task indicators; normalization applies only to raw39.
    mean[39:] = 0.0
    std[39:] = 1.0
    train_files = [
        files[index].relative_to(data_dir).as_posix() for index in train_groups
    ]
    validation_files = [
        files[index].relative_to(data_dir).as_posix() for index in validation_groups
    ]
    split_descriptor = {
        "schema_version": "reim-episode-grouped-stratified-split-v1",
        "seed": int(args.seed),
        "validation_fraction": float(args.validation_fraction),
        "train_files": train_files,
        "validation_files": validation_files,
    }
    split_sha256 = _canonical_sha256(split_descriptor)
    recovery_provenance = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": args.benchmark.upper(),
        "task_vocabulary": task_vocabulary,
        "task_vocabulary_sha256": vocabulary_sha256,
        "dataset_manifest_sha256": manifest_sha256,
        "data_schema_version": RECOVERY_DATA_SCHEMA_VERSION,
        "dataset_type": RECOVERY_DATASET_TYPE,
        "dataset_complete": True,
        "target_per_task": OFFICIAL_TARGET_PER_TASK,
        "task_bank_sha256": manifest["task_bank_sha256"],
        "collection_protocol_fingerprint_sha256": manifest[
            "protocol_fingerprint_sha256"
        ],
        "act_checkpoint_sha256": manifest["act_checkpoint_sha256"],
        "detector_checkpoint_sha256": manifest["detector_checkpoint_sha256"],
        "split": split_descriptor,
        "split_sha256": split_sha256,
        "training_config": training_config,
        "training_config_sha256": training_config_sha256,
        "source_training": {
            "algorithm": SOURCE_ALGORITHM,
            "num_timesteps": 0,
            "supervised_update_only": True,
        },
    }
    model = ImitationRecoveryPolicy(
        state_dim,
        action_dim,
        hidden_dims=tuple(args.hidden_dims),
        activation="tanh",
        observation_mean=mean,
        observation_std=std,
        provenance=recovery_provenance,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_function = nn.SmoothL1Loss()
    elementwise_loss = nn.SmoothL1Loss(reduction="none")

    train_dataset = TensorDataset(
        torch.from_numpy(train_states), torch.from_numpy(train_actions)
    )
    counts = np.bincount(train_task_ids, minlength=expected_tasks).astype(np.float64)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Training recovery data has no rows for task IDs {missing}")
    weights = _task_episode_balanced_weights(states, group_task_ids, train_groups)
    if len(weights) != len(train_dataset):
        raise RuntimeError(
            "Task-balanced sampler weights do not align with training rows"
        )
    task_sampling_mass = np.bincount(
        train_task_ids, weights=weights, minlength=expected_tasks
    )
    if not np.allclose(task_sampling_mass, np.ones(expected_tasks), atol=1e-12):
        raise RuntimeError("Inverse-frequency sampler is not task balanced")
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )
    validation_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(validation_states),
            torch.from_numpy(validation_actions),
            torch.from_numpy(validation_task_ids),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )

    latest = output.with_name(output.stem + "_latest" + output.suffix)
    history: list[dict[str, float]] = []
    start_epoch = 1
    best_loss = float("inf")
    best_epoch = 0
    best_model_state_dict: dict[str, torch.Tensor] | None = None
    stale = 0
    if args.resume:
        resume_path = (
            latest
            if args.resume == "auto"
            else Path(args.resume).expanduser().resolve()
        )
        # Generator/RNG states must remain CPU ByteTensors. Optimizer state is
        # migrated to parameter devices by ``load_state_dict``.
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Recovery resume checkpoint must be a mapping")
        required_resume_keys = {
            "model_state_dict",
            "optimizer_state_dict",
            "epoch",
            "seed",
            "dataset_manifest_sha256",
            "train_files",
            "validation_files",
            "split_sha256",
            "training_config_sha256",
            "best_model_state_dict",
            "best_epoch",
            "sampler_generator_state",
            "rng_state",
        }
        missing_resume = required_resume_keys.difference(checkpoint)
        if missing_resume:
            raise ValueError(
                "--resume requires a recovery training-state checkpoint, not a "
                f"deployment policy; missing {sorted(missing_resume)}"
            )
        requested_hidden_dims = list(map(int, args.hidden_dims))
        resume_mismatches = []
        for label, stored, requested in (
            ("schema", checkpoint.get("schema_version"), SCHEMA_VERSION),
            ("benchmark", checkpoint.get("benchmark"), args.benchmark.upper()),
            ("seed", int(checkpoint.get("seed", -1)), args.seed),
            (
                "dataset manifest",
                checkpoint.get("dataset_manifest_sha256"),
                manifest_sha256,
            ),
            ("task vocabulary", checkpoint.get("task_vocabulary"), task_vocabulary),
            (
                "task vocabulary hash",
                checkpoint.get("task_vocabulary_sha256"),
                vocabulary_sha256,
            ),
            ("hidden dimensions", checkpoint.get("hidden_dims"), requested_hidden_dims),
            ("train split", checkpoint.get("train_files"), train_files),
            ("validation split", checkpoint.get("validation_files"), validation_files),
            ("split fingerprint", checkpoint.get("split_sha256"), split_sha256),
            (
                "training configuration",
                checkpoint.get("training_config_sha256"),
                training_config_sha256,
            ),
            ("state dimension", int(checkpoint.get("state_dim", -1)), state_dim),
            ("action dimension", int(checkpoint.get("action_dim", -1)), action_dim),
        ):
            if stored != requested:
                resume_mismatches.append(
                    f"{label}: stored={stored!r}, requested={requested!r}"
                )
        if resume_mismatches:
            raise ValueError(
                "Incompatible recovery resume: " + "; ".join(resume_mismatches)
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint["epoch"]) + 1
        if int(checkpoint["epoch"]) > int(args.epochs):
            raise ValueError("--epochs cannot be below the resumed epoch")
        if len(history) != int(checkpoint["epoch"]) or any(
            not isinstance(row, Mapping) or int(row.get("epoch", -1)) != index
            for index, row in enumerate(history, start=1)
        ):
            raise ValueError("Recovery resume history is not contiguous with its epoch")
        best_loss = float(checkpoint.get("best_validation_loss", best_loss))
        best_epoch = int(checkpoint["best_epoch"])
        raw_best_state = checkpoint["best_model_state_dict"]
        if not isinstance(raw_best_state, Mapping):
            raise ValueError("Recovery resume checkpoint has invalid best model state")
        best_model_state_dict = {
            str(key): torch.as_tensor(value).detach().cpu().clone()
            for key, value in raw_best_state.items()
        }
        stale = int(checkpoint.get("epochs_without_improvement", 0))
        if (
            not 1 <= best_epoch <= int(checkpoint["epoch"])
            or not math.isfinite(best_loss)
            or not math.isclose(
                float(history[best_epoch - 1]["validation_loss"]),
                best_loss,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise ValueError("Recovery resume best-model metadata is inconsistent")
        generator.set_state(checkpoint["sampler_generator_state"])
        restore_rng_state(checkpoint["rng_state"])
        if stale >= int(args.patience):
            # The stored run has already met its frozen early-stop condition;
            # exact resume must not perform an extra optimizer update.
            start_epoch = int(args.epochs) + 1

    raw_dim = 39
    for epoch in tqdm(
        range(start_epoch, args.epochs + 1), desc=f"{args.benchmark} recovery"
    ):
        model.train()
        train_sum = 0.0
        for batch_states, batch_actions in train_loader:
            batch_states = batch_states.to(device, non_blocking=True)
            batch_actions = batch_actions.to(device, non_blocking=True)
            if args.state_noise_std > 0:
                noisy = batch_states.clone()
                noisy[:, :raw_dim] += (
                    torch.randn_like(noisy[:, :raw_dim]) * args.state_noise_std
                )
            else:
                noisy = batch_states
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model.mean_action(noisy), batch_actions)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_sum += float(loss.detach()) * len(batch_states)
        model.eval()
        validation_sum = 0.0
        validation_task_sums = np.zeros(expected_tasks, dtype=np.float64)
        validation_task_counts = np.zeros(expected_tasks, dtype=np.int64)
        with torch.inference_mode():
            for batch_states, batch_actions, batch_task_ids in validation_loader:
                batch_states = batch_states.to(device, non_blocking=True)
                batch_actions = batch_actions.to(device, non_blocking=True)
                row_losses = elementwise_loss(
                    model.mean_action(batch_states), batch_actions
                ).mean(dim=1)
                validation_sum += float(row_losses.sum())
                task_values = batch_task_ids.numpy()
                loss_values = row_losses.detach().cpu().numpy()
                validation_task_sums += np.bincount(
                    task_values, weights=loss_values, minlength=expected_tasks
                )
                validation_task_counts += np.bincount(
                    task_values, minlength=expected_tasks
                )
        train_loss = train_sum / len(train_dataset)
        validation_loss_micro = validation_sum / len(validation_states)
        if np.any(validation_task_counts == 0):
            missing = np.flatnonzero(validation_task_counts == 0).tolist()
            raise RuntimeError(f"Validation split has no rows for task IDs {missing}")
        validation_loss = float(np.mean(validation_task_sums / validation_task_counts))
        if not all(
            math.isfinite(value)
            for value in (train_loss, validation_loss, validation_loss_micro)
        ):
            raise RuntimeError("Recovery training produced a non-finite loss")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_loss_micro": validation_loss_micro,
            }
        )
        improved = validation_loss < best_loss - args.min_delta
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
            model.provenance["best_epoch"] = epoch
            model.provenance["best_validation_loss"] = best_loss
            model.provenance["best_validation_loss_micro"] = validation_loss_micro
            best_model_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            model.save(output)
        else:
            stale += 1
        _atomic_torch_save(
            {
                "schema_version": SCHEMA_VERSION,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "best_validation_loss": best_loss,
                "best_epoch": best_epoch,
                "best_model_state_dict": best_model_state_dict,
                "epochs_without_improvement": stale,
                "benchmark": args.benchmark.upper(),
                "seed": args.seed,
                "state_dim": state_dim,
                "action_dim": action_dim,
                "hidden_dims": list(map(int, args.hidden_dims)),
                "task_vocabulary": task_vocabulary,
                "task_vocabulary_sha256": vocabulary_sha256,
                "dataset_manifest_sha256": manifest_sha256,
                "data_schema_version": RECOVERY_DATA_SCHEMA_VERSION,
                "dataset_type": RECOVERY_DATASET_TYPE,
                "act_checkpoint_sha256": manifest["act_checkpoint_sha256"],
                "detector_checkpoint_sha256": manifest["detector_checkpoint_sha256"],
                "split_sha256": split_sha256,
                "training_config": training_config,
                "training_config_sha256": training_config_sha256,
                "train_files": train_files,
                "validation_files": validation_files,
                "sampler_generator_state": generator.get_state(),
                "rng_state": capture_rng_state(),
            },
            latest,
        )
        logger.info(
            "epoch=%d train=%.6f validation_macro=%.6f validation_micro=%.6f best=%.6f",
            epoch,
            train_loss,
            validation_loss,
            validation_loss_micro,
            best_loss,
        )
        if stale >= args.patience:
            break

    if best_model_state_dict is None or best_epoch <= 0 or not math.isfinite(best_loss):
        raise RuntimeError("Recovery training did not produce a valid best checkpoint")
    best_history = next(
        (row for row in history if int(row["epoch"]) == best_epoch), None
    )
    if best_history is None:
        raise RuntimeError("Best recovery epoch is absent from training history")
    model.load_state_dict(best_model_state_dict, strict=True)
    model.provenance["best_epoch"] = best_epoch
    model.provenance["best_validation_loss"] = best_loss
    model.provenance["best_validation_loss_micro"] = float(
        best_history["validation_loss_micro"]
    )
    # Re-materialize the deployment artifact from the authenticated best state.
    # This also makes resume robust to a missing/stale side artifact.
    model.save(output)

    history_path = Path(args.history).expanduser().resolve()
    curve_path = Path(args.curve).expanduser().resolve()
    _save_history(history, history_path, curve_path)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": args.benchmark.upper(),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "task_count": expected_tasks,
        "train_groups": len(train_groups),
        "validation_groups": len(validation_groups),
        "train_rows": len(train_states),
        "validation_rows": len(validation_states),
        "best_validation_loss": best_loss,
        "best_validation_loss_micro": float(best_history["validation_loss_micro"]),
        "best_epoch": best_epoch,
        "epochs_completed": history[-1]["epoch"] if history else 0,
        "checkpoint": str(output),
        "checkpoint_sha256": file_sha256(output),
        "dataset_manifest_sha256": manifest_sha256,
        "data_schema_version": RECOVERY_DATA_SCHEMA_VERSION,
        "dataset_type": RECOVERY_DATASET_TYPE,
        "act_checkpoint_sha256": manifest["act_checkpoint_sha256"],
        "detector_checkpoint_sha256": manifest["detector_checkpoint_sha256"],
        "split_sha256": split_sha256,
        "training_config_sha256": training_config_sha256,
        "sampling_task_mass": task_sampling_mass.tolist(),
        "source_file_count": len(files),
        "task_vocabulary": task_vocabulary,
    }
    atomic_json_dump(summary, args.summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=("MT10", "MT50"))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--curve", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--log-file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=(512, 512, 256))
    parser.add_argument("--state-noise-std", type=float, default=0.005)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        help="Resume from PATH, or the latest training-state checkpoint if omitted.",
    )
    return parser


def main() -> None:
    summary = train(build_parser().parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
