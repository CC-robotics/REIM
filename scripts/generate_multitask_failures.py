#!/usr/bin/env python3
"""Generate task-conditioned causal risk data on MT10 or MT50.

Failure supervision is deliberately task agnostic. During data collection only,
the matching Meta-World scripted expert supplies a counterfactual corrective
action. Large ACT/expert disagreement marks behavioral deviation; unsuccessful
episode tails mark terminal failure. A causal LSTM later predicts the resulting
future-window labels without access to the expert.

Resume is fail closed. A collection can only be extended after its immutable
protocol fingerprint, task bank, model hash, complete file inventory, and every
shard digest have been verified. ``rollouts_per_task`` is intentionally not
part of the immutable fingerprint: increasing it is the sole supported resume
mutation, while decreasing it is rejected because stale shards would otherwise
remain visible to downstream glob-based loaders.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.io import atomic_save_npz, atomic_write_json, file_sha256
from models.bc_policy import ACTPolicy
from utils.common import configure_logging, seed_everything, select_device


LOGGER = logging.getLogger("reim.generate_multitask_failures")
SCHEMA_VERSION = "reim-multitask-failures-v2"
DATASET_TYPE = "task_conditioned_behavioral_deviation_risk"
RAW_OBSERVATION_DIM = 39
ACTION_DIM = 4
OFFICIAL_VARIANTS_PER_TASK = 50
SUPPORTED_BENCHMARKS = {"MT10": 10, "MT50": 50}
_SHARD_PATTERN = re.compile(r"^task_(\d{2})/failure_(\d{4,})\.npz$")


@dataclass(frozen=True)
class FailureGenerationComponents:
    """Injectable external dependencies used by fast protocol tests."""

    benchmark_factory: Callable[[str, int], Any]
    policy_loader: Callable[[Path, str], Any]
    expert_policy_map: Mapping[str, Callable[[], Any]]
    metaworld_version: str


def _benchmark(name: str, seed: int) -> Any:
    import metaworld

    constructor = getattr(metaworld, name.upper(), None)
    if constructor is None:
        raise ValueError(f"Unsupported Meta-World benchmark {name!r}")
    return constructor(seed=seed)


def _default_components() -> FailureGenerationComponents:
    from metaworld.policies import ENV_POLICY_MAP

    return FailureGenerationComponents(
        benchmark_factory=_benchmark,
        policy_loader=lambda path, device: ACTPolicy.from_checkpoint(
            path, map_location=device
        ),
        expert_policy_map=ENV_POLICY_MAP,
        metaworld_version=importlib.metadata.version("metaworld"),
    )


def _toy_components() -> FailureGenerationComponents:
    """Explicit deterministic CI backend; never a silent fallback."""

    from env import toy_multitask

    return FailureGenerationComponents(
        benchmark_factory=lambda name, seed: getattr(
            toy_multitask, name.upper()
        )(seed=seed),
        policy_loader=lambda path, device: ACTPolicy.from_checkpoint(
            path, map_location=device
        ),
        expert_policy_map=toy_multitask.ENV_POLICY_MAP,
        metaworld_version=toy_multitask.TOY_VERSION,
    )


def _components_for_backend(backend: str) -> FailureGenerationComponents:
    if backend == "toy":
        return _toy_components()
    if backend == "metaworld":
        return _default_components()
    raise ValueError(f"Unsupported backend {backend!r}; use 'metaworld' or 'toy'.")


def _condition(raw: np.ndarray, task_id: int, task_count: int) -> np.ndarray:
    one_hot = np.zeros(task_count, dtype=np.float32)
    one_hot[task_id] = 1.0
    return np.concatenate([np.asarray(raw, dtype=np.float32), one_hot])


def _future_labels(events: np.ndarray, horizon: int) -> np.ndarray:
    result = np.zeros(len(events), dtype=np.bool_)
    for index in range(len(events)):
        result[index] = bool(np.any(events[index : index + horizon + 1]))
    return result


def _episode_path(output_dir: Path, task_id: int, rollout_index: int) -> Path:
    return output_dir / f"task_{task_id:02d}" / f"failure_{rollout_index:04d}.npz"


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _task_vocabulary_sha256(task_vocabulary: Sequence[str]) -> str:
    return _canonical_json_sha256(list(task_vocabulary))


def _task_payload_sha256(task: Any) -> str:
    try:
        payload = bytes(task.data)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Meta-World task payload must be bytes-like") from exc
    return hashlib.sha256(payload).hexdigest()


def _prepare_task_bank(
    benchmark: Any,
    *,
    benchmark_name: str,
) -> tuple[list[str], dict[str, list[Any]], str]:
    task_vocabulary = [str(name) for name in benchmark.train_classes.keys()]
    expected_tasks = SUPPORTED_BENCHMARKS[benchmark_name]
    if len(task_vocabulary) != expected_tasks:
        raise RuntimeError(
            f"{benchmark_name} returned {len(task_vocabulary)} tasks, "
            f"expected {expected_tasks}"
        )
    if len(set(task_vocabulary)) != len(task_vocabulary):
        raise RuntimeError("Meta-World returned duplicate task names")

    tasks_by_name: dict[str, list[Any]] = {
        name: [] for name in task_vocabulary
    }
    for task in benchmark.train_tasks:
        task_name = str(task.env_name)
        if task_name not in tasks_by_name:
            raise RuntimeError(
                f"Benchmark task bank contains unknown task {task_name!r}"
            )
        tasks_by_name[task_name].append(task)
    invalid_counts = {
        name: len(tasks)
        for name, tasks in tasks_by_name.items()
        if len(tasks) != OFFICIAL_VARIANTS_PER_TASK
    }
    if invalid_counts:
        raise RuntimeError(
            "Official MT10/MT50 task banks require exactly "
            f"{OFFICIAL_VARIANTS_PER_TASK} variants per task; got {invalid_counts}"
        )

    bank_payload = [
        {
            "task_id": task_id,
            "task_name": task_name,
            "variant_payload_sha256": [
                _task_payload_sha256(task) for task in tasks_by_name[task_name]
            ],
        }
        for task_id, task_name in enumerate(task_vocabulary)
    ]
    return task_vocabulary, tasks_by_name, _canonical_json_sha256(bank_payload)


def _validate_arguments(args: argparse.Namespace) -> str:
    benchmark_name = str(args.benchmark).upper()
    if benchmark_name not in SUPPORTED_BENCHMARKS:
        raise ValueError(f"Unsupported benchmark {benchmark_name!r}")
    if bool(args.resume) and bool(args.overwrite):
        raise ValueError("--resume and --overwrite are mutually exclusive")
    integer_fields = {
        "rollouts_per_task": args.rollouts_per_task,
        "max_steps": args.max_steps,
        "prediction_horizon": args.prediction_horizon,
        "terminal_positive_horizon": args.terminal_positive_horizon,
    }
    for field, value in integer_fields.items():
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"--{field.replace('_', '-')} must be an integer")
    if int(args.rollouts_per_task) <= 0:
        raise ValueError("--rollouts-per-task must be positive")
    if int(args.max_steps) <= 0:
        raise ValueError("--max-steps must be positive")
    if int(args.prediction_horizon) < 0:
        raise ValueError("--prediction-horizon must be non-negative")
    if int(args.terminal_positive_horizon) < 0:
        raise ValueError("--terminal-positive-horizon must be non-negative")
    float_fields = {
        "noise_level": args.noise_level,
        "action_std_scale": args.action_std_scale,
        "observation_std_scale": args.observation_std_scale,
        "disagreement_threshold": args.disagreement_threshold,
    }
    for field, value in float_fields.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"--{field.replace('_', '-')} must be finite") from exc
        if not np.isfinite(numeric):
            raise ValueError(f"--{field.replace('_', '-')} must be finite")
    if float(args.noise_level) < 0.0:
        raise ValueError("--noise-level must be non-negative")
    if float(args.action_std_scale) < 0.0 or float(args.observation_std_scale) < 0.0:
        raise ValueError("noise scales must be non-negative")
    if float(args.disagreement_threshold) < 0.0:
        raise ValueError("--disagreement-threshold must be non-negative")
    return benchmark_name


def _protocol_payload(
    args: argparse.Namespace,
    *,
    benchmark_name: str,
    metaworld_version: str,
    task_vocabulary: Sequence[str],
    task_bank_sha256: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    task_count = len(task_vocabulary)
    noise_level = float(args.noise_level)
    action_std_scale = float(args.action_std_scale)
    observation_std_scale = float(args.observation_std_scale)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_type": DATASET_TYPE,
        "benchmark": benchmark_name,
        "metaworld_version": str(metaworld_version),
        "benchmark_seed": int(args.benchmark_seed),
        "collection_seed": int(args.seed),
        "benchmark_task_bank_sha256": task_bank_sha256,
        "task_vocabulary": list(task_vocabulary),
        "task_vocabulary_sha256": _task_vocabulary_sha256(task_vocabulary),
        "act_checkpoint_sha256": checkpoint_sha256,
        "observation_schema": "raw39_plus_official_task_one_hot",
        "state_dim": RAW_OBSERVATION_DIM + task_count,
        "action_dim": ACTION_DIM,
        "max_episode_steps": int(args.max_steps),
        "noise_parameters": {
            "noise_level": noise_level,
            "action_std_scale": action_std_scale,
            "observation_std_scale": observation_std_scale,
            "action_noise_std": action_std_scale * noise_level,
            "observation_noise_std": observation_std_scale * noise_level,
            "object_position_noise": False,
        },
        "label_parameters": {
            "expert_action_disagreement_l1": float(
                args.disagreement_threshold
            ),
            "prediction_horizon": int(args.prediction_horizon),
            "terminal_positive_horizon": int(args.terminal_positive_horizon),
        },
    }


def _provenance_differences(
    stored: Any,
    requested: Any,
    *,
    prefix: str = "provenance",
) -> list[str]:
    if isinstance(stored, Mapping) and isinstance(requested, Mapping):
        differences: list[str] = []
        for key in sorted(set(stored) | set(requested)):
            child = f"{prefix}.{key}"
            if key not in stored:
                differences.append(f"{child}: missing from stored manifest")
            elif key not in requested:
                differences.append(f"{child}: unexpected stored field")
            else:
                differences.extend(
                    _provenance_differences(
                        stored[key], requested[key], prefix=child
                    )
                )
        return differences
    if stored != requested:
        return [f"{prefix}: stored={stored!r}, requested={requested!r}"]
    return []


def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"Dataset manifest is not a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot parse existing dataset manifest {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset manifest must be a JSON object: {path}")
    return payload


def _top_level_protocol_fields(protocol: Mapping[str, Any]) -> dict[str, Any]:
    noise = protocol["noise_parameters"]
    labels = protocol["label_parameters"]
    return {
        "schema_version": protocol["schema_version"],
        "dataset_type": protocol["dataset_type"],
        "benchmark": protocol["benchmark"],
        "metaworld_version": protocol["metaworld_version"],
        "benchmark_seed": protocol["benchmark_seed"],
        "seed": protocol["collection_seed"],
        "task_vocabulary": protocol["task_vocabulary"],
        "task_vocabulary_sha256": protocol["task_vocabulary_sha256"],
        "benchmark_task_bank_sha256": protocol[
            "benchmark_task_bank_sha256"
        ],
        "act_checkpoint_sha256": protocol["act_checkpoint_sha256"],
        "observation_schema": protocol["observation_schema"],
        "state_dim": protocol["state_dim"],
        "action_dim": protocol["action_dim"],
        "max_episode_steps": protocol["max_episode_steps"],
        "noise_level": noise["noise_level"],
        "action_std_scale": noise["action_std_scale"],
        "observation_std_scale": noise["observation_std_scale"],
        "action_noise_std": noise["action_noise_std"],
        "observation_noise_std": noise["observation_noise_std"],
        "object_position_noise": noise["object_position_noise"],
        "expert_action_disagreement_l1": labels[
            "expert_action_disagreement_l1"
        ],
        "prediction_horizon": labels["prediction_horizon"],
        "terminal_positive_horizon": labels[
            "terminal_positive_horizon"
        ],
    }


def _validate_resume_manifest(
    manifest: Mapping[str, Any],
    *,
    requested_protocol: Mapping[str, Any],
    requested_rollouts_per_task: int,
) -> int:
    stored_protocol = manifest.get("provenance")
    if not isinstance(stored_protocol, Mapping):
        raise ValueError(
            "Cannot resume without a structured provenance fingerprint; "
            "use --overwrite to create a v2 dataset"
        )
    stored_fingerprint = manifest.get("provenance_fingerprint_sha256")
    recomputed_stored_fingerprint = _canonical_json_sha256(stored_protocol)
    if stored_fingerprint != recomputed_stored_fingerprint:
        raise ValueError(
            "Stored provenance fingerprint does not match the manifest payload"
        )
    requested_fingerprint = _canonical_json_sha256(requested_protocol)
    if stored_fingerprint != requested_fingerprint:
        differences = _provenance_differences(
            stored_protocol, requested_protocol
        )
        detail = "; ".join(differences[:12]) or "fingerprint differs"
        raise ValueError(
            "Cannot resume with incompatible provenance: " + detail
        )

    top_level_mismatches = [
        f"{key}: stored={manifest.get(key)!r}, requested={value!r}"
        for key, value in _top_level_protocol_fields(requested_protocol).items()
        if manifest.get(key) != value
    ]
    if top_level_mismatches:
        raise ValueError(
            "Manifest protocol fields disagree with its provenance: "
            + "; ".join(top_level_mismatches)
        )

    try:
        stored_rollouts = int(manifest["rollouts_per_task"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Manifest has invalid rollouts_per_task") from exc
    if stored_rollouts <= 0:
        raise ValueError("Manifest rollouts_per_task must be positive")
    if requested_rollouts_per_task < stored_rollouts:
        raise ValueError(
            "Cannot decrease rollouts_per_task during resume "
            f"({stored_rollouts} -> {requested_rollouts_per_task}); stale shards "
            "would remain"
        )
    return stored_rollouts


def _dataset_shards(output_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in output_dir.glob("task_*/failure_*.npz")
        if path.is_file() or path.is_symlink()
    )


def _safe_manifest_relative_path(value: Any) -> Path:
    text = str(value)
    relative = Path(text)
    if (
        not text
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != text
        or _SHARD_PATTERN.fullmatch(text) is None
    ):
        raise ValueError(f"Manifest contains unsafe or non-canonical file path {text!r}")
    return relative


def _clear_owned_dataset(
    output_dir: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any] | None,
) -> None:
    shards = _dataset_shards(output_dir)
    if manifest is None:
        if shards:
            raise ValueError(
                "Refusing --overwrite: failure shards exist without a manifest "
                "that proves dataset ownership"
            )
        return
    if manifest.get("dataset_type") != DATASET_TYPE:
        raise FileExistsError(
            "Refusing --overwrite: manifest belongs to dataset_type="
            f"{manifest.get('dataset_type')!r}, not {DATASET_TYPE!r}"
        )

    # Do not follow arbitrary manifest paths during a destructive operation.
    # Dataset ownership is established by dataset_type above, and only the
    # generator's canonical on-disk shard namespace is removed. This also lets
    # --overwrite recover old v1 manifests affected by the historical relative
    # path bug without risking unrelated files.
    for path in shards:
        if path.exists() or path.is_symlink():
            path.unlink()
    for task_dir in sorted(output_dir.glob("task_*"), reverse=True):
        if task_dir.is_dir() and not any(task_dir.iterdir()):
            task_dir.rmdir()
    residual = _dataset_shards(output_dir)
    if residual:
        raise RuntimeError(
            "Overwrite cleanup left failure shards: "
            + ", ".join(str(path) for path in residual[:5])
        )
    manifest_path.unlink()
    if manifest_path.exists():
        raise RuntimeError("Overwrite cleanup could not remove manifest.json")


def _scalar(archive: Any, key: str, *, path: Path) -> Any:
    value = np.asarray(archive[key])
    if value.size != 1:
        raise ValueError(f"{path}: {key} must be scalar")
    return value.reshape(-1)[0]


def _read_shard_summary(
    path: Path,
    *,
    output_dir: Path,
    task_count: int,
    expected_task_id: int,
    expected_task_name: str,
    expected_rollout_index: int,
    expected_task_variant: int,
    expected_task_payload_sha256: str,
    expected_episode_seed: int,
) -> dict[str, Any]:
    expected_path = _episode_path(
        output_dir, expected_task_id, expected_rollout_index
    )
    if path != expected_path:
        raise ValueError(
            f"Shard path {path} does not match canonical path {expected_path}"
        )
    if path.is_symlink():
        raise ValueError(f"Refusing symlinked dataset shard {path}")
    required = {
        "states",
        "raw_observations",
        "actions",
        "expert_actions",
        "action_disagreement_l1",
        "risk_events",
        "labels",
        "rewards",
        "success",
        "task_id",
        "task_name",
        "task_variant",
        "task_payload_sha256",
        "episode_seed",
        "schema_version",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"{path}: missing arrays {sorted(missing)}")
            states = np.asarray(archive["states"])
            raw_observations = np.asarray(archive["raw_observations"])
            actions = np.asarray(archive["actions"])
            expert_actions = np.asarray(archive["expert_actions"])
            disagreements = np.asarray(archive["action_disagreement_l1"])
            risk_events = np.asarray(archive["risk_events"])
            labels = np.asarray(archive["labels"])
            rewards = np.asarray(archive["rewards"])
            trajectory_arrays = {
                "states": states,
                "raw_observations": raw_observations,
                "actions": actions,
                "expert_actions": expert_actions,
                "action_disagreement_l1": disagreements,
                "risk_events": risk_events,
                "labels": labels,
                "rewards": rewards,
            }
            scalar_arrays = [
                key for key, value in trajectory_arrays.items() if value.ndim == 0
            ]
            if scalar_arrays:
                raise ValueError(
                    f"{path}: arrays lack a trajectory dimension {scalar_arrays}"
                )
            length = len(states)
            expected_lengths = {
                "raw_observations": len(raw_observations),
                "actions": len(actions),
                "expert_actions": len(expert_actions),
                "action_disagreement_l1": len(disagreements),
                "risk_events": len(risk_events),
                "labels": len(labels),
                "rewards": len(rewards),
            }
            if length <= 0 or any(value != length for value in expected_lengths.values()):
                raise ValueError(
                    f"{path}: inconsistent or empty trajectory lengths "
                    f"(states={length}, others={expected_lengths})"
                )
            if states.shape != (length, RAW_OBSERVATION_DIM + task_count):
                raise ValueError(f"{path}: invalid states shape {states.shape}")
            if raw_observations.shape != (length, RAW_OBSERVATION_DIM):
                raise ValueError(
                    f"{path}: invalid raw_observations shape {raw_observations.shape}"
                )
            if actions.shape != (length, ACTION_DIM) or expert_actions.shape != (
                length,
                ACTION_DIM,
            ):
                raise ValueError(f"{path}: invalid action tensor shape")
            vector_shapes = {
                "action_disagreement_l1": disagreements.shape,
                "risk_events": risk_events.shape,
                "labels": labels.shape,
                "rewards": rewards.shape,
            }
            invalid_vectors = {
                key: shape
                for key, shape in vector_shapes.items()
                if shape != (length,)
            }
            if invalid_vectors:
                raise ValueError(
                    f"{path}: invalid per-step vector shapes {invalid_vectors}"
                )

            task_id = int(_scalar(archive, "task_id", path=path))
            task_name = str(_scalar(archive, "task_name", path=path))
            task_variant = int(_scalar(archive, "task_variant", path=path))
            payload_sha256 = str(
                _scalar(archive, "task_payload_sha256", path=path)
            )
            episode_seed = int(_scalar(archive, "episode_seed", path=path))
            schema_version = str(_scalar(archive, "schema_version", path=path))
            success = bool(_scalar(archive, "success", path=path))
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(str(path)):
            raise
        raise ValueError(f"Cannot validate dataset shard {path}: {exc}") from exc

    expected_metadata = {
        "task_id": (task_id, expected_task_id),
        "task_name": (task_name, expected_task_name),
        "task_variant": (task_variant, expected_task_variant),
        "task_payload_sha256": (
            payload_sha256,
            expected_task_payload_sha256,
        ),
        "episode_seed": (episode_seed, expected_episode_seed),
        "schema_version": (schema_version, SCHEMA_VERSION),
    }
    mismatches = [
        f"{key}: stored={stored!r}, expected={expected!r}"
        for key, (stored, expected) in expected_metadata.items()
        if stored != expected
    ]
    if mismatches:
        raise ValueError(f"{path}: shard provenance mismatch: " + "; ".join(mismatches))

    return {
        "file": path.relative_to(output_dir).as_posix(),
        "task_id": expected_task_id,
        "task_name": expected_task_name,
        "rollout_index": expected_rollout_index,
        "length": length,
        "success": success,
        "positive_labels": int(np.asarray(labels, dtype=np.bool_).sum()),
        "sha256": file_sha256(path),
    }


def _validate_manifest_file_inventory(
    manifest: Mapping[str, Any],
    *,
    output_dir: Path,
    task_vocabulary: Sequence[str],
    tasks_by_name: Mapping[str, Sequence[Any]],
    stored_rollouts_per_task: int,
    seed: int,
) -> dict[tuple[int, int], dict[str, Any]]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("Manifest files inventory must be a list")
    complete = bool(manifest.get("complete", True))
    expected_count = len(task_vocabulary) * stored_rollouts_per_task
    if complete:
        if len(files) != expected_count:
            raise ValueError(
                f"Manifest file inventory has {len(files)} entries, expected {expected_count}"
            )
        slots: list[tuple[int, int]] = [
            (task_id, rollout_index)
            for task_id in range(len(task_vocabulary))
            for rollout_index in range(stored_rollouts_per_task)
        ]
    else:
        if len(files) > expected_count:
            raise ValueError(
                f"Manifest file inventory has {len(files)} entries, "
                f"expected at most {expected_count}"
            )
        partial_slots: list[tuple[int, int]] = []
        for index, entry in enumerate(files):
            if not isinstance(entry, Mapping):
                raise ValueError(f"Manifest files[{index}] is not an object")
            relative = _safe_manifest_relative_path(entry.get("file"))
            match = _SHARD_PATTERN.fullmatch(relative.as_posix())
            if match is None:  # pragma: no cover - guarded above
                raise ValueError(f"Manifest file entry {relative} is not a shard path")
            task_id = int(match.group(1))
            rollout_index = int(match.group(2))
            if task_id >= len(task_vocabulary):
                raise ValueError(
                    f"Manifest file entry {relative} exceeds the task vocabulary"
                )
            if rollout_index >= stored_rollouts_per_task:
                raise ValueError(
                    f"Manifest file entry {relative} exceeds rollouts_per_task"
                )
            partial_slots.append((task_id, rollout_index))
        if len(set(partial_slots)) != len(partial_slots):
            raise ValueError("Manifest file inventory contains duplicate slots")
        slots = sorted(partial_slots)

    manifest_by_path: dict[str, Mapping[str, Any]] = {}
    for index, entry in enumerate(files):
        if not isinstance(entry, Mapping):
            raise ValueError(f"Manifest files[{index}] is not an object")
        relative = _safe_manifest_relative_path(entry.get("file"))
        relative_text = relative.as_posix()
        if relative_text in manifest_by_path:
            raise ValueError(f"Duplicate manifest file entry {relative_text}")
        manifest_by_path[relative_text] = entry

    disk_by_path = {
        path.relative_to(output_dir).as_posix(): path
        for path in _dataset_shards(output_dir)
    }
    missing = sorted(set(manifest_by_path) - set(disk_by_path))
    stale = sorted(set(disk_by_path) - set(manifest_by_path))
    if missing or stale:
        detail: list[str] = []
        if missing:
            detail.append(f"missing shards={missing[:5]}")
        if stale:
            detail.append(f"stale shards={stale[:5]}")
        raise ValueError("Dataset file inventory mismatch: " + "; ".join(detail))

    summaries: dict[tuple[int, int], dict[str, Any]] = {}
    expected_relative_order: list[str] = []
    for task_id, rollout_index in slots:
        task_name = task_vocabulary[task_id]
        task_variants = tasks_by_name[task_name]
        path = _episode_path(output_dir, task_id, rollout_index)
        relative = path.relative_to(output_dir).as_posix()
        expected_relative_order.append(relative)
        entry = manifest_by_path.get(relative)
        if entry is None:
            raise ValueError(f"Manifest is missing expected shard {relative}")
        stored_sha256 = str(entry.get("sha256", ""))
        actual_sha256 = file_sha256(path)
        if stored_sha256 != actual_sha256:
            raise ValueError(
                f"Shard SHA256 mismatch for {relative}: "
                f"stored={stored_sha256!r}, actual={actual_sha256!r}"
            )
        task_variant = rollout_index % len(task_variants)
        summary = _read_shard_summary(
            path,
            output_dir=output_dir,
            task_count=len(task_vocabulary),
            expected_task_id=task_id,
            expected_task_name=task_name,
            expected_rollout_index=rollout_index,
            expected_task_variant=task_variant,
            expected_task_payload_sha256=_task_payload_sha256(
                task_variants[task_variant]
            ),
            expected_episode_seed=seed + task_id * 100_000 + rollout_index,
        )
        for key in (
            "file",
            "task_id",
            "task_name",
            "rollout_index",
            "length",
            "success",
            "positive_labels",
            "sha256",
        ):
            if entry.get(key) != summary[key]:
                raise ValueError(
                    f"Manifest entry {relative} has invalid {key}: "
                    f"stored={entry.get(key)!r}, actual={summary[key]!r}"
                )
        summaries[(task_id, rollout_index)] = summary

    stored_order = [str(entry["file"]) for entry in files]
    if stored_order != expected_relative_order:
        raise ValueError("Manifest files inventory is not in canonical task/rollout order")
    expected_episode_count = expected_count if complete else len(files)
    if manifest.get("episodes") != expected_episode_count:
        raise ValueError("Manifest episode count disagrees with its file inventory")
    rows = sum(item["length"] for item in summaries.values())
    positives = sum(item["positive_labels"] for item in summaries.values())
    if manifest.get("rows") != rows:
        raise ValueError("Manifest row count disagrees with its file inventory")
    expected_positive_rate = positives / max(1, rows)
    try:
        stored_positive_rate = float(manifest["positive_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Manifest positive_rate is invalid") from exc
    if not np.isclose(stored_positive_rate, expected_positive_rate, rtol=0.0, atol=1e-15):
        raise ValueError("Manifest positive_rate disagrees with its file inventory")
    return summaries


def _build_manifest(
    *,
    protocol: Mapping[str, Any],
    checkpoint_path: Path,
    rollouts_per_task: int,
    entries: Mapping[tuple[int, int], Mapping[str, Any]],
    complete: bool = True,
) -> dict[str, Any]:
    ordered_entries = [entries[key] for key in sorted(entries)]
    task_vocabulary = list(protocol["task_vocabulary"])
    total_rows = sum(int(entry["length"]) for entry in ordered_entries)
    total_positive = sum(
        int(entry["positive_labels"]) for entry in ordered_entries
    )
    per_task: dict[str, Any] = {}
    for task_name in task_vocabulary:
        selected = [
            entry for entry in ordered_entries if entry["task_name"] == task_name
        ]
        rows = sum(int(entry["length"]) for entry in selected)
        per_task[task_name] = {
            "episodes": len(selected),
            "success_rate": float(
                np.mean([bool(entry["success"]) for entry in selected])
            )
            if selected
            else 0.0,
            "rows": rows,
            "positive_rate": sum(
                int(entry["positive_labels"]) for entry in selected
            )
            / max(1, rows),
        }

    manifest: dict[str, Any] = {
        **_top_level_protocol_fields(protocol),
        "provenance": dict(protocol),
        "provenance_fingerprint_sha256": _canonical_json_sha256(protocol),
        "rollouts_per_task": int(rollouts_per_task),
        "episodes": len(ordered_entries),
        "rows": total_rows,
        "positive_rate": total_positive / max(1, total_rows),
        "act_checkpoint": str(checkpoint_path),
        "per_task": per_task,
        "files": ordered_entries,
        "complete": bool(complete),
    }
    return manifest


def generate(
    args: argparse.Namespace,
    *,
    components: FailureGenerationComponents | None = None,
) -> dict[str, Any]:
    """Generate or strictly resume a balanced MT10/MT50 risk dataset."""

    benchmark_name = _validate_arguments(args)
    seed_everything(int(args.seed))
    device = select_device(args.device)
    logger = configure_logging(
        "generate_multitask_failures",
        args.log_file
        or f"results/logs/{benchmark_name.lower()}_failures.log",
    )
    components = components or _components_for_backend(
        str(getattr(args, "backend", "metaworld"))
    )
    benchmark = components.benchmark_factory(
        benchmark_name, int(args.benchmark_seed)
    )
    task_vocabulary, tasks_by_name, task_bank_sha256 = _prepare_task_bank(
        benchmark, benchmark_name=benchmark_name
    )
    missing_experts = [
        name for name in task_vocabulary if name not in components.expert_policy_map
    ]
    if missing_experts:
        raise RuntimeError(
            "Scripted expert map lacks tasks: " + ", ".join(missing_experts)
        )

    checkpoint_path = Path(args.act_checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"ACT checkpoint does not exist: {checkpoint_path}")
    checkpoint_sha256 = file_sha256(checkpoint_path)
    protocol = _protocol_payload(
        args,
        benchmark_name=benchmark_name,
        metaworld_version=components.metaworld_version,
        task_vocabulary=task_vocabulary,
        task_bank_sha256=task_bank_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )
    act = components.policy_loader(checkpoint_path, device)
    if int(act.state_dim) != RAW_OBSERVATION_DIM + len(task_vocabulary) or int(
        act.action_dim
    ) != ACTION_DIM:
        raise ValueError(
            f"ACT dimensions {act.state_dim}->{act.action_dim} do not match "
            f"{benchmark_name} ({RAW_OBSERVATION_DIM + len(task_vocabulary)}"
            f"->{ACTION_DIM})"
        )
    act_provenance = getattr(act, "provenance", None)
    if isinstance(act, ACTPolicy) and not isinstance(act_provenance, Mapping):
        raise ValueError("ACT checkpoint lacks ordered multi-task provenance")
    if isinstance(act_provenance, Mapping):
        if list(act_provenance.get("task_vocabulary", [])) != task_vocabulary:
            raise ValueError("ACT checkpoint task vocabulary does not match benchmark")
        if act_provenance.get("task_vocabulary_sha256") != _task_vocabulary_sha256(
            task_vocabulary
        ):
            raise ValueError("ACT checkpoint task vocabulary hash does not match")
        stored_benchmark = act_provenance.get("benchmark")
        if stored_benchmark is not None and str(stored_benchmark).upper() != benchmark_name:
            raise ValueError("ACT checkpoint benchmark provenance does not match")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    previous_manifest = _load_manifest(manifest_path)
    existing_shards = _dataset_shards(output_dir)

    if bool(args.overwrite):
        _clear_owned_dataset(
            output_dir, manifest_path, previous_manifest
        )
        previous_manifest = None
        existing_shards = []
    elif (previous_manifest is not None or existing_shards) and not bool(args.resume):
        raise FileExistsError(
            f"{output_dir} already contains a failure dataset; "
            "pass --resume or --overwrite"
        )
    elif bool(args.resume) and existing_shards and previous_manifest is None:
        raise ValueError(
            "Cannot resume failure shards without manifest.json provenance"
        )

    entries: dict[tuple[int, int], dict[str, Any]] = {}
    if previous_manifest is not None:
        if not bool(args.resume):
            raise FileExistsError(
                f"{output_dir} already contains a failure dataset"
            )
        stored_rollouts = _validate_resume_manifest(
            previous_manifest,
            requested_protocol=protocol,
            requested_rollouts_per_task=int(args.rollouts_per_task),
        )
        entries = _validate_manifest_file_inventory(
            previous_manifest,
            output_dir=output_dir,
            task_vocabulary=task_vocabulary,
            tasks_by_name=tasks_by_name,
            stored_rollouts_per_task=stored_rollouts,
            seed=int(args.seed),
        )

    def _write_progress_manifest(*, complete: bool) -> dict[str, Any]:
        progress_manifest = _build_manifest(
            protocol=protocol,
            checkpoint_path=checkpoint_path,
            rollouts_per_task=int(args.rollouts_per_task),
            entries=entries,
            complete=complete,
        )
        atomic_write_json(manifest_path, progress_manifest)
        return progress_manifest

    # Establish provenance before the first expensive rollout so an
    # interrupted run can be resumed instead of restarted from scratch.
    _write_progress_manifest(complete=False)

    action_std = float(protocol["noise_parameters"]["action_noise_std"])
    observation_std = float(
        protocol["noise_parameters"]["observation_noise_std"]
    )
    for task_id, task_name in enumerate(task_vocabulary):
        environment_type = benchmark.train_classes[task_name]
        env = environment_type(render_mode=None)
        expert = components.expert_policy_map[task_name]()
        task_variants = tasks_by_name[task_name]
        try:
            progress = tqdm(
                range(int(args.rollouts_per_task)),
                desc=f"{benchmark_name} failures {task_name}",
                leave=False,
            )
            for rollout_index in progress:
                key = (task_id, rollout_index)
                if key in entries:
                    continue
                destination = _episode_path(
                    output_dir, task_id, rollout_index
                )
                if destination.exists() or destination.is_symlink():
                    raise ValueError(
                        f"Refusing to overwrite stale shard {destination}; "
                        "use --overwrite after validating dataset ownership"
                    )
                task_variant = rollout_index % len(task_variants)
                task = task_variants[task_variant]
                env.set_task(task)
                episode_seed = int(args.seed) + task_id * 100_000 + rollout_index
                raw, _ = env.reset(seed=episode_seed)
                act.reset()
                action_rng = np.random.default_rng(episode_seed + 10_000_000)
                observation_rng = np.random.default_rng(
                    episode_seed + 20_000_000
                )
                states: list[np.ndarray] = []
                raw_observations: list[np.ndarray] = []
                actions: list[np.ndarray] = []
                expert_actions: list[np.ndarray] = []
                disagreements: list[float] = []
                rewards: list[float] = []
                success = False
                for _ in range(int(args.max_steps)):
                    raw = np.asarray(raw, dtype=np.float32)
                    if raw.shape != (RAW_OBSERVATION_DIM,):
                        raise ValueError(
                            f"{task_name} emitted observation shape {raw.shape}; "
                            f"expected {(RAW_OBSERVATION_DIM,)}"
                        )
                    observed_raw = raw.copy()
                    if observation_std > 0.0:
                        observed_raw += observation_rng.normal(
                            0.0,
                            observation_std,
                            size=RAW_OBSERVATION_DIM,
                        ).astype(np.float32)
                    state = _condition(
                        observed_raw, task_id, len(task_vocabulary)
                    )
                    nominal = np.asarray(
                        act.act(state), dtype=np.float32
                    ).reshape(ACTION_DIM)
                    corrective = np.clip(
                        np.asarray(
                            expert.get_action(raw), dtype=np.float32
                        ).reshape(ACTION_DIM),
                        -1.0,
                        1.0,
                    )
                    executed = nominal.copy()
                    if action_std > 0.0:
                        executed += action_rng.normal(
                            0.0, action_std, size=ACTION_DIM
                        ).astype(np.float32)
                    executed = np.clip(executed, -1.0, 1.0)
                    next_raw, reward, terminated, truncated, info = env.step(
                        executed
                    )
                    states.append(state)
                    raw_observations.append(raw.copy())
                    actions.append(nominal)
                    expert_actions.append(corrective)
                    disagreements.append(
                        float(np.mean(np.abs(nominal - corrective)))
                    )
                    rewards.append(float(reward))
                    success = bool(info.get("success", False))
                    raw = next_raw
                    if success or terminated or truncated:
                        break

                disagreement_array = np.asarray(
                    disagreements, dtype=np.float32
                )
                events = disagreement_array >= float(
                    args.disagreement_threshold
                )
                if not success:
                    events[
                        max(
                            0,
                            len(events) - int(args.terminal_positive_horizon),
                        ) :
                    ] = True
                labels = _future_labels(
                    events, int(args.prediction_horizon)
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                task_sha256 = _task_payload_sha256(task)
                atomic_save_npz(
                    destination,
                    states=np.stack(states).astype(np.float32),
                    raw_observations=np.stack(raw_observations).astype(
                        np.float32
                    ),
                    actions=np.stack(actions).astype(np.float32),
                    expert_actions=np.stack(expert_actions).astype(np.float32),
                    action_disagreement_l1=disagreement_array,
                    risk_events=events.astype(np.bool_),
                    labels=labels.astype(np.bool_),
                    rewards=np.asarray(rewards, dtype=np.float32),
                    success=np.asarray(success, dtype=np.bool_),
                    task_id=np.asarray(task_id, dtype=np.int16),
                    task_name=np.asarray(task_name),
                    task_variant=np.asarray(task_variant, dtype=np.int16),
                    task_payload_sha256=np.asarray(task_sha256),
                    episode_seed=np.asarray(episode_seed, dtype=np.int64),
                    schema_version=np.asarray(SCHEMA_VERSION),
                )
                entries[key] = _read_shard_summary(
                    destination,
                    output_dir=output_dir,
                    task_count=len(task_vocabulary),
                    expected_task_id=task_id,
                    expected_task_name=task_name,
                    expected_rollout_index=rollout_index,
                    expected_task_variant=task_variant,
                    expected_task_payload_sha256=task_sha256,
                    expected_episode_seed=episode_seed,
                )
                _write_progress_manifest(complete=False)
        finally:
            env.close()

    expected_entries = len(task_vocabulary) * int(args.rollouts_per_task)
    if len(entries) != expected_entries:
        raise RuntimeError(
            f"Expected {expected_entries} failure shards, found {len(entries)}"
        )
    expected_keys = {
        (task_id, rollout_index)
        for task_id in range(len(task_vocabulary))
        for rollout_index in range(int(args.rollouts_per_task))
    }
    if set(entries) != expected_keys:
        raise RuntimeError("Failure shard task/rollout slots are incomplete")

    manifest = _write_progress_manifest(complete=True)
    logger.info(
        "%s failure dataset: episodes=%d rows=%d positive=%.2f%% "
        "ACT-success=%.2f%% fingerprint=%s",
        benchmark_name,
        manifest["episodes"],
        manifest["rows"],
        100.0 * manifest["positive_rate"],
        100.0
        * float(np.mean([entry["success"] for entry in manifest["files"]])),
        manifest["provenance_fingerprint_sha256"],
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=("MT10", "MT50"))
    parser.add_argument("--benchmark-seed", type=int, required=True)
    parser.add_argument("--act-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rollouts-per-task", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--noise-level", type=float, default=0.2)
    parser.add_argument("--action-std-scale", type=float, default=0.40)
    parser.add_argument("--observation-std-scale", type=float, default=0.025)
    parser.add_argument("--disagreement-threshold", type=float, default=0.35)
    parser.add_argument("--prediction-horizon", type=int, default=10)
    parser.add_argument("--terminal-positive-horizon", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backend",
        choices=("metaworld", "toy"),
        default="metaworld",
        help=(
            "'toy' selects the explicit deterministic CI benchmark "
            "(env/toy_multitask.py). It is never selected implicitly and its "
            "outputs are engineering artifacts, not benchmark evidence."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-file")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    try:
        manifest = generate(build_parser().parse_args())
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "benchmark": manifest["benchmark"],
                "episodes": manifest["episodes"],
                "rows": manifest["rows"],
                "positive_rate": manifest["positive_rate"],
                "provenance_fingerprint_sha256": manifest[
                    "provenance_fingerprint_sha256"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
