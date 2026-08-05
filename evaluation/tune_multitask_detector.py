#!/usr/bin/env python3
"""Tune an MT10/MT50 detector gate on an independent validation bank.

The tuner is intentionally separate from both detector training and final task
evaluation.  It accepts only a ``generate_multitask_failures.py`` v2 dataset,
verifies the manifest whitelist and every shard digest, checks detector/task
provenance, rebuilds causal histories, and selects one shared deployment
threshold.  The final evaluation bank is never opened by this program.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.io import atomic_write_json, file_sha256
from models.failure_detector import FailureDetector
from trainers.data import (
    FailureData,
    MULTITASK_CALIBRATION_SCHEMA,
    audit_calibrated_multitask_failure_bank,
    load_failure_data,
)
from utils.common import (
    configure_logging,
    load_yaml,
    resolve_path,
    seed_everything,
    select_device,
)


LOGGER = logging.getLogger("reim.tune_multitask_detector")
SCHEMA_VERSION = "reim-multitask-failures-v2"
DATASET_TYPE = "task_conditioned_behavioral_deviation_risk"
OUTPUT_SCHEMA_VERSION = "reim-multitask-detector-threshold-v1"
DETECTOR_TRAINING_SCHEMA = "reim-failure-detector-training-v2"
RAW_OBSERVATION_DIM = 39
SUPPORTED_BENCHMARKS = {"MT10": 10, "MT50": 50}
DEFAULT_VALIDATION_SEEDS = {"MT10": 20264010, "MT50": 20264050}
DEFAULT_FINAL_SEEDS = {"MT10": 20265010, "MT50": 20265050}
_SHARD_PATTERN = re.compile(r"^task_(\d{2})/failure_(\d{4,})\.npz$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _torch_load(path: Path, map_location: str = "cpu") -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise TypeError("Detector checkpoint must be a mapping with provenance")
    return payload


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


def _v2_top_level_fields(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct generator v2's duplicated, human-readable protocol fields."""

    try:
        noise = provenance["noise_parameters"]
        labels = provenance["label_parameters"]
        return {
            "schema_version": provenance["schema_version"],
            "dataset_type": provenance["dataset_type"],
            "benchmark": provenance["benchmark"],
            "metaworld_version": provenance["metaworld_version"],
            "benchmark_seed": provenance["benchmark_seed"],
            "seed": provenance["collection_seed"],
            "task_vocabulary": provenance["task_vocabulary"],
            "task_vocabulary_sha256": provenance["task_vocabulary_sha256"],
            "benchmark_task_bank_sha256": provenance[
                "benchmark_task_bank_sha256"
            ],
            "act_checkpoint_sha256": provenance["act_checkpoint_sha256"],
            "observation_schema": provenance["observation_schema"],
            "state_dim": provenance["state_dim"],
            "action_dim": provenance["action_dim"],
            "max_episode_steps": provenance["max_episode_steps"],
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
            "terminal_positive_horizon": labels["terminal_positive_horizon"],
        }
    except (KeyError, TypeError) as exc:
        raise ValueError("Validation manifest has incomplete v2 provenance") from exc


def _scalar(archive: Any, key: str, *, path: Path) -> Any:
    value = np.asarray(archive[key])
    if value.size != 1:
        raise ValueError(f"{path}: {key} must be scalar")
    return value.reshape(-1)[0]


def _safe_relative_shard(value: Any) -> Path:
    text = str(value)
    relative = Path(text)
    if (
        not text
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != text
        or _SHARD_PATTERN.fullmatch(text) is None
    ):
        raise ValueError(f"Unsafe or non-canonical manifest shard path {text!r}")
    return relative


def _require_sha256(value: Any, *, field: str) -> str:
    digest = str(value)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return digest


def _audit_shard(
    path: Path,
    entry: Mapping[str, Any],
    *,
    task_vocabulary: Sequence[str],
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Validation shard must be a regular, non-symlink file: {path}")
    stored_digest = _require_sha256(entry.get("sha256"), field=f"{path} sha256")
    actual_digest = file_sha256(path)
    if actual_digest != stored_digest:
        raise ValueError(
            f"Validation shard SHA256 mismatch for {path}: "
            f"stored={stored_digest}, actual={actual_digest}"
        )

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
            states = np.asarray(archive["states"], dtype=np.float32)
            raw = np.asarray(archive["raw_observations"], dtype=np.float32)
            actions = np.asarray(archive["actions"], dtype=np.float32)
            expert_actions = np.asarray(archive["expert_actions"], dtype=np.float32)
            disagreements = np.asarray(
                archive["action_disagreement_l1"], dtype=np.float32
            )
            events = np.asarray(archive["risk_events"])
            labels = np.asarray(archive["labels"])
            rewards = np.asarray(archive["rewards"], dtype=np.float32)
            task_id = int(_scalar(archive, "task_id", path=path))
            task_name = str(_scalar(archive, "task_name", path=path))
            schema_version = str(_scalar(archive, "schema_version", path=path))
    except (OSError, KeyError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(str(path)):
            raise
        raise ValueError(f"Cannot validate shard {path}: {exc}") from exc

    length = len(states) if states.ndim > 0 else 0
    state_dim = RAW_OBSERVATION_DIM + len(task_vocabulary)
    expected_shapes = {
        "states": (length, state_dim),
        "raw_observations": (length, RAW_OBSERVATION_DIM),
        "actions": (length, 4),
        "expert_actions": (length, 4),
        "action_disagreement_l1": (length,),
        "risk_events": (length,),
        "labels": (length,),
        "rewards": (length,),
    }
    actual_shapes = {
        "states": states.shape,
        "raw_observations": raw.shape,
        "actions": actions.shape,
        "expert_actions": expert_actions.shape,
        "action_disagreement_l1": disagreements.shape,
        "risk_events": events.shape,
        "labels": labels.shape,
        "rewards": rewards.shape,
    }
    invalid_shapes = {
        key: (actual_shapes[key], shape)
        for key, shape in expected_shapes.items()
        if actual_shapes[key] != shape
    }
    if length <= 0 or invalid_shapes:
        raise ValueError(f"{path}: empty or invalid trajectory shapes {invalid_shapes}")
    if not all(
        np.isfinite(array).all()
        for array in (states, raw, actions, expert_actions, disagreements, rewards)
    ):
        raise ValueError(f"{path}: trajectory contains non-finite values")
    if not np.isin(labels, (0, 1, False, True)).all() or not np.isin(
        events, (0, 1, False, True)
    ).all():
        raise ValueError(f"{path}: labels and risk_events must be binary")
    if not 0 <= task_id < len(task_vocabulary):
        raise ValueError(f"{path}: task_id {task_id} is out of range")
    if task_name != task_vocabulary[task_id]:
        raise ValueError(f"{path}: task_id/task_name disagree with vocabulary order")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"{path}: shard schema is {schema_version!r}, expected v2")

    one_hot = states[:, RAW_OBSERVATION_DIM:]
    expected_one_hot = np.zeros(len(task_vocabulary), dtype=np.float32)
    expected_one_hot[task_id] = 1.0
    if not np.allclose(one_hot, expected_one_hot[None, :], rtol=0.0, atol=1e-6):
        raise ValueError(f"{path}: task one-hot is not constant or vocabulary-aligned")

    entry_metadata = {
        "task_id": task_id,
        "task_name": task_name,
        "length": length,
        "positive_labels": int(np.asarray(labels, dtype=np.bool_).sum()),
    }
    mismatches = [
        f"{key}: manifest={entry.get(key)!r}, shard={value!r}"
        for key, value in entry_metadata.items()
        if entry.get(key) != value
    ]
    if mismatches:
        raise ValueError(f"{path}: manifest metadata mismatch: " + "; ".join(mismatches))
    return {**entry_metadata, "sha256": actual_digest}


def audit_validation_bank(
    data_dir: str | Path,
    *,
    benchmark: str,
    expected_validation_seed: int,
    expected_benchmark_seed: int,
    forbidden_final_seed: int,
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    """Verify the complete v2 manifest whitelist and validation-bank identity."""

    benchmark = benchmark.upper()
    if benchmark not in SUPPORTED_BENCHMARKS:
        raise ValueError(f"Unsupported benchmark {benchmark!r}")
    if expected_validation_seed == forbidden_final_seed:
        raise ValueError("Validation and final-evaluation bank seeds must be distinct")
    directory = resolve_path(data_dir).resolve()
    audited_bank = audit_calibrated_multitask_failure_bank(
        directory,
        expected_role="validation",
        expected_mode="frozen-task-thresholds",
    )
    manifest = audited_bank.manifest
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Threshold tuning requires a failure-dataset v2 manifest")
    if manifest.get("dataset_type") != DATASET_TYPE:
        raise ValueError("Manifest is not a task-conditioned behavioral-risk dataset")
    if manifest.get("complete") is not True:
        raise ValueError("Validation manifest is not marked complete")
    if str(manifest.get("benchmark", "")).upper() != benchmark:
        raise ValueError("Validation manifest benchmark does not match --benchmark")

    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Validation manifest lacks structured v2 provenance")
    stored_fingerprint = _require_sha256(
        manifest.get("provenance_fingerprint_sha256"),
        field="provenance_fingerprint_sha256",
    )
    if _canonical_json_sha256(provenance) != stored_fingerprint:
        raise ValueError("Validation manifest provenance fingerprint mismatch")
    top_level_mismatches = [
        key
        for key, value in _v2_top_level_fields(provenance).items()
        if manifest.get(key) != value
    ]
    if top_level_mismatches:
        raise ValueError(
            "Validation top-level protocol disagrees with v2 provenance: "
            + ", ".join(top_level_mismatches)
        )
    _require_sha256(
        manifest.get("benchmark_task_bank_sha256"),
        field="benchmark_task_bank_sha256",
    )
    _require_sha256(
        manifest.get("act_checkpoint_sha256"), field="act_checkpoint_sha256"
    )

    try:
        collection_seed = int(manifest["seed"])
        provenance_collection_seed = int(provenance["collection_seed"])
        benchmark_seed = int(manifest["benchmark_seed"])
        provenance_benchmark_seed = int(provenance["benchmark_seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Validation manifest bank seeds are missing or invalid") from exc
    if collection_seed != provenance_collection_seed:
        raise ValueError("Validation collection seed disagrees with provenance")
    if benchmark_seed != provenance_benchmark_seed:
        raise ValueError("Validation benchmark seed disagrees with provenance")
    if collection_seed == forbidden_final_seed or benchmark_seed == forbidden_final_seed:
        raise ValueError("Refusing to tune on the reserved final-evaluation bank")
    if collection_seed != int(expected_validation_seed):
        raise ValueError(
            f"Validation collection seed {collection_seed} does not match the "
            f"registered validation seed {expected_validation_seed}"
        )
    if benchmark_seed != int(expected_benchmark_seed):
        raise ValueError(
            f"Validation benchmark seed {benchmark_seed} does not match the "
            f"registered validation benchmark seed {expected_benchmark_seed}"
        )

    task_vocabulary = manifest.get("task_vocabulary")
    if not isinstance(task_vocabulary, list) or not all(
        isinstance(name, str) and name for name in task_vocabulary
    ):
        raise ValueError("Manifest task_vocabulary must be an ordered string list")
    expected_task_count = SUPPORTED_BENCHMARKS[benchmark]
    if len(task_vocabulary) != expected_task_count or len(set(task_vocabulary)) != len(
        task_vocabulary
    ):
        raise ValueError(
            f"{benchmark} requires {expected_task_count} unique ordered task names"
        )
    vocabulary_digest = _task_vocabulary_sha256(task_vocabulary)
    if manifest.get("task_vocabulary_sha256") != vocabulary_digest:
        raise ValueError("Manifest task vocabulary SHA256 mismatch")
    if provenance.get("task_vocabulary") != task_vocabulary or provenance.get(
        "task_vocabulary_sha256"
    ) != vocabulary_digest:
        raise ValueError("Manifest and provenance task vocabularies disagree")
    state_dim = RAW_OBSERVATION_DIM + expected_task_count
    if int(manifest.get("state_dim", -1)) != state_dim or int(
        provenance.get("state_dim", -1)
    ) != state_dim:
        raise ValueError("Validation state_dim does not match raw39 + task one-hot")

    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Validation manifest files inventory must be non-empty")
    try:
        rollouts_per_task = int(manifest["rollouts_per_task"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Manifest rollouts_per_task is invalid") from exc
    if rollouts_per_task <= 0 or len(entries) != expected_task_count * rollouts_per_task:
        raise ValueError("Manifest is not a complete task-balanced validation bank")

    listed_paths: list[Path] = []
    listed_text: list[str] = []
    summaries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"Manifest files[{index}] is not an object")
        relative = _safe_relative_shard(raw_entry.get("file"))
        relative_text = relative.as_posix()
        if relative_text in seen:
            raise ValueError(f"Duplicate manifest shard {relative_text}")
        seen.add(relative_text)
        match = _SHARD_PATTERN.fullmatch(relative_text)
        assert match is not None
        path_task_id = int(match.group(1))
        rollout_index = int(match.group(2))
        expected_task_id = index // rollouts_per_task
        expected_rollout = index % rollouts_per_task
        if path_task_id != expected_task_id or rollout_index != expected_rollout:
            raise ValueError("Manifest files are not in canonical task/rollout order")
        if raw_entry.get("rollout_index") != rollout_index:
            raise ValueError(f"{relative_text}: rollout_index metadata mismatch")
        absolute = directory / relative
        listed_paths.append(absolute)
        listed_text.append(relative_text)
        summaries.append(
            _audit_shard(absolute, raw_entry, task_vocabulary=task_vocabulary)
        )

    disk_paths = sorted(
        path
        for path in directory.rglob("*.npz")
        if path.is_file() or path.is_symlink()
    )
    disk_text = [path.relative_to(directory).as_posix() for path in disk_paths]
    missing = sorted(set(listed_text) - set(disk_text))
    stale = sorted(set(disk_text) - set(listed_text))
    if missing or stale:
        raise ValueError(
            "Validation file inventory mismatch: "
            f"missing={missing[:5]}, stale={stale[:5]}"
        )
    if disk_text != listed_text:
        raise ValueError("Validation files are not in manifest canonical order")

    rows = sum(item["length"] for item in summaries)
    positives = sum(item["positive_labels"] for item in summaries)
    if manifest.get("episodes") != len(entries) or manifest.get("rows") != rows:
        raise ValueError("Manifest aggregate counts disagree with its whitelist")
    try:
        positive_rate = float(manifest["positive_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Manifest positive_rate is invalid") from exc
    if not math.isclose(positive_rate, positives / rows, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("Manifest positive_rate disagrees with its whitelist")
    if tuple(path.resolve() for path in listed_paths) != tuple(
        path.resolve() for path in audited_bank.whitelist
    ):
        raise ValueError("Validation audit whitelist implementations disagree")
    return manifest, tuple(listed_paths)


def _load_audited_failure_data(
    data_dir: Path,
    whitelist: Sequence[Path],
    *,
    sequence_length: int,
    task_vocabulary: Sequence[str],
) -> tuple[FailureData, np.ndarray]:
    data = load_failure_data(data_dir, sequence_length)
    resolved_files = tuple(path.resolve() for path in data.files)
    expected_files = tuple(path.resolve() for path in whitelist)
    if resolved_files != expected_files:
        raise ValueError("Causal loader files differ from the audited manifest whitelist")
    if len(np.unique(data.groups)) != len(whitelist):
        raise ValueError("Each v2 validation shard must contain exactly one trajectory")

    group_task_ids = np.asarray(
        [index // (len(whitelist) // len(task_vocabulary)) for index in range(len(whitelist))],
        dtype=np.int64,
    )
    groups = np.asarray(data.groups, dtype=np.int64)
    if groups.min(initial=0) < 0 or groups.max(initial=-1) >= len(group_task_ids):
        raise ValueError("Causal loader returned invalid trajectory group identifiers")
    task_ids = group_task_ids[groups]
    if data.windows.shape[-1] != RAW_OBSERVATION_DIM + len(task_vocabulary):
        raise ValueError("Causal windows have an incompatible state dimension")
    one_hot = data.windows[:, :, RAW_OBSERVATION_DIM:]
    for index, (task_id, length) in enumerate(zip(task_ids, data.lengths)):
        expected = np.zeros(len(task_vocabulary), dtype=np.float32)
        expected[int(task_id)] = 1.0
        if not np.allclose(
            one_hot[index, : int(length)], expected[None, :], rtol=0.0, atol=1e-6
        ):
            raise ValueError("Causal window task one-hot disagrees with manifest order")
    return data, task_ids


def _binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(probabilities) >= float(threshold)
    tp = int(np.sum((labels == 1) & predictions))
    fp = int(np.sum((labels == 0) & predictions))
    fn = int(np.sum((labels == 1) & ~predictions))
    tn = int(np.sum((labels == 0) & ~predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_positive_rate": float(np.mean(predictions)),
    }


def binary_auroc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    """Compute tie-aware Mann--Whitney AUROC without a sklearn dependency."""

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(probabilities, kind="mergesort")
    sorted_scores = probabilities[order]
    ranks = np.empty(len(labels), dtype=np.float64)
    start = 0
    while start < len(labels):
        stop = start + 1
        while stop < len(labels) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    positive_rank_sum = float(ranks[labels == 1].sum())
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def binary_auprc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    """Compute threshold-grouped average precision (area under stepwise PR)."""

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = probabilities[order]
    distinct_ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(labels) - 1]
    true_positives = np.cumsum(sorted_labels)[distinct_ends]
    predicted_positives = distinct_ends + 1
    recall = true_positives / positives
    precision = true_positives / predicted_positives
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increment * precision))


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 15,
) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if bins <= 0:
        raise ValueError("ECE bins must be positive")
    if len(labels) == 0 or len(labels) != len(probabilities):
        raise ValueError("ECE requires equally sized, non-empty arrays")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    bin_ids = np.minimum((probabilities * bins).astype(np.int64), bins - 1)
    ece = 0.0
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(probabilities[mask].mean()) - float(labels[mask].mean())
            )
    return float(ece)


def threshold_grid_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    task_ids: np.ndarray,
    thresholds: Sequence[float],
    *,
    task_count: int,
    precision_floor: float,
    ece_bins: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    task_ids = np.asarray(task_ids, dtype=np.int64)
    if not (len(labels) == len(probabilities) == len(task_ids)) or len(labels) == 0:
        raise ValueError("Labels, probabilities, and task_ids must be equally sized")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Threshold labels must be binary")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("Threshold probabilities must be finite and in [0, 1]")
    if not 0.0 <= precision_floor <= 1.0:
        raise ValueError("precision_floor must lie in [0, 1]")
    threshold_values = np.asarray(list(thresholds), dtype=np.float64)
    if (
        threshold_values.ndim != 1
        or len(threshold_values) == 0
        or not np.isfinite(threshold_values).all()
        or np.any((threshold_values < 0.0) | (threshold_values > 1.0))
        or np.any(np.diff(threshold_values) <= 0.0)
    ):
        raise ValueError("Threshold grid must be finite, unique, sorted, and in [0, 1]")
    expected_tasks = set(range(task_count))
    observed_tasks = set(np.unique(task_ids).tolist())
    if observed_tasks != expected_tasks:
        raise ValueError(
            f"Validation rows cover task ids {sorted(observed_tasks)}, "
            f"expected {sorted(expected_tasks)}"
        )

    per_task_ranking: list[dict[str, Any]] = []
    task_aurocs: list[float] = []
    task_auprcs: list[float] = []
    task_eces: list[float] = []
    for task_id in range(task_count):
        selected = task_ids == task_id
        auroc = binary_auroc(labels[selected], probabilities[selected])
        auprc = binary_auprc(labels[selected], probabilities[selected])
        ece = expected_calibration_error(
            labels[selected], probabilities[selected], bins=ece_bins
        )
        if auroc is not None:
            task_aurocs.append(auroc)
        if auprc is not None:
            task_auprcs.append(auprc)
        task_eces.append(ece)
        per_task_ranking.append(
            {
                "task_id": task_id,
                "samples": int(selected.sum()),
                "positive_rate": float(labels[selected].mean()),
                "auroc": auroc,
                "auprc": auprc,
                "ece": ece,
            }
        )

    ranking = {
        "micro_auroc": binary_auroc(labels, probabilities),
        "micro_auprc": binary_auprc(labels, probabilities),
        "micro_ece": expected_calibration_error(labels, probabilities, bins=ece_bins),
        "task_macro_auroc": float(np.mean(task_aurocs)) if task_aurocs else None,
        "task_macro_auprc": float(np.mean(task_auprcs)) if task_auprcs else None,
        "task_macro_ece": float(np.mean(task_eces)),
        "auroc_eligible_task_count": len(task_aurocs),
        "auprc_eligible_task_count": len(task_auprcs),
        "ece_bins": int(ece_bins),
    }

    grid: list[dict[str, Any]] = []
    for threshold in threshold_values:
        per_task = [
            _binary_metrics(
                labels[task_ids == task_id],
                probabilities[task_ids == task_id],
                float(threshold),
            )
            for task_id in range(task_count)
        ]
        micro = _binary_metrics(labels, probabilities, float(threshold))
        row: dict[str, Any] = {
            "threshold": float(threshold),
            "task_macro_precision": float(np.mean([item["precision"] for item in per_task])),
            "task_macro_recall": float(np.mean([item["recall"] for item in per_task])),
            "task_macro_f1": float(np.mean([item["f1"] for item in per_task])),
            "micro_precision": micro["precision"],
            "micro_recall": micro["recall"],
            "micro_f1": micro["f1"],
            "predicted_positive_rate": micro["predicted_positive_rate"],
            "precision_constraint_satisfied": bool(
                np.mean([item["precision"] for item in per_task]) + 1e-12
                >= precision_floor
            ),
            **ranking,
        }
        grid.append(row)

    feasible = [row for row in grid if row["precision_constraint_satisfied"]]
    candidates = feasible if feasible else grid
    selected = max(
        candidates,
        key=lambda row: (
            row["task_macro_f1"],
            row["task_macro_recall"],
            row["task_macro_precision"],
            row["threshold"],
        ),
    )
    selection = {
        **selected,
        "precision_floor": float(precision_floor),
        "selection_status": (
            "precision_floor_satisfied"
            if feasible
            else "fallback_no_threshold_satisfied_precision_floor"
        ),
        "tie_break_order": [
            "task_macro_f1",
            "task_macro_recall",
            "task_macro_precision",
            "higher_threshold",
        ],
    }
    return grid, selection, per_task_ranking


def _parse_thresholds(args: argparse.Namespace) -> list[float]:
    explicit = getattr(args, "thresholds", None)
    if explicit:
        try:
            values = [float(value.strip()) for value in str(explicit).split(",")]
        except ValueError as exc:
            raise ValueError("--thresholds must be a comma-separated float list") from exc
        return sorted(set(values))
    minimum = float(args.threshold_min)
    maximum = float(args.threshold_max)
    count = int(args.threshold_count)
    if count <= 0 or not 0.0 <= minimum <= maximum <= 1.0:
        raise ValueError("Threshold bounds/count are invalid")
    return np.linspace(minimum, maximum, count, dtype=np.float64).tolist()


def _protocol_bank_seeds(args: argparse.Namespace, benchmark: str) -> tuple[int, int, int]:
    config_value = getattr(args, "protocol_config", None)
    config_path = resolve_path(
        config_value or f"configs/multitask/{benchmark.lower()}.yaml"
    )
    config: dict[str, Any] = {}
    if config_path.is_file():
        config = load_yaml(config_path)
        configured_benchmark = str(config.get("benchmark", benchmark)).upper()
        if configured_benchmark != benchmark:
            raise ValueError("Protocol config benchmark does not match --benchmark")
    elif config_value:
        raise FileNotFoundError(f"Protocol config not found: {config_path}")
    banks = config.get("banks", {}) if isinstance(config.get("banks", {}), Mapping) else {}
    validation_seed = int(
        args.validation_bank_seed
        if getattr(args, "validation_bank_seed", None) is not None
        else banks.get("validation", DEFAULT_VALIDATION_SEEDS[benchmark])
    )
    validation_benchmark_seed = int(
        args.validation_benchmark_seed
        if getattr(args, "validation_benchmark_seed", None) is not None
        else validation_seed
    )
    final_seed = int(banks.get("final_evaluation", DEFAULT_FINAL_SEEDS[benchmark]))
    return validation_seed, validation_benchmark_seed, final_seed


def _validate_detector_checkpoint(
    checkpoint_path: Path,
    *,
    benchmark: str,
    task_vocabulary: Sequence[str],
    validation_manifest: Mapping[str, Any],
    validation_manifest_sha256: str,
    device: str,
) -> tuple[FailureDetector, Mapping[str, Any], str]:
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise ValueError("Detector checkpoint must be a regular, non-symlink file")
    checkpoint = _torch_load(checkpoint_path, "cpu")
    vocabulary_digest = _task_vocabulary_sha256(task_vocabulary)
    if str(checkpoint.get("benchmark", "")).upper() != benchmark:
        raise ValueError("Detector checkpoint benchmark provenance mismatch")
    if checkpoint.get("task_vocabulary") != list(task_vocabulary):
        raise ValueError("Detector checkpoint task vocabulary/order mismatch")
    if checkpoint.get("task_vocabulary_sha256") != vocabulary_digest:
        raise ValueError("Detector checkpoint task vocabulary SHA256 mismatch")
    if checkpoint.get("detector_training_schema") != DETECTOR_TRAINING_SCHEMA:
        raise ValueError(
            "Detector checkpoint lacks the calibrated multi-task training schema"
        )
    expected_training_fields = {
        "data_schema_version": SCHEMA_VERSION,
        "dataset_type": DATASET_TYPE,
        "dataset_role": "training",
        "label_calibration_mode": "fit-task-quantile",
    }
    mismatched_training_fields = [
        field
        for field, expected in expected_training_fields.items()
        if checkpoint.get(field) != expected
    ]
    if mismatched_training_fields:
        raise ValueError(
            "Detector checkpoint calibration provenance is missing or invalid: "
            + ", ".join(mismatched_training_fields)
        )
    training_manifest_digest = _require_sha256(
        checkpoint.get("data_manifest_sha256"), field="checkpoint data_manifest_sha256"
    )
    if training_manifest_digest == validation_manifest_sha256:
        raise ValueError(
            "Detector checkpoint was trained from this validation manifest; "
            "an independent validation bank is required"
        )
    _require_sha256(
        checkpoint.get("dataset_fingerprint_sha256"),
        field="checkpoint dataset_fingerprint_sha256",
    )
    training_calibration_digest = _require_sha256(
        checkpoint.get("label_calibration_fingerprint_sha256"),
        field="checkpoint label_calibration_fingerprint_sha256",
    )
    training_source_digest = _require_sha256(
        checkpoint.get("label_calibration_source_sha256"),
        field="checkpoint label_calibration_source_sha256",
    )
    training_calibration = checkpoint.get("label_calibration")
    if not isinstance(training_calibration, Mapping):
        raise ValueError("Detector checkpoint lacks full label_calibration provenance")
    if _canonical_json_sha256(training_calibration) != training_calibration_digest:
        raise ValueError("Detector checkpoint label_calibration fingerprint mismatch")
    expected_calibration_fields = {
        "schema_version": MULTITASK_CALIBRATION_SCHEMA,
        "mode": "fit-task-quantile",
        "dataset_role": "training",
        "benchmark": benchmark,
        "task_vocabulary_sha256": vocabulary_digest,
        "comparison": "greater_than_or_equal",
        "quantile_method": "linear",
    }
    mismatched_calibration_fields = [
        field
        for field, expected in expected_calibration_fields.items()
        if training_calibration.get(field) != expected
    ]
    if mismatched_calibration_fields:
        raise ValueError(
            "Detector checkpoint training calibration is invalid: "
            + ", ".join(mismatched_calibration_fields)
        )
    try:
        training_quantile = float(training_calibration["quantile"])
        checkpoint_quantile = float(checkpoint["label_calibration_quantile"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Detector checkpoint calibration quantile is invalid") from exc
    if (
        not math.isfinite(training_quantile)
        or not 0.0 < training_quantile < 1.0
        or checkpoint_quantile != training_quantile
    ):
        raise ValueError("Detector checkpoint calibration quantile provenance mismatch")
    training_source = training_calibration.get("calibration_source")
    if not isinstance(training_source, Mapping):
        raise ValueError("Detector checkpoint training calibration source is missing")
    if (
        training_source.get("kind") != "training_bank_raw_action_disagreement"
        or training_source.get("sha256") != training_source_digest
        or training_calibration.get("target_raw_disagreement_sha256")
        != training_source_digest
    ):
        raise ValueError("Detector checkpoint training calibration source mismatch")
    training_thresholds = training_calibration.get("task_thresholds")
    if not isinstance(training_thresholds, list) or len(training_thresholds) != len(
        task_vocabulary
    ):
        raise ValueError("Detector checkpoint lacks one threshold per benchmark task")
    for task_id, (task_name, threshold_entry) in enumerate(
        zip(task_vocabulary, training_thresholds, strict=True)
    ):
        if not isinstance(threshold_entry, Mapping):
            raise ValueError("Detector checkpoint task threshold must be an object")
        try:
            threshold = float(threshold_entry["threshold"])
            sample_count = int(threshold_entry["samples"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Detector checkpoint task threshold is malformed") from exc
        if (
            threshold_entry.get("task_id") != task_id
            or threshold_entry.get("task_name") != task_name
            or not math.isfinite(threshold)
            or threshold < 0.0
            or sample_count <= 0
        ):
            raise ValueError(
                "Detector checkpoint task thresholds are not in official order"
            )

    validation_calibration = validation_manifest.get("label_calibration")
    if not isinstance(validation_calibration, Mapping):
        raise ValueError("Validation bank lacks frozen calibration provenance")
    validation_source = validation_calibration.get("calibration_source")
    if not isinstance(validation_source, Mapping):
        raise ValueError("Validation bank lacks a frozen training calibration source")
    validation_source_digest = _require_sha256(
        validation_source.get("sha256"), field="validation calibration source sha256"
    )
    validation_source_manifest_digest = _require_sha256(
        validation_source.get("manifest_sha256"),
        field="validation calibration source manifest_sha256",
    )
    if validation_source_digest != training_calibration_digest:
        raise ValueError(
            "Validation bank does not reuse this detector's training calibration"
        )
    if validation_source_manifest_digest != training_manifest_digest:
        raise ValueError(
            "Validation bank calibration source is not this detector's training manifest"
        )
    try:
        validation_quantile = float(validation_calibration["quantile"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Validation calibration quantile is invalid") from exc
    if validation_quantile != training_quantile:
        raise ValueError("Validation and detector-training quantiles differ")
    if validation_calibration.get("task_thresholds") != training_thresholds:
        raise ValueError(
            "Validation bank task thresholds differ from detector-training thresholds"
        )
    expected_state_dim = RAW_OBSERVATION_DIM + len(task_vocabulary)
    if int(checkpoint.get("state_dim", -1)) != expected_state_dim:
        raise ValueError("Detector checkpoint state_dim does not match benchmark tasks")
    try:
        sequence_length = int(checkpoint["sequence_length"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Detector checkpoint lacks a valid sequence_length") from exc
    if sequence_length <= 0:
        raise ValueError("Detector checkpoint sequence_length must be positive")
    model = FailureDetector.from_checkpoint(checkpoint_path, map_location=device)
    if model.state_dim != expected_state_dim or model.sequence_length != sequence_length:
        raise ValueError("Reloaded detector architecture disagrees with checkpoint metadata")
    return model, checkpoint, file_sha256(checkpoint_path)


def _infer_probabilities(
    model: FailureDetector,
    data: FailureData,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model.eval()
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(data.labels), batch_size):
            stop = min(start + batch_size, len(data.labels))
            windows = torch.from_numpy(data.windows[start:stop]).to(device)
            lengths = torch.from_numpy(data.lengths[start:stop]).to(device)
            batches.append(torch.sigmoid(model(windows, lengths)).cpu().numpy())
    probabilities = np.concatenate(batches).astype(np.float64, copy=False)
    if not np.isfinite(probabilities).all():
        raise ValueError("Detector emitted non-finite validation probabilities")
    return probabilities


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty threshold grid")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def tune(args: argparse.Namespace) -> dict[str, Any]:
    benchmark = str(args.benchmark).upper()
    if benchmark not in SUPPORTED_BENCHMARKS:
        raise ValueError(f"Unsupported benchmark {benchmark!r}")
    seed_everything(int(args.seed), deterministic_torch=True)
    device = select_device(args.device)
    logger = configure_logging(
        "tune_multitask_detector",
        args.log_file or f"results/logs/{benchmark.lower()}_detector_threshold.log",
    )
    validation_seed, validation_benchmark_seed, final_seed = _protocol_bank_seeds(
        args, benchmark
    )
    data_dir = resolve_path(args.validation_data).resolve()
    manifest, whitelist = audit_validation_bank(
        data_dir,
        benchmark=benchmark,
        expected_validation_seed=validation_seed,
        expected_benchmark_seed=validation_benchmark_seed,
        forbidden_final_seed=final_seed,
    )
    manifest_path = data_dir / "manifest.json"
    manifest_digest = file_sha256(manifest_path)
    task_vocabulary = list(manifest["task_vocabulary"])
    checkpoint_path = resolve_path(args.detector_checkpoint).resolve()
    model, checkpoint, checkpoint_digest = _validate_detector_checkpoint(
        checkpoint_path,
        benchmark=benchmark,
        task_vocabulary=task_vocabulary,
        validation_manifest=manifest,
        validation_manifest_sha256=manifest_digest,
        device=device,
    )
    sequence_length = int(checkpoint["sequence_length"])
    data, task_ids = _load_audited_failure_data(
        data_dir,
        whitelist,
        sequence_length=sequence_length,
        task_vocabulary=task_vocabulary,
    )
    probabilities = _infer_probabilities(
        model, data, batch_size=int(args.batch_size), device=device
    )
    thresholds = _parse_thresholds(args)
    grid, selection, per_task_ranking = threshold_grid_metrics(
        data.labels,
        probabilities,
        task_ids,
        thresholds,
        task_count=len(task_vocabulary),
        precision_floor=float(args.precision_floor),
        ece_bins=int(args.ece_bins),
    )

    selected_threshold = float(selection["threshold"])
    selected_per_task: list[dict[str, Any]] = []
    for task_id, task_name in enumerate(task_vocabulary):
        selected = task_ids == task_id
        selected_per_task.append(
            {
                **per_task_ranking[task_id],
                "task_name": task_name,
                **_binary_metrics(
                    data.labels[selected], probabilities[selected], selected_threshold
                ),
            }
        )

    csv_rows = [
        {
            "benchmark": benchmark,
            "threshold": row["threshold"],
            "task_macro_precision": row["task_macro_precision"],
            "task_macro_recall": row["task_macro_recall"],
            "task_macro_f1": row["task_macro_f1"],
            "micro_precision": row["micro_precision"],
            "micro_recall": row["micro_recall"],
            "micro_f1": row["micro_f1"],
            "predicted_positive_rate": row["predicted_positive_rate"],
            "precision_constraint_satisfied": row[
                "precision_constraint_satisfied"
            ],
            "selected": bool(math.isclose(row["threshold"], selected_threshold)),
            "task_macro_auroc": row["task_macro_auroc"],
            "task_macro_auprc": row["task_macro_auprc"],
            "task_macro_ece": row["task_macro_ece"],
            "micro_auroc": row["micro_auroc"],
            "micro_auprc": row["micro_auprc"],
            "micro_ece": row["micro_ece"],
        }
        for row in grid
    ]
    output_json = resolve_path(
        args.output_json
        or f"results/tables/{benchmark.lower()}_detector_threshold.json"
    )
    output_csv = resolve_path(
        args.output_csv
        or f"results/tables/{benchmark.lower()}_detector_threshold_grid.csv"
    )
    payload: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "benchmark": benchmark,
        "bank_role": "validation_only",
        "final_bank_accessed": False,
        "selection": selection,
        "ranking_and_calibration": {
            key: selection[key]
            for key in (
                "micro_auroc",
                "micro_auprc",
                "micro_ece",
                "task_macro_auroc",
                "task_macro_auprc",
                "task_macro_ece",
                "auroc_eligible_task_count",
                "auprc_eligible_task_count",
                "ece_bins",
            )
        },
        "dataset": {
            "task_count": len(task_vocabulary),
            "trajectory_count": len(whitelist),
            "sample_count": int(len(data.labels)),
            "positive_rate": float(data.labels.mean()),
            "sequence_length": sequence_length,
            "task_vocabulary": task_vocabulary,
            "per_task": selected_per_task,
        },
        "provenance": {
            "validation_manifest": str(manifest_path),
            "validation_manifest_sha256": manifest_digest,
            "validation_provenance_fingerprint_sha256": manifest[
                "provenance_fingerprint_sha256"
            ],
            "validation_collection_seed": validation_seed,
            "validation_benchmark_seed": validation_benchmark_seed,
            "reserved_final_evaluation_seed": final_seed,
            "detector_checkpoint": str(checkpoint_path),
            "detector_checkpoint_sha256": checkpoint_digest,
            "detector_training_manifest_sha256": checkpoint[
                "data_manifest_sha256"
            ],
            "detector_training_dataset_fingerprint_sha256": checkpoint[
                "dataset_fingerprint_sha256"
            ],
            "detector_training_calibration_fingerprint_sha256": checkpoint[
                "label_calibration_fingerprint_sha256"
            ],
            "validation_dataset_fingerprint_sha256": manifest[
                "dataset_fingerprint_sha256"
            ],
            "validation_calibration_fingerprint_sha256": manifest[
                "label_calibration_fingerprint_sha256"
            ],
            "task_vocabulary_sha256": manifest["task_vocabulary_sha256"],
            "manifest_whitelist_sha256": _canonical_json_sha256(
                [
                    {"file": entry["file"], "sha256": entry["sha256"]}
                    for entry in manifest["files"]
                ]
            ),
            "seed": int(args.seed),
        },
        "threshold_grid": grid,
        "outputs": {"json": str(output_json), "csv": str(output_csv)},
    }
    atomic_write_json(output_json, payload)
    _atomic_write_csv(output_csv, csv_rows)
    logger.info(
        "%s validation-only threshold=%.4f macro-F1=%.4f macro-P=%.4f "
        "macro-R=%.4f status=%s samples=%d",
        benchmark,
        selected_threshold,
        selection["task_macro_f1"],
        selection["task_macro_precision"],
        selection["task_macro_recall"],
        selection["selection_status"],
        len(data.labels),
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=("MT10", "MT50"))
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--protocol-config")
    parser.add_argument("--validation-bank-seed", type=int)
    parser.add_argument("--validation-benchmark-seed", type=int)
    parser.add_argument("--thresholds", help="Optional sorted comma-separated grid")
    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-count", type=int, default=99)
    parser.add_argument("--precision-floor", type=float, default=0.60)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")
    parser.add_argument("--log-file")
    return parser


def main() -> None:
    payload = tune(build_parser().parse_args())
    print(json.dumps(payload["selection"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
