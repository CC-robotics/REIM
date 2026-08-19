#!/usr/bin/env python3
"""Paired shared-policy evaluation on the official MT10 or MT50 task suites.

Clean evaluation follows Meta-World's goal-observable multi-task protocol:
raw 39D observations, an appended task one-hot, official ``info['success']``,
and a 500-step horizon.  A nonzero noise level is an explicitly separate REIM
robustness extension using only task-universal action and observation noise.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import sys
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.io import atomic_write_json, file_sha256
from evaluation.multitask_metrics import (
    aggregate_multitask_metrics,
    paired_task_stratified_bootstrap_delta,
)
from models.bc_policy import ACTPolicy, MLPBCPolicy
from models.failure_detector import FailureDetector
from models.imitation_recovery_policy import ImitationRecoveryPolicy
from utils.common import configure_logging, seed_everything, select_device

SCHEMA_VERSION = "reim-multitask-evaluation-v2"
RUN_SIDECAR_SCHEMA_VERSION = "reim-multitask-evaluation-run-v1"
OFFICIAL_MAX_EPISODE_STEPS = 500
OFFICIAL_TASK_VARIANTS_PER_CLASS = 50
OFFICIAL_CLEAN_CONDITION = "official_clean"
DEMONSTRATION_SCHEMA = "reim-multitask-demonstrations-v1"
DEMONSTRATION_DATASET_TYPE = "balanced_multitask_scripted_expert_demonstrations"
DETECTOR_TRAINING_SCHEMA = "reim-failure-detector-training-v2"
FAILURE_DATA_SCHEMA = "reim-multitask-failures-v2"
FAILURE_DATASET_TYPE = "task_conditioned_behavioral_deviation_risk"
CALIBRATION_SCHEMA = "reim-task-conditional-risk-calibration-v1"
RECOVERY_TRAINING_SCHEMA = "reim-multitask-recovery-training-v1"
METHODS = ("mlp_bc", "act", "act_retry", "heuristic_recovery", "reim")
DEFAULT_OFFICIAL_METHODS = ("mlp_bc", "act", "heuristic_recovery", "reim")
METHOD_LABELS = {
    "mlp_bc": "MT-MLP BC",
    "act": "MT-ACT",
    "act_retry": "MT-ACT + Retry",
    "heuristic_recovery": "MT-ACT + Heuristic-Gated Learned Recovery",
    "reim": "MT-REIM",
}
CSV_FIELDS = (
    "run_fingerprint",
    "benchmark",
    "condition",
    "task_name",
    "task_id",
    "task_variant",
    "method",
    "success",
    "intervention_count",
    "recovery_success",
    "steps",
    "paired_episode_id",
    "episode_seed",
    "task_payload_sha256",
    "max_failure_probability",
    "trigger_step",
    "attempt_count",
    "recovery_steps_total",
)


def _benchmark(name: str, seed: int):
    import metaworld

    return getattr(metaworld, name)(seed=seed)


def _backend_components(backend: str) -> tuple[Any, str]:
    """Resolve the explicit benchmark backend; 'toy' is never implicit."""

    if backend == "toy":
        from env import toy_multitask

        return toy_multitask, toy_multitask.TOY_VERSION
    if backend == "metaworld":
        import metaworld

        return metaworld, importlib.metadata.version("metaworld")
    raise ValueError(f"Unsupported backend {backend!r}; use 'metaworld' or 'toy'.")


def _condition(raw: np.ndarray, task_id: int, task_count: int) -> np.ndarray:
    one_hot = np.zeros(task_count, dtype=np.float32)
    one_hot[task_id] = 1.0
    return np.concatenate([np.asarray(raw, dtype=np.float32), one_hot])


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_integer(
    name: str,
    value: Any,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if (
        result != value
        or result < minimum
        or (maximum is not None and result > maximum)
    ):
        interval = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be {interval}")
    return result


def _require_finite(
    name: str,
    value: Any,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if (
        not math.isfinite(result)
        or result < minimum
        or (maximum is not None and result > maximum)
    ):
        interval = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be finite and {interval}")
    return result


def _validate_evaluation_arguments(
    args: argparse.Namespace,
    task_count: int,
) -> tuple[str, tuple[str, ...], tuple[int, ...]]:
    """Normalize and fail closed on every protocol-affecting CLI value."""

    condition = str(args.condition).strip()
    if not condition:
        raise ValueError("--condition must be non-empty")
    methods = tuple(str(method) for method in args.methods)
    if not methods:
        raise ValueError("--methods must contain at least one method")
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    if len(set(methods)) != len(methods):
        raise ValueError("--methods must not contain duplicates")
    mlp_checkpoint = getattr(args, "mlp_checkpoint", None)
    if "mlp_bc" in methods and not str(mlp_checkpoint or "").strip():
        raise ValueError("--mlp-checkpoint is required when selecting mlp_bc")

    _require_integer(
        "--episodes-per-task",
        args.episodes_per_task,
        minimum=1,
        maximum=OFFICIAL_TASK_VARIANTS_PER_CLASS,
    )
    _require_integer(
        "--max-steps",
        args.max_steps,
        minimum=1,
        maximum=OFFICIAL_MAX_EPISODE_STEPS,
    )
    _require_integer("--benchmark-seed", args.benchmark_seed, minimum=0)
    _require_integer("--seed", args.seed, minimum=0)
    _require_integer(
        "--recovery-budget",
        args.recovery_budget,
        minimum=1,
        maximum=int(args.max_steps),
    )
    heuristic_min_steps = _require_integer(
        "--heuristic-min-steps", args.heuristic_min_steps, minimum=0
    )
    heuristic_window = _require_integer(
        "--heuristic-window", args.heuristic_window, minimum=1
    )
    if heuristic_min_steps >= int(args.max_steps):
        raise ValueError("--heuristic-min-steps must be smaller than --max-steps")
    if heuristic_window > int(args.max_steps):
        raise ValueError("--heuristic-window must not exceed --max-steps")
    _require_integer("--bootstrap-samples", args.bootstrap_samples, minimum=1)
    noise_level = _require_finite("--noise-level", args.noise_level, minimum=0.0)
    _require_finite("--action-std-scale", args.action_std_scale, minimum=0.0)
    _require_finite("--observation-std-scale", args.observation_std_scale, minimum=0.0)
    trigger_threshold = _require_finite(
        "--threshold", args.threshold, minimum=0.0, maximum=1.0
    )
    release_threshold = _require_finite(
        "--release-threshold",
        getattr(args, "release_threshold", 0.15),
        minimum=0.0,
        maximum=1.0,
    )
    if release_threshold >= trigger_threshold:
        raise ValueError("--release-threshold must be smaller than --threshold")
    _require_integer(
        "--release-patience",
        getattr(args, "release_patience", 5),
        minimum=1,
        maximum=int(args.max_steps),
    )
    _require_integer(
        "--min-recovery-steps",
        getattr(args, "min_recovery_steps", 5),
        minimum=1,
        maximum=int(args.recovery_budget),
    )
    _require_integer(
        "--intervention-cooldown",
        getattr(args, "intervention_cooldown", 10),
        minimum=0,
        maximum=int(args.max_steps),
    )
    _require_finite("--heuristic-tolerance", args.heuristic_tolerance, minimum=0.0)

    requested_task_ids = tuple(
        _require_integer(
            "--task-ids",
            value,
            minimum=0,
            maximum=task_count - 1,
        )
        for value in (args.task_ids or ())
    )
    task_ids = requested_task_ids or tuple(range(task_count))
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("--task-ids must not contain duplicates")
    task_ids = tuple(sorted(task_ids))

    _require_finite(
        "derived action noise standard deviation",
        noise_level * float(args.action_std_scale),
        minimum=0.0,
    )
    _require_finite(
        "derived observation noise standard deviation",
        noise_level * float(args.observation_std_scale),
        minimum=0.0,
    )

    # A full retry is a second episode, so it cannot be reported as an official
    # single-attempt clean Meta-World evaluation.
    if noise_level == 0.0 and "act_retry" in methods:
        raise ValueError(
            "act_retry is not permitted in a zero-noise clean evaluation; "
            "run it in a separately named non-clean robustness protocol"
        )
    return condition, methods, task_ids


def _task_payload_sha256(task: Any) -> str:
    return hashlib.sha256(bytes(task.data)).hexdigest()


def _task_bank_sha256(
    task_vocabulary: Sequence[str],
    tasks_by_name: Mapping[str, Sequence[Any]],
) -> str:
    ordered_bank: list[dict[str, Any]] = []
    for task_id, task_name in enumerate(task_vocabulary):
        variants = list(tasks_by_name.get(task_name, ()))
        if not variants:
            raise RuntimeError(f"Meta-World task bank has no variants for {task_name}")
        ordered_bank.append(
            {
                "task_id": task_id,
                "task_name": task_name,
                "variant_payload_sha256": [
                    _task_payload_sha256(task) for task in variants
                ],
            }
        )
    return _canonical_sha256(ordered_bank)


def _build_run_protocol(
    *,
    args: argparse.Namespace,
    condition: str,
    methods: Sequence[str],
    task_ids: Sequence[int],
    task_vocabulary: Sequence[str],
    task_bank_sha256: str,
    checkpoint_sha256: Mapping[str, str],
    metaworld_version: str,
    execution_device: str,
) -> dict[str, Any]:
    return {
        "evaluation_schema_version": SCHEMA_VERSION,
        "benchmark": str(args.benchmark),
        "condition": condition,
        "metaworld_version": metaworld_version,
        "execution_device": execution_device,
        "benchmark_seed": int(args.benchmark_seed),
        "episode_seed_base": int(args.seed),
        "task_bank_sha256": task_bank_sha256,
        "task_vocabulary": list(task_vocabulary),
        "task_vocabulary_sha256": _canonical_sha256(list(task_vocabulary)),
        "task_ids": list(task_ids),
        "methods": list(methods),
        "episodes_per_task": int(args.episodes_per_task),
        "max_episode_steps": int(args.max_steps),
        "noise_level": float(args.noise_level),
        "action_std_scale": float(args.action_std_scale),
        "observation_std_scale": float(args.observation_std_scale),
        "object_position_noise": False,
        "detector_threshold": float(args.threshold),
        "release_threshold": float(getattr(args, "release_threshold", 0.15)),
        "release_patience": int(getattr(args, "release_patience", 5)),
        "min_recovery_steps": int(getattr(args, "min_recovery_steps", 5)),
        "intervention_cooldown": int(getattr(args, "intervention_cooldown", 10)),
        "recovery_budget": int(args.recovery_budget),
        "heuristic_min_steps": int(args.heuristic_min_steps),
        "heuristic_window": int(args.heuristic_window),
        "heuristic_tolerance": float(args.heuristic_tolerance),
        "bootstrap_samples": int(args.bootstrap_samples),
        "checkpoint_sha256": dict(checkpoint_sha256),
    }


def _run_sidecar_path(output_csv: Path) -> Path:
    return output_csv.with_suffix(output_csv.suffix + ".run.json")


def _prepare_run_artifacts(
    output_csv: Path,
    protocol: Mapping[str, Any],
    *,
    resume: bool,
) -> tuple[str, Path, list[dict[str, Any]]]:
    """Create or validate an immutable one-protocol-per-CSV run sidecar."""

    run_fingerprint = _canonical_sha256(protocol)
    sidecar_path = _run_sidecar_path(output_csv)
    expected_sidecar = {
        "schema_version": RUN_SIDECAR_SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "protocol": dict(protocol),
    }
    csv_exists = output_csv.is_file()
    sidecar_exists = sidecar_path.is_file()
    if not resume:
        if csv_exists or sidecar_exists:
            raise FileExistsError(
                f"Evaluation artifacts already exist for {output_csv}; "
                "pass --resume only with the identical run protocol"
            )
        atomic_write_json(sidecar_path, expected_sidecar)
        return run_fingerprint, sidecar_path, []

    if csv_exists and not sidecar_exists:
        raise ValueError(
            f"Cannot resume {output_csv} without immutable run sidecar {sidecar_path}"
        )
    if sidecar_exists:
        try:
            stored_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot validate run sidecar {sidecar_path}") from exc
        if stored_sidecar != expected_sidecar:
            stored_fingerprint = (
                stored_sidecar.get("run_fingerprint")
                if isinstance(stored_sidecar, Mapping)
                else None
            )
            raise ValueError(
                "Resume protocol fingerprint mismatch: "
                f"stored={stored_fingerprint!r}, requested={run_fingerprint!r}"
            )
    else:
        atomic_write_json(sidecar_path, expected_sidecar)
    rows = _read_csv(output_csv) if csv_exists else []
    return run_fingerprint, sidecar_path, rows


def _expected_row_specifications(
    *,
    benchmark: str,
    condition: str,
    methods: Sequence[str],
    task_ids: Sequence[int],
    task_vocabulary: Sequence[str],
    tasks_by_name: Mapping[str, Sequence[Any]],
    episodes_per_task: int,
    seed: int,
    run_fingerprint: str,
) -> dict[tuple[str, int, str], dict[str, Any]]:
    expected: dict[tuple[str, int, str], dict[str, Any]] = {}
    for task_id in task_ids:
        task_name = task_vocabulary[task_id]
        variants = list(tasks_by_name[task_name])
        if episodes_per_task > len(variants):
            raise ValueError(
                f"{task_name} exposes {len(variants)} task variants, fewer than "
                f"--episodes-per-task={episodes_per_task}"
            )
        for episode_index in range(episodes_per_task):
            task_variant = episode_index
            task_hash = _task_payload_sha256(variants[task_variant])
            episode_seed = seed + task_id * 100_000 + episode_index
            paired_id = f"{task_id:02d}-{episode_index:04d}-{task_hash[:12]}"
            for method in methods:
                label = METHOD_LABELS[method]
                key = (label, task_id, paired_id)
                expected[key] = {
                    "run_fingerprint": run_fingerprint,
                    "benchmark": benchmark,
                    "condition": condition,
                    "task_name": task_name,
                    "task_id": task_id,
                    "task_variant": task_variant,
                    "method": label,
                    "method_key": method,
                    "paired_episode_id": paired_id,
                    "episode_seed": episode_seed,
                    "task_payload_sha256": task_hash,
                }
    return expected


def _row_integer(row: Mapping[str, Any], field: str, row_number: int) -> int:
    try:
        numeric = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"resume CSV row {row_number}: invalid {field}") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"resume CSV row {row_number}: invalid {field}")
    return int(numeric)


def _validate_protocol_rows(
    rows: Sequence[Mapping[str, Any]],
    expected: Mapping[tuple[str, int, str], Mapping[str, Any]],
    *,
    run_fingerprint: str,
    max_steps: int,
) -> set[tuple[str, int, str]]:
    """Reject foreign, stale, duplicate, or malformed rows before resuming."""

    completed: set[tuple[str, int, str]] = set()
    seen_row_keys: set[tuple[str, int, str]] = set()
    required = set(CSV_FIELDS)
    text_fields = (
        "run_fingerprint",
        "benchmark",
        "condition",
        "task_name",
        "method",
        "paired_episode_id",
        "task_payload_sha256",
    )
    integer_fields = ("task_id", "task_variant", "episode_seed")
    for row_number, row in enumerate(rows, start=1):
        missing = required - set(row)
        if missing:
            raise ValueError(
                f"resume CSV row {row_number}: missing fields {sorted(missing)}"
            )
        unexpected = set(row) - required
        if unexpected:
            raise ValueError(
                f"resume CSV row {row_number}: unexpected fields {sorted(unexpected)}"
            )
        if str(row["run_fingerprint"]) != run_fingerprint:
            raise ValueError(
                f"resume CSV row {row_number}: row belongs to another run fingerprint"
            )
        task_id = _row_integer(row, "task_id", row_number)
        key = (str(row["method"]), task_id, str(row["paired_episode_id"]))
        specification = expected.get(key)
        if specification is None:
            raise ValueError(
                f"resume CSV row {row_number}: row is outside the current run protocol"
            )
        if key in seen_row_keys:
            raise ValueError(
                f"resume CSV row {row_number}: duplicate method/task/episode row"
            )
        for field in text_fields:
            if str(row[field]) != str(specification[field]):
                raise ValueError(
                    f"resume CSV row {row_number}: {field} does not match run protocol"
                )
        for field in integer_fields:
            if _row_integer(row, field, row_number) != int(specification[field]):
                raise ValueError(
                    f"resume CSV row {row_number}: {field} does not match run protocol"
                )

        success = _row_integer(row, "success", row_number)
        interventions = _row_integer(row, "intervention_count", row_number)
        recovery_success = _row_integer(row, "recovery_success", row_number)
        steps = _row_integer(row, "steps", row_number)
        attempt_count = _row_integer(row, "attempt_count", row_number)
        trigger_step = _row_integer(row, "trigger_step", row_number)
        method_key = str(specification["method_key"])
        if success not in (0, 1) or interventions < 0:
            raise ValueError(f"resume CSV row {row_number}: invalid outcome counts")
        if recovery_success < 0 or recovery_success > interventions:
            raise ValueError(f"resume CSV row {row_number}: invalid recovery count")
        maximum_steps = max_steps * (2 if method_key == "act_retry" else 1)
        if not 1 <= steps <= maximum_steps:
            raise ValueError(f"resume CSV row {row_number}: steps outside protocol")
        valid_attempt_counts = (1, 2) if method_key == "act_retry" else (1,)
        if attempt_count not in valid_attempt_counts:
            raise ValueError(f"resume CSV row {row_number}: invalid attempt_count")
        if trigger_step < -1 or trigger_step >= max_steps:
            raise ValueError(f"resume CSV row {row_number}: invalid trigger_step")
        try:
            probability = float(row["max_failure_probability"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"resume CSV row {row_number}: invalid max_failure_probability"
            ) from exc
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"resume CSV row {row_number}: invalid max_failure_probability"
            )
        completed.add((str(specification["method_key"]), task_id, key[2]))
        seen_row_keys.add(key)
    return completed


def _official_clean_eligibility(
    *,
    condition: str,
    noise_level: float,
    max_steps: int,
    task_ids: Sequence[int],
    task_count: int,
    methods: Sequence[str],
    episodes_per_task: int,
    completed: set[tuple[str, int, str]],
) -> dict[str, dict[str, Any]]:
    full_suite = tuple(sorted(task_ids)) == tuple(range(task_count))
    run_reasons: list[str] = []
    if condition != OFFICIAL_CLEAN_CONDITION:
        run_reasons.append("non_official_condition")
    if noise_level != 0.0:
        run_reasons.append("nonzero_noise")
    if max_steps != OFFICIAL_MAX_EPISODE_STEPS:
        run_reasons.append("non_official_horizon")
    if episodes_per_task != OFFICIAL_TASK_VARIANTS_PER_CLASS:
        run_reasons.append("non_official_episodes_per_task")
    if not full_suite:
        run_reasons.append("partial_task_suite")
    if "act_retry" in methods:
        run_reasons.append("run_contains_retry")

    expected_per_method = len(task_ids) * episodes_per_task
    result: dict[str, dict[str, Any]] = {}
    for method in methods:
        completed_rows = sum(1 for key in completed if key[0] == method)
        reasons = list(run_reasons)
        if method == "act_retry" and "retry_method" not in reasons:
            reasons.append("retry_method")
        if completed_rows != expected_per_method:
            reasons.append("incomplete_rows")
        result[method] = {
            "label": METHOD_LABELS[method],
            "eligible": not reasons,
            "reasons": reasons,
            "completed_rows": completed_rows,
            "expected_rows": expected_per_method,
        }
    return result


def _publication_readiness(
    *,
    benchmark: str,
    official_clean_protocol: bool,
) -> dict[str, Any]:
    """Separate rollout validity from the external data-isolation gate.

    Evaluation deliberately does not require the five-bank audit before a run:
    doing so would make diagnostic rollouts impossible.  Conversely, a valid
    rollout protocol alone must never be presented as publication-ready.
    """

    reasons = ["external_five_bank_payload_isolation_audit_required"]
    if not official_clean_protocol:
        reasons.insert(0, "official_clean_rollout_protocol_ineligible")
    return {
        "eligible": False,
        "rollout_protocol_eligible": bool(official_clean_protocol),
        "audit_required": True,
        "audit_scope": (
            "demonstrations,failure_training,failure_validation,"
            "recovery,final_evaluation"
        ),
        "audit_benchmark": benchmark,
        "external_audit_consumed": False,
        "reasons": reasons,
    }


def _risk(
    detector: FailureDetector,
    history: deque[np.ndarray],
) -> float:
    length = min(len(history), detector.sequence_length)
    window = np.zeros((detector.sequence_length, detector.state_dim), dtype=np.float32)
    window[:length] = np.stack(list(history)[-length:])
    probability = detector.predict_proba(
        window[None, ...], np.asarray([length], dtype=np.int64)
    )
    return float(probability.detach().cpu().numpy().reshape(-1)[0])


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


_CHECKPOINT_METADATA_FIELDS = frozenset(
    {
        "policy_type",
        "format_version",
        "state_dim",
        "action_dim",
        "benchmark",
        "task_vocabulary",
        "task_vocabulary_sha256",
        "data_manifest_sha256",
        "detector_training_schema",
        "data_schema_version",
        "dataset_type",
        "dataset_role",
        "label_calibration_mode",
        "label_calibration_quantile",
        "label_calibration_fingerprint_sha256",
        "label_calibration_source_sha256",
        "dataset_fingerprint_sha256",
        "label_calibration",
    }
)


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    """Load only the provenance fields needed for fail-closed evaluation.

    The policy loaders intentionally expose a small backwards-compatible
    provenance subset.  Evaluation needs stricter, model-specific contracts,
    so it independently extracts the checkpoint's top-level metadata.  Tensor
    payloads are discarded before this function returns.
    """

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint {path} is not a metadata mapping")
    return {
        key: checkpoint[key]
        for key in _CHECKPOINT_METADATA_FIELDS
        if key in checkpoint
    }


def _require_provenance_sha256(
    checkpoint_name: str,
    provenance: Mapping[str, Any],
    field: str,
) -> str:
    value = provenance.get(field)
    if not _is_sha256(value):
        raise ValueError(
            f"{checkpoint_name} checkpoint lacks a valid {field} SHA-256"
        )
    return str(value)


def _validate_multitask_provenance(
    checkpoint_name: str,
    provenance: Any,
    *,
    benchmark: str,
    task_vocabulary: Sequence[str],
    checkpoint_kind: str,
    linked_checkpoint_sha256: Mapping[str, str] | None = None,
) -> None:
    """Fail closed on model-specific multi-task training provenance."""

    if not isinstance(provenance, Mapping):
        raise ValueError(f"{checkpoint_name} checkpoint lacks multi-task provenance")
    if checkpoint_kind not in {"act", "detector", "recovery", "mlp_bc"}:
        raise ValueError(f"Unknown checkpoint provenance kind {checkpoint_kind!r}")
    expected_vocabulary_sha256 = _canonical_sha256(list(task_vocabulary))
    if provenance.get("task_vocabulary") != list(task_vocabulary):
        raise ValueError(
            f"{checkpoint_name} checkpoint task vocabulary does not match benchmark"
        )
    if provenance.get("task_vocabulary_sha256") != expected_vocabulary_sha256:
        raise ValueError(
            f"{checkpoint_name} checkpoint task vocabulary hash does not match benchmark"
        )
    if provenance.get("benchmark") != benchmark:
        raise ValueError(f"{checkpoint_name} checkpoint benchmark does not match")

    if checkpoint_kind == "act":
        if provenance.get("policy_type") != "ACT":
            raise ValueError(f"{checkpoint_name} checkpoint is not an explicit ACT policy")
        if provenance.get("state_dim") != 39 + len(task_vocabulary):
            raise ValueError(f"{checkpoint_name} checkpoint state_dim does not match")
        if provenance.get("action_dim") != 4:
            raise ValueError(f"{checkpoint_name} checkpoint action_dim does not match")
        _require_provenance_sha256(
            checkpoint_name, provenance, "data_manifest_sha256"
        )
        return

    if checkpoint_kind == "detector":
        if provenance.get("detector_training_schema") != DETECTOR_TRAINING_SCHEMA:
            raise ValueError(
                f"{checkpoint_name} checkpoint has invalid detector training schema"
            )
        if provenance.get("data_schema_version") != FAILURE_DATA_SCHEMA:
            raise ValueError(
                f"{checkpoint_name} checkpoint has invalid failure data schema"
            )
        if provenance.get("dataset_type") != FAILURE_DATASET_TYPE:
            raise ValueError(
                f"{checkpoint_name} checkpoint has invalid failure dataset type"
            )
        if provenance.get("dataset_role") != "training":
            raise ValueError(
                f"{checkpoint_name} checkpoint was not trained on a training-role bank"
            )
        if provenance.get("label_calibration_mode") != "fit-task-quantile":
            raise ValueError(
                f"{checkpoint_name} checkpoint has invalid label calibration mode"
            )
        try:
            quantile = float(provenance.get("label_calibration_quantile"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{checkpoint_name} checkpoint lacks a calibration quantile"
            ) from exc
        if not math.isfinite(quantile) or not 0.0 < quantile < 1.0:
            raise ValueError(
                f"{checkpoint_name} checkpoint has invalid calibration quantile"
            )
        _require_provenance_sha256(
            checkpoint_name, provenance, "data_manifest_sha256"
        )
        calibration_fingerprint = _require_provenance_sha256(
            checkpoint_name,
            provenance,
            "label_calibration_fingerprint_sha256",
        )
        calibration_source_sha256 = _require_provenance_sha256(
            checkpoint_name,
            provenance,
            "label_calibration_source_sha256",
        )
        _require_provenance_sha256(
            checkpoint_name, provenance, "dataset_fingerprint_sha256"
        )
        calibration = provenance.get("label_calibration")
        if not isinstance(calibration, Mapping):
            raise ValueError(
                f"{checkpoint_name} checkpoint lacks full label calibration provenance"
            )
        if _canonical_sha256(dict(calibration)) != calibration_fingerprint:
            raise ValueError(
                f"{checkpoint_name} checkpoint label calibration fingerprint mismatch"
            )
        if (
            calibration.get("schema_version") != CALIBRATION_SCHEMA
            or calibration.get("mode") != "fit-task-quantile"
            or calibration.get("dataset_role") != "training"
            or calibration.get("benchmark") != benchmark
            or calibration.get("task_vocabulary_sha256")
            != expected_vocabulary_sha256
            or calibration.get("quantile") != quantile
        ):
            raise ValueError(
                f"{checkpoint_name} checkpoint label calibration metadata is inconsistent"
            )
        calibration_source = calibration.get("calibration_source")
        if (
            not isinstance(calibration_source, Mapping)
            or calibration_source.get("sha256") != calibration_source_sha256
        ):
            raise ValueError(
                f"{checkpoint_name} checkpoint calibration source hash is inconsistent"
            )
        return

    if checkpoint_kind == "recovery":
        if provenance.get("schema_version") != RECOVERY_TRAINING_SCHEMA:
            raise ValueError(
                f"{checkpoint_name} checkpoint has invalid recovery training schema"
            )
        _require_provenance_sha256(
            checkpoint_name, provenance, "dataset_manifest_sha256"
        )
        source_training = provenance.get("source_training")
        if (
            not isinstance(source_training, Mapping)
            or source_training.get("algorithm") != "task_balanced_smooth_l1"
        ):
            raise ValueError(
                f"{checkpoint_name} checkpoint has invalid recovery training provenance"
            )
        for field, expected in (linked_checkpoint_sha256 or {}).items():
            stored = provenance.get(field)
            if stored is not None and stored != expected:
                raise ValueError(
                    f"{checkpoint_name} checkpoint {field} does not match evaluation"
                )
        return

    if provenance.get("training_schema") != "reim-multitask-mlp-training-v1":
        raise ValueError(f"{checkpoint_name} checkpoint has invalid training provenance")
    if provenance.get("data_schema_version") != DEMONSTRATION_SCHEMA:
        raise ValueError(f"{checkpoint_name} checkpoint has invalid data manifest schema")
    if provenance.get("dataset_type") != DEMONSTRATION_DATASET_TYPE:
        raise ValueError(f"{checkpoint_name} checkpoint has invalid dataset provenance")
    _require_provenance_sha256(checkpoint_name, provenance, "data_manifest_sha256")
    _require_provenance_sha256(checkpoint_name, provenance, "split_sha256")


def _noise_arrays(
    seed: int,
    max_steps: int,
    action_std: float,
    observation_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    action_rng = np.random.default_rng(seed + 70_000_000)
    observation_rng = np.random.default_rng(seed + 80_000_000)
    return (
        action_rng.normal(0.0, action_std, size=(max_steps, 4)).astype(np.float32),
        observation_rng.normal(0.0, observation_std, size=(max_steps, 39)).astype(
            np.float32
        ),
    )


def _rollout(
    *,
    env: Any,
    task: Any,
    task_id: int,
    task_count: int,
    method: str,
    episode_seed: int,
    max_steps: int,
    action_noise: np.ndarray,
    observation_noise: np.ndarray,
    act: ACTPolicy,
    detector: FailureDetector,
    recovery: ImitationRecoveryPolicy,
    threshold: float,
    release_threshold: float = 0.15,
    release_patience: int = 5,
    min_recovery_steps: int = 5,
    intervention_cooldown: int = 0,
    recovery_budget: int,
    heuristic_min_steps: int,
    heuristic_window: int,
    heuristic_tolerance: float,
    mlp: MLPBCPolicy | None = None,
) -> dict[str, Any]:
    env.set_task(task)
    raw, _ = env.reset(seed=episode_seed)
    if method == "mlp_bc":
        if mlp is None:
            raise ValueError("mlp_bc rollout requires an MLPBCPolicy")
        mlp.reset()
    else:
        act.reset()
    history: deque[np.ndarray] = deque(maxlen=detector.sequence_length)
    rewards: deque[float] = deque(maxlen=heuristic_window)
    recovery_active = False
    recovery_steps = 0
    safe_streak = 0
    cooldown_until = 0
    intervention_count = 0
    recovery_success = 0
    recovery_steps_total = 0
    trigger_step = -1
    max_probability = 0.0
    success = False
    executed_steps = 0
    for step in range(max_steps):
        observed_raw = np.asarray(raw, dtype=np.float32) + observation_noise[step]
        state = _condition(observed_raw, task_id, task_count)
        history.append(state)
        probability = _risk(detector, history) if method == "reim" else 0.0
        max_probability = max(max_probability, probability)
        heuristic_trigger = False
        if (
            method == "heuristic_recovery"
            and step >= heuristic_min_steps
            and len(rewards) == heuristic_window
        ):
            values = np.asarray(rewards, dtype=np.float64)
            heuristic_trigger = float(values.max() - values.min()) < heuristic_tolerance
        learned_trigger = method == "reim" and probability >= threshold
        if (
            recovery_active
            and method == "reim"
            and recovery_steps >= min_recovery_steps
        ):
            safe_streak = safe_streak + 1 if probability <= release_threshold else 0
            if safe_streak >= release_patience:
                recovery_active = False
                recovery_success += 1
                cooldown_until = step + intervention_cooldown
                act.reset()
        if (
            not recovery_active
            and step >= cooldown_until
            and (heuristic_trigger or learned_trigger)
        ):
            recovery_active = True
            recovery_steps = 0
            safe_streak = 0
            intervention_count += 1
            if trigger_step < 0:
                trigger_step = step
        if recovery_active:
            intended = np.asarray(recovery.act(state), dtype=np.float32).reshape(4)
            recovery_steps += 1
            recovery_steps_total += 1
        elif method == "mlp_bc":
            assert mlp is not None
            intended = np.asarray(mlp.act(state), dtype=np.float32).reshape(4)
        else:
            intended = np.asarray(act.act(state), dtype=np.float32).reshape(4)
        executed = np.clip(intended + action_noise[step], -1.0, 1.0)
        raw, reward, terminated, truncated, info = env.step(executed)
        executed_steps += 1
        rewards.append(float(reward))
        success = bool(info.get("success", False))
        if success:
            if recovery_active:
                recovery_success += 1
            break
        if recovery_active and recovery_steps >= recovery_budget:
            recovery_active = False
            safe_streak = 0
            cooldown_until = step + intervention_cooldown
            act.reset()
        if terminated or truncated:
            break
    return {
        "success": success,
        "intervention_count": intervention_count,
        "recovery_success": recovery_success,
        "steps": executed_steps,
        "max_failure_probability": max_probability,
        "trigger_step": trigger_step,
        "recovery_steps_total": recovery_steps_total,
    }


def _retry_rollout(
    *,
    primary: dict[str, Any],
    env: Any,
    task: Any,
    task_id: int,
    task_count: int,
    episode_seed: int,
    max_steps: int,
    action_std: float,
    observation_std: float,
    act: ACTPolicy,
    detector: FailureDetector,
    recovery: ImitationRecoveryPolicy,
    threshold: float,
    release_threshold: float,
    release_patience: int,
    min_recovery_steps: int,
    intervention_cooldown: int,
    recovery_budget: int,
    heuristic_min_steps: int,
    heuristic_window: int,
    heuristic_tolerance: float,
) -> dict[str, Any]:
    if primary["success"]:
        return {**primary, "attempt_count": 1}
    retry_seed = episode_seed + 900_000_000
    action_noise, observation_noise = _noise_arrays(
        retry_seed, max_steps, action_std, observation_std
    )
    retry = _rollout(
        env=env,
        task=task,
        task_id=task_id,
        task_count=task_count,
        method="act",
        episode_seed=retry_seed,
        max_steps=max_steps,
        action_noise=action_noise,
        observation_noise=observation_noise,
        act=act,
        detector=detector,
        recovery=recovery,
        threshold=threshold,
        release_threshold=release_threshold,
        release_patience=release_patience,
        min_recovery_steps=min_recovery_steps,
        intervention_cooldown=intervention_cooldown,
        recovery_budget=recovery_budget,
        heuristic_min_steps=heuristic_min_steps,
        heuristic_window=heuristic_window,
        heuristic_tolerance=heuristic_tolerance,
    )
    return {
        **retry,
        "steps": primary["steps"] + retry["steps"],
        "intervention_count": 1,
        "recovery_success": int(retry["success"]),
        "attempt_count": 2,
        "recovery_steps_total": primary["recovery_steps_total"]
        + retry["recovery_steps_total"],
    }


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    # Validate seed values before either NumPy or Meta-World consumes them.
    _require_integer("--benchmark-seed", args.benchmark_seed, minimum=0)
    _require_integer("--seed", args.seed, minimum=0)
    backend_module, backend_version = _backend_components(
        str(getattr(args, "backend", "metaworld"))
    )
    benchmark = getattr(backend_module, args.benchmark)(seed=args.benchmark_seed)
    task_vocabulary = list(benchmark.train_classes.keys())
    task_count = len(task_vocabulary)
    expected = 10 if args.benchmark == "MT10" else 50
    if task_count != expected:
        raise RuntimeError(f"Expected {expected} tasks, got {task_count}")
    condition, methods, task_ids = _validate_evaluation_arguments(args, task_count)
    seed_everything(args.seed)
    device = select_device(args.device)
    logger = configure_logging(
        "evaluate_multitask",
        args.log_file or f"results/logs/{args.benchmark.lower()}_{condition}.log",
    )

    tasks_by_name = {
        name: [task for task in benchmark.train_tasks if task.env_name == name]
        for name in task_vocabulary
    }
    task_bank_sha256 = _task_bank_sha256(task_vocabulary, tasks_by_name)
    for task_name, variants in tasks_by_name.items():
        if len(variants) != OFFICIAL_TASK_VARIANTS_PER_CLASS:
            raise RuntimeError(
                f"Official {args.benchmark} bank expected "
                f"{OFFICIAL_TASK_VARIANTS_PER_CLASS} variants for {task_name}, "
                f"found {len(variants)}"
            )

    act_path = Path(args.act_checkpoint).expanduser().resolve()
    detector_path = Path(args.detector_checkpoint).expanduser().resolve()
    recovery_path = Path(args.recovery_checkpoint).expanduser().resolve()
    checkpoint_sha256 = {
        "act": file_sha256(act_path),
        "detector": file_sha256(detector_path),
        "recovery": file_sha256(recovery_path),
    }
    mlp_path: Path | None = None
    mlp: MLPBCPolicy | None = None
    if "mlp_bc" in methods:
        mlp_path = Path(args.mlp_checkpoint).expanduser().resolve()
        checkpoint_sha256["mlp_bc"] = file_sha256(mlp_path)
        mlp = MLPBCPolicy.from_checkpoint(mlp_path, map_location=device)
    act = ACTPolicy.from_checkpoint(act_path, map_location=device)
    detector = FailureDetector.from_checkpoint(
        detector_path, map_location=device, state_dim=39 + task_count
    )
    recovery = ImitationRecoveryPolicy.load(recovery_path, device=device)
    if {act.state_dim, detector.state_dim, recovery.state_dim} != {39 + task_count}:
        raise ValueError("Checkpoint observation dimensions do not match benchmark")
    if mlp is not None and mlp.state_dim != 39 + task_count:
        raise ValueError("MLP checkpoint observation dimensions do not match benchmark")
    if act.action_dim != 4 or recovery.action_dim != 4:
        raise ValueError("Checkpoint action dimensions do not match Meta-World")
    if mlp is not None and mlp.action_dim != 4:
        raise ValueError("MLP checkpoint action dimensions do not match Meta-World")
    _validate_multitask_provenance(
        "ACT",
        _checkpoint_metadata(act_path),
        benchmark=args.benchmark,
        task_vocabulary=task_vocabulary,
        checkpoint_kind="act",
    )
    _validate_multitask_provenance(
        "detector",
        _checkpoint_metadata(detector_path),
        benchmark=args.benchmark,
        task_vocabulary=task_vocabulary,
        checkpoint_kind="detector",
    )
    _validate_multitask_provenance(
        "recovery",
        getattr(recovery, "provenance", None),
        benchmark=args.benchmark,
        task_vocabulary=task_vocabulary,
        checkpoint_kind="recovery",
        linked_checkpoint_sha256={
            "act_checkpoint_sha256": checkpoint_sha256["act"],
            "detector_checkpoint_sha256": checkpoint_sha256["detector"],
        },
    )
    if mlp is not None:
        _validate_multitask_provenance(
            "MLP-BC",
            getattr(mlp, "provenance", None),
            benchmark=args.benchmark,
            task_vocabulary=task_vocabulary,
            checkpoint_kind="mlp_bc",
        )

    metaworld_version = backend_version
    protocol = _build_run_protocol(
        args=args,
        condition=condition,
        methods=methods,
        task_ids=task_ids,
        task_vocabulary=task_vocabulary,
        task_bank_sha256=task_bank_sha256,
        checkpoint_sha256=checkpoint_sha256,
        metaworld_version=metaworld_version,
        execution_device=device,
    )
    output_csv = Path(args.output_csv).expanduser().resolve()
    run_fingerprint, run_sidecar, rows = _prepare_run_artifacts(
        output_csv, protocol, resume=args.resume
    )
    expected_specifications = _expected_row_specifications(
        benchmark=args.benchmark,
        condition=condition,
        methods=methods,
        task_ids=task_ids,
        task_vocabulary=task_vocabulary,
        tasks_by_name=tasks_by_name,
        episodes_per_task=args.episodes_per_task,
        seed=args.seed,
        run_fingerprint=run_fingerprint,
    )
    completed = _validate_protocol_rows(
        rows,
        expected_specifications,
        run_fingerprint=run_fingerprint,
        max_steps=args.max_steps,
    )
    action_std = args.action_std_scale * args.noise_level
    observation_std = args.observation_std_scale * args.noise_level
    for task_id in task_ids:
        task_name = task_vocabulary[task_id]
        env = benchmark.train_classes[task_name](render_mode=None)
        try:
            progress = tqdm(
                range(args.episodes_per_task),
                desc=f"{args.benchmark} {condition} {task_name}",
                leave=False,
            )
            for episode_index in progress:
                task_variant = episode_index
                task = tasks_by_name[task_name][task_variant]
                task_hash = _task_payload_sha256(task)
                episode_seed = args.seed + task_id * 100_000 + episode_index
                paired_id = f"{task_id:02d}-{episode_index:04d}-{task_hash[:12]}"
                action_noise, observation_noise = _noise_arrays(
                    episode_seed, args.max_steps, action_std, observation_std
                )
                # Retry reuses the exact first ACT attempt, avoiding accidental
                # differences between its primary attempt and the ACT baseline.
                act_primary: dict[str, Any] | None = None
                for method in methods:
                    if (method, task_id, paired_id) in completed:
                        continue
                    if method == "act_retry":
                        if act_primary is None:
                            act_primary = _rollout(
                                env=env,
                                task=task,
                                task_id=task_id,
                                task_count=task_count,
                                method="act",
                                episode_seed=episode_seed,
                                max_steps=args.max_steps,
                                action_noise=action_noise,
                                observation_noise=observation_noise,
                                act=act,
                                detector=detector,
                                recovery=recovery,
                                threshold=args.threshold,
                                release_threshold=args.release_threshold,
                                release_patience=args.release_patience,
                                min_recovery_steps=args.min_recovery_steps,
                                intervention_cooldown=args.intervention_cooldown,
                                recovery_budget=args.recovery_budget,
                                heuristic_min_steps=args.heuristic_min_steps,
                                heuristic_window=args.heuristic_window,
                                heuristic_tolerance=args.heuristic_tolerance,
                                mlp=mlp,
                            )
                        result = _retry_rollout(
                            primary=act_primary,
                            env=env,
                            task=task,
                            task_id=task_id,
                            task_count=task_count,
                            episode_seed=episode_seed,
                            max_steps=args.max_steps,
                            action_std=action_std,
                            observation_std=observation_std,
                            act=act,
                            detector=detector,
                            recovery=recovery,
                            threshold=args.threshold,
                            release_threshold=args.release_threshold,
                            release_patience=args.release_patience,
                            min_recovery_steps=args.min_recovery_steps,
                            intervention_cooldown=args.intervention_cooldown,
                            recovery_budget=args.recovery_budget,
                            heuristic_min_steps=args.heuristic_min_steps,
                            heuristic_window=args.heuristic_window,
                            heuristic_tolerance=args.heuristic_tolerance,
                        )
                    else:
                        result = _rollout(
                            env=env,
                            task=task,
                            task_id=task_id,
                            task_count=task_count,
                            method=method,
                            episode_seed=episode_seed,
                            max_steps=args.max_steps,
                            action_noise=action_noise,
                            observation_noise=observation_noise,
                            act=act,
                            detector=detector,
                            recovery=recovery,
                            threshold=args.threshold,
                            release_threshold=args.release_threshold,
                            release_patience=args.release_patience,
                            min_recovery_steps=args.min_recovery_steps,
                            intervention_cooldown=args.intervention_cooldown,
                            recovery_budget=args.recovery_budget,
                            heuristic_min_steps=args.heuristic_min_steps,
                            heuristic_window=args.heuristic_window,
                            heuristic_tolerance=args.heuristic_tolerance,
                            mlp=mlp,
                        )
                        result["attempt_count"] = 1
                        if method == "act":
                            act_primary = result.copy()
                    row = {
                        "run_fingerprint": run_fingerprint,
                        "benchmark": args.benchmark,
                        "condition": condition,
                        "task_name": task_name,
                        "task_id": task_id,
                        "task_variant": task_variant,
                        "method": METHOD_LABELS[method],
                        "success": int(result["success"]),
                        "intervention_count": int(result["intervention_count"]),
                        "recovery_success": int(result["recovery_success"]),
                        "steps": int(result["steps"]),
                        "paired_episode_id": paired_id,
                        "episode_seed": episode_seed,
                        "task_payload_sha256": task_hash,
                        "max_failure_probability": float(
                            result["max_failure_probability"]
                        ),
                        "trigger_step": int(result["trigger_step"]),
                        "attempt_count": int(result["attempt_count"]),
                    }
                    rows.append(row)
                    completed.add((method, task_id, paired_id))
            _write_csv(output_csv, rows)
        finally:
            env.close()

    # Validate newly generated rows too, then aggregate exclusively over this
    # immutable protocol fingerprint. Foreign rows are rejected above rather
    # than silently entering any estimand.
    completed = _validate_protocol_rows(
        rows,
        expected_specifications,
        run_fingerprint=run_fingerprint,
        max_steps=args.max_steps,
    )
    current_rows = [
        row for row in rows if str(row["run_fingerprint"]) == run_fingerprint
    ]
    expected_rows = len(expected_specifications)
    if len(current_rows) != expected_rows and not args.allow_partial:
        raise RuntimeError(
            f"Expected {expected_rows} evaluation rows, found {len(current_rows)}"
        )

    aggregates: dict[str, Any] = {}
    for method in methods:
        label = METHOD_LABELS[method]
        selected = [row for row in current_rows if row["method"] == label]
        if selected:
            aggregates[method] = aggregate_multitask_metrics(selected)
    paired: dict[str, Any] = {}
    if "act" in aggregates:
        reference = [
            row for row in current_rows if row["method"] == METHOD_LABELS["act"]
        ]
        for method in methods:
            if method == "act" or method not in aggregates:
                continue
            candidate = [
                row for row in current_rows if row["method"] == METHOD_LABELS[method]
            ]
            reference_keys = {
                (str(row["task_id"]), str(row["paired_episode_id"]))
                for row in reference
            }
            candidate_keys = {
                (str(row["task_id"]), str(row["paired_episode_id"]))
                for row in candidate
            }
            if reference_keys != candidate_keys and args.allow_partial:
                continue
            paired[method] = paired_task_stratified_bootstrap_delta(
                reference,
                candidate,
                metric="success",
                n_bootstrap=args.bootstrap_samples,
                seed=args.seed + 2026,
            )
    eligibility = _official_clean_eligibility(
        condition=condition,
        noise_level=float(args.noise_level),
        max_steps=int(args.max_steps),
        task_ids=task_ids,
        task_count=task_count,
        methods=methods,
        episodes_per_task=int(args.episodes_per_task),
        completed=completed,
    )
    official_clean_protocol = bool(eligibility) and all(
        item["eligible"] for item in eligibility.values()
    )
    publication_readiness = _publication_readiness(
        benchmark=args.benchmark,
        official_clean_protocol=official_clean_protocol,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "run_sidecar": str(run_sidecar),
        "run_sidecar_sha256": file_sha256(run_sidecar),
        "benchmark": args.benchmark,
        "condition": condition,
        "official_clean_protocol": official_clean_protocol,
        "official_clean_protocol_scope": "rollout_protocol_only",
        "official_clean_eligibility_by_method": eligibility,
        "publication_eligible": publication_readiness["eligible"],
        "publication_audit_required": publication_readiness["audit_required"],
        "publication_readiness": publication_readiness,
        "robustness_extension": args.noise_level != 0.0,
        "metaworld_version": metaworld_version,
        "benchmark_seed": args.benchmark_seed,
        "seed": args.seed,
        "task_bank_sha256": task_bank_sha256,
        "observation_schema": "raw39_plus_official_task_one_hot",
        "task_vocabulary": task_vocabulary,
        "task_ids": list(task_ids),
        "max_episode_steps": args.max_steps,
        "episodes_per_task": args.episodes_per_task,
        "noise_level": args.noise_level,
        "action_noise_std": action_std,
        "observation_noise_std": observation_std,
        "object_position_noise": False,
        "detector_threshold": args.threshold,
        "release_threshold": args.release_threshold,
        "release_patience": args.release_patience,
        "min_recovery_steps": args.min_recovery_steps,
        "intervention_cooldown": args.intervention_cooldown,
        "recovery_budget": args.recovery_budget,
        "methods": list(methods),
        "episode_csv": str(output_csv),
        "episode_csv_sha256": file_sha256(output_csv),
        "checkpoints": {
            "act": {"path": str(act_path), "sha256": checkpoint_sha256["act"]},
            "detector": {
                "path": str(detector_path),
                "sha256": checkpoint_sha256["detector"],
            },
            "recovery": {
                "path": str(recovery_path),
                "sha256": checkpoint_sha256["recovery"],
            },
        },
        "aggregates": aggregates,
        "paired_vs_act": paired,
    }
    if mlp_path is not None:
        summary["checkpoints"]["mlp_bc"] = {
            "path": str(mlp_path),
            "sha256": checkpoint_sha256["mlp_bc"],
        }
    atomic_write_json(Path(args.output_summary).expanduser().resolve(), summary)
    logger.info(
        "%s %s complete: %d rows; MT-REIM macro success=%s",
        args.benchmark,
        condition,
        len(current_rows),
        aggregates.get("reim", {}).get("summary", {}).get("success_rate_task_macro"),
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=("MT10", "MT50"))
    parser.add_argument("--condition", required=True)
    parser.add_argument("--benchmark-seed", type=int, required=True)
    parser.add_argument("--act-checkpoint", required=True)
    parser.add_argument("--mlp-checkpoint")
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--recovery-checkpoint", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_OFFICIAL_METHODS)
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--action-std-scale", type=float, default=0.40)
    parser.add_argument("--observation-std-scale", type=float, default=0.025)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--release-threshold", type=float, default=0.15)
    parser.add_argument("--release-patience", type=int, default=5)
    parser.add_argument("--min-recovery-steps", type=int, default=5)
    parser.add_argument("--intervention-cooldown", type=int, default=10)
    parser.add_argument("--recovery-budget", type=int, default=250)
    parser.add_argument("--heuristic-min-steps", type=int, default=30)
    parser.add_argument("--heuristic-window", type=int, default=20)
    parser.add_argument("--heuristic-tolerance", type=float, default=0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--task-ids", type=int, nargs="*")
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
    parser.add_argument("--allow-partial", action="store_true")
    return parser


def main() -> None:
    summary = evaluate(build_parser().parse_args())
    print(
        json.dumps(
            {
                "benchmark": summary["benchmark"],
                "condition": summary["condition"],
                "methods": {
                    name: value["summary"]["success_rate_task_macro"]
                    for name, value in summary["aggregates"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
