#!/usr/bin/env python3
"""Fail-closed, read-only separation audit for MT10/MT50 data banks.

The multi-task study reserves independent Meta-World task banks for expert
demonstrations, failure training, detector validation, recovery training, and
final evaluation.  A different integer seed is not, by itself, evidence that
the sampled task payloads differ.  This command therefore reconstructs every
official 50-variant-per-task bank from its recorded benchmark seed and hashes
the native ``Task.data`` bytes.  Embedded protocol fingerprints, file digests,
materialized NPZ metadata, and final-evaluation CSV rows are then checked
against those reconstructed banks before pairwise overlap is measured.

The command never mutates audited evidence. A successful audit is emitted as
JSON on stdout and may also be persisted atomically with ``--output-json``;
any missing, partial, stale, inconsistent, or overlapping evidence raises an
error and returns a non-zero exit status without publishing a new report.
Raw episode RNG integers may intentionally be reused across *different* task
payload banks for paired noise.  They are reported, but leakage is defined on
the complete sampling-unit identity ``(task_name, task_payload, episode_seed)``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.io import atomic_write_json, file_sha256


SUPPORTED_BENCHMARKS = {"MT10": 10, "MT50": 50}
OFFICIAL_VARIANTS_PER_TASK = 50
OFFICIAL_MAX_EPISODE_STEPS = 500

DEMONSTRATION_SCHEMA = "reim-multitask-demonstrations-v1"
DEMONSTRATION_DATASET_TYPE = (
    "balanced_multitask_scripted_expert_demonstrations"
)
FAILURE_SCHEMA = "reim-multitask-failures-v2"
FAILURE_DATASET_TYPE = "task_conditioned_behavioral_deviation_risk"
RECOVERY_SCHEMA = "reim-multitask-trigger-aligned-recovery-v2"
RECOVERY_DATASET_TYPE = "online_detector_triggered_expert_continuations"
EVALUATION_SCHEMA = "reim-multitask-evaluation-v2"
EVALUATION_SIDECAR_SCHEMA = "reim-multitask-evaluation-run-v1"

STAGE_LABELS = (
    "demonstrations",
    "failure_training",
    "failure_validation",
    "recovery_training",
    "final_evaluation",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class BankAuditError(ValueError):
    """Raised when provenance is insufficient or contradicts stored data."""


@dataclass(frozen=True)
class TaskBankSnapshot:
    """Native Meta-World bank reconstructed from one benchmark seed."""

    benchmark: str
    benchmark_seed: int
    metaworld_version: str
    task_vocabulary: tuple[str, ...]
    payloads_by_task: tuple[tuple[str, ...], ...]
    failure_evaluation_bank_sha256: str
    recovery_bank_sha256: str
    content_sha256: str

    @property
    def payloads(self) -> frozenset[str]:
        return frozenset(
            payload
            for task_payloads in self.payloads_by_task
            for payload in task_payloads
        )

    def payload(self, task_id: int, variant: int) -> str:
        if not 0 <= task_id < len(self.task_vocabulary):
            raise BankAuditError(f"task_id {task_id} is outside the bank")
        if not 0 <= variant < OFFICIAL_VARIANTS_PER_TASK:
            raise BankAuditError(f"task variant {variant} is outside [0,49]")
        return self.payloads_by_task[task_id][variant]


@dataclass(frozen=True)
class StageEvidence:
    """Validated provenance extracted from one immutable study stage."""

    label: str
    source_paths: tuple[str, ...]
    source_sha256: tuple[str, ...]
    benchmark_seed: int
    embedded_fingerprint_sha256: str | None
    audit_descriptor_sha256: str
    task_bank_content_sha256: str
    reserved_payloads: frozenset[str]
    materialized_payloads: frozenset[str]
    episode_seeds: frozenset[int]
    sampling_unit_identities: frozenset[tuple[str, str, int]]
    materialized_records: int


BenchmarkLoader = Callable[[str, int], tuple[Any, str]]


def _canonical_sha256(payload: Any) -> str:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BankAuditError("provenance is not canonical-JSON serializable") from exc
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, context: str) -> str:
    result = str(value)
    if SHA256_PATTERN.fullmatch(result) is None:
        raise BankAuditError(f"{context} is not a lowercase SHA-256 digest")
    return result


def _require_int(
    value: Any,
    context: str,
    *,
    minimum: int = 0,
    allow_decimal_string: bool = False,
) -> int:
    if isinstance(value, bool):
        raise BankAuditError(f"{context} must be an integer")
    if isinstance(value, str):
        if not allow_decimal_string or re.fullmatch(r"0|[1-9][0-9]*", value) is None:
            raise BankAuditError(f"{context} must be an integer")
        result = int(value)
        if result < minimum:
            raise BankAuditError(f"{context} must be an integer >= {minimum}")
        return result
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise BankAuditError(f"{context} must be an integer") from exc
    if result != value or result < minimum:
        raise BankAuditError(f"{context} must be an integer >= {minimum}")
    return result


def _require_regular_file(path: Path, context: str) -> Path:
    resolved = path.expanduser().resolve()
    if path.expanduser().is_symlink() or not resolved.is_file():
        raise BankAuditError(f"{context} must be an existing non-symlink file: {path}")
    return resolved


def _load_json(path: Path, context: str) -> tuple[Path, dict[str, Any]]:
    resolved = _require_regular_file(path, context)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BankAuditError(f"cannot parse {context}: {resolved}") from exc
    if not isinstance(value, dict):
        raise BankAuditError(f"{context} must contain a JSON object")
    return resolved, value


def _safe_member(root: Path, value: Any, context: str) -> Path:
    text = str(value)
    relative = Path(text)
    if (
        not text
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != text
    ):
        raise BankAuditError(f"{context} has unsafe relative path {text!r}")
    candidate = root / relative
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise BankAuditError(f"{context} escapes its dataset directory") from exc
    if candidate.is_symlink() or not resolved.is_file():
        raise BankAuditError(f"{context} is missing or symlinked: {candidate}")
    return resolved


def _default_benchmark_loader(name: str, seed: int) -> tuple[Any, str]:
    try:
        import metaworld
    except ImportError as exc:  # pragma: no cover - deployment-only path
        raise BankAuditError(
            "Meta-World is required to reconstruct task payloads"
        ) from exc
    try:
        benchmark_type = getattr(metaworld, name)
    except AttributeError as exc:
        raise BankAuditError(f"installed Meta-World does not expose {name}") from exc
    try:
        version = importlib.metadata.version("metaworld")
    except importlib.metadata.PackageNotFoundError as exc:
        raise BankAuditError(
            "installed Meta-World has no package version; provenance cannot be verified"
        ) from exc
    return benchmark_type(seed=seed), version


def _task_payload_sha256(task: Any) -> str:
    try:
        payload = bytes(task.data)
    except (AttributeError, TypeError, ValueError) as exc:
        raise BankAuditError("Meta-World Task.data is not bytes-like") from exc
    return hashlib.sha256(payload).hexdigest()


def _toy_benchmark_loader(name: str, seed: int) -> tuple[Any, str]:
    """Explicit deterministic CI backend; never a silent fallback."""

    from env import toy_multitask

    try:
        benchmark_type = getattr(toy_multitask, name)
    except AttributeError as exc:
        raise BankAuditError(f"toy backend does not expose {name}") from exc
    return benchmark_type(seed=seed), toy_multitask.TOY_VERSION


def _loader_for_backend(backend: str) -> BenchmarkLoader:
    if backend == "toy":
        return _toy_benchmark_loader
    if backend == "metaworld":
        return _default_benchmark_loader
    raise BankAuditError(f"Unsupported backend {backend!r}; use 'metaworld' or 'toy'")


def _snapshot(
    benchmark_name: str,
    benchmark_seed: int,
    *,
    benchmark_loader: BenchmarkLoader,
) -> TaskBankSnapshot:
    benchmark, metaworld_version = benchmark_loader(
        benchmark_name, benchmark_seed
    )
    try:
        task_vocabulary = tuple(str(name) for name in benchmark.train_classes.keys())
        train_tasks = list(benchmark.train_tasks)
    except (AttributeError, TypeError) as exc:
        raise BankAuditError("benchmark lacks train_classes/train_tasks") from exc
    expected_tasks = SUPPORTED_BENCHMARKS[benchmark_name]
    if len(task_vocabulary) != expected_tasks or len(set(task_vocabulary)) != expected_tasks:
        raise BankAuditError(
            f"{benchmark_name} must expose {expected_tasks} unique train task classes"
        )
    grouped: dict[str, list[Any]] = {name: [] for name in task_vocabulary}
    for task in train_tasks:
        task_name = str(getattr(task, "env_name", ""))
        if task_name not in grouped:
            raise BankAuditError(f"task bank contains unknown task {task_name!r}")
        grouped[task_name].append(task)
    counts = {name: len(values) for name, values in grouped.items()}
    if any(value != OFFICIAL_VARIANTS_PER_TASK for value in counts.values()):
        raise BankAuditError(
            f"official {benchmark_name} requires 50 variants/task; got {counts}"
        )
    payloads_by_task = tuple(
        tuple(_task_payload_sha256(task) for task in grouped[name])
        for name in task_vocabulary
    )
    all_payloads = [value for values in payloads_by_task for value in values]
    if len(set(all_payloads)) != len(all_payloads):
        raise BankAuditError(
            f"reconstructed {benchmark_name} seed {benchmark_seed} has duplicate "
            "native task payloads"
        )
    failure_descriptor = [
        {
            "task_id": task_id,
            "task_name": task_name,
            "variant_payload_sha256": list(payloads_by_task[task_id]),
        }
        for task_id, task_name in enumerate(task_vocabulary)
    ]
    recovery_descriptor = [
        {
            "task_id": task_id,
            "task_name": task_name,
            "variant_sha256s": list(payloads_by_task[task_id]),
        }
        for task_id, task_name in enumerate(task_vocabulary)
    ]
    content_descriptor = [
        [task_name, list(payloads_by_task[task_id])]
        for task_id, task_name in enumerate(task_vocabulary)
    ]
    return TaskBankSnapshot(
        benchmark=benchmark_name,
        benchmark_seed=benchmark_seed,
        metaworld_version=str(metaworld_version),
        task_vocabulary=task_vocabulary,
        payloads_by_task=payloads_by_task,
        failure_evaluation_bank_sha256=_canonical_sha256(failure_descriptor),
        recovery_bank_sha256=_canonical_sha256(recovery_descriptor),
        content_sha256=_canonical_sha256(content_descriptor),
    )


def _check_common_manifest(
    manifest: Mapping[str, Any],
    snapshot: TaskBankSnapshot,
    *,
    schema_version: str,
    dataset_type: str,
    context: str,
) -> None:
    expected = {
        "schema_version": schema_version,
        "dataset_type": dataset_type,
        "benchmark": snapshot.benchmark,
        "metaworld_version": snapshot.metaworld_version,
        "task_vocabulary": list(snapshot.task_vocabulary),
        "task_vocabulary_sha256": _canonical_sha256(
            list(snapshot.task_vocabulary)
        ),
        "complete": True,
    }
    differences = [
        f"{key}: stored={manifest.get(key)!r}, expected={value!r}"
        for key, value in expected.items()
        if manifest.get(key) != value
    ]
    if differences:
        raise BankAuditError(f"{context} provenance mismatch: " + "; ".join(differences))


def _npz_scalar(archive: Any, name: str, path: Path) -> Any:
    if name not in archive:
        raise BankAuditError(f"{path} lacks scalar {name}")
    value = np.asarray(archive[name])
    if value.size != 1:
        raise BankAuditError(f"{path}:{name} must be scalar")
    return value.reshape(-1)[0]


def _verify_digest(path: Path, expected: Any, context: str) -> str:
    expected_digest = _require_sha256(expected, f"{context}.sha256")
    actual = file_sha256(path)
    if actual != expected_digest:
        raise BankAuditError(
            f"{context} digest mismatch: stored={expected_digest}, actual={actual}"
        )
    return actual


def _make_evidence(
    *,
    label: str,
    source_paths: Sequence[Path],
    snapshot: TaskBankSnapshot,
    embedded_fingerprint: str | None,
    materialized: Sequence[tuple[str, str, int]],
) -> StageEvidence:
    if not materialized:
        raise BankAuditError(f"{label} has no materialized records")
    source_digests = tuple(file_sha256(path) for path in source_paths)
    if embedded_fingerprint is not None:
        _require_sha256(embedded_fingerprint, f"{label} embedded fingerprint")
    descriptor = {
        "label": label,
        "benchmark": snapshot.benchmark,
        "benchmark_seed": snapshot.benchmark_seed,
        "task_bank_content_sha256": snapshot.content_sha256,
        "source_sha256": list(source_digests),
        "embedded_fingerprint_sha256": embedded_fingerprint,
    }
    identities = frozenset(materialized)
    if len(identities) != len(materialized):
        raise BankAuditError(
            f"{label} contains duplicate materialized sampling-unit identities"
        )
    return StageEvidence(
        label=label,
        source_paths=tuple(str(path) for path in source_paths),
        source_sha256=source_digests,
        benchmark_seed=snapshot.benchmark_seed,
        embedded_fingerprint_sha256=embedded_fingerprint,
        audit_descriptor_sha256=_canonical_sha256(descriptor),
        task_bank_content_sha256=snapshot.content_sha256,
        reserved_payloads=snapshot.payloads,
        materialized_payloads=frozenset(value[1] for value in identities),
        episode_seeds=frozenset(value[2] for value in identities),
        sampling_unit_identities=identities,
        materialized_records=len(materialized),
    )


def _audit_demonstrations(
    path: Path,
    benchmark_name: str,
    *,
    benchmark_loader: BenchmarkLoader,
) -> StageEvidence:
    manifest_path, manifest = _load_json(path, "demonstration manifest")
    benchmark_seed = _require_int(
        manifest.get("seed"), "demonstrations.seed"
    )
    snapshot = _snapshot(
        benchmark_name, benchmark_seed, benchmark_loader=benchmark_loader
    )
    _check_common_manifest(
        manifest,
        snapshot,
        schema_version=DEMONSTRATION_SCHEMA,
        dataset_type=DEMONSTRATION_DATASET_TYPE,
        context="demonstrations",
    )
    trajectories = manifest.get("trajectories")
    if not isinstance(trajectories, list) or not trajectories:
        raise BankAuditError("demonstrations.trajectories must be a non-empty list")
    statistics = manifest.get("statistics")
    if not isinstance(statistics, Mapping):
        raise BankAuditError("demonstrations.statistics is missing")
    expected_count = len(snapshot.task_vocabulary) * _require_int(
        manifest.get("protocol", {}).get("episodes_per_task")
        if isinstance(manifest.get("protocol"), Mapping)
        else None,
        "demonstrations.protocol.episodes_per_task",
        minimum=1,
    )
    if len(trajectories) != expected_count:
        raise BankAuditError(
            f"demonstrations has {len(trajectories)} trajectories; expected {expected_count}"
        )
    if statistics.get("successful_trajectories") != expected_count:
        raise BankAuditError("demonstration statistics disagree with trajectory inventory")

    root = manifest_path.parent
    slots: set[tuple[int, int]] = set()
    materialized: list[tuple[str, str, int]] = []
    for index, raw_entry in enumerate(trajectories):
        if not isinstance(raw_entry, Mapping):
            raise BankAuditError(f"demonstration trajectory {index} is not an object")
        entry = dict(raw_entry)
        task_id = _require_int(entry.get("task_id"), f"trajectory[{index}].task_id")
        trajectory_index = _require_int(
            entry.get("trajectory_index"), f"trajectory[{index}].trajectory_index"
        )
        slot = (task_id, trajectory_index)
        if slot in slots:
            raise BankAuditError(f"duplicate demonstration slot {slot}")
        slots.add(slot)
        task_name = str(entry.get("task_name"))
        if task_id >= len(snapshot.task_vocabulary) or snapshot.task_vocabulary[task_id] != task_name:
            raise BankAuditError(f"trajectory[{index}] has inconsistent task identity")
        variant = _require_int(
            entry.get("task_variant"),
            f"trajectory[{index}].task_variant",
        )
        payload_hash = snapshot.payload(task_id, variant)
        episode_seed = _require_int(entry.get("seed"), f"trajectory[{index}].seed")
        member = _safe_member(root, entry.get("file"), f"trajectory[{index}].file")
        _verify_digest(member, entry.get("sha256"), f"trajectory[{index}]")
        try:
            with np.load(member, allow_pickle=False) as archive:
                actual = {
                    "task_id": int(_npz_scalar(archive, "task_id", member)),
                    "task_name": str(_npz_scalar(archive, "task_name", member)),
                    "task_variant": int(_npz_scalar(archive, "task_variant", member)),
                    "seed": int(_npz_scalar(archive, "seed", member)),
                }
        except (OSError, ValueError, KeyError) as exc:
            if isinstance(exc, BankAuditError):
                raise
            raise BankAuditError(f"cannot validate demonstration shard {member}") from exc
        expected = {
            "task_id": task_id,
            "task_name": task_name,
            "task_variant": variant,
            "seed": episode_seed,
        }
        if actual != expected:
            raise BankAuditError(
                f"trajectory[{index}] manifest/NPZ metadata mismatch: "
                f"stored={entry}, actual={actual}"
            )
        materialized.append((task_name, payload_hash, episode_seed))
    expected_slots = {
        (task_id, trajectory_index)
        for task_id in range(len(snapshot.task_vocabulary))
        for trajectory_index in range(expected_count // len(snapshot.task_vocabulary))
    }
    if slots != expected_slots:
        raise BankAuditError("demonstration task/trajectory slots are incomplete")
    return _make_evidence(
        label="demonstrations",
        source_paths=[manifest_path],
        snapshot=snapshot,
        embedded_fingerprint=None,
        materialized=materialized,
    )


def _audit_failure_bank(
    path: Path,
    benchmark_name: str,
    *,
    label: str,
    benchmark_loader: BenchmarkLoader,
) -> StageEvidence:
    manifest_path, manifest = _load_json(path, f"{label} manifest")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise BankAuditError(f"{label} lacks structured provenance")
    fingerprint = _require_sha256(
        manifest.get("provenance_fingerprint_sha256"),
        f"{label}.provenance_fingerprint_sha256",
    )
    if fingerprint != _canonical_sha256(provenance):
        raise BankAuditError(f"{label} provenance fingerprint is invalid")
    benchmark_seed = _require_int(
        provenance.get("benchmark_seed"), f"{label}.provenance.benchmark_seed"
    )
    snapshot = _snapshot(
        benchmark_name, benchmark_seed, benchmark_loader=benchmark_loader
    )
    _check_common_manifest(
        manifest,
        snapshot,
        schema_version=FAILURE_SCHEMA,
        dataset_type=FAILURE_DATASET_TYPE,
        context=label,
    )
    protocol_expected = {
        "schema_version": FAILURE_SCHEMA,
        "dataset_type": FAILURE_DATASET_TYPE,
        "benchmark": benchmark_name,
        "metaworld_version": snapshot.metaworld_version,
        "benchmark_seed": benchmark_seed,
        "task_vocabulary": list(snapshot.task_vocabulary),
        "task_vocabulary_sha256": _canonical_sha256(list(snapshot.task_vocabulary)),
        "benchmark_task_bank_sha256": snapshot.failure_evaluation_bank_sha256,
    }
    for key, value in protocol_expected.items():
        if provenance.get(key) != value:
            raise BankAuditError(
                f"{label}.provenance.{key}={provenance.get(key)!r}; expected {value!r}"
            )
    if manifest.get("benchmark_task_bank_sha256") != snapshot.failure_evaluation_bank_sha256:
        raise BankAuditError(f"{label} top-level task-bank hash is invalid")
    collection_seed = _require_int(
        provenance.get("collection_seed"), f"{label}.provenance.collection_seed"
    )
    if manifest.get("seed") != collection_seed or manifest.get("benchmark_seed") != benchmark_seed:
        raise BankAuditError(f"{label} top-level seeds contradict provenance")
    rollouts = _require_int(
        manifest.get("rollouts_per_task"), f"{label}.rollouts_per_task", minimum=1
    )
    files = manifest.get("files")
    expected_count = len(snapshot.task_vocabulary) * rollouts
    if not isinstance(files, list) or len(files) != expected_count:
        raise BankAuditError(f"{label} file inventory is incomplete")
    if manifest.get("episodes") != expected_count:
        raise BankAuditError(f"{label} episode count contradicts inventory")

    root = manifest_path.parent
    slots: set[tuple[int, int]] = set()
    materialized: list[tuple[str, str, int]] = []
    for index, raw_entry in enumerate(files):
        if not isinstance(raw_entry, Mapping):
            raise BankAuditError(f"{label}.files[{index}] is not an object")
        entry = dict(raw_entry)
        task_id = _require_int(entry.get("task_id"), f"{label}.files[{index}].task_id")
        rollout = _require_int(
            entry.get("rollout_index"), f"{label}.files[{index}].rollout_index"
        )
        slot = (task_id, rollout)
        if slot in slots:
            raise BankAuditError(f"{label} has duplicate rollout slot {slot}")
        slots.add(slot)
        if task_id >= len(snapshot.task_vocabulary) or rollout >= rollouts:
            raise BankAuditError(f"{label} has an out-of-range rollout slot {slot}")
        task_name = snapshot.task_vocabulary[task_id]
        if entry.get("task_name") != task_name:
            raise BankAuditError(f"{label} entry {index} has inconsistent task name")
        variant = rollout % OFFICIAL_VARIANTS_PER_TASK
        payload_hash = snapshot.payload(task_id, variant)
        episode_seed = collection_seed + task_id * 100_000 + rollout
        member = _safe_member(root, entry.get("file"), f"{label}.files[{index}].file")
        _verify_digest(member, entry.get("sha256"), f"{label}.files[{index}]")
        try:
            with np.load(member, allow_pickle=False) as archive:
                actual = {
                    "task_id": int(_npz_scalar(archive, "task_id", member)),
                    "task_name": str(_npz_scalar(archive, "task_name", member)),
                    "task_variant": int(_npz_scalar(archive, "task_variant", member)),
                    "task_payload_sha256": str(
                        _npz_scalar(archive, "task_payload_sha256", member)
                    ),
                    "episode_seed": int(
                        _npz_scalar(archive, "episode_seed", member)
                    ),
                    "schema_version": str(
                        _npz_scalar(archive, "schema_version", member)
                    ),
                }
        except (OSError, ValueError, KeyError) as exc:
            if isinstance(exc, BankAuditError):
                raise
            raise BankAuditError(f"cannot validate failure shard {member}") from exc
        expected = {
            "task_id": task_id,
            "task_name": task_name,
            "task_variant": variant,
            "task_payload_sha256": payload_hash,
            "episode_seed": episode_seed,
            "schema_version": FAILURE_SCHEMA,
        }
        if actual != expected:
            raise BankAuditError(
                f"{label}.files[{index}] NPZ provenance mismatch: "
                f"actual={actual}, expected={expected}"
            )
        materialized.append((task_name, payload_hash, episode_seed))
    expected_slots = {
        (task_id, rollout)
        for task_id in range(len(snapshot.task_vocabulary))
        for rollout in range(rollouts)
    }
    if slots != expected_slots:
        raise BankAuditError(f"{label} rollout slots are incomplete")
    return _make_evidence(
        label=label,
        source_paths=[manifest_path],
        snapshot=snapshot,
        embedded_fingerprint=fingerprint,
        materialized=materialized,
    )


def _recovery_episode_seed(base_seed: int, task_id: int, attempt_index: int) -> int:
    if not 0 <= attempt_index < 1_000_000:
        raise BankAuditError("recovery attempt index is outside its seed namespace")
    return (
        (base_seed % 1_000_000_000) * 100_000_000
        + task_id * 1_000_000
        + attempt_index
    )


def _audit_recovery(
    path: Path,
    benchmark_name: str,
    *,
    benchmark_loader: BenchmarkLoader,
) -> StageEvidence:
    label = "recovery_training"
    manifest_path, manifest = _load_json(path, "recovery manifest")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise BankAuditError("recovery manifest lacks structured protocol")
    fingerprint = _require_sha256(
        manifest.get("protocol_fingerprint_sha256"),
        "recovery.protocol_fingerprint_sha256",
    )
    if fingerprint != _canonical_sha256(protocol):
        raise BankAuditError("recovery protocol fingerprint is invalid")
    benchmark_seed = _require_int(
        protocol.get("benchmark_seed"), "recovery.protocol.benchmark_seed"
    )
    snapshot = _snapshot(
        benchmark_name, benchmark_seed, benchmark_loader=benchmark_loader
    )
    _check_common_manifest(
        manifest,
        snapshot,
        schema_version=RECOVERY_SCHEMA,
        dataset_type=RECOVERY_DATASET_TYPE,
        context="recovery",
    )
    protocol_expected = {
        "schema_version": RECOVERY_SCHEMA,
        "dataset_type": RECOVERY_DATASET_TYPE,
        "benchmark": benchmark_name,
        "metaworld_version": snapshot.metaworld_version,
        "benchmark_seed": benchmark_seed,
        "task_vocabulary": list(snapshot.task_vocabulary),
        "task_vocabulary_sha256": _canonical_sha256(list(snapshot.task_vocabulary)),
        "task_bank_sha256": snapshot.recovery_bank_sha256,
        "official_goal_variants_per_task": OFFICIAL_VARIANTS_PER_TASK,
    }
    for key, value in protocol_expected.items():
        if protocol.get(key) != value:
            raise BankAuditError(
                f"recovery.protocol.{key}={protocol.get(key)!r}; expected {value!r}"
            )
    if manifest.get("task_bank_sha256") != snapshot.recovery_bank_sha256:
        raise BankAuditError("recovery top-level task-bank hash is invalid")
    rollout_seed = _require_int(
        protocol.get("rollout_seed"), "recovery.protocol.rollout_seed"
    )
    if manifest.get("seed") != rollout_seed or manifest.get("benchmark_seed") != benchmark_seed:
        raise BankAuditError("recovery top-level seeds contradict protocol")
    target = _require_int(
        protocol.get("target_per_task"), "recovery.protocol.target_per_task", minimum=1
    )
    files = manifest.get("files")
    expected_count = len(snapshot.task_vocabulary) * target
    if not isinstance(files, list) or len(files) != expected_count:
        raise BankAuditError("recovery file inventory is incomplete")
    if manifest.get("successful_continuations") != expected_count:
        raise BankAuditError("recovery continuation count contradicts inventory")

    root = manifest_path.parent
    slots: set[tuple[int, int]] = set()
    materialized: list[tuple[str, str, int]] = []
    for index, raw_entry in enumerate(files):
        if not isinstance(raw_entry, Mapping):
            raise BankAuditError(f"recovery.files[{index}] is not an object")
        entry = dict(raw_entry)
        task_id = _require_int(entry.get("task_id"), f"recovery.files[{index}].task_id")
        shard_index = _require_int(
            entry.get("shard_index"), f"recovery.files[{index}].shard_index"
        )
        slot = (task_id, shard_index)
        if slot in slots:
            raise BankAuditError(f"recovery has duplicate shard slot {slot}")
        slots.add(slot)
        if task_id >= len(snapshot.task_vocabulary) or shard_index >= target:
            raise BankAuditError(f"recovery has an out-of-range shard slot {slot}")
        task_name = snapshot.task_vocabulary[task_id]
        if entry.get("task_name") != task_name:
            raise BankAuditError(f"recovery entry {index} has inconsistent task name")
        attempt_index = _require_int(
            entry.get("attempt_index"), f"recovery.files[{index}].attempt_index"
        )
        variant = attempt_index % OFFICIAL_VARIANTS_PER_TASK
        payload_hash = snapshot.payload(task_id, variant)
        episode_seed = _recovery_episode_seed(rollout_seed, task_id, attempt_index)
        if entry.get("task_variant") != variant:
            raise BankAuditError(f"recovery entry {index} has invalid task variant")
        if entry.get("task_payload_sha256") != payload_hash:
            raise BankAuditError(f"recovery entry {index} has invalid task payload hash")
        if entry.get("episode_seed") != episode_seed:
            raise BankAuditError(f"recovery entry {index} has invalid episode seed")
        member = _safe_member(root, entry.get("file"), f"recovery.files[{index}].file")
        _verify_digest(member, entry.get("sha256"), f"recovery.files[{index}]")
        try:
            with np.load(member, allow_pickle=False) as archive:
                actual = {
                    "task_id": int(_npz_scalar(archive, "task_id", member)),
                    "task_name": str(_npz_scalar(archive, "task_name", member)),
                    "task_variant": int(_npz_scalar(archive, "task_variant", member)),
                    "task_payload_sha256": str(
                        _npz_scalar(archive, "task_payload_sha256", member)
                    ),
                    "episode_seed": int(
                        _npz_scalar(archive, "episode_seed", member)
                    ),
                    "attempt_index": int(
                        _npz_scalar(archive, "attempt_index", member)
                    ),
                    "shard_index": int(
                        _npz_scalar(archive, "shard_index", member)
                    ),
                    "protocol_fingerprint_sha256": str(
                        _npz_scalar(
                            archive, "protocol_fingerprint_sha256", member
                        )
                    ),
                    "schema_version": str(
                        _npz_scalar(archive, "schema_version", member)
                    ),
                }
        except (OSError, ValueError, KeyError) as exc:
            if isinstance(exc, BankAuditError):
                raise
            raise BankAuditError(f"cannot validate recovery shard {member}") from exc
        expected = {
            "task_id": task_id,
            "task_name": task_name,
            "task_variant": variant,
            "task_payload_sha256": payload_hash,
            "episode_seed": episode_seed,
            "attempt_index": attempt_index,
            "shard_index": shard_index,
            "protocol_fingerprint_sha256": fingerprint,
            "schema_version": RECOVERY_SCHEMA,
        }
        if actual != expected:
            raise BankAuditError(
                f"recovery.files[{index}] NPZ provenance mismatch: "
                f"actual={actual}, expected={expected}"
            )
        materialized.append((task_name, payload_hash, episode_seed))
    expected_slots = {
        (task_id, shard_index)
        for task_id in range(len(snapshot.task_vocabulary))
        for shard_index in range(target)
    }
    if slots != expected_slots:
        raise BankAuditError("recovery shard slots are incomplete")
    return _make_evidence(
        label=label,
        source_paths=[manifest_path],
        snapshot=snapshot,
        embedded_fingerprint=fingerprint,
        materialized=materialized,
    )


def _audit_final_evaluation(
    sidecar_path: Path,
    csv_path: Path,
    benchmark_name: str,
    *,
    benchmark_loader: BenchmarkLoader,
    expected_episodes_per_task: int = OFFICIAL_VARIANTS_PER_TASK,
) -> StageEvidence:
    sidecar_resolved, sidecar = _load_json(
        sidecar_path, "final evaluation run sidecar"
    )
    csv_resolved = _require_regular_file(csv_path, "final evaluation episode CSV")
    if sidecar.get("schema_version") != EVALUATION_SIDECAR_SCHEMA:
        raise BankAuditError("final evaluation sidecar schema is stale or unknown")
    protocol = sidecar.get("protocol")
    if not isinstance(protocol, Mapping):
        raise BankAuditError("final evaluation sidecar lacks structured protocol")
    fingerprint = _require_sha256(
        sidecar.get("run_fingerprint"), "final evaluation run_fingerprint"
    )
    if fingerprint != _canonical_sha256(protocol):
        raise BankAuditError("final evaluation run fingerprint is invalid")
    if protocol.get("evaluation_schema_version") != EVALUATION_SCHEMA:
        raise BankAuditError("final evaluation protocol schema is stale or unknown")
    benchmark_seed = _require_int(
        protocol.get("benchmark_seed"), "final evaluation benchmark_seed"
    )
    snapshot = _snapshot(
        benchmark_name, benchmark_seed, benchmark_loader=benchmark_loader
    )
    expected_protocol = {
        "benchmark": benchmark_name,
        "metaworld_version": snapshot.metaworld_version,
        "task_vocabulary": list(snapshot.task_vocabulary),
        "task_vocabulary_sha256": _canonical_sha256(list(snapshot.task_vocabulary)),
        "task_bank_sha256": snapshot.failure_evaluation_bank_sha256,
        "task_ids": list(range(len(snapshot.task_vocabulary))),
        "episodes_per_task": expected_episodes_per_task,
        "max_episode_steps": OFFICIAL_MAX_EPISODE_STEPS,
        "noise_level": 0.0,
        "object_position_noise": False,
    }
    for key, value in expected_protocol.items():
        if protocol.get(key) != value:
            raise BankAuditError(
                f"final evaluation protocol {key}={protocol.get(key)!r}; "
                f"expected {value!r}"
            )
    methods = protocol.get("methods")
    if (
        not isinstance(methods, list)
        or not methods
        or any(not isinstance(value, str) or not value for value in methods)
        or len(set(methods)) != len(methods)
    ):
        raise BankAuditError("final evaluation methods are missing or invalid")
    episode_seed_base = _require_int(
        protocol.get("episode_seed_base"), "final evaluation episode_seed_base"
    )
    try:
        with csv_resolved.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required_columns = {
                "run_fingerprint",
                "benchmark",
                "task_name",
                "task_id",
                "task_variant",
                "method",
                "paired_episode_id",
                "episode_seed",
                "task_payload_sha256",
            }
            if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
                raise BankAuditError("final evaluation CSV lacks provenance columns")
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise BankAuditError(f"cannot parse final evaluation CSV {csv_resolved}") from exc
    expected_count = (
        len(snapshot.task_vocabulary)
        * expected_episodes_per_task
        * len(methods)
    )
    if len(rows) != expected_count:
        raise BankAuditError(
            f"final evaluation has {len(rows)} rows; expected {expected_count}"
        )

    cells: dict[tuple[int, int], set[str]] = {}
    seen_rows: set[tuple[int, int, str]] = set()
    materialized_by_episode: dict[
        tuple[int, int], tuple[str, str, int]
    ] = {}
    for index, row in enumerate(rows, start=2):
        if row.get("run_fingerprint") != fingerprint:
            raise BankAuditError(f"evaluation CSV row {index} has a foreign fingerprint")
        if row.get("benchmark") != benchmark_name:
            raise BankAuditError(f"evaluation CSV row {index} has wrong benchmark")
        task_id = _require_int(
            row.get("task_id"),
            f"evaluation row {index} task_id",
            allow_decimal_string=True,
        )
        variant = _require_int(
            row.get("task_variant"),
            f"evaluation row {index} task_variant",
            allow_decimal_string=True,
        )
        if task_id >= len(snapshot.task_vocabulary):
            raise BankAuditError(f"evaluation CSV row {index} task_id is out of range")
        task_name = snapshot.task_vocabulary[task_id]
        if row.get("task_name") != task_name:
            raise BankAuditError(f"evaluation CSV row {index} has wrong task name")
        payload_hash = snapshot.payload(task_id, variant)
        if row.get("task_payload_sha256") != payload_hash:
            raise BankAuditError(f"evaluation CSV row {index} has wrong payload hash")
        episode_seed = _require_int(
            row.get("episode_seed"),
            f"evaluation row {index} episode_seed",
            allow_decimal_string=True,
        )
        expected_seed = episode_seed_base + task_id * 100_000 + variant
        if episode_seed != expected_seed:
            raise BankAuditError(f"evaluation CSV row {index} has wrong episode seed")
        expected_paired_id = f"{task_id:02d}-{variant:04d}-{payload_hash[:12]}"
        if row.get("paired_episode_id") != expected_paired_id:
            raise BankAuditError(f"evaluation CSV row {index} has wrong paired ID")
        method = str(row.get("method", ""))
        if not method:
            raise BankAuditError(f"evaluation CSV row {index} has empty method")
        row_key = (task_id, variant, method)
        if row_key in seen_rows:
            raise BankAuditError(f"evaluation CSV has duplicate row {row_key}")
        seen_rows.add(row_key)
        cells.setdefault((task_id, variant), set()).add(method)
        materialized_by_episode[(task_id, variant)] = (
            task_name,
            payload_hash,
            episode_seed,
        )
    method_labels = set(next(iter(cells.values()))) if cells else set()
    if len(method_labels) != len(methods):
        raise BankAuditError("evaluation CSV method labels do not match protocol count")
    expected_cells = {
        (task_id, variant)
        for task_id in range(len(snapshot.task_vocabulary))
        for variant in range(expected_episodes_per_task)
    }
    if set(cells) != expected_cells or any(value != method_labels for value in cells.values()):
        raise BankAuditError("evaluation CSV task/variant/method grid is incomplete")
    # Evaluation methods share paired task variants and RNG streams.  Record
    # each episode once so within-stage CRN pairing is not mistaken for leakage.
    return _make_evidence(
        label="final_evaluation",
        source_paths=[sidecar_resolved, csv_resolved],
        snapshot=snapshot,
        embedded_fingerprint=fingerprint,
        materialized=[materialized_by_episode[key] for key in sorted(materialized_by_episode)],
    )


def _pairwise_report(evidence: Sequence[StageEvidence]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(evidence):
        for right in evidence[left_index + 1 :]:
            reserved_overlap = left.reserved_payloads & right.reserved_payloads
            materialized_overlap = (
                left.materialized_payloads & right.materialized_payloads
            )
            identity_overlap = (
                left.sampling_unit_identities & right.sampling_unit_identities
            )
            seed_overlap = left.episode_seeds & right.episode_seeds
            pairs.append(
                {
                    "left": left.label,
                    "right": right.label,
                    "benchmark_seed_equal": left.benchmark_seed == right.benchmark_seed,
                    "task_bank_content_fingerprint_equal": (
                        left.task_bank_content_sha256
                        == right.task_bank_content_sha256
                    ),
                    "reserved_task_payload_overlap_count": len(reserved_overlap),
                    "materialized_task_payload_overlap_count": len(
                        materialized_overlap
                    ),
                    "sampling_unit_identity_overlap_count": len(identity_overlap),
                    "raw_episode_seed_overlap_count": len(seed_overlap),
                    "raw_episode_seed_reuse_is_leakage": bool(identity_overlap),
                    "overlap_examples": {
                        "task_payload_sha256": sorted(reserved_overlap)[:3],
                        "sampling_unit_identity": [
                            list(item) for item in sorted(identity_overlap)[:3]
                        ],
                        "raw_episode_seed": sorted(seed_overlap)[:3],
                    },
                }
            )
    return pairs


def audit(
    args: argparse.Namespace,
    *,
    benchmark_loader: BenchmarkLoader | None = None,
) -> dict[str, Any]:
    """Validate all five banks and return a read-only overlap report."""

    benchmark_name = str(args.benchmark).upper()
    if benchmark_name not in SUPPORTED_BENCHMARKS:
        raise BankAuditError("--benchmark must be MT10 or MT50")
    loader = benchmark_loader or _loader_for_backend(
        str(getattr(args, "backend", "metaworld"))
    )
    evidence = [
        _audit_demonstrations(
            Path(args.demonstrations), benchmark_name, benchmark_loader=loader
        ),
        _audit_failure_bank(
            Path(args.failure_train),
            benchmark_name,
            label="failure_training",
            benchmark_loader=loader,
        ),
        _audit_failure_bank(
            Path(args.failure_validation),
            benchmark_name,
            label="failure_validation",
            benchmark_loader=loader,
        ),
        _audit_recovery(
            Path(args.recovery), benchmark_name, benchmark_loader=loader
        ),
        _audit_final_evaluation(
            Path(args.final_evaluation_sidecar),
            Path(args.final_evaluation_csv),
            benchmark_name,
            benchmark_loader=loader,
            expected_episodes_per_task=int(
                getattr(args, "expected_episodes_per_task", OFFICIAL_VARIANTS_PER_TASK)
            ),
        ),
    ]
    if tuple(item.label for item in evidence) != STAGE_LABELS:
        raise AssertionError("internal stage ordering error")
    pairwise = _pairwise_report(evidence)
    checks = {
        "all_artifacts_complete_and_digest_verified": True,
        "benchmark_bank_seeds_pairwise_distinct": (
            len({item.benchmark_seed for item in evidence}) == len(evidence)
        ),
        "task_bank_content_fingerprints_pairwise_distinct": (
            len({item.task_bank_content_sha256 for item in evidence})
            == len(evidence)
        ),
        "reserved_native_task_payloads_pairwise_disjoint": all(
            item["reserved_task_payload_overlap_count"] == 0 for item in pairwise
        ),
        "materialized_native_task_payloads_pairwise_disjoint": all(
            item["materialized_task_payload_overlap_count"] == 0
            for item in pairwise
        ),
        "materialized_sampling_unit_identities_pairwise_disjoint": all(
            item["sampling_unit_identity_overlap_count"] == 0 for item in pairwise
        ),
        "artifact_protocol_descriptors_pairwise_distinct": (
            len({item.audit_descriptor_sha256 for item in evidence})
            == len(evidence)
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise BankAuditError(
            "bank-separation audit failed: " + ", ".join(failed)
        )
    raw_seed_pairs = [
        f"{item['left']}::{item['right']}"
        for item in pairwise
        if item["raw_episode_seed_overlap_count"] > 0
    ]
    return {
        "schema_version": "reim-multitask-bank-separation-audit-v1",
        "read_only": True,
        "passed": True,
        "benchmark": benchmark_name,
        "separation_definition": {
            "reserved_bank": "SHA-256(Task.data) over all 50 variants/task",
            "materialized_sampling_unit": [
                "task_name",
                "task_payload_sha256",
                "episode_seed",
            ],
            "raw_episode_seed_note": (
                "An RNG integer reused with a disjoint native task payload is "
                "reported but is not a repeated sampling unit."
            ),
        },
        "checks": checks,
        "stages": {
            item.label: {
                "source_paths": list(item.source_paths),
                "source_sha256": list(item.source_sha256),
                "benchmark_seed": item.benchmark_seed,
                "embedded_fingerprint_sha256": (
                    item.embedded_fingerprint_sha256
                ),
                "audit_descriptor_sha256": item.audit_descriptor_sha256,
                "task_bank_content_sha256": item.task_bank_content_sha256,
                "reserved_task_payloads": len(item.reserved_payloads),
                "materialized_task_payloads": len(item.materialized_payloads),
                "materialized_records": item.materialized_records,
                "unique_episode_seeds": len(item.episode_seeds),
            }
            for item in evidence
        },
        "pairwise": pairwise,
        "raw_episode_seed_reuse_pairs": raw_seed_pairs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=("MT10", "MT50"))
    parser.add_argument(
        "--backend",
        choices=("metaworld", "toy"),
        default="metaworld",
        help=(
            "'toy' selects the explicit deterministic CI benchmark "
            "(env/toy_multitask.py) when reconstructing task payloads."
        ),
    )
    parser.add_argument("--demonstrations", required=True, help="manifest.json")
    parser.add_argument("--failure-train", required=True, help="manifest.json")
    parser.add_argument("--failure-validation", required=True, help="manifest.json")
    parser.add_argument("--recovery", required=True, help="manifest.json")
    parser.add_argument(
        "--final-evaluation-sidecar",
        required=True,
        help="immutable *.csv.run.json sidecar",
    )
    parser.add_argument(
        "--final-evaluation-csv", required=True, help="complete clean episode CSV"
    )
    parser.add_argument(
        "--output-json",
        help=(
            "Optional persistent audit report. Written atomically only after all "
            "five banks pass."
        ),
    )
    parser.add_argument(
        "--expected-episodes-per-task",
        type=int,
        default=OFFICIAL_VARIANTS_PER_TASK,
        help=(
            "Final-evaluation episodes per task. Defaults to the publication "
            "protocol's 50; the explicit CI smoke profile passes its smaller "
            "preregistered count."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = audit(args)
        if args.output_json:
            output_path = Path(args.output_json).expanduser()
            if output_path.is_symlink():
                raise BankAuditError(
                    f"refusing to replace symlinked audit report: {output_path}"
                )
            atomic_write_json(output_path, result)
    except (BankAuditError, OSError) as exc:
        print(f"bank audit failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
