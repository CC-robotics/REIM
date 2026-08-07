#!/usr/bin/env python3
"""Fail-closed merge of disjoint MT10/MT50 evaluation shards.

Each input is the episode CSV and JSON summary emitted by
``evaluation/evaluate_multitask.py`` for a disjoint ``--task-ids`` subset.
The merger validates the summary -> CSV and summary -> run-sidecar digest
chains, rejects every protocol difference except ``task_ids``, verifies the
complete paired episode grid, and recomputes all statistics from raw rows.

The output is a normal full-suite evaluator CSV, run sidecar, and summary.
Rows receive the newly derived full-suite run fingerprint, so the artifacts
can be consumed unchanged by ``audit_multitask_banks.py`` and the publication
asset gate.  No output is installed until every shard has passed validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.io import file_sha256, json_compatible
from evaluation import evaluate_multitask as evaluator
from evaluation.multitask_metrics import (
    aggregate_multitask_metrics,
    paired_task_stratified_bootstrap_delta,
)


MERGE_SCHEMA_VERSION = "reim-multitask-evaluation-merge-v1"
SUITE_TASK_COUNT = {"MT10": 10, "MT50": 50}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class EvaluationShardMergeError(ValueError):
    """Raised when shard evidence cannot support an exact full-suite merge."""


@dataclass(frozen=True)
class ShardEvidence:
    summary_path: Path
    csv_path: Path
    sidecar_path: Path
    summary: Mapping[str, Any]
    protocol: Mapping[str, Any]
    task_ids: tuple[int, ...]
    rows: tuple[dict[str, str], ...]
    completed: frozenset[tuple[str, int, str]]


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvaluationShardMergeError(
            "protocol is not canonical-JSON serializable"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, context: str) -> str:
    result = str(value)
    if SHA256_RE.fullmatch(result) is None:
        raise EvaluationShardMergeError(
            f"{context} is not a lowercase SHA-256 digest"
        )
    return result


def _require_int(
    value: Any,
    context: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool):
        raise EvaluationShardMergeError(f"{context} must be an integer")
    try:
        numeric = float(value)
        result = int(numeric)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvaluationShardMergeError(f"{context} must be an integer") from exc
    if (
        not math.isfinite(numeric)
        or numeric != result
        or result < minimum
        or (maximum is not None and result > maximum)
    ):
        raise EvaluationShardMergeError(f"{context} must be an integer in range")
    return result


def _require_float(
    value: Any,
    context: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationShardMergeError(f"{context} must be numeric") from exc
    if (
        not math.isfinite(result)
        or (minimum is not None and result < minimum)
        or (maximum is not None and result > maximum)
    ):
        raise EvaluationShardMergeError(f"{context} must be finite and in range")
    return result


def _regular_file(path: Path, context: str) -> Path:
    expanded = path.expanduser()
    resolved = expanded.resolve()
    if expanded.is_symlink() or not resolved.is_file():
        raise EvaluationShardMergeError(
            f"{context} must be an existing non-symlink file: {path}"
        )
    return resolved


def _load_json(path: Path, context: str) -> tuple[Path, dict[str, Any]]:
    resolved = _regular_file(path, context)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationShardMergeError(f"cannot parse {context}: {resolved}") from exc
    if not isinstance(value, dict):
        raise EvaluationShardMergeError(f"{context} must contain a JSON object")
    return resolved, value


def _load_csv(path: Path, context: str) -> tuple[Path, list[dict[str, str]]]:
    resolved = _regular_file(path, context)
    try:
        with resolved.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(evaluator.CSV_FIELDS):
                raise EvaluationShardMergeError(
                    f"{context} header does not exactly match evaluation schema"
                )
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise EvaluationShardMergeError(f"cannot parse {context}: {resolved}") from exc
    if not rows:
        raise EvaluationShardMergeError(f"{context} is empty")
    return resolved, rows


def _resolve_recorded_path(value: Any, anchor: Path, context: str) -> Path:
    recorded = Path(str(value)).expanduser()
    if not recorded.is_absolute():
        recorded = anchor.parent / recorded
    return _regular_file(recorded, context)


def _assert_equivalent(actual: Any, expected: Any, context: str) -> None:
    """Compare stored statistics with a raw-row recomputation."""

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise EvaluationShardMergeError(
                f"{context} keys differ from raw-row recomputation"
            )
        for key, value in expected.items():
            _assert_equivalent(actual[key], value, f"{context}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise EvaluationShardMergeError(
                f"{context} length differs from raw-row recomputation"
            )
        for index, value in enumerate(expected):
            _assert_equivalent(actual[index], value, f"{context}[{index}]")
        return
    if isinstance(expected, (float,)):
        value = _require_float(actual, context)
        if not math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-12):
            raise EvaluationShardMergeError(
                f"{context} differs from raw-row recomputation"
            )
        return
    if actual != expected:
        raise EvaluationShardMergeError(
            f"{context} differs from raw-row recomputation"
        )


def _summary_protocol_checks(
    summary: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    context: str,
) -> None:
    expected = {
        "schema_version": protocol.get("evaluation_schema_version"),
        "benchmark": protocol.get("benchmark"),
        "condition": protocol.get("condition"),
        "metaworld_version": protocol.get("metaworld_version"),
        "benchmark_seed": protocol.get("benchmark_seed"),
        "seed": protocol.get("episode_seed_base"),
        "task_bank_sha256": protocol.get("task_bank_sha256"),
        "task_vocabulary": protocol.get("task_vocabulary"),
        "task_ids": protocol.get("task_ids"),
        "max_episode_steps": protocol.get("max_episode_steps"),
        "episodes_per_task": protocol.get("episodes_per_task"),
        "noise_level": protocol.get("noise_level"),
        "object_position_noise": protocol.get("object_position_noise"),
        "detector_threshold": protocol.get("detector_threshold"),
        "release_threshold": protocol.get("release_threshold"),
        "release_patience": protocol.get("release_patience"),
        "min_recovery_steps": protocol.get("min_recovery_steps"),
        "intervention_cooldown": protocol.get("intervention_cooldown"),
        "recovery_budget": protocol.get("recovery_budget"),
        "methods": protocol.get("methods"),
    }
    differences = [
        key for key, value in expected.items() if summary.get(key) != value
    ]
    if differences:
        raise EvaluationShardMergeError(
            f"{context} summary/sidecar mismatch in {sorted(differences)}"
        )
    noise = _require_float(protocol.get("noise_level"), f"{context}.noise_level", minimum=0)
    action_scale = _require_float(
        protocol.get("action_std_scale"), f"{context}.action_std_scale", minimum=0
    )
    observation_scale = _require_float(
        protocol.get("observation_std_scale"),
        f"{context}.observation_std_scale",
        minimum=0,
    )
    for key, expected_value in (
        ("action_noise_std", noise * action_scale),
        ("observation_noise_std", noise * observation_scale),
    ):
        actual = _require_float(summary.get(key), f"{context}.{key}", minimum=0)
        if not math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=1e-12):
            raise EvaluationShardMergeError(f"{context}.{key} is inconsistent")
    if summary.get("robustness_extension") is not (noise != 0.0):
        raise EvaluationShardMergeError(
            f"{context}.robustness_extension is inconsistent"
        )
    if summary.get("observation_schema") != "raw39_plus_official_task_one_hot":
        raise EvaluationShardMergeError(f"{context} observation schema is unsupported")


def _validate_checkpoints(
    summary: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    summary_path: Path,
    context: str,
) -> dict[str, dict[str, str]]:
    checkpoint_hashes = protocol.get("checkpoint_sha256")
    records = summary.get("checkpoints")
    if not isinstance(checkpoint_hashes, Mapping) or not isinstance(records, Mapping):
        raise EvaluationShardMergeError(f"{context} lacks checkpoint provenance")
    expected_names = {"mlp_bc", "act", "detector", "recovery"}
    if set(checkpoint_hashes) != expected_names or set(records) != expected_names:
        raise EvaluationShardMergeError(
            f"{context} checkpoint set must be exactly {sorted(expected_names)}"
        )
    normalized: dict[str, dict[str, str]] = {}
    for name in sorted(expected_names):
        digest = _require_sha256(
            checkpoint_hashes[name], f"{context}.checkpoint_sha256.{name}"
        )
        record = records[name]
        if not isinstance(record, Mapping):
            raise EvaluationShardMergeError(f"{context}.checkpoints.{name} is invalid")
        if _require_sha256(
            record.get("sha256"), f"{context}.checkpoints.{name}.sha256"
        ) != digest:
            raise EvaluationShardMergeError(
                f"{context} checkpoint digest differs between summary and sidecar"
            )
        checkpoint_path = _resolve_recorded_path(
            record.get("path"), summary_path, f"{context} {name} checkpoint"
        )
        if file_sha256(checkpoint_path) != digest:
            raise EvaluationShardMergeError(f"{context} {name} checkpoint was modified")
        normalized[name] = {"path": str(checkpoint_path), "sha256": digest}
    return normalized


def _validate_rows(
    rows: Sequence[dict[str, str]],
    protocol: Mapping[str, Any],
    *,
    run_fingerprint: str,
    context: str,
) -> frozenset[tuple[str, int, str]]:
    benchmark = str(protocol["benchmark"])
    condition = str(protocol["condition"])
    task_vocabulary = tuple(str(value) for value in protocol["task_vocabulary"])
    task_ids = tuple(int(value) for value in protocol["task_ids"])
    methods = tuple(str(value) for value in protocol["methods"])
    episodes_per_task = int(protocol["episodes_per_task"])
    seed = int(protocol["episode_seed_base"])
    labels = {key: evaluator.METHOD_LABELS[key] for key in methods}
    reverse_labels = {value: key for key, value in labels.items()}
    expected_count = len(task_ids) * episodes_per_task * len(methods)
    if len(rows) != expected_count:
        raise EvaluationShardMergeError(
            f"{context} has {len(rows)} rows; expected {expected_count}"
        )

    cells: dict[tuple[int, int], dict[str, dict[str, str]]] = {}
    identities: dict[tuple[int, int], tuple[str, int, str, str]] = {}
    specifications: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=2):
        prefix = f"{context} CSV row {row_number}"
        if row.get("run_fingerprint") != run_fingerprint:
            raise EvaluationShardMergeError(f"{prefix} has a foreign run fingerprint")
        if row.get("benchmark") != benchmark or row.get("condition") != condition:
            raise EvaluationShardMergeError(f"{prefix} has foreign protocol fields")
        task_id = _require_int(
            row.get("task_id"), prefix + ".task_id", minimum=0,
            maximum=len(task_vocabulary) - 1,
        )
        variant = _require_int(
            row.get("task_variant"), prefix + ".task_variant", minimum=0,
            maximum=episodes_per_task - 1,
        )
        if task_id not in task_ids:
            raise EvaluationShardMergeError(f"{prefix} is outside declared task_ids")
        task_name = task_vocabulary[task_id]
        if row.get("task_name") != task_name:
            raise EvaluationShardMergeError(f"{prefix} has the wrong task name")
        method_label = str(row.get("method", ""))
        method_key = reverse_labels.get(method_label)
        if method_key is None:
            raise EvaluationShardMergeError(f"{prefix} has an unknown method label")
        payload = _require_sha256(
            row.get("task_payload_sha256"), prefix + ".task_payload_sha256"
        )
        episode_seed = _require_int(row.get("episode_seed"), prefix + ".episode_seed")
        expected_seed = seed + task_id * 100_000 + variant
        if episode_seed != expected_seed:
            raise EvaluationShardMergeError(f"{prefix} violates the episode seed bank")
        paired_id = f"{task_id:02d}-{variant:04d}-{payload[:12]}"
        if row.get("paired_episode_id") != paired_id:
            raise EvaluationShardMergeError(f"{prefix} has an invalid paired episode ID")

        slot = (task_id, variant)
        identity = (task_name, episode_seed, payload, paired_id)
        previous_identity = identities.setdefault(slot, identity)
        if previous_identity != identity:
            raise EvaluationShardMergeError(
                f"{prefix} violates within-slot common-random-number pairing"
            )
        slot_rows = cells.setdefault(slot, {})
        if method_key in slot_rows:
            raise EvaluationShardMergeError(
                f"{prefix} duplicates task/variant/method slot"
            )
        slot_rows[method_key] = row
        specifications[(method_label, task_id, paired_id)] = {
            "run_fingerprint": run_fingerprint,
            "benchmark": benchmark,
            "condition": condition,
            "task_name": task_name,
            "task_id": task_id,
            "task_variant": variant,
            "method": method_label,
            "method_key": method_key,
            "paired_episode_id": paired_id,
            "episode_seed": episode_seed,
            "task_payload_sha256": payload,
        }

    expected_slots = {
        (task_id, variant)
        for task_id in task_ids
        for variant in range(episodes_per_task)
    }
    if set(cells) != expected_slots or any(
        set(method_rows) != set(methods) for method_rows in cells.values()
    ):
        raise EvaluationShardMergeError(f"{context} paired episode grid is incomplete")
    try:
        completed = evaluator._validate_protocol_rows(
            rows,
            specifications,
            run_fingerprint=run_fingerprint,
            max_steps=int(protocol["max_episode_steps"]),
        )
    except (TypeError, ValueError) as exc:
        raise EvaluationShardMergeError(f"{context} contains invalid outcome rows") from exc
    return frozenset(completed)


def _recompute_statistics(
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    methods = tuple(str(value) for value in protocol["methods"])
    aggregates: dict[str, Any] = {}
    selected: dict[str, list[Mapping[str, Any]]] = {}
    for method in methods:
        label = evaluator.METHOD_LABELS[method]
        selected[method] = [row for row in rows if row["method"] == label]
        aggregates[method] = aggregate_multitask_metrics(selected[method])
    paired: dict[str, Any] = {}
    if "act" in selected:
        for method in methods:
            if method == "act":
                continue
            paired[method] = paired_task_stratified_bootstrap_delta(
                selected["act"],
                selected[method],
                metric="success",
                n_bootstrap=int(protocol["bootstrap_samples"]),
                seed=int(protocol["episode_seed_base"]) + 2026,
            )
    return aggregates, paired


def _validate_shard(summary_path: Path, csv_path: Path) -> ShardEvidence:
    summary_path, summary = _load_json(summary_path, "shard summary")
    csv_path, rows = _load_csv(csv_path, "shard episode CSV")
    context = summary_path.name
    if summary.get("schema_version") != evaluator.SCHEMA_VERSION:
        raise EvaluationShardMergeError(f"{context} evaluation schema is stale")
    if file_sha256(csv_path) != _require_sha256(
        summary.get("episode_csv_sha256"), f"{context}.episode_csv_sha256"
    ):
        raise EvaluationShardMergeError(
            f"{context} shard CSV SHA-256 does not match its summary"
        )
    recorded_csv = _resolve_recorded_path(
        summary.get("episode_csv"), summary_path, f"{context} recorded episode CSV"
    )
    if recorded_csv != csv_path:
        raise EvaluationShardMergeError(f"{context} points to another episode CSV")

    sidecar_path = csv_path.with_suffix(csv_path.suffix + ".run.json")
    sidecar_path = _regular_file(sidecar_path, f"{context} run sidecar")
    recorded_sidecar = _resolve_recorded_path(
        summary.get("run_sidecar"), summary_path, f"{context} recorded run sidecar"
    )
    if recorded_sidecar != sidecar_path:
        raise EvaluationShardMergeError(f"{context} points to another run sidecar")
    if file_sha256(sidecar_path) != _require_sha256(
        summary.get("run_sidecar_sha256"), f"{context}.run_sidecar_sha256"
    ):
        raise EvaluationShardMergeError(f"{context} run sidecar digest mismatch")
    _, sidecar = _load_json(sidecar_path, f"{context} run sidecar")
    if sidecar.get("schema_version") != evaluator.RUN_SIDECAR_SCHEMA_VERSION:
        raise EvaluationShardMergeError(f"{context} run sidecar schema is stale")
    protocol = sidecar.get("protocol")
    if not isinstance(protocol, Mapping):
        raise EvaluationShardMergeError(f"{context} lacks a structured protocol")
    protocol = dict(protocol)
    run_fingerprint = _require_sha256(
        sidecar.get("run_fingerprint"), f"{context}.run_fingerprint"
    )
    if run_fingerprint != _canonical_sha256(protocol):
        raise EvaluationShardMergeError(f"{context} run fingerprint is invalid")
    if summary.get("run_fingerprint") != run_fingerprint:
        raise EvaluationShardMergeError(
            f"{context} summary and sidecar fingerprints differ"
        )

    benchmark = str(protocol.get("benchmark"))
    if benchmark not in SUITE_TASK_COUNT:
        raise EvaluationShardMergeError(f"{context} has unsupported benchmark")
    task_count = SUITE_TASK_COUNT[benchmark]
    vocabulary = protocol.get("task_vocabulary")
    if (
        not isinstance(vocabulary, list)
        or len(vocabulary) != task_count
        or len(set(vocabulary)) != task_count
        or any(not isinstance(value, str) or not value for value in vocabulary)
    ):
        raise EvaluationShardMergeError(f"{context} task vocabulary is incomplete")
    if protocol.get("task_vocabulary_sha256") != _canonical_sha256(vocabulary):
        raise EvaluationShardMergeError(f"{context} task vocabulary digest is invalid")
    _require_sha256(protocol.get("task_bank_sha256"), f"{context}.task_bank_sha256")
    task_ids_value = protocol.get("task_ids")
    if not isinstance(task_ids_value, list) or not task_ids_value:
        raise EvaluationShardMergeError(f"{context} task_ids are missing")
    task_ids = tuple(
        _require_int(value, f"{context}.task_ids", maximum=task_count - 1)
        for value in task_ids_value
    )
    if task_ids != tuple(sorted(set(task_ids))):
        raise EvaluationShardMergeError(
            f"{context} task_ids must be unique and sorted"
        )
    if tuple(protocol.get("methods") or ()) != tuple(evaluator.DEFAULT_OFFICIAL_METHODS):
        raise EvaluationShardMergeError(
            f"{context} methods must be the canonical publication method set"
        )
    if _require_int(protocol.get("episodes_per_task"), f"{context}.episodes_per_task") != evaluator.OFFICIAL_TASK_VARIANTS_PER_CLASS:
        raise EvaluationShardMergeError(f"{context} must contain 50 episodes/task")
    if _require_int(protocol.get("max_episode_steps"), f"{context}.max_episode_steps") != evaluator.OFFICIAL_MAX_EPISODE_STEPS:
        raise EvaluationShardMergeError(f"{context} must use the 500-step horizon")
    if not str(protocol.get("condition", "")).strip():
        raise EvaluationShardMergeError(f"{context} condition is empty")
    _require_int(protocol.get("benchmark_seed"), f"{context}.benchmark_seed")
    _require_int(protocol.get("episode_seed_base"), f"{context}.episode_seed_base")
    _require_float(
        protocol.get("detector_threshold"),
        f"{context}.detector_threshold",
        minimum=0,
        maximum=1,
    )
    _require_int(protocol.get("bootstrap_samples"), f"{context}.bootstrap_samples", minimum=1)
    if protocol.get("evaluation_schema_version") != evaluator.SCHEMA_VERSION:
        raise EvaluationShardMergeError(f"{context} protocol schema is stale")

    _summary_protocol_checks(summary, protocol, context=context)
    _validate_checkpoints(
        summary,
        protocol,
        summary_path=summary_path,
        context=context,
    )
    completed = _validate_rows(
        rows,
        protocol,
        run_fingerprint=run_fingerprint,
        context=context,
    )
    aggregates, paired = _recompute_statistics(rows, protocol)
    _assert_equivalent(summary.get("aggregates"), aggregates, context + ".aggregates")
    _assert_equivalent(summary.get("paired_vs_act"), paired, context + ".paired_vs_act")
    eligibility = evaluator._official_clean_eligibility(
        condition=str(protocol["condition"]),
        noise_level=float(protocol["noise_level"]),
        max_steps=int(protocol["max_episode_steps"]),
        task_ids=task_ids,
        task_count=task_count,
        methods=tuple(protocol["methods"]),
        episodes_per_task=int(protocol["episodes_per_task"]),
        completed=set(completed),
    )
    _assert_equivalent(
        summary.get("official_clean_eligibility_by_method"),
        eligibility,
        context + ".official_clean_eligibility_by_method",
    )
    official = bool(eligibility) and all(value["eligible"] for value in eligibility.values())
    if summary.get("official_clean_protocol") is not official:
        raise EvaluationShardMergeError(f"{context} official-clean flag is inconsistent")
    readiness = evaluator._publication_readiness(
        benchmark=benchmark,
        official_clean_protocol=official,
    )
    _assert_equivalent(
        summary.get("publication_readiness"), readiness, context + ".publication_readiness"
    )
    if summary.get("publication_eligible") is not readiness["eligible"]:
        raise EvaluationShardMergeError(f"{context} publication flag is inconsistent")
    if summary.get("publication_audit_required") is not readiness["audit_required"]:
        raise EvaluationShardMergeError(f"{context} audit flag is inconsistent")

    return ShardEvidence(
        summary_path=summary_path,
        csv_path=csv_path,
        sidecar_path=sidecar_path,
        summary=summary,
        protocol=protocol,
        task_ids=task_ids,
        rows=tuple(rows),
        completed=completed,
    )


def _json_bytes(value: Any) -> bytes:
    text = json.dumps(
        json_compatible(value),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    return text.encode("utf-8")


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=evaluator.CSV_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _protocol_differences(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> list[str]:
    keys = (set(reference) | set(candidate)) - {"task_ids"}
    return sorted(key for key in keys if reference.get(key) != candidate.get(key))


def merge_evaluation_shards(
    shards: Sequence[tuple[Path, Path]],
    *,
    output_csv: Path,
    output_summary: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate and merge ``(summary, CSV)`` shard pairs."""

    if len(shards) < 2:
        raise EvaluationShardMergeError("at least two evaluation shards are required")
    evidence = [_validate_shard(Path(summary), Path(csv_path)) for summary, csv_path in shards]
    reference = evidence[0]
    benchmark = str(reference.protocol["benchmark"])
    task_count = SUITE_TASK_COUNT[benchmark]
    task_owner: dict[int, Path] = {}
    for item in evidence:
        differences = _protocol_differences(reference.protocol, item.protocol)
        if differences:
            raise EvaluationShardMergeError(
                "shard protocol mismatch outside task_ids in "
                f"{item.summary_path}: {differences}"
            )
        for task_id in item.task_ids:
            previous = task_owner.get(task_id)
            if previous is not None:
                raise EvaluationShardMergeError(
                    f"task_id {task_id} overlaps between {previous} and {item.summary_path}"
                )
            task_owner[task_id] = item.summary_path
    expected_tasks = set(range(task_count))
    actual_tasks = set(task_owner)
    if actual_tasks != expected_tasks:
        missing = sorted(expected_tasks - actual_tasks)
        unexpected = sorted(actual_tasks - expected_tasks)
        raise EvaluationShardMergeError(
            f"shards do not cover the full {benchmark} task suite; "
            f"missing={missing}, unexpected={unexpected}"
        )

    output_csv = output_csv.expanduser().resolve()
    output_summary = output_summary.expanduser().resolve()
    output_sidecar = evaluator._run_sidecar_path(output_csv)
    output_paths = (output_csv, output_summary, output_sidecar)
    if len(set(output_paths)) != len(output_paths):
        raise EvaluationShardMergeError("output CSV, summary, and sidecar must differ")
    input_paths = {
        path
        for item in evidence
        for path in (item.summary_path, item.csv_path, item.sidecar_path)
    }
    if any(path in input_paths for path in output_paths):
        raise EvaluationShardMergeError("output paths must not overwrite input shards")
    for path in output_paths:
        if path.is_symlink():
            raise EvaluationShardMergeError(f"refusing symlink output {path}")
        if path.exists() and not overwrite:
            raise FileExistsError(f"output already exists: {path}")

    protocol = dict(reference.protocol)
    protocol["task_ids"] = list(range(task_count))
    run_fingerprint = _canonical_sha256(protocol)
    sidecar = {
        "schema_version": evaluator.RUN_SIDECAR_SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "protocol": protocol,
    }
    method_order = {
        evaluator.METHOD_LABELS[method]: index
        for index, method in enumerate(protocol["methods"])
    }
    rows: list[dict[str, Any]] = []
    for item in evidence:
        for source in item.rows:
            row = dict(source)
            row["run_fingerprint"] = run_fingerprint
            rows.append(row)
    rows.sort(
        key=lambda row: (
            int(row["task_id"]),
            int(row["task_variant"]),
            method_order[str(row["method"])],
        )
    )
    csv_payload = _csv_bytes(rows)
    sidecar_payload = _json_bytes(sidecar)
    csv_sha256 = hashlib.sha256(csv_payload).hexdigest()
    sidecar_sha256 = hashlib.sha256(sidecar_payload).hexdigest()

    # Revalidate the rewritten grid against the merged protocol before writing.
    completed = _validate_rows(
        [{key: str(value) for key, value in row.items()} for row in rows],
        protocol,
        run_fingerprint=run_fingerprint,
        context="merged full-suite evaluation",
    )
    aggregates, paired = _recompute_statistics(rows, protocol)
    methods = tuple(str(value) for value in protocol["methods"])
    eligibility = evaluator._official_clean_eligibility(
        condition=str(protocol["condition"]),
        noise_level=float(protocol["noise_level"]),
        max_steps=int(protocol["max_episode_steps"]),
        task_ids=tuple(range(task_count)),
        task_count=task_count,
        methods=methods,
        episodes_per_task=int(protocol["episodes_per_task"]),
        completed=set(completed),
    )
    official_clean_protocol = bool(eligibility) and all(
        value["eligible"] for value in eligibility.values()
    )
    readiness = evaluator._publication_readiness(
        benchmark=benchmark,
        official_clean_protocol=official_clean_protocol,
    )
    checkpoints = _validate_checkpoints(
        reference.summary,
        reference.protocol,
        summary_path=reference.summary_path,
        context=reference.summary_path.name,
    )
    merge_sources = [
        {
            "summary": str(item.summary_path),
            "summary_sha256": file_sha256(item.summary_path),
            "episode_csv": str(item.csv_path),
            "episode_csv_sha256": file_sha256(item.csv_path),
            "run_sidecar": str(item.sidecar_path),
            "run_sidecar_sha256": file_sha256(item.sidecar_path),
            "source_run_fingerprint": str(item.summary["run_fingerprint"]),
            "task_ids": list(item.task_ids),
        }
        for item in sorted(evidence, key=lambda value: value.task_ids)
    ]
    summary: dict[str, Any] = {
        "schema_version": evaluator.SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "run_sidecar": str(output_sidecar),
        "run_sidecar_sha256": sidecar_sha256,
        "benchmark": benchmark,
        "condition": protocol["condition"],
        "official_clean_protocol": official_clean_protocol,
        "official_clean_protocol_scope": "rollout_protocol_only",
        "official_clean_eligibility_by_method": eligibility,
        "publication_eligible": readiness["eligible"],
        "publication_audit_required": readiness["audit_required"],
        "publication_readiness": readiness,
        "robustness_extension": float(protocol["noise_level"]) != 0.0,
        "metaworld_version": protocol["metaworld_version"],
        "benchmark_seed": protocol["benchmark_seed"],
        "seed": protocol["episode_seed_base"],
        "task_bank_sha256": protocol["task_bank_sha256"],
        "observation_schema": "raw39_plus_official_task_one_hot",
        "task_vocabulary": list(protocol["task_vocabulary"]),
        "task_ids": list(range(task_count)),
        "max_episode_steps": protocol["max_episode_steps"],
        "episodes_per_task": protocol["episodes_per_task"],
        "noise_level": protocol["noise_level"],
        "action_noise_std": (
            float(protocol["noise_level"]) * float(protocol["action_std_scale"])
        ),
        "observation_noise_std": (
            float(protocol["noise_level"])
            * float(protocol["observation_std_scale"])
        ),
        "object_position_noise": False,
        "detector_threshold": protocol["detector_threshold"],
        "release_threshold": protocol["release_threshold"],
        "release_patience": protocol["release_patience"],
        "min_recovery_steps": protocol["min_recovery_steps"],
        "intervention_cooldown": protocol["intervention_cooldown"],
        "recovery_budget": protocol["recovery_budget"],
        "methods": list(methods),
        "episode_csv": str(output_csv),
        "episode_csv_sha256": csv_sha256,
        "checkpoints": checkpoints,
        "aggregates": aggregates,
        "paired_vs_act": paired,
        "merge_provenance": {
            "schema_version": MERGE_SCHEMA_VERSION,
            "source_shard_count": len(evidence),
            "source_shards": merge_sources,
        },
    }
    summary_payload = _json_bytes(summary)

    # A summary is the commit marker: the complete sidecar and CSV are installed
    # first, and the digest-binding summary is atomically installed last.
    _atomic_write_bytes(output_sidecar, sidecar_payload)
    _atomic_write_bytes(output_csv, csv_payload)
    _atomic_write_bytes(output_summary, summary_payload)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard",
        action="append",
        nargs=2,
        required=True,
        metavar=("SUMMARY_JSON", "EPISODE_CSV"),
        help="repeat once per disjoint evaluator shard",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = merge_evaluation_shards(
        [(Path(summary), Path(csv_path)) for summary, csv_path in args.shard],
        output_csv=Path(args.output_csv),
        output_summary=Path(args.output_summary),
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "benchmark": summary["benchmark"],
                "condition": summary["condition"],
                "task_count": len(summary["task_ids"]),
                "episode_csv_sha256": summary["episode_csv_sha256"],
                "official_clean_protocol": summary["official_clean_protocol"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
