#!/usr/bin/env python3
"""Fail-closed publication assets for the MT10/MT50 REIM study.

This module is deliberately separate from the evaluator.  Evaluation summaries
only certify that a rollout obeyed the official-clean *rollout* protocol; they
explicitly do not certify isolation of the demonstration, failure-training,
failure-validation, recovery-training, and final-evaluation task banks.  The
paper gate opens only after this command validates both official-clean suites,
all five task-universal disturbance conditions, and both external five-bank
audit reports.

No publication-facing file is replaced until every input has passed.  The
single TeX gate file, ``multitask_results.tex``, is installed last, so a fresh
checkout or an incomplete experiment renders no numerical MT claim.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.multitask_metrics import (
    aggregate_multitask_metrics,
    paired_task_stratified_bootstrap_delta,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_SCHEMA = "reim-multitask-evaluation-v2"
RUN_SIDECAR_SCHEMA = "reim-multitask-evaluation-run-v1"
AUDIT_SCHEMA = "reim-multitask-bank-separation-audit-v1"
OFFICIAL_EPISODES_PER_TASK = 50
OFFICIAL_MAX_STEPS = 500
SUITE_TASK_COUNT = {"MT10": 10, "MT50": 50}
ROBUSTNESS_LEVELS = (0.0, 0.1, 0.2, 0.3, 0.4)
ACTION_STD_SCALE = 0.40
OBSERVATION_STD_SCALE = 0.025
CONFIDENCE = 0.95
MIN_BOOTSTRAP_SAMPLES = 2_000
CANONICAL_METHODS = (
    "mlp_bc",
    "act",
    "heuristic_recovery",
    "reim",
)
METHOD_LABELS = {
    "mlp_bc": "MT-MLP BC",
    "act": "MT-ACT",
    "heuristic_recovery": "MT-ACT + Heuristic-Gated Learned Recovery",
    "reim": "MT-REIM",
}
SHORT_LABELS = {
    "mlp_bc": "MT-MLP BC",
    "act": "MT-ACT",
    "heuristic_recovery": "Heuristic gate",
    "reim": "MT-REIM",
}
STAGES = (
    "demonstrations",
    "failure_training",
    "failure_validation",
    "recovery_training",
    "final_evaluation",
)
AUDIT_CHECKS = (
    "all_artifacts_complete_and_digest_verified",
    "benchmark_bank_seeds_pairwise_distinct",
    "task_bank_content_fingerprints_pairwise_distinct",
    "reserved_native_task_payloads_pairwise_disjoint",
    "materialized_native_task_payloads_pairwise_disjoint",
    "materialized_sampling_unit_identities_pairwise_disjoint",
    "artifact_protocol_descriptors_pairwise_distinct",
)
CSV_REQUIRED_FIELDS = frozenset(
    {
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
    }
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class PublicationGateError(ValueError):
    """Raised when an artifact cannot support a publication claim."""


@dataclass(frozen=True)
class ConditionEvidence:
    benchmark: str
    condition: str
    noise_level: float
    summary_path: Path
    episode_path: Path
    summary: Mapping[str, Any]
    rows: tuple[Mapping[str, str], ...]
    statistics: Mapping[str, Mapping[str, float | None]]
    identity_grid: Mapping[tuple[int, int], tuple[str, str, int, str]]


@dataclass(frozen=True)
class SuiteEvidence:
    benchmark: str
    clean: ConditionEvidence
    robustness: tuple[ConditionEvidence, ...]
    audit_path: Path
    audit: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _regular_file(path: Path, context: str) -> Path:
    expanded = path.expanduser()
    resolved = expanded.resolve()
    if expanded.is_symlink() or not resolved.is_file():
        raise PublicationGateError(
            f"{context} must be an existing non-symlink file: {path}"
        )
    return resolved


def _load_json(path: Path, context: str) -> tuple[Path, dict[str, Any]]:
    resolved = _regular_file(path, context)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationGateError(f"cannot parse {context}: {resolved}") from exc
    if not isinstance(value, dict):
        raise PublicationGateError(f"{context} must contain a JSON object")
    return resolved, value


def _load_csv(path: Path, context: str) -> tuple[Path, list[dict[str, str]]]:
    resolved = _regular_file(path, context)
    try:
        with resolved.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not CSV_REQUIRED_FIELDS.issubset(
                reader.fieldnames
            ):
                missing = CSV_REQUIRED_FIELDS - set(reader.fieldnames or ())
                raise PublicationGateError(
                    f"{context} lacks fields {sorted(missing)}"
                )
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise PublicationGateError(f"cannot parse {context}: {resolved}") from exc
    if not rows:
        raise PublicationGateError(f"{context} is empty")
    return resolved, rows


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PublicationGateError(f"{context} must be an object")
    return value


def _sequence(value: Any, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PublicationGateError(f"{context} must be an array")
    return value


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise PublicationGateError(f"{context} must be an integer")
    try:
        result = int(value)
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise PublicationGateError(f"{context} must be an integer") from exc
    if result != numeric or result < minimum:
        raise PublicationGateError(f"{context} must be an integer >= {minimum}")
    return result


def _number(value: Any, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PublicationGateError(f"{context} must be numeric") from exc
    if not math.isfinite(result):
        raise PublicationGateError(f"{context} must be finite")
    return result


def _binary(value: Any, context: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise PublicationGateError(f"{context} must be binary, got {value!r}")


def _digest(value: Any, context: str) -> str:
    text = str(value)
    if SHA256_RE.fullmatch(text) is None:
        raise PublicationGateError(f"{context} is not a lowercase SHA-256 digest")
    return text


def _same_float(actual: Any, expected: float, context: str) -> None:
    value = _number(actual, context)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
        raise PublicationGateError(
            f"{context}={value!r}; expected {expected!r}"
        )


def _assert_equivalent(actual: Any, expected: Any, context: str) -> None:
    """Recursively compare stored statistics against raw-record recomputation."""

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise PublicationGateError(
                f"{context} keys differ from raw-record recomputation"
            )
        for key, value in expected.items():
            _assert_equivalent(actual[key], value, f"{context}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise PublicationGateError(
                f"{context} length differs from raw-record recomputation"
            )
        for index, value in enumerate(expected):
            _assert_equivalent(actual[index], value, f"{context}[{index}]")
        return
    if isinstance(expected, (float, np.floating)):
        if not math.isclose(
            _number(actual, context),
            float(expected),
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise PublicationGateError(
                f"{context} differs from raw-record recomputation"
            )
        return
    if actual != expected:
        raise PublicationGateError(
            f"{context}={actual!r}; recomputed value is {expected!r}"
        )


def _resolve_recorded_path(value: Any, summary_path: Path, context: str) -> Path:
    recorded = Path(str(value)).expanduser()
    if not recorded.is_absolute():
        recorded = (summary_path.parent / recorded).resolve()
    else:
        recorded = recorded.resolve()
    return _regular_file(recorded, context)


def _validate_sidecar(summary: Mapping[str, Any], summary_path: Path) -> None:
    sidecar_path = _resolve_recorded_path(
        summary.get("run_sidecar"), summary_path, "evaluation run sidecar"
    )
    expected_digest = _digest(
        summary.get("run_sidecar_sha256"), "summary.run_sidecar_sha256"
    )
    if _sha256(sidecar_path) != expected_digest:
        raise PublicationGateError("evaluation run sidecar digest mismatch")
    _, sidecar = _load_json(sidecar_path, "evaluation run sidecar")
    if sidecar.get("schema_version") != RUN_SIDECAR_SCHEMA:
        raise PublicationGateError("evaluation run sidecar schema is stale")
    if sidecar.get("run_fingerprint") != summary.get("run_fingerprint"):
        raise PublicationGateError("summary and sidecar run fingerprints differ")
    protocol = _mapping(sidecar.get("protocol"), "sidecar.protocol")
    expected = {
        "benchmark": summary.get("benchmark"),
        "condition": summary.get("condition"),
        "benchmark_seed": summary.get("benchmark_seed"),
        "task_bank_sha256": summary.get("task_bank_sha256"),
        "episodes_per_task": summary.get("episodes_per_task"),
        "max_episode_steps": summary.get("max_episode_steps"),
        "noise_level": summary.get("noise_level"),
        "methods": summary.get("methods"),
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise PublicationGateError(
                f"sidecar.protocol.{key} does not match the summary"
            )
    fingerprint = _digest(summary.get("run_fingerprint"), "run_fingerprint")
    if _canonical_sha256(protocol) != fingerprint:
        raise PublicationGateError("evaluation run fingerprint is invalid")


def _validate_checkpoints(
    summary: Mapping[str, Any],
    summary_path: Path,
    *,
    digest_cache: dict[Path, str],
) -> dict[str, str]:
    checkpoints = _mapping(summary.get("checkpoints"), "summary.checkpoints")
    expected = {"mlp_bc", "act", "detector", "recovery"}
    if set(checkpoints) != expected:
        raise PublicationGateError(
            f"summary.checkpoints must contain exactly {sorted(expected)}"
        )
    result: dict[str, str] = {}
    for name in sorted(expected):
        record = _mapping(checkpoints[name], f"checkpoints.{name}")
        path = _resolve_recorded_path(
            record.get("path"), summary_path, f"{name} checkpoint"
        )
        claimed = _digest(record.get("sha256"), f"checkpoints.{name}.sha256")
        actual = digest_cache.setdefault(path, _sha256(path))
        if actual != claimed:
            raise PublicationGateError(f"{name} checkpoint digest mismatch")
        result[name] = claimed
    return result


def _validate_episode_grid(
    rows: Sequence[Mapping[str, str]],
    *,
    benchmark: str,
    condition: str,
    run_fingerprint: str,
    vocabulary: Sequence[str],
) -> dict[tuple[int, int], tuple[str, str, int, str]]:
    task_count = SUITE_TASK_COUNT[benchmark]
    expected_count = (
        task_count * OFFICIAL_EPISODES_PER_TASK * len(CANONICAL_METHODS)
    )
    if len(rows) != expected_count:
        raise PublicationGateError(
            f"{benchmark} {condition} has {len(rows)} rows; expected {expected_count}"
        )
    labels = set(METHOD_LABELS.values())
    cells: dict[tuple[int, int], set[str]] = {}
    identities: dict[tuple[int, int], tuple[str, str, int, str]] = {}
    seen: set[tuple[int, int, str]] = set()
    for row_number, row in enumerate(rows, start=2):
        prefix = f"{benchmark} {condition} CSV row {row_number}"
        if row.get("run_fingerprint") != run_fingerprint:
            raise PublicationGateError(f"{prefix} has a foreign run fingerprint")
        if row.get("benchmark") != benchmark or row.get("condition") != condition:
            raise PublicationGateError(f"{prefix} has foreign protocol fields")
        task_id = _integer(row.get("task_id"), f"{prefix}.task_id")
        variant = _integer(row.get("task_variant"), f"{prefix}.task_variant")
        if task_id >= task_count or variant >= OFFICIAL_EPISODES_PER_TASK:
            raise PublicationGateError(f"{prefix} has an out-of-range task slot")
        task_name = str(row.get("task_name", ""))
        if task_name != vocabulary[task_id]:
            raise PublicationGateError(f"{prefix} task identity is inconsistent")
        method = str(row.get("method", ""))
        if method not in labels:
            raise PublicationGateError(f"{prefix} has unknown method {method!r}")
        key = (task_id, variant, method)
        if key in seen:
            raise PublicationGateError(f"{prefix} duplicates {key}")
        seen.add(key)
        paired_id = str(row.get("paired_episode_id", ""))
        if not paired_id:
            raise PublicationGateError(f"{prefix} has empty paired_episode_id")
        episode_seed = _integer(row.get("episode_seed"), f"{prefix}.episode_seed")
        payload = _digest(
            row.get("task_payload_sha256"), f"{prefix}.task_payload_sha256"
        )
        identity = (task_name, paired_id, episode_seed, payload)
        slot = (task_id, variant)
        previous = identities.setdefault(slot, identity)
        if previous != identity:
            raise PublicationGateError(
                f"{prefix} violates within-episode common-random-number pairing"
            )
        cells.setdefault(slot, set()).add(method)
        success = _binary(row.get("success"), f"{prefix}.success")
        del success
        interventions = _integer(
            row.get("intervention_count"), f"{prefix}.intervention_count"
        )
        recoveries = _integer(
            row.get("recovery_success"), f"{prefix}.recovery_success"
        )
        if recoveries > interventions:
            raise PublicationGateError(f"{prefix} recovery count exceeds interventions")
        if method in {METHOD_LABELS["mlp_bc"], METHOD_LABELS["act"]} and (
            interventions != 0 or recoveries != 0
        ):
            raise PublicationGateError(
                f"{prefix} nominal policy reports a recovery intervention"
            )
        steps = _integer(row.get("steps"), f"{prefix}.steps", minimum=1)
        if steps > OFFICIAL_MAX_STEPS:
            raise PublicationGateError(f"{prefix} exceeds the official horizon")
    expected_slots = {
        (task_id, variant)
        for task_id in range(task_count)
        for variant in range(OFFICIAL_EPISODES_PER_TASK)
    }
    if set(cells) != expected_slots or any(value != labels for value in cells.values()):
        raise PublicationGateError(
            f"{benchmark} {condition} task/variant/method grid is incomplete"
        )
    return identities


def _bootstrap_seed(base: int, *parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).digest()
    return (base + int.from_bytes(digest[:4], "little")) % (2**32)


def _task_macro_success_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    """Within-task episode bootstrap for a fixed official task suite."""

    grouped: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows, start=1):
        key = (str(row["task_id"]), str(row["task_name"]))
        grouped.setdefault(key, []).append(int(_binary(row["success"], f"row {index}")))
    if not grouped:
        raise PublicationGateError("cannot bootstrap an empty condition")
    rates = [float(np.mean(values)) for values in grouped.values()]
    point = float(np.mean(rates))
    rng = np.random.default_rng(seed)
    estimates = np.zeros(n_bootstrap, dtype=np.float64)
    for values in grouped.values():
        successes = int(sum(values))
        count = len(values)
        estimates += rng.binomial(count, successes / count, size=n_bootstrap) / count
    estimates /= len(grouped)
    alpha = (1.0 - CONFIDENCE) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return point, float(low), float(high)


def _validate_condition(
    *,
    benchmark: str,
    condition: str,
    noise_level: float,
    summary_path: Path,
    episode_path: Path,
    n_bootstrap: int,
    bootstrap_seed: int,
    digest_cache: dict[Path, str],
) -> tuple[ConditionEvidence, dict[str, str]]:
    summary_path, summary = _load_json(summary_path, f"{benchmark} {condition} summary")
    episode_path, rows = _load_csv(episode_path, f"{benchmark} {condition} episodes")
    if summary.get("schema_version") != EVALUATION_SCHEMA:
        raise PublicationGateError(f"{benchmark} {condition} evaluation schema is stale")
    if summary.get("benchmark") != benchmark or summary.get("condition") != condition:
        raise PublicationGateError(f"{benchmark} {condition} summary identity mismatch")
    if summary.get("observation_schema") != "raw39_plus_official_task_one_hot":
        raise PublicationGateError(f"{benchmark} observation schema is not publishable")
    _same_float(summary.get("noise_level"), noise_level, "summary.noise_level")
    _same_float(
        summary.get("action_noise_std"),
        noise_level * ACTION_STD_SCALE,
        "summary.action_noise_std",
    )
    _same_float(
        summary.get("observation_noise_std"),
        noise_level * OBSERVATION_STD_SCALE,
        "summary.observation_noise_std",
    )
    if summary.get("object_position_noise") is not False:
        raise PublicationGateError("multi-task evaluation used object teleportation")
    if _integer(summary.get("max_episode_steps"), "max_episode_steps") != OFFICIAL_MAX_STEPS:
        raise PublicationGateError("multi-task evaluation horizon is not 500")
    if (
        _integer(summary.get("episodes_per_task"), "episodes_per_task")
        != OFFICIAL_EPISODES_PER_TASK
    ):
        raise PublicationGateError("multi-task evaluation is not 50 episodes/task")
    task_count = SUITE_TASK_COUNT[benchmark]
    vocabulary = [
        str(value)
        for value in _sequence(
            summary.get("task_vocabulary"), "task_vocabulary"
        )
    ]
    if len(vocabulary) != task_count or len(set(vocabulary)) != task_count:
        raise PublicationGateError(f"{benchmark} task vocabulary is incomplete")
    task_ids = [
        _integer(value, "task_ids")
        for value in _sequence(summary.get("task_ids"), "task_ids")
    ]
    if task_ids != list(range(task_count)):
        raise PublicationGateError(f"{benchmark} does not evaluate the full ordered suite")
    methods = tuple(str(value) for value in _sequence(summary.get("methods"), "methods"))
    if methods != CANONICAL_METHODS:
        raise PublicationGateError(
            f"{benchmark} {condition} methods are {methods}; expected {CANONICAL_METHODS}"
        )
    if summary.get("publication_eligible") is not False:
        raise PublicationGateError(
            "evaluator must not self-certify publication eligibility"
        )
    if summary.get("publication_audit_required") is not True:
        raise PublicationGateError("summary does not require the external five-bank audit")
    readiness = _mapping(summary.get("publication_readiness"), "publication_readiness")
    if readiness.get("external_audit_consumed") is not False:
        raise PublicationGateError("evaluation summary claims to consume an external audit")
    official = condition == "official_clean"
    if summary.get("official_clean_protocol") is not official:
        raise PublicationGateError(
            f"{benchmark} {condition} official-clean rollout flag is inconsistent"
        )
    if readiness.get("rollout_protocol_eligible") is not official:
        raise PublicationGateError("publication readiness rollout flag is inconsistent")
    if summary.get("official_clean_protocol_scope") != "rollout_protocol_only":
        raise PublicationGateError("official-clean scope is not explicitly rollout-only")
    if summary.get("robustness_extension") is not (noise_level != 0.0):
        raise PublicationGateError("robustness-extension flag is inconsistent")
    eligibility = _mapping(
        summary.get("official_clean_eligibility_by_method"),
        "official_clean_eligibility_by_method",
    )
    if set(eligibility) != set(CANONICAL_METHODS):
        raise PublicationGateError("method eligibility map is incomplete")
    expected_rows_per_method = task_count * OFFICIAL_EPISODES_PER_TASK
    for method in CANONICAL_METHODS:
        record = _mapping(eligibility[method], f"eligibility.{method}")
        if bool(record.get("eligible")) is not official:
            raise PublicationGateError(f"eligibility.{method} is inconsistent")
        if (
            _integer(
                record.get("completed_rows"),
                f"eligibility.{method}.completed_rows",
            )
            != expected_rows_per_method
        ):
            raise PublicationGateError(f"eligibility.{method} row count is incomplete")
        if (
            _integer(
                record.get("expected_rows"),
                f"eligibility.{method}.expected_rows",
            )
            != expected_rows_per_method
        ):
            raise PublicationGateError(f"eligibility.{method} expected row count drifted")

    claimed_episode_path = _resolve_recorded_path(
        summary.get("episode_csv"), summary_path, "summary episode CSV"
    )
    if claimed_episode_path != episode_path:
        raise PublicationGateError("summary points to a different episode CSV")
    if _sha256(episode_path) != _digest(
        summary.get("episode_csv_sha256"), "episode_csv_sha256"
    ):
        raise PublicationGateError("episode CSV digest mismatch")
    _validate_sidecar(summary, summary_path)
    checkpoint_digests = _validate_checkpoints(
        summary, summary_path, digest_cache=digest_cache
    )
    run_fingerprint = _digest(summary.get("run_fingerprint"), "run_fingerprint")
    identities = _validate_episode_grid(
        rows,
        benchmark=benchmark,
        condition=condition,
        run_fingerprint=run_fingerprint,
        vocabulary=vocabulary,
    )

    stored_aggregates = _mapping(summary.get("aggregates"), "summary.aggregates")
    if set(stored_aggregates) != set(CANONICAL_METHODS):
        raise PublicationGateError("summary aggregates are incomplete")
    stored_paired = _mapping(summary.get("paired_vs_act"), "summary.paired_vs_act")
    if set(stored_paired) != set(CANONICAL_METHODS) - {"act"}:
        raise PublicationGateError("summary paired-vs-ACT statistics are incomplete")

    selected: dict[str, list[Mapping[str, Any]]] = {}
    statistics: dict[str, dict[str, float | None]] = {}
    for method in CANONICAL_METHODS:
        label = METHOD_LABELS[method]
        method_rows = [row for row in rows if row["method"] == label]
        selected[method] = method_rows
        recomputed = aggregate_multitask_metrics(method_rows)
        _assert_equivalent(
            stored_aggregates[method],
            recomputed,
            f"summary.aggregates.{method}",
        )
        point, low, high = _task_macro_success_interval(
            method_rows,
            n_bootstrap=n_bootstrap,
            seed=_bootstrap_seed(
                bootstrap_seed, benchmark, condition, method, "absolute"
            ),
        )
        macro = _number(
            recomputed["summary"]["success_rate_task_macro"],
            f"{method}.success_rate_task_macro",
        )
        if not math.isclose(point, macro, rel_tol=0.0, abs_tol=1e-12):
            raise PublicationGateError("bootstrap point estimate is inconsistent")
        statistics[method] = {
            "success": point,
            "success_ci_lower": low,
            "success_ci_upper": high,
            "worst_quartile": _number(
                recomputed["summary"]["success_rate_worst_quartile"],
                f"{method}.success_rate_worst_quartile",
            ),
            "intervention": _number(
                recomputed["summary"]["intervention_episode_rate_task_macro"],
                f"{method}.intervention_episode_rate_task_macro",
            ),
            "delta_vs_act": None,
            "delta_ci_lower": None,
            "delta_ci_upper": None,
        }

    for method in CANONICAL_METHODS:
        if method == "act":
            continue
        stored = _mapping(stored_paired[method], f"paired_vs_act.{method}")
        bootstrap_count = _integer(
            stored.get("n_bootstrap"),
            "paired.n_bootstrap",
            minimum=MIN_BOOTSTRAP_SAMPLES,
        )
        paired_seed = _integer(stored.get("seed"), "paired.seed")
        recomputed = paired_task_stratified_bootstrap_delta(
            selected["act"],
            selected[method],
            metric="success",
            confidence=CONFIDENCE,
            n_bootstrap=bootstrap_count,
            seed=paired_seed,
        )
        _assert_equivalent(stored, recomputed, f"summary.paired_vs_act.{method}")
        statistics[method]["delta_vs_act"] = float(recomputed["delta"])
        statistics[method]["delta_ci_lower"] = float(recomputed["ci_lower"])
        statistics[method]["delta_ci_upper"] = float(recomputed["ci_upper"])

    return (
        ConditionEvidence(
            benchmark=benchmark,
            condition=condition,
            noise_level=noise_level,
            summary_path=summary_path,
            episode_path=episode_path,
            summary=summary,
            rows=tuple(rows),
            statistics=statistics,
            identity_grid=identities,
        ),
        checkpoint_digests,
    )


def _validate_audit(
    audit_path: Path,
    *,
    benchmark: str,
    clean_episode_path: Path,
) -> tuple[Path, Mapping[str, Any]]:
    audit_path, audit = _load_json(audit_path, f"{benchmark} five-bank audit")
    if audit.get("schema_version") != AUDIT_SCHEMA:
        raise PublicationGateError(f"{benchmark} five-bank audit schema is stale")
    if audit.get("benchmark") != benchmark or audit.get("passed") is not True:
        raise PublicationGateError(f"{benchmark} five-bank audit did not pass")
    if audit.get("read_only") is not True:
        raise PublicationGateError(f"{benchmark} bank audit is not read-only")
    checks = _mapping(audit.get("checks"), f"{benchmark} audit.checks")
    if set(checks) != set(AUDIT_CHECKS) or not all(
        checks[name] is True for name in AUDIT_CHECKS
    ):
        raise PublicationGateError(f"{benchmark} five-bank checks are incomplete")
    stages = _mapping(audit.get("stages"), f"{benchmark} audit.stages")
    if set(stages) != set(STAGES):
        raise PublicationGateError(f"{benchmark} five-bank stage set is incomplete")
    seeds: set[int] = set()
    fingerprints: set[str] = set()
    for stage in STAGES:
        record = _mapping(stages[stage], f"audit.stages.{stage}")
        seeds.add(_integer(record.get("benchmark_seed"), f"{stage}.benchmark_seed"))
        fingerprints.add(
            _digest(
                record.get("task_bank_content_sha256"),
                f"{stage}.task_bank_content_sha256",
            )
        )
        if (
            _integer(
                record.get("reserved_task_payloads"),
                f"{stage}.reserved_task_payloads",
                minimum=1,
            )
            != SUITE_TASK_COUNT[benchmark] * OFFICIAL_EPISODES_PER_TASK
        ):
            raise PublicationGateError(f"{benchmark} {stage} reserved bank is incomplete")
        if (
            _integer(
                record.get("materialized_records"),
                f"{stage}.materialized_records",
                minimum=1,
            )
            <= 0
        ):
            raise PublicationGateError(f"{benchmark} {stage} has no materialized data")
    if len(seeds) != len(STAGES) or len(fingerprints) != len(STAGES):
        raise PublicationGateError(f"{benchmark} five banks are not distinct")

    pairwise = _sequence(audit.get("pairwise"), f"{benchmark} audit.pairwise")
    expected_pairs = {
        frozenset((left, right))
        for index, left in enumerate(STAGES)
        for right in STAGES[index + 1 :]
    }
    actual_pairs: set[frozenset[str]] = set()
    for index, value in enumerate(pairwise):
        row = _mapping(value, f"audit.pairwise[{index}]")
        pair = frozenset((str(row.get("left")), str(row.get("right"))))
        actual_pairs.add(pair)
        for key in (
            "reserved_task_payload_overlap_count",
            "materialized_task_payload_overlap_count",
            "sampling_unit_identity_overlap_count",
        ):
            if _integer(row.get(key), f"audit.pairwise[{index}].{key}") != 0:
                raise PublicationGateError(f"{benchmark} audit reports bank overlap")
        if row.get("benchmark_seed_equal") is not False:
            raise PublicationGateError(f"{benchmark} audit reports repeated bank seed")
        if row.get("task_bank_content_fingerprint_equal") is not False:
            raise PublicationGateError(f"{benchmark} audit reports repeated task bank")
    if actual_pairs != expected_pairs or len(pairwise) != len(expected_pairs):
        raise PublicationGateError(f"{benchmark} pairwise audit grid is incomplete")

    final_stage = _mapping(stages["final_evaluation"], "final_evaluation stage")
    paths = [
        Path(str(value)).expanduser().resolve()
        for value in _sequence(
            final_stage.get("source_paths"), "final_evaluation.source_paths"
        )
    ]
    digests = [
        _digest(value, "final_evaluation.source_sha256")
        for value in _sequence(
            final_stage.get("source_sha256"),
            "final_evaluation.source_sha256",
        )
    ]
    if len(paths) != len(digests):
        raise PublicationGateError("final-evaluation audit sources are malformed")
    clean_digest = _sha256(clean_episode_path)
    if not any(
        path == clean_episode_path.resolve() and digest == clean_digest
        for path, digest in zip(paths, digests, strict=True)
    ):
        raise PublicationGateError(
            f"{benchmark} five-bank audit does not bind the supplied clean CSV"
        )
    if (
        _integer(
            final_stage.get("materialized_records"),
            "final_evaluation.materialized_records",
        )
        != SUITE_TASK_COUNT[benchmark] * OFFICIAL_EPISODES_PER_TASK
    ):
        raise PublicationGateError(f"{benchmark} final evaluation bank is incomplete")
    return audit_path, audit


def _condition_paths(tables_dir: Path, slug: str, stem: str) -> tuple[Path, Path]:
    return (
        tables_dir / f"{slug}_{stem}_summary.json",
        tables_dir / f"{slug}_{stem}_episodes.csv",
    )


def _noise_tag(level: float) -> str:
    # Must match scripts/run_multitask_pipeline.py:_noise_tag, which names the
    # evaluation artifacts this validator consumes (e.g. disturbed_noise_20).
    return f"{int(round(level * 100)):02d}"


def validate_suite(
    *,
    benchmark: str,
    tables_dir: Path,
    audit_path: Path,
    n_bootstrap: int,
    bootstrap_seed: int,
    digest_cache: dict[Path, str] | None = None,
) -> SuiteEvidence:
    """Validate one complete suite without writing any output."""

    if benchmark not in SUITE_TASK_COUNT:
        raise PublicationGateError(f"unsupported benchmark {benchmark!r}")
    if n_bootstrap < MIN_BOOTSTRAP_SAMPLES:
        raise PublicationGateError(
            f"n_bootstrap must be >= {MIN_BOOTSTRAP_SAMPLES} for publication"
        )
    digest_cache = {} if digest_cache is None else digest_cache
    slug = benchmark.lower()
    clean_summary, clean_episodes = _condition_paths(tables_dir, slug, "clean")
    clean, checkpoint_reference = _validate_condition(
        benchmark=benchmark,
        condition="official_clean",
        noise_level=0.0,
        summary_path=clean_summary,
        episode_path=clean_episodes,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        digest_cache=digest_cache,
    )
    robustness: list[ConditionEvidence] = []
    run_fingerprints = {str(clean.summary["run_fingerprint"])}
    for level in ROBUSTNESS_LEVELS:
        tag = _noise_tag(level)
        summary_path, episode_path = _condition_paths(
            tables_dir, slug, f"disturbed_noise_{tag}"
        )
        item, checkpoints = _validate_condition(
            benchmark=benchmark,
            condition=f"robustness_noise_{tag}",
            noise_level=level,
            summary_path=summary_path,
            episode_path=episode_path,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
            digest_cache=digest_cache,
        )
        if checkpoints != checkpoint_reference:
            raise PublicationGateError(
                f"{benchmark} clean and robustness runs use different checkpoints"
            )
        for key in (
            "benchmark_seed",
            "task_bank_sha256",
            "task_vocabulary",
            "task_ids",
            "detector_threshold",
            "release_threshold",
            "release_patience",
            "min_recovery_steps",
            "intervention_cooldown",
            "recovery_budget",
        ):
            if item.summary.get(key) != clean.summary.get(key):
                raise PublicationGateError(
                    f"{benchmark} condition drift in frozen field {key!r}"
                )
        if item.identity_grid != clean.identity_grid:
            raise PublicationGateError(
                f"{benchmark} robustness condition does not reuse the paired final bank"
            )
        fingerprint = str(item.summary["run_fingerprint"])
        if fingerprint in run_fingerprints:
            raise PublicationGateError(f"{benchmark} condition fingerprints collide")
        run_fingerprints.add(fingerprint)
        robustness.append(item)
    audit_path, audit = _validate_audit(
        audit_path,
        benchmark=benchmark,
        clean_episode_path=clean.episode_path,
    )
    return SuiteEvidence(
        benchmark=benchmark,
        clean=clean,
        robustness=tuple(robustness),
        audit_path=audit_path,
        audit=audit,
    )


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}\\%"


def _interval(low: float, high: float) -> str:
    return f"[{100.0 * low:.1f}, {100.0 * high:.1f}]\\%"


def _pp(value: float) -> str:
    return f"{100.0 * value:+.1f}\\,pp"


def _pp_interval(low: float, high: float) -> str:
    return f"[{100.0 * low:+.1f}, {100.0 * high:+.1f}]\\,pp"


def _macro_prefix(benchmark: str, method: str) -> str:
    suite = "MTTen" if benchmark == "MT10" else "MTFifty"
    method_name = {
        "mlp_bc": "MLP",
        "act": "ACT",
        "heuristic_recovery": "Heuristic",
        "reim": "REIM",
    }[method]
    return suite + method_name


def _render_gate_tex(suites: Sequence[SuiteEvidence], manifest_sha: str) -> str:
    lines = [
        "% Generated by visualization/generate_multitask_paper_assets.py.",
        "% The publication gate validated MT10 + MT50 clean, five robustness",
        "% conditions per suite, and both independent five-bank audits.",
        f"% input-manifest-sha256: {manifest_sha}",
        r"\REIMMultiTaskResultstrue",
    ]
    for suite in suites:
        for method in CANONICAL_METHODS:
            values = suite.clean.statistics[method]
            prefix = _macro_prefix(suite.benchmark, method)
            lines.extend(
                [
                    rf"\renewcommand{{\{prefix}CleanSuccess}}"
                    rf"{{{_percent(float(values['success']))}}}",
                    rf"\renewcommand{{\{prefix}CleanSuccessCI}}"
                    "{"
                    + _interval(
                        float(values["success_ci_lower"]),
                        float(values["success_ci_upper"]),
                    )
                    + "}",
                    rf"\renewcommand{{\{prefix}WorstQuartile}}"
                    rf"{{{_percent(float(values['worst_quartile']))}}}",
                ]
            )
            if method in {"heuristic_recovery", "reim"}:
                lines.append(
                    rf"\renewcommand{{\{prefix}Intervention}}"
                    rf"{{{_percent(float(values['intervention']))}}}"
                )
            if method != "act":
                lines.extend(
                    [
                        rf"\renewcommand{{\{prefix}DeltaVsACT}}"
                        rf"{{{_pp(float(values['delta_vs_act']))}}}",
                        rf"\renewcommand{{\{prefix}DeltaVsACTCI}}"
                        "{"
                        + _pp_interval(
                            float(values["delta_ci_lower"]),
                            float(values["delta_ci_upper"]),
                        )
                        + "}",
                    ]
                )
    return "\n".join(lines) + "\n"


def _summary_rows(
    suites: Sequence[SuiteEvidence],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clean_rows: list[dict[str, Any]] = []
    robustness_rows: list[dict[str, Any]] = []
    for suite in suites:
        conditions = (suite.clean, *suite.robustness)
        for condition in conditions:
            target = clean_rows if condition is suite.clean else robustness_rows
            for method in CANONICAL_METHODS:
                values = condition.statistics[method]
                target.append(
                    {
                        "benchmark": suite.benchmark,
                        "condition": condition.condition,
                        "noise_level": condition.noise_level,
                        "method": METHOD_LABELS[method],
                        "task_macro_success": values["success"],
                        "success_ci_lower": values["success_ci_lower"],
                        "success_ci_upper": values["success_ci_upper"],
                        "worst_quartile_success": values["worst_quartile"],
                        "task_macro_intervention": values["intervention"],
                        "delta_vs_act": values["delta_vs_act"],
                        "delta_ci_lower": values["delta_ci_lower"],
                        "delta_ci_upper": values["delta_ci_upper"],
                        "task_count": SUITE_TASK_COUNT[suite.benchmark],
                        "episodes_per_task": OFFICIAL_EPISODES_PER_TASK,
                    }
                )
    return clean_rows, robustness_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise PublicationGateError(f"refusing to write empty CSV {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_robustness(suites: Sequence[SuiteEvidence], output: Path) -> None:
    colors = {
        "mlp_bc": "#737B85",
        "act": "#2457A6",
        "heuristic_recovery": "#D48A1F",
        "reim": "#17805C",
    }
    markers = {
        "mlp_bc": "o",
        "act": "s",
        "heuristic_recovery": "D",
        "reim": "^",
    }
    linestyles = {
        "mlp_bc": ":",
        "act": "--",
        "heuristic_recovery": "-.",
        "reim": "-",
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), sharey=True)
    for ax, suite in zip(axes, suites, strict=True):
        x = np.asarray([item.noise_level * 100.0 for item in suite.robustness])
        for method in CANONICAL_METHODS:
            values = np.asarray(
                [float(item.statistics[method]["success"]) * 100.0 for item in suite.robustness]
            )
            low = np.asarray(
                [
                    float(item.statistics[method]["success_ci_lower"])
                    * 100.0
                    for item in suite.robustness
                ]
            )
            high = np.asarray(
                [
                    float(item.statistics[method]["success_ci_upper"])
                    * 100.0
                    for item in suite.robustness
                ]
            )
            ax.errorbar(
                x,
                values,
                yerr=np.vstack((values - low, high - values)),
                color=colors[method],
                linestyle=linestyles[method],
                marker=markers[method],
                markersize=4.0,
                linewidth=1.45 if method == "reim" else 1.05,
                capsize=2.0,
                capthick=0.75,
                label=SHORT_LABELS[method],
                zorder=4 if method == "reim" else 3,
            )
        ax.set_title(suite.benchmark)
        ax.set_xlabel("Injected noise level (%)")
        ax.set_xticks([0, 10, 20, 30, 40])
        ax.set_xlim(-2, 42)
        ax.set_ylim(-2, 102)
        ax.grid(axis="y", color="#D9DEE5", linewidth=0.55, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Task-macro success (%)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        frameon=False,
        columnspacing=1.15,
        handlelength=2.3,
    )
    fig.text(
        0.5,
        0.005,
        "REIM robustness extension — task-universal action/observation noise; "
        "not official Meta-World scores",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#3F4650",
    )
    fig.tight_layout(rect=(0.0, 0.065, 1.0, 0.89), w_pad=1.4)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _input_manifest(
    suites: Sequence[SuiteEvidence],
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    inputs: dict[str, dict[str, Any]] = {}
    for suite in suites:
        paths = [suite.audit_path, suite.clean.summary_path, suite.clean.episode_path]
        for item in suite.robustness:
            paths.extend((item.summary_path, item.episode_path))
        for path in paths:
            inputs[str(path)] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    return {
        "schema_version": "reim-multitask-paper-inputs-v1",
        "gate_requirements": {
            "benchmarks": ["MT10", "MT50"],
            "official_clean": True,
            "robustness_noise_levels": list(ROBUSTNESS_LEVELS),
            "five_bank_audit": True,
        },
        "statistics": {
            "primary_estimand": "task_macro_success",
            "confidence": CONFIDENCE,
            "absolute_interval": "within_task_episode_percentile_bootstrap",
            "paired_delta_interval": "paired_episode_bootstrap_within_fixed_tasks",
            "bootstrap_samples": n_bootstrap,
            "bootstrap_seed": bootstrap_seed,
        },
        "inputs": inputs,
    }


def _output_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def generate_assets(
    *,
    tables_dir: Path,
    audits_dir: Path,
    assets_dir: Path,
    n_bootstrap: int = 5_000,
    bootstrap_seed: int = 20260806,
    compile_pdf: bool = False,
) -> dict[str, Any]:
    """Validate all evidence and atomically open the multi-task paper gate."""

    digest_cache: dict[Path, str] = {}
    suites = [
        validate_suite(
            benchmark=benchmark,
            tables_dir=tables_dir,
            audit_path=audits_dir / f"{benchmark.lower()}_bank_separation.json",
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
            digest_cache=digest_cache,
        )
        for benchmark in ("MT10", "MT50")
    ]
    input_manifest = _input_manifest(
        suites, n_bootstrap=n_bootstrap, bootstrap_seed=bootstrap_seed
    )
    input_manifest_sha = _canonical_sha256(input_manifest)
    clean_rows, robustness_rows = _summary_rows(suites)

    assets_dir = assets_dir.expanduser().resolve()
    assets_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        "gate": assets_dir / "multitask_results.tex",
        "clean_csv": assets_dir / "multitask_clean_statistics.csv",
        "robustness_csv": assets_dir / "multitask_robustness_statistics.csv",
        "figure_png": assets_dir / "Figure_multitask_robustness.png",
        "figure_pdf": assets_dir / "Figure_multitask_robustness.pdf",
        "manifest": assets_dir / "multitask_results_manifest.json",
    }
    paper_pdf = assets_dir / "reim_results.pdf"
    rollback_targets = list(targets.values()) + ([paper_pdf] if compile_pdf else [])
    previous = {
        path: (path.read_bytes() if path.is_file() else None)
        for path in rollback_targets
    }
    try:
        with tempfile.TemporaryDirectory(prefix="reim-mt-paper-", dir=assets_dir) as raw_tmp:
            tmp = Path(raw_tmp)
            _write_csv(tmp / "clean.csv", clean_rows)
            _write_csv(tmp / "robustness.csv", robustness_rows)
            _plot_robustness(suites, tmp / "robustness.png")
            gate_text = _render_gate_tex(suites, input_manifest_sha)
            (tmp / "gate.tex").write_text(gate_text, encoding="utf-8")
            staged = {
                targets["clean_csv"]: tmp / "clean.csv",
                targets["robustness_csv"]: tmp / "robustness.csv",
                targets["figure_png"]: tmp / "robustness.png",
                targets["figure_pdf"]: tmp / "robustness.pdf",
            }
            for destination, source in staged.items():
                os.replace(source, destination)
            # The TeX switch is intentionally installed after every dependency.
            os.replace(tmp / "gate.tex", targets["gate"])

        if compile_pdf:
            subprocess.run(
                [str(PROJECT_ROOT / "compile_paper.sh")],
                cwd=PROJECT_ROOT,
                check=True,
            )
            _regular_file(paper_pdf, "compiled paper PDF")

        outputs = {
            key: _output_record(path)
            for key, path in targets.items()
            if key != "manifest"
        }
        if compile_pdf:
            outputs["paper_pdf"] = _output_record(paper_pdf)
        manifest = {
            **input_manifest,
            "schema_version": "reim-multitask-paper-results-v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "publication_gate": {
                "eligible": True,
                "reason": (
                    "MT10 and MT50 official-clean rollouts, five robustness "
                    "conditions, and both five-bank audits passed"
                ),
                "official_clean_scope": "known-task_goal-observable",
                "robustness_scope": "non_official_reim_extension",
            },
            "input_manifest_sha256": input_manifest_sha,
            "outputs": outputs,
        }
        tmp_manifest = targets["manifest"].with_suffix(".json.tmp")
        tmp_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_manifest, targets["manifest"])
        return manifest
    except BaseException:
        for path, content in previous.items():
            if content is None:
                if path.exists():
                    path.unlink()
            else:
                temporary = path.with_name(path.name + ".rollback.tmp")
                temporary.write_bytes(content)
                os.replace(temporary, path)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "tables",
    )
    parser.add_argument(
        "--audits-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "audits",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=PROJECT_ROOT / "paper_assets",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260806)
    parser.add_argument(
        "--compile-pdf",
        action="store_true",
        help="Compile paper_assets/reim_results.pdf after opening the gate.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = generate_assets(
            tables_dir=args.tables_dir,
            audits_dir=args.audits_dir,
            assets_dir=args.assets_dir,
            n_bootstrap=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            compile_pdf=args.compile_pdf,
        )
    except (PublicationGateError, OSError, subprocess.CalledProcessError) as exc:
        print(f"multi-task paper gate closed: {exc}")
        return 2
    print(json.dumps(manifest["publication_gate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
