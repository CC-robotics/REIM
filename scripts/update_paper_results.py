#!/usr/bin/env python3
"""Validate the frozen full protocol and atomically fill paper result macros.

This command is intentionally a strict publication gate.  It reads canonical
summary files *and* their per-episode companions, recomputes the sufficient
statistics, checks paired seeds and recovery semantics, validates the detector
at the deployed 0.20 operating point, and only then changes
``\\REIMFinalResults`` to true.  A failed validation never modifies a paper
asset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable, Hashable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_EPISODES = 1_000
ROBUSTNESS_EPISODES = 200
MAIN_NOISE = 0.20
NOISE_LEVELS = (0.0, 0.1, 0.2, 0.3, 0.4)
DEPLOYMENT_THRESHOLD = 0.20
FINAL_RECOVERY_DEFINITION = (
    "task_success_while_recovery_active_per_intervention"
)
RESET_RECOVERY_DEFINITION = "post_reset_task_success_per_reset"

BASELINE_METHODS = (
    "ACT",
    "ACT + Random Reset",
    "ACT + Heuristic Recovery",
    "REIM (ACT + Detector + Recovery)",
)
ABLATION_VARIANTS = {
    "A": "ACT",
    "B": "ACT + Detector",
    "C": "ACT + Recovery",
    "D": "REIM",
}
ROBUSTNESS_METHODS = ("ACT", "REIM (ACT + Detector + Recovery)")

SUMMARY_COLUMNS = (
    "Method",
    "Success Rate",
    "Success CI Lower",
    "Success CI Upper",
    "Recovery Rate",
    "Recovery CI Lower",
    "Recovery CI Upper",
    "Average Steps",
    "Episodes",
    "Successes",
    "Recovery Attempts",
    "Recovery Successes",
    "Detector Triggers",
    "Backend",
    "Benchmark Eligible",
    "Profile",
    "Noise Level",
    "Recovery Definition",
    "Intervened Episodes",
    "Evaluation Seed Start",
    "Evaluation Seed End",
    "Episode Bank SHA256",
    "Episode Bank File SHA256",
    "CRN Episode Specifications Verified",
)
RAW_COLUMNS = (
    "method",
    "episode",
    "seed",
    "backend",
    "success",
    "steps",
    "recovery_attempts",
    "recovery_successes",
    "recovery_definition",
    "detector_triggers",
    "Profile",
    "noise_level",
    "episode_specification_sha256",
    "episode_bank_sha256",
    "metaworld_task_sha256",
    "retry_specification_sha256s",
    "retry_task_sha256s",
)


class PaperResultError(ValueError):
    """Raised when an artifact is not valid final-paper evidence."""


def _read_csv(path: Path, required: Sequence[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise PaperResultError(f"missing required measured result: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = [name for name in required if name not in columns]
        if missing:
            raise PaperResultError(
                f"{path} is missing columns: {', '.join(missing)}"
            )
        rows = [dict(row) for row in reader]
    if not rows:
        raise PaperResultError(f"{path} is empty")
    return rows


def _float(
    row: Mapping[str, Any],
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise PaperResultError(f"invalid value for {key!r}: {row.get(key)!r}") from exc
    if not math.isfinite(value):
        raise PaperResultError(f"{key!r} must be finite")
    if minimum is not None and value < minimum:
        raise PaperResultError(f"{key!r}={value} is below {minimum}")
    if maximum is not None and value > maximum:
        raise PaperResultError(f"{key!r}={value} is above {maximum}")
    return value


def _int(row: Mapping[str, Any], key: str, *, minimum: int = 0) -> int:
    value = _float(row, key)
    rounded = int(round(value))
    if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise PaperResultError(f"{key!r}={value} is not an integer")
    if rounded < minimum:
        raise PaperResultError(f"{key!r}={rounded} is below {minimum}")
    return rounded


def _truth(value: Any, *, field: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise PaperResultError(f"{field} has invalid Boolean value {value!r}")


def _close(actual: float, expected: float, *, label: str, atol: float = 1e-10) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol):
        raise PaperResultError(f"{label}: reported {actual}, recomputed {expected}")


def _sha256_text(value: Any, *, field: str) -> str:
    normalized = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise PaperResultError(f"{field} is not a SHA256 digest: {value!r}")
    return normalized


def _validate_ci(row: Mapping[str, Any], value_key: str, low_key: str, high_key: str) -> None:
    value = _float(row, value_key, minimum=0.0, maximum=1.0)
    low = _float(row, low_key, minimum=0.0, maximum=1.0)
    high = _float(row, high_key, minimum=0.0, maximum=1.0)
    if low > value or value > high:
        raise PaperResultError(
            f"invalid interval ordering for {value_key}: {low}, {value}, {high}"
        )


def _expected_recovery_definition(method: str, attempts: int) -> str:
    if attempts == 0:
        return "not_applicable"
    if "Random Reset" in method:
        return RESET_RECOVERY_DEFINITION
    return FINAL_RECOVERY_DEFINITION


def _validate_summary_metadata(
    row: Mapping[str, Any],
    *,
    method: str,
    episodes: int,
    noise: float,
    seeds: set[int],
) -> None:
    if row.get("Backend", "").strip().lower() != "metaworld":
        raise PaperResultError(f"{method}: Backend must be metaworld")
    if row.get("Profile", "").strip().lower() != "full":
        raise PaperResultError(f"{method}: Profile must be full")
    if not _truth(row.get("Benchmark Eligible", ""), field="Benchmark Eligible"):
        raise PaperResultError(f"{method}: result is not benchmark eligible")
    if _int(row, "Episodes") != episodes:
        raise PaperResultError(f"{method}: expected {episodes} summary episodes")
    _close(_float(row, "Noise Level"), noise, label=f"{method} noise level")
    if _int(row, "Evaluation Seed Start") != min(seeds):
        raise PaperResultError(f"{method}: incorrect Evaluation Seed Start")
    if _int(row, "Evaluation Seed End") != max(seeds):
        raise PaperResultError(f"{method}: incorrect Evaluation Seed End")
    if not _truth(
        row.get("CRN Episode Specifications Verified", ""),
        field="CRN Episode Specifications Verified",
    ):
        raise PaperResultError(f"{method}: CRN episode specifications are unverified")
    _sha256_text(
        row.get("Episode Bank SHA256", ""),
        field=f"{method} Episode Bank SHA256",
    )
    _sha256_text(
        row.get("Episode Bank File SHA256", ""),
        field=f"{method} Episode Bank File SHA256",
    )
    _validate_ci(
        row, "Success Rate", "Success CI Lower", "Success CI Upper"
    )
    _validate_ci(
        row, "Recovery Rate", "Recovery CI Lower", "Recovery CI Upper"
    )


def _validate_condition(
    *,
    summary: Mapping[str, Any],
    raw: Sequence[Mapping[str, Any]],
    method: str,
    episodes: int,
    noise: float,
    expected_seeds: set[int],
) -> None:
    _validate_summary_metadata(
        summary,
        method=method,
        episodes=episodes,
        noise=noise,
        seeds=expected_seeds,
    )
    if len(raw) != episodes:
        raise PaperResultError(f"{method}: expected {episodes} raw episodes")

    seeds: list[int] = []
    episode_ids: list[int] = []
    successes = 0
    steps = 0
    attempts = 0
    recovery_successes = 0
    detector_triggers = 0
    intervened = 0
    summary_bank_sha = _sha256_text(
        summary.get("Episode Bank SHA256", ""),
        field=f"{method} summary episode bank",
    )
    for record in raw:
        if str(record.get("method", "")) != method:
            raise PaperResultError(f"{method}: raw method label mismatch")
        if str(record.get("backend", "")).strip().lower() != "metaworld":
            raise PaperResultError(f"{method}: raw Backend must be metaworld")
        if str(record.get("Profile", "")).strip().lower() != "full":
            raise PaperResultError(f"{method}: raw Profile must be full")
        _close(
            _float(record, "noise_level"),
            noise,
            label=f"{method} raw noise level",
        )
        seed = _int(record, "seed")
        episode = _int(record, "episode")
        success = _truth(record.get("success", ""), field="success")
        step_count = _int(record, "steps", minimum=1)
        attempt_count = _int(record, "recovery_attempts")
        recovered_count = _int(record, "recovery_successes")
        trigger_count = _int(record, "detector_triggers")
        specification_sha = _sha256_text(
            record.get("episode_specification_sha256", ""),
            field=f"{method} seed {seed} episode specification",
        )
        del specification_sha
        bank_sha = _sha256_text(
            record.get("episode_bank_sha256", ""),
            field=f"{method} seed {seed} episode bank",
        )
        if bank_sha != summary_bank_sha:
            raise PaperResultError(
                f"{method} seed {seed}: raw and summary episode-bank hashes differ"
            )
        _sha256_text(
            record.get("metaworld_task_sha256", ""),
            field=f"{method} seed {seed} Meta-World task",
        )
        raw_definition = str(record.get("recovery_definition", "")).strip()
        if "Random Reset" in method:
            expected_raw_definition = RESET_RECOVERY_DEFINITION
        elif method in {
            "ACT + Heuristic Recovery",
            "REIM (ACT + Detector + Recovery)",
            "ACT + Recovery",
            "REIM",
        }:
            expected_raw_definition = FINAL_RECOVERY_DEFINITION
        else:
            expected_raw_definition = ""
        if raw_definition != expected_raw_definition:
            raise PaperResultError(
                f"{method} seed {seed}: raw recovery_definition="
                f"{raw_definition!r}; expected {expected_raw_definition!r}"
            )
        if recovered_count > attempt_count:
            raise PaperResultError(
                f"{method} seed {seed}: recovery successes exceed interventions"
            )
        if (
            recovered_count > 0
            and "Random Reset" not in method
            and not success
        ):
            raise PaperResultError(
                f"{method} seed {seed}: completion during recovery must imply "
                "final task success"
            )
        retry_specs = str(
            record.get("retry_specification_sha256s", "")
        ).strip()
        retry_tasks = str(record.get("retry_task_sha256s", "")).strip()
        if "Random Reset" in method and attempt_count > 0:
            for field, value in (
                ("retry specification", retry_specs),
                ("retry task", retry_tasks),
            ):
                digests = [item for item in value.split(";") if item]
                if len(digests) != attempt_count:
                    raise PaperResultError(
                        f"{method} seed {seed}: expected {attempt_count} "
                        f"{field} hashes, got {len(digests)}"
                    )
                for digest in digests:
                    _sha256_text(
                        digest,
                        field=f"{method} seed {seed} {field}",
                    )
        elif retry_specs or retry_tasks:
            raise PaperResultError(
                f"{method} seed {seed}: unexpected retry provenance"
            )
        seeds.append(seed)
        episode_ids.append(episode)
        successes += int(success)
        steps += step_count
        attempts += attempt_count
        recovery_successes += recovered_count
        detector_triggers += trigger_count
        intervened += int(attempt_count > 0)

    if len(set(seeds)) != episodes or set(seeds) != expected_seeds:
        raise PaperResultError(f"{method}: raw seeds are not the frozen paired set")
    if set(episode_ids) != set(range(episodes)):
        raise PaperResultError(f"{method}: raw episode ids must be 0..{episodes - 1}")

    reported_episodes = _int(summary, "Episodes")
    reported_successes = _int(summary, "Successes")
    reported_attempts = _int(summary, "Recovery Attempts")
    reported_recovered = _int(summary, "Recovery Successes")
    if reported_successes != successes:
        raise PaperResultError(f"{method}: Successes does not match raw episodes")
    if reported_attempts != attempts:
        raise PaperResultError(f"{method}: Recovery Attempts does not match raw episodes")
    if reported_recovered != recovery_successes:
        raise PaperResultError(
            f"{method}: Recovery Successes does not match raw episodes"
        )
    if _int(summary, "Detector Triggers") != detector_triggers:
        raise PaperResultError(f"{method}: Detector Triggers does not match raw episodes")
    if _int(summary, "Intervened Episodes") != intervened:
        raise PaperResultError(f"{method}: Intervened Episodes does not match raw episodes")

    _close(
        _float(summary, "Success Rate"),
        successes / reported_episodes,
        label=f"{method} success rate",
    )
    expected_recovery = recovery_successes / attempts if attempts else 0.0
    _close(
        _float(summary, "Recovery Rate"),
        expected_recovery,
        label=f"{method} recovery rate",
    )
    _close(
        _float(summary, "Average Steps"),
        steps / reported_episodes,
        label=f"{method} average steps",
        atol=1e-8,
    )
    expected_definition = _expected_recovery_definition(method, attempts)
    actual_definition = str(summary.get("Recovery Definition", "")).strip()
    if actual_definition != expected_definition:
        raise PaperResultError(
            f"{method}: Recovery Definition={actual_definition!r}; "
            f"expected {expected_definition!r}"
        )


def _index_unique(
    rows: Sequence[Mapping[str, Any]],
    key: Callable[[Mapping[str, Any]], Hashable],
    *,
    source: str,
) -> dict[Hashable, Mapping[str, Any]]:
    indexed: dict[Hashable, Mapping[str, Any]] = {}
    for row in rows:
        item_key = key(row)
        if item_key in indexed:
            raise PaperResultError(f"{source} has duplicate condition {item_key!r}")
        indexed[item_key] = row
    return indexed


def _validate_baseline_and_ablation(
    *,
    baseline: Sequence[Mapping[str, Any]],
    baseline_raw: Sequence[Mapping[str, Any]],
    ablation: Sequence[Mapping[str, Any]],
    ablation_raw: Sequence[Mapping[str, Any]],
    evaluation_seed: int,
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    dict[str, dict[int, bool]],
]:
    expected_seeds = set(range(evaluation_seed, evaluation_seed + MAIN_EPISODES))
    baseline_index = _index_unique(
        baseline, lambda row: str(row["Method"]), source="baseline.csv"
    )
    if set(baseline_index) != set(BASELINE_METHODS):
        raise PaperResultError("baseline.csv does not contain the canonical four methods")
    raw_by_method: dict[str, list[Mapping[str, Any]]] = {
        method: [] for method in BASELINE_METHODS
    }
    for row in baseline_raw:
        method = str(row.get("method", ""))
        if method not in raw_by_method:
            raise PaperResultError(f"baseline_episodes.csv has extra method {method!r}")
        raw_by_method[method].append(row)
    if len(baseline_raw) != MAIN_EPISODES * len(BASELINE_METHODS):
        raise PaperResultError("baseline_episodes.csv must contain exactly 4,000 rows")
    for method in BASELINE_METHODS:
        _validate_condition(
            summary=baseline_index[method],
            raw=raw_by_method[method],
            method=method,
            episodes=MAIN_EPISODES,
            noise=MAIN_NOISE,
            expected_seeds=expected_seeds,
        )
    baseline_by_seed = {
        method: {_int(row, "seed"): row for row in rows}
        for method, rows in raw_by_method.items()
    }
    for seed in sorted(expected_seeds):
        reference = baseline_by_seed["ACT"][seed]
        for method in BASELINE_METHODS[1:]:
            candidate = baseline_by_seed[method][seed]
            for field in (
                "episode_specification_sha256",
                "episode_bank_sha256",
                "metaworld_task_sha256",
            ):
                if str(candidate[field]) != str(reference[field]):
                    raise PaperResultError(
                        f"baseline seed {seed}: {method} differs from ACT in {field}"
                    )

    ablation_index = _index_unique(
        ablation, lambda row: str(row["Variant"]), source="ablation.csv"
    )
    if set(ablation_index) != set(ABLATION_VARIANTS):
        raise PaperResultError("ablation.csv must contain variants A--D")
    raw_ablation: dict[str, list[Mapping[str, Any]]] = {
        method: [] for method in ABLATION_VARIANTS.values()
    }
    for variant, method in ABLATION_VARIANTS.items():
        row = ablation_index[variant]
        if str(row.get("Method", "")) != method:
            raise PaperResultError(
                f"ablation variant {variant} must be labeled {method!r}"
            )
    for row in ablation_raw:
        method = str(row.get("method", ""))
        if method not in raw_ablation:
            raise PaperResultError(f"ablation_episodes.csv has extra method {method!r}")
        expected_variant = next(
            variant for variant, label in ABLATION_VARIANTS.items() if label == method
        )
        if str(row.get("variant", "")) != expected_variant:
            raise PaperResultError(
                f"ablation raw method {method!r} has incorrect variant"
            )
        raw_ablation[method].append(row)
    if len(ablation_raw) != MAIN_EPISODES * len(ABLATION_VARIANTS):
        raise PaperResultError("ablation_episodes.csv must contain exactly 4,000 rows")
    for variant, method in ABLATION_VARIANTS.items():
        _validate_condition(
            summary=ablation_index[variant],
            raw=raw_ablation[method],
            method=method,
            episodes=MAIN_EPISODES,
            noise=MAIN_NOISE,
            expected_seeds=expected_seeds,
        )

    paired: dict[str, dict[int, bool]] = {}
    for method, rows in raw_by_method.items():
        paired[method] = {
            _int(row, "seed"): _truth(row["success"], field="success")
            for row in rows
        }
    return baseline_index, ablation_index, paired


def _validate_robustness(
    *,
    summaries: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    evaluation_seed: int,
) -> dict[tuple[str, float], Mapping[str, Any]]:
    expected_keys = {
        (method, level) for method in ROBUSTNESS_METHODS for level in NOISE_LEVELS
    }
    expected_seeds = set(
        range(evaluation_seed, evaluation_seed + ROBUSTNESS_EPISODES)
    )

    def summary_key(row: Mapping[str, Any]) -> tuple[str, float]:
        return str(row["Method"]), round(_float(row, "Noise Level"), 8)

    indexed = _index_unique(summaries, summary_key, source="robustness.csv")
    if set(indexed) != expected_keys:
        raise PaperResultError(
            "robustness.csv must contain ACT and REIM at five frozen noise levels"
        )
    grouped: dict[tuple[str, float], list[Mapping[str, Any]]] = {
        key: [] for key in expected_keys
    }
    for row in raw_rows:
        key = (str(row.get("method", "")), round(_float(row, "noise_level"), 8))
        if key not in grouped:
            raise PaperResultError(f"robustness_episodes.csv has extra condition {key}")
        grouped[key].append(row)
    if len(raw_rows) != len(expected_keys) * ROBUSTNESS_EPISODES:
        raise PaperResultError("robustness_episodes.csv must contain exactly 2,000 rows")
    for method, level in sorted(expected_keys):
        _validate_condition(
            summary=indexed[(method, level)],
            raw=grouped[(method, level)],
            method=method,
            episodes=ROBUSTNESS_EPISODES,
            noise=level,
            expected_seeds=expected_seeds,
        )
    for level in NOISE_LEVELS:
        act_by_seed = {
            _int(row, "seed"): row for row in grouped[("ACT", level)]
        }
        reim_by_seed = {
            _int(row, "seed"): row
            for row in grouped[
                ("REIM (ACT + Detector + Recovery)", level)
            ]
        }
        for seed in sorted(expected_seeds):
            for field in (
                "episode_specification_sha256",
                "episode_bank_sha256",
                "metaworld_task_sha256",
            ):
                if str(act_by_seed[seed][field]) != str(
                    reim_by_seed[seed][field]
                ):
                    raise PaperResultError(
                        f"robustness noise={level:g} seed={seed}: "
                        f"ACT/REIM differ in {field}"
                    )
    return indexed


def _validate_confusion_metrics(
    metrics: Mapping[str, Any],
    *,
    label: str,
    expected_threshold: float,
) -> tuple[int, int, int, int]:
    threshold = _float(metrics, "threshold", minimum=0.0, maximum=1.0)
    _close(threshold, expected_threshold, label=f"{label} threshold", atol=1e-12)
    matrix = np.asarray(metrics.get("confusion_matrix"), dtype=np.float64)
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
        raise PaperResultError(f"{label}: confusion_matrix must be finite 2x2")
    rounded = np.rint(matrix).astype(np.int64)
    if not np.allclose(matrix, rounded) or np.any(rounded < 0):
        raise PaperResultError(f"{label}: confusion matrix must contain counts")
    tn, fp, fn, tp = (int(value) for value in rounded.reshape(-1))
    samples = tn + fp + fn + tp
    if _int(metrics, "samples", minimum=1) != samples:
        raise PaperResultError(f"{label}: sample count does not match confusion matrix")
    for key, expected in (("tn", tn), ("fp", fp), ("fn", fn), ("tp", tp)):
        if key in metrics and _int(metrics, key) != expected:
            raise PaperResultError(f"{label}: {key} does not match confusion matrix")
    accuracy = (tn + tp) / samples
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    for key, expected in (
        ("accuracy", accuracy),
        ("precision", precision),
        ("recall", recall),
        ("f1", f1),
    ):
        _close(
            _float(metrics, key, minimum=0.0, maximum=1.0),
            expected,
            label=f"{label} {key}",
            atol=1e-9,
        )
    return tn, fp, fn, tp


def _validate_detector(path: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not path.is_file():
        raise PaperResultError(f"missing detector metrics: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise PaperResultError("detector_metrics.json must contain an object")
    _validate_confusion_metrics(
        payload, label="detector standard operating point", expected_threshold=0.50
    )
    deployment = payload.get("deployment_threshold_metrics")
    if not isinstance(deployment, Mapping):
        raise PaperResultError(
            "detector_metrics.json lacks deployment_threshold_metrics"
        )
    _validate_confusion_metrics(
        deployment,
        label="detector deployment operating point",
        expected_threshold=DEPLOYMENT_THRESHOLD,
    )
    if _int(payload, "samples", minimum=1) != _int(
        deployment, "samples", minimum=1
    ):
        raise PaperResultError("detector operating points use different samples")
    return payload, deployment


def _validate_operation_trace(path: Path) -> Mapping[str, Any]:
    """Validate the qualitative simulator figure and return its measured fields."""

    if not path.is_file():
        raise PaperResultError(f"missing operation-trace metadata: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise PaperResultError("operation trace metadata must contain an object")
    if payload.get("schema_version") != "reim-paired-simulation-rollout-figure-v3":
        raise PaperResultError("operation trace uses an unexpected schema")
    provenance = str(payload.get("provenance_statement", "")).lower()
    if "not physical-robot photographs" not in provenance:
        raise PaperResultError("operation trace does not disclose simulation provenance")
    paired = payload.get("paired_protocol")
    act = payload.get("act")
    reim = payload.get("reim")
    if not all(isinstance(item, Mapping) for item in (paired, act, reim)):
        raise PaperResultError("operation trace lacks paired ACT/REIM metadata")
    if not _truth(paired.get("same_seed", ""), field="operation same_seed"):
        raise PaperResultError("operation ACT and REIM traces are not seed paired")
    _close(
        _float(paired, "initial_object_goal_max_abs_delta", minimum=0.0),
        0.0,
        label="operation initial object/goal pairing",
        atol=1e-12,
    )
    _close(
        _float(paired, "object_displacement_max_abs_delta", minimum=0.0),
        0.0,
        label="operation disturbance pairing",
        atol=1e-12,
    )
    if _truth(act.get("success", ""), field="operation ACT success"):
        raise PaperResultError("operation trace is not an ACT failure")
    if not _truth(reim.get("success", ""), field="operation REIM success"):
        raise PaperResultError("operation trace is not a REIM success")
    if _int(reim, "detector_triggers", minimum=1) != 1:
        raise PaperResultError("operation trace must contain exactly one trigger")
    if _int(reim, "recovery_attempts", minimum=1) != 1:
        raise PaperResultError("operation trace must contain exactly one intervention")
    if _int(reim, "recovery_successes", minimum=1) != 1:
        raise PaperResultError("operation trace intervention did not complete the task")
    recovery_steps = _int(reim, "recovery_steps", minimum=1)
    reim_steps = _int(reim, "steps", minimum=1)
    if recovery_steps > reim_steps:
        raise PaperResultError("operation recovery steps exceed total REIM steps")
    _close(
        _float(payload, "noise_level", minimum=0.0),
        MAIN_NOISE,
        label="operation noise level",
    )
    _close(
        _float(payload, "failure_threshold", minimum=0.0, maximum=1.0),
        DEPLOYMENT_THRESHOLD,
        label="operation deployment threshold",
    )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise PaperResultError("operation trace has no artifact provenance")
    required_artifacts = {
        "results/figures/recovery_operation_sequence.png",
        "results/figures/recovery_operation_sequence.pdf",
        "paper_assets/Figure5_operation_sequence.png",
        "paper_assets/Figure5_operation_sequence.pdf",
    }
    if not required_artifacts.issubset(artifacts):
        raise PaperResultError("operation trace omits required composite artifacts")
    for relative_path, metadata in artifacts.items():
        if not isinstance(metadata, Mapping):
            raise PaperResultError(f"invalid operation artifact metadata: {relative_path}")
        artifact_path = PROJECT_ROOT / str(relative_path)
        if not artifact_path.is_file():
            raise PaperResultError(f"missing operation artifact: {artifact_path}")
        expected_sha = _sha256_text(
            metadata.get("sha256", ""),
            field=f"operation artifact {relative_path}",
        )
        if _sha256(artifact_path) != expected_sha:
            raise PaperResultError(f"operation artifact hash drift: {artifact_path}")
    return payload


def _validate_matched_gate(path: Path) -> Mapping[str, Any]:
    """Validate the separate post-freeze heuristic-vs-LSTM gate audit."""

    if not path.is_file():
        raise PaperResultError(f"missing matched-gate audit: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise PaperResultError("matched-gate audit must contain an object")
    validation = payload.get("validation")
    protocol = payload.get("protocol")
    methods = payload.get("methods")
    comparisons = payload.get("paired_success_comparisons")
    if not all(
        isinstance(item, Mapping)
        for item in (validation, protocol, methods, comparisons)
    ):
        raise PaperResultError("matched-gate audit lacks required sections")
    if not _truth(validation.get("all_checks_passed", ""), field="matched gate"):
        raise PaperResultError("matched-gate audit did not pass")
    if (
        str(protocol.get("backend", "")).lower() != "metaworld"
        or _int(protocol, "episodes") != ROBUSTNESS_EPISODES
        or _int(protocol, "episode_seed_start") != 8_200_042
        or _int(protocol, "episode_seed_end") != 8_200_241
    ):
        raise PaperResultError("matched-gate audit protocol drifted")
    _close(
        _float(protocol, "noise_level"),
        MAIN_NOISE,
        label="matched-gate noise",
    )
    bank_path = Path(str(protocol.get("episode_bank", "")))
    if not bank_path.is_file():
        raise PaperResultError(f"matched-gate episode bank is missing: {bank_path}")
    expected_bank_file_sha = _sha256_text(
        protocol.get("episode_bank_file_sha256", ""),
        field="matched-gate episode-bank file",
    )
    if _sha256(bank_path) != expected_bank_file_sha:
        raise PaperResultError("matched-gate episode-bank file hash drifted")
    _sha256_text(
        protocol.get("episode_bank_sha256", ""),
        field="matched-gate episode bank",
    )
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping) or len(inputs) != 3:
        raise PaperResultError("matched-gate audit must record three raw inputs")
    for name, record in inputs.items():
        if not isinstance(record, Mapping):
            raise PaperResultError(f"matched-gate input {name} is invalid")
        raw_path = Path(str(record.get("path", "")))
        if not raw_path.is_file() or _int(record, "rows") != ROBUSTNESS_EPISODES:
            raise PaperResultError(f"matched-gate input {name} is missing or incomplete")
        expected_sha = _sha256_text(
            record.get("sha256", ""), field=f"matched-gate input {name}"
        )
        if _sha256(raw_path) != expected_sha:
            raise PaperResultError(f"matched-gate input hash drift: {raw_path}")
    expected_methods = {
        "heuristic_gate": "ACT + Heuristic Recovery",
        "reim_tau_0.175": "REIM (ACT + Detector + Recovery)",
        "reim_tau_0.20": "REIM (ACT + Detector + Recovery)",
    }
    if set(methods) != set(expected_methods):
        raise PaperResultError("matched-gate audit has unexpected methods")
    for name, expected_method in expected_methods.items():
        record = methods[name]
        if record.get("method") != expected_method:
            raise PaperResultError(f"matched-gate method mismatch for {name}")
        if _int(record, "episodes") != ROBUSTNESS_EPISODES:
            raise PaperResultError(f"matched-gate episode count mismatch for {name}")
        for key in ("success_rate", "intervention_rate", "recovery_rate"):
            _float(record, key, minimum=0.0, maximum=1.0)
    expected_comparisons = {
        "reim_tau_0.175_minus_heuristic",
        "reim_tau_0.20_minus_heuristic",
        "reim_tau_0.175_minus_reim_tau_0.20",
    }
    if set(comparisons) != expected_comparisons:
        raise PaperResultError("matched-gate paired comparisons are incomplete")
    for name, record in comparisons.items():
        episodes = _int(record, "episodes")
        wins = _int(record, "wins")
        losses = _int(record, "losses")
        delta = _float(record, "paired_delta", minimum=-1.0, maximum=1.0)
        _close(delta, (wins - losses) / episodes, label=f"{name} paired delta")
        interval = record.get("paired_delta_bootstrap_95_ci")
        if not isinstance(interval, Sequence) or len(interval) != 2:
            raise PaperResultError(f"{name} has no paired interval")
        low, high = (float(value) for value in interval)
        if low > delta or delta > high:
            raise PaperResultError(f"{name} paired interval is not ordered")
        _float(
            record,
            "exact_two_sided_mcnemar_binomial_p",
            minimum=0.0,
            maximum=1.0,
        )
    return payload


def _bootstrap_gain(differences: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(20260726)
    estimates = np.empty(10_000, dtype=np.float64)
    for start in range(0, len(estimates), 500):
        stop = min(start + 500, len(estimates))
        ids = rng.integers(
            0, differences.size, size=(stop - start, differences.size)
        )
        estimates[start:stop] = differences[ids].mean(axis=1)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def _paired_gain(
    *,
    paired: Mapping[str, Mapping[int, bool]],
    comparison_path: Path,
    baseline_reim: Mapping[str, Any],
    evaluation_seed: int,
) -> tuple[float, float, float, int, int]:
    act = paired["ACT"]
    reim = paired["REIM (ACT + Detector + Recovery)"]
    if set(act) != set(reim):
        raise PaperResultError("ACT and REIM raw episodes are not seed paired")
    seeds = sorted(act)
    differences = np.asarray(
        [float(reim[seed]) - float(act[seed]) for seed in seeds],
        dtype=np.float64,
    )
    rescued = sum(not act[seed] and reim[seed] for seed in seeds)
    harmed = sum(act[seed] and not reim[seed] for seed in seeds)
    gain = float(differences.mean())
    _close(
        gain,
        (rescued - harmed) / len(seeds),
        label="paired rescued/harmed difference",
    )
    _close(
        gain,
        _float(baseline_reim, "Success Rate")
        - sum(act.values()) / len(act),
        label="paired gain versus summary rates",
    )

    if comparison_path.is_file():
        comparison = _read_csv(
            comparison_path,
            (
                "Method",
                "Success Gain vs ACT",
                "Gain CI Lower",
                "Gain CI Upper",
                "Episodes",
                "Backend",
                "Profile",
                "Evaluation Seed Start",
                "Evaluation Seed End",
            ),
        )
        selected = [
            row
            for row in comparison
            if row["Method"] == "REIM (ACT + Detector + Recovery)"
        ]
        if len(selected) != 1:
            raise PaperResultError("comparison.csv needs exactly one REIM row")
        row = selected[0]
        if (
            _int(row, "Episodes") != MAIN_EPISODES
            or str(row["Backend"]).lower() != "metaworld"
            or str(row["Profile"]).lower() != "full"
            or _int(row, "Evaluation Seed Start") != evaluation_seed
            or _int(row, "Evaluation Seed End")
            != evaluation_seed + MAIN_EPISODES - 1
        ):
            raise PaperResultError("comparison.csv does not match the final paired run")
        reported_gain = _float(row, "Success Gain vs ACT", minimum=-1.0, maximum=1.0)
        low = _float(row, "Gain CI Lower", minimum=-1.0, maximum=1.0)
        high = _float(row, "Gain CI Upper", minimum=-1.0, maximum=1.0)
        _close(reported_gain, gain, label="comparison.csv paired gain")
        if low > reported_gain or reported_gain > high:
            raise PaperResultError("comparison.csv gain interval is not ordered")
    else:
        low, high = _bootstrap_gain(differences)
    return gain, low, high, rescued, harmed


def _paired_delta(
    *,
    paired: Mapping[str, Mapping[int, bool]],
    contender: str,
    reference: str,
) -> tuple[float, float, float, int, int]:
    """Recompute a paired success-rate difference and bootstrap interval."""
    contender_rows = paired[contender]
    reference_rows = paired[reference]
    if set(contender_rows) != set(reference_rows):
        raise PaperResultError(
            f"{contender} and {reference} raw episodes are not seed paired"
        )
    seeds = sorted(contender_rows)
    differences = np.asarray(
        [
            float(contender_rows[seed]) - float(reference_rows[seed])
            for seed in seeds
        ],
        dtype=np.float64,
    )
    wins = sum(
        contender_rows[seed] and not reference_rows[seed] for seed in seeds
    )
    losses = sum(
        reference_rows[seed] and not contender_rows[seed] for seed in seeds
    )
    gain = float(differences.mean())
    _close(
        gain,
        (wins - losses) / len(seeds),
        label=f"paired {contender} minus {reference} gain",
    )
    low, high = _bootstrap_gain(differences)
    return gain, low, high, wins, losses


def _percentage(value: float) -> str:
    return f"{100.0 * value:.1f}\\%"


def _percentage_ci(low: float, high: float) -> str:
    return f"[{100.0 * low:.1f}, {100.0 * high:.1f}]\\%"


def _signed_pp(value: float) -> str:
    return f"{100.0 * value:+.1f}\\,pp"


def _signed_pp_ci(low: float, high: float) -> str:
    return f"[{100.0 * low:+.1f}, {100.0 * high:+.1f}]\\,pp"


def _tex_int(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


def _row_macros(prefix: str, row: Mapping[str, Any]) -> dict[str, str]:
    episodes = _int(row, "Episodes", minimum=1)
    attempts = _int(row, "Recovery Attempts")
    intervened = _int(row, "Intervened Episodes")
    values = {
        f"Final{prefix}Success": _percentage(_float(row, "Success Rate")),
        f"Final{prefix}SuccessCI": _percentage_ci(
            _float(row, "Success CI Lower"), _float(row, "Success CI Upper")
        ),
        f"Final{prefix}Intervention": _percentage(intervened / episodes),
        f"Final{prefix}Steps": f"{_float(row, 'Average Steps'):.1f}",
    }
    if attempts:
        values[f"Final{prefix}Recovery"] = _percentage(
            _float(row, "Recovery Rate")
        )
        values[f"Final{prefix}RecoveryCI"] = _percentage_ci(
            _float(row, "Recovery CI Lower"), _float(row, "Recovery CI Upper")
        )
    return values


def _build_macro_values(
    *,
    baseline: Mapping[str, Mapping[str, Any]],
    ablation: Mapping[str, Mapping[str, Any]],
    robustness: Mapping[tuple[str, float], Mapping[str, Any]],
    detector: Mapping[str, Any],
    deployment: Mapping[str, Any],
    operation: Mapping[str, Any],
    matched_gate: Mapping[str, Any],
    gain: tuple[float, float, float, int, int],
    gain_vs_reset: tuple[float, float, float, int, int],
    gain_vs_rl: tuple[float, float, float, int, int],
) -> dict[str, str]:
    values: dict[str, str] = {}
    act = baseline["ACT"]
    values.update(_row_macros("ACT", act))
    values["FinalACTRecovery"] = r"\NotApplicable"
    values["FinalACTRecoveryCI"] = ""
    values["FinalACTIntervention"] = r"\NotApplicable"
    values.update(_row_macros("Reset", baseline["ACT + Random Reset"]))

    rl_values = _row_macros(
        "RLRecovery", baseline["ACT + Heuristic Recovery"]
    )
    values.update(rl_values)
    # The manuscript uses ``Rate`` in this legacy internal macro name.
    values["FinalRLRecoveryRate"] = values.pop("FinalRLRecoveryRecovery")
    values["FinalRLRecoveryRateCI"] = values.pop("FinalRLRecoveryRecoveryCI")

    reim_values = _row_macros(
        "REIM", baseline["REIM (ACT + Detector + Recovery)"]
    )
    values.update(reim_values)

    gain_value, gain_low, gain_high, rescued, harmed = gain
    rl_intervened = _int(
        baseline["ACT + Heuristic Recovery"], "Intervened Episodes"
    )
    reim_intervened = _int(
        baseline["REIM (ACT + Detector + Recovery)"], "Intervened Episodes"
    )
    if reim_intervened > rl_intervened or rl_intervened <= 0:
        intervention_reduction = (
            (rl_intervened - reim_intervened) / rl_intervened
            if rl_intervened
            else 0.0
        )
    else:
        intervention_reduction = (
            rl_intervened - reim_intervened
        ) / rl_intervened
    intervention_delta = (
        reim_intervened - rl_intervened
    ) / _int(
        baseline["ACT + Heuristic Recovery"], "Episodes", minimum=1
    )
    values.update(
        {
            "FinalREIMGain": _signed_pp(gain_value),
            "FinalREIMGainCI": _signed_pp_ci(gain_low, gain_high),
            "FinalREIMRescuedEpisodes": _tex_int(rescued),
            "FinalREIMHarmedEpisodes": _tex_int(harmed),
            "FinalREIMInterventionDeltaVsRL": _signed_pp(intervention_delta),
            "FinalREIMInterventionReductionVsRL": _percentage(
                intervention_reduction
            ),
            "FinalREIMFewerIntervenedEpisodes": _tex_int(
                rl_intervened - reim_intervened
            ),
            "FinalREIMGainVsReset": _signed_pp(gain_vs_reset[0]),
            "FinalREIMGainVsResetCI": _signed_pp_ci(
                gain_vs_reset[1], gain_vs_reset[2]
            ),
            "FinalREIMDeltaVsRL": _signed_pp(gain_vs_rl[0]),
            "FinalREIMDeltaVsRLCI": _signed_pp_ci(
                gain_vs_rl[1], gain_vs_rl[2]
            ),
        }
    )

    for variant in ("A", "B", "C", "D"):
        row = ablation[variant]
        attempts = _int(row, "Recovery Attempts")
        values[f"Ablation{variant}Success"] = _percentage(
            _float(row, "Success Rate")
        )
        values[f"Ablation{variant}Recovery"] = (
            _percentage(_float(row, "Recovery Rate"))
            if attempts
            else r"\NotApplicable"
        )
        values[f"Ablation{variant}Steps"] = f"{_float(row, 'Average Steps'):.1f}"

    for method, prefix in (
        ("ACT", "FinalACT"),
        ("REIM (ACT + Detector + Recovery)", "FinalREIM"),
    ):
        values[f"{prefix}SuccessAtZeroNoise"] = _percentage(
            _float(robustness[(method, 0.0)], "Success Rate")
        )
        values[f"{prefix}SuccessAtFortyNoise"] = _percentage(
            _float(robustness[(method, 0.4)], "Success Rate")
        )
    for level, suffix in (
        (0.1, "Ten"),
        (0.2, "Twenty"),
        (0.3, "Thirty"),
        (0.4, "Forty"),
    ):
        values[f"FinalREIMGainAt{suffix}Noise"] = _signed_pp(
            _float(
                robustness[("REIM (ACT + Detector + Recovery)", level)],
                "Success Rate",
            )
            - _float(robustness[("ACT", level)], "Success Rate")
        )

    gate_methods = matched_gate["methods"]
    gate_comparisons = matched_gate["paired_success_comparisons"]
    heuristic_gate = gate_methods["heuristic_gate"]
    gate_tau_0175 = gate_methods["reim_tau_0.175"]
    gate_tau_020 = gate_methods["reim_tau_0.20"]
    comparison_0175 = gate_comparisons["reim_tau_0.175_minus_heuristic"]
    comparison_020 = gate_comparisons["reim_tau_0.20_minus_heuristic"]
    intervention_comparison = matched_gate[
        "tau_0.175_minus_heuristic_intervention_rate"
    ]
    values.update(
        {
            "GateHeuristicSuccess": _percentage(
                _float(heuristic_gate, "success_rate")
            ),
            "GateHeuristicIntervention": _percentage(
                _float(heuristic_gate, "intervention_rate")
            ),
            "GateTauSeventeenFiveSuccess": _percentage(
                _float(gate_tau_0175, "success_rate")
            ),
            "GateTauSeventeenFiveIntervention": _percentage(
                _float(gate_tau_0175, "intervention_rate")
            ),
            "GateTauTwentySuccess": _percentage(
                _float(gate_tau_020, "success_rate")
            ),
            "GateTauTwentyIntervention": _percentage(
                _float(gate_tau_020, "intervention_rate")
            ),
            "GateTauSeventeenFiveGainVsHeuristic": _signed_pp(
                _float(comparison_0175, "paired_delta")
            ),
            "GateTauSeventeenFiveGainVsHeuristicCI": _signed_pp_ci(
                float(comparison_0175["paired_delta_bootstrap_95_ci"][0]),
                float(comparison_0175["paired_delta_bootstrap_95_ci"][1]),
            ),
            "GateTauSeventeenFiveP": (
                f"{_float(comparison_0175, 'exact_two_sided_mcnemar_binomial_p'):.4f}"
            ),
            "GateTauTwentyGainVsHeuristic": _signed_pp(
                _float(comparison_020, "paired_delta")
            ),
            "GateTauTwentyGainVsHeuristicCI": _signed_pp_ci(
                float(comparison_020["paired_delta_bootstrap_95_ci"][0]),
                float(comparison_020["paired_delta_bootstrap_95_ci"][1]),
            ),
            "GateTauTwentyP": (
                f"{_float(comparison_020, 'exact_two_sided_mcnemar_binomial_p'):.4f}"
            ),
            "GateTauSeventeenFiveInterventionDelta": _signed_pp(
                _float(intervention_comparison, "paired_delta")
            ),
            "GateTauSeventeenFiveInterventionDeltaCI": _signed_pp_ci(
                float(
                    intervention_comparison[
                        "paired_delta_bootstrap_95_ci"
                    ][0]
                ),
                float(
                    intervention_comparison[
                        "paired_delta_bootstrap_95_ci"
                    ][1]
                ),
            ),
        }
    )

    values.update(
        {
            "DetectorAccuracyHalf": _percentage(_float(detector, "accuracy")),
            "DetectorPrecisionHalf": _percentage(_float(detector, "precision")),
            "DetectorRecallHalf": _percentage(_float(detector, "recall")),
            "DetectorFoneHalf": _percentage(_float(detector, "f1")),
            "DetectorDiagnosticThreshold": f"{_float(deployment, 'threshold'):.2f}",
            "DetectorAccuracyDiagnostic": _percentage(
                _float(deployment, "accuracy")
            ),
            "DetectorPrecisionDiagnostic": _percentage(
                _float(deployment, "precision")
            ),
            "DetectorRecallDiagnostic": _percentage(_float(deployment, "recall")),
            "DetectorFoneDiagnostic": _percentage(_float(deployment, "f1")),
            "DetectorTNDiagnostic": _tex_int(_int(deployment, "tn")),
            "DetectorFPDiagnostic": _tex_int(_int(deployment, "fp")),
            "DetectorFNDiagnostic": _tex_int(_int(deployment, "fn")),
            "DetectorTPDiagnostic": _tex_int(_int(deployment, "tp")),
        }
    )
    operation_act = operation["act"]
    operation_reim = operation["reim"]
    values.update(
        {
            "OperationTraceSeed": _tex_int(_int(operation, "seed", minimum=1)),
            "OperationTraceNoise": _percentage(
                _float(operation, "noise_level", minimum=0.0)
            ),
            "OperationTraceThreshold": (
                f"{_float(operation, 'failure_threshold', minimum=0.0, maximum=1.0):.2f}"
            ),
            "OperationTraceACTSteps": _tex_int(
                _int(operation_act, "steps", minimum=1)
            ),
            "OperationTraceREIMSteps": _tex_int(
                _int(operation_reim, "steps", minimum=1)
            ),
            "OperationTraceRecoverySteps": _tex_int(
                _int(operation_reim, "recovery_steps", minimum=1)
            ),
        }
    )
    return values


STATIC_PROTOCOL_LINES = (
    r"\newcommand{\CurriculumTriggerThreshold}{0.10}",
    r"\newcommand{\FinalControllerTrigger}{0.20}",
    r"\newcommand{\FinalControllerRelease}{0}",
    r"\newcommand{\FinalControllerMinSteps}{150}",
    r"\newcommand{\FinalControllerBudget}{150}",
    r"\newcommand{\FinalControllerClearSteps}{200}",
    r"\newcommand{\PPOEpochsPerUpdate}{5}",
    r"\newcommand{\PPOLearningRate}{3\times10^{-5}}",
    r"\newcommand{\PPOClip}{0.10}",
    r"\newcommand{\PPOEntropyCoefficient}{0.0001}",
)


def _render_macros(source: str, values: Mapping[str, str]) -> str:
    for line in STATIC_PROTOCOL_LINES:
        if line not in source:
            raise PaperResultError(
                f"reim_macros.tex protocol drifted; missing literal: {line}"
            )
    rendered = source
    for name, value in values.items():
        pattern = re.compile(
            rf"^\\newcommand\{{\\{re.escape(name)}\}}\{{.*\}}$", re.MULTILINE
        )
        replacement = rf"\newcommand{{\{name}}}{{{value}}}"
        rendered, count = pattern.subn(lambda _: replacement, rendered)
        if count != 1:
            raise PaperResultError(
                f"reim_macros.tex must define \\{name} exactly once; found {count}"
            )
    switches = len(
        re.findall(r"^\\REIMFinalResults(?:true|false)$", rendered, re.MULTILINE)
    )
    if switches != 1:
        raise PaperResultError(
            "reim_macros.tex must contain exactly one final-results switch"
        )
    rendered = re.sub(
        r"^\\REIMFinalResults(?:true|false)$",
        r"\\REIMFinalResultstrue",
        rendered,
        flags=re.MULTILINE,
    )
    return rendered


def _baseline_table() -> str:
    return r"""\begin{table*}[t]
  \centering
  \caption{Paired comparison on the simulated Meta-World
  \REIMTask{} task. Every method uses the same \PlannedMainEpisodes{} seeds.
  Success and intervention outcomes are percentages with \ConfidenceLevel{}
  intervals. For Heuristic Recovery/REIM, recovery is task completion while
  recovery has control, divided by intervention count; Random Reset reports post-reset
  completion per reset and is not directly comparable.}
  \label{tab:final-baseline}
  \small
  \setlength{\tabcolsep}{5pt}
  \begin{tabular}{lcccc}
    \toprule
    Method & Task success $\uparrow$ & Intervention outcome $\uparrow$ &
    Intervention rate & Steps $\downarrow$ \\
    \midrule
    ACT &
    \FinalACTSuccess{} \FinalACTSuccessCI &
    \FinalACTRecovery{} &
    \FinalACTIntervention{} &
    \FinalACTSteps{} \\
    ACT + Random Reset &
    \FinalResetSuccess{} \FinalResetSuccessCI &
    \FinalResetRecovery{} \FinalResetRecoveryCI &
    \FinalResetIntervention{} &
    \FinalResetSteps{} \\
    ACT + Heuristic Recovery &
    \FinalRLRecoverySuccess{} \FinalRLRecoverySuccessCI &
    \FinalRLRecoveryRate{} \FinalRLRecoveryRateCI &
    \FinalRLRecoveryIntervention{} &
    \FinalRLRecoverySteps{} \\
    REIM (ours) &
    \FinalREIMSuccess{} \FinalREIMSuccessCI &
    \FinalREIMRecovery{} \FinalREIMRecoveryCI &
    \FinalREIMIntervention{} &
    \FinalREIMSteps{} \\
    \bottomrule
  \end{tabular}
  \vspace{1mm}\parbox{0.96\textwidth}{\footnotesize
  Paired REIM gain over ACT: \FinalREIMGain{}
  (\ConfidenceLevel{} CI \FinalREIMGainCI);
  \FinalREIMRescuedEpisodes{} ACT failures are rescued and
  \FinalREIMHarmedEpisodes{} ACT successes are harmed. REIM intervenes in
  \FinalREIMFewerIntervenedEpisodes{} fewer episodes than heuristic-gated recovery
  (\FinalREIMInterventionReductionVsRL{} relative reduction).}
\end{table*}
"""


def _ablation_table() -> str:
    return r"""\begin{table}[t]
  \centering
  \caption{Recovery and gate ablation under the paired protocol of
  Table~\ref{tab:final-baseline}. The heuristic and LSTM rows use the same
  trigger-aligned recovery actor, isolating the arbitration rule.}
  \label{tab:final-ablation}
  \small
  \setlength{\tabcolsep}{3.5pt}
  \begin{tabular}{lcc}
    \toprule
    Configuration & Success $\uparrow$ & Intervention \\
    \midrule
    ACT (no recovery) & \AblationASuccess & -- \\
    Heuristic gate + recovery & \AblationCSuccess &
      \FinalRLRecoveryIntervention \\
    REIM (LSTM gate) & \AblationDSuccess & \FinalREIMIntervention \\
    \bottomrule
  \end{tabular}
\end{table}
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_many(payloads: Mapping[Path, str]) -> None:
    temporary_paths: list[tuple[Path, Path]] = []
    try:
        for path, content in payloads.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.paper-results.tmp")
            temporary.write_text(content, encoding="utf-8")
            temporary_paths.append((temporary, path))
        for temporary, path in temporary_paths:
            temporary.replace(path)
    finally:
        for temporary, _ in temporary_paths:
            if temporary.exists():
                temporary.unlink()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    tables = PROJECT_ROOT / "results" / "tables"
    assets = PROJECT_ROOT / "paper_assets"
    parser.add_argument("--baseline", type=Path, default=tables / "baseline.csv")
    parser.add_argument(
        "--baseline-episodes",
        type=Path,
        default=tables / "baseline_episodes.csv",
    )
    parser.add_argument("--ablation", type=Path, default=tables / "ablation.csv")
    parser.add_argument(
        "--ablation-episodes",
        type=Path,
        default=tables / "ablation_episodes.csv",
    )
    parser.add_argument("--robustness", type=Path, default=tables / "robustness.csv")
    parser.add_argument(
        "--robustness-episodes",
        type=Path,
        default=tables / "robustness_episodes.csv",
    )
    parser.add_argument(
        "--detector-metrics",
        type=Path,
        default=tables / "detector_metrics.json",
    )
    parser.add_argument(
        "--operation-metadata",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "figures"
        / "recovery_operation_sequence.json",
    )
    parser.add_argument(
        "--matched-gate-audit",
        type=Path,
        default=tables / "gate_matched_comparison.json",
    )
    parser.add_argument("--comparison", type=Path, default=tables / "comparison.csv")
    parser.add_argument("--macros", type=Path, default=assets / "reim_macros.tex")
    parser.add_argument(
        "--table1", type=Path, default=assets / "Table1_final_baseline.tex"
    )
    parser.add_argument(
        "--table2", type=Path, default=assets / "Table2_final_ablation.tex"
    )
    parser.add_argument(
        "--expected-evaluation-seed", type=int, default=8_000_042
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="run every validation without modifying paper assets",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    baseline = _read_csv(args.baseline, SUMMARY_COLUMNS)
    baseline_raw = _read_csv(args.baseline_episodes, RAW_COLUMNS)
    ablation = _read_csv(args.ablation, ("Variant", *SUMMARY_COLUMNS))
    ablation_raw = _read_csv(
        args.ablation_episodes, (*RAW_COLUMNS, "variant")
    )
    robustness = _read_csv(args.robustness, SUMMARY_COLUMNS)
    robustness_raw = _read_csv(args.robustness_episodes, RAW_COLUMNS)

    baseline_index, ablation_index, paired = _validate_baseline_and_ablation(
        baseline=baseline,
        baseline_raw=baseline_raw,
        ablation=ablation,
        ablation_raw=ablation_raw,
        evaluation_seed=args.expected_evaluation_seed,
    )
    robustness_index = _validate_robustness(
        summaries=robustness,
        raw_rows=robustness_raw,
        evaluation_seed=args.expected_evaluation_seed,
    )
    detector, deployment = _validate_detector(args.detector_metrics)
    operation = _validate_operation_trace(args.operation_metadata)
    matched_gate = _validate_matched_gate(args.matched_gate_audit)
    paired_gain = _paired_gain(
        paired=paired,
        comparison_path=args.comparison,
        baseline_reim=baseline_index["REIM (ACT + Detector + Recovery)"],
        evaluation_seed=args.expected_evaluation_seed,
    )
    paired_gain_vs_reset = _paired_delta(
        paired=paired,
        contender="REIM (ACT + Detector + Recovery)",
        reference="ACT + Random Reset",
    )
    paired_gain_vs_rl = _paired_delta(
        paired=paired,
        contender="REIM (ACT + Detector + Recovery)",
        reference="ACT + Heuristic Recovery",
    )
    if not args.macros.is_file():
        raise PaperResultError(f"missing macro template: {args.macros}")
    macro_source = args.macros.read_text(encoding="utf-8")
    values = _build_macro_values(
        baseline=baseline_index,
        ablation=ablation_index,
        robustness=robustness_index,
        detector=detector,
        deployment=deployment,
        operation=operation,
        matched_gate=matched_gate,
        gain=paired_gain,
        gain_vs_reset=paired_gain_vs_reset,
        gain_vs_rl=paired_gain_vs_rl,
    )
    rendered_macros = _render_macros(macro_source, values)

    input_paths = (
        args.baseline,
        args.baseline_episodes,
        args.ablation,
        args.ablation_episodes,
        args.robustness,
        args.robustness_episodes,
        args.detector_metrics,
        args.operation_metadata,
        args.matched_gate_audit,
        *((args.comparison,) if args.comparison.is_file() else ()),
    )
    report = {
        "validated": True,
        "write_enabled": not args.check_only,
        "evaluation_seed_start": args.expected_evaluation_seed,
        "main_episodes_per_method": MAIN_EPISODES,
        "robustness_episodes_per_condition": ROBUSTNESS_EPISODES,
        "deployment_threshold": DEPLOYMENT_THRESHOLD,
        "recovery_definition": FINAL_RECOVERY_DEFINITION,
        "operation_trace_seed": _int(operation, "seed", minimum=1),
        "operation_trace_schema": str(operation["schema_version"]),
        "matched_gate_episode_seed_start": _int(
            matched_gate["protocol"], "episode_seed_start", minimum=1
        ),
        "matched_gate_episode_bank_sha256": str(
            matched_gate["protocol"]["episode_bank_sha256"]
        ),
        "paired_reim_gain": paired_gain[0],
        "paired_reim_gain_ci": [paired_gain[1], paired_gain[2]],
        "paired_reim_gain_vs_random_reset": paired_gain_vs_reset[0],
        "paired_reim_gain_vs_random_reset_ci": [
            paired_gain_vs_reset[1],
            paired_gain_vs_reset[2],
        ],
        "paired_reim_delta_vs_rl": paired_gain_vs_rl[0],
        "paired_reim_delta_vs_rl_ci": [
            paired_gain_vs_rl[1],
            paired_gain_vs_rl[2],
        ],
        "rescued_episodes": paired_gain[3],
        "harmed_episodes": paired_gain[4],
        "reim_intervened_episodes": _int(
            baseline_index["REIM (ACT + Detector + Recovery)"],
            "Intervened Episodes",
        ),
        "rl_intervened_episodes": _int(
            baseline_index["ACT + Heuristic Recovery"],
            "Intervened Episodes",
        ),
        "inputs": {str(path.resolve()): _sha256(path) for path in input_paths},
    }
    if args.check_only:
        print(json.dumps(report, indent=2))
        return 0

    table1 = _baseline_table()
    table2 = _ablation_table()
    manifest_path = args.macros.parent / "paper_results_manifest.json"
    payloads = {
        args.macros: rendered_macros,
        args.table1: table1,
        args.table2: table2,
        args.macros.parent / "Table1_baseline.tex": table1,
        args.macros.parent / "Table2_ablation.tex": table2,
        manifest_path: json.dumps(report, indent=2) + "\n",
    }
    _atomic_write_many(payloads)
    print(json.dumps({**report, "outputs": [str(path) for path in payloads]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PaperResultError, json.JSONDecodeError) as error:
        print(f"paper result validation failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
