"""Publication-facing metrics for Meta-World multi-task evaluations.

The module deliberately keeps two notions of aggregation separate:

``micro``
    Every episode receives equal weight.
``task_macro``
    Every task receives equal weight, irrespective of its episode count.

Recovery success is a ratio of successful recoveries to interventions, not a
ratio of episodes.  A task with no intervention has an undefined recovery
rate and is excluded from the task-macro recovery average; the returned
``recovery_eligible_task_count`` makes that denominator explicit.

All public functions accept ordinary mappings so that raw ``csv.DictReader``
rows can be consumed without a pandas dependency.  Validation is intentionally
strict: malformed counts, duplicate paired episode identifiers, or incomplete
paired method banks raise instead of silently changing the estimand.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from evaluation.metrics import wilson_score_interval


REQUIRED_FIELDS = frozenset(
    {
        "benchmark",
        "task_name",
        "task_id",
        "method",
        "success",
        "intervention_count",
        "recovery_success",
        "steps",
        "paired_episode_id",
    }
)

_BOOTSTRAP_METRICS = frozenset(
    {
        "success",
        "intervention",
        "intervention_count",
        "recovery_success",
        "steps",
    }
)


@dataclass(frozen=True, slots=True)
class _Episode:
    benchmark: str
    task_name: str
    task_id: str
    method: str
    success: bool
    intervention_count: int
    recovery_success: int
    steps: float
    paired_episode_id: str

    @property
    def task_key(self) -> tuple[str, str]:
        return self.task_id, self.task_name


def _required_text(value: Any, *, field: str, row_number: int) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"row {row_number}: {field} must be non-empty")
    return text


def _binary(value: Any, *, field: str, row_number: int) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes"}:
            return True
        if normalized in {"0", "false", "no"}:
            return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if math.isfinite(numeric) and numeric in {0.0, 1.0}:
            return bool(int(numeric))
    raise ValueError(f"row {row_number}: {field} must be binary, got {value!r}")


def _nonnegative_integer(value: Any, *, field: str, row_number: int) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"row {row_number}: {field} must be a non-negative integer"
        ) from exc
    if not math.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
        raise ValueError(
            f"row {row_number}: {field} must be a non-negative integer"
        )
    return int(numeric)


def _nonnegative_float(value: Any, *, field: str, row_number: int) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"row {row_number}: {field} must be a non-negative finite number"
        ) from exc
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(
            f"row {row_number}: {field} must be a non-negative finite number"
        )
    return numeric


def _parse_records(
    records: Iterable[Mapping[str, Any]],
    *,
    context: str = "records",
) -> list[_Episode]:
    parsed: list[_Episode] = []
    task_names_by_identity: dict[tuple[str, str], str] = {}
    for row_number, row in enumerate(records, start=1):
        if not isinstance(row, Mapping):
            raise TypeError(f"{context} row {row_number} is not a mapping")
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            raise ValueError(
                f"{context} row {row_number}: missing fields {sorted(missing)}"
            )

        benchmark = _required_text(
            row["benchmark"], field="benchmark", row_number=row_number
        )
        task_name = _required_text(
            row["task_name"], field="task_name", row_number=row_number
        )
        task_id = _required_text(row["task_id"], field="task_id", row_number=row_number)
        method = _required_text(row["method"], field="method", row_number=row_number)
        paired_episode_id = _required_text(
            row["paired_episode_id"],
            field="paired_episode_id",
            row_number=row_number,
        )
        success = _binary(row["success"], field="success", row_number=row_number)
        intervention_count = _nonnegative_integer(
            row["intervention_count"],
            field="intervention_count",
            row_number=row_number,
        )
        recovery_success = _nonnegative_integer(
            row["recovery_success"],
            field="recovery_success",
            row_number=row_number,
        )
        if recovery_success > intervention_count:
            raise ValueError(
                f"{context} row {row_number}: recovery_success cannot exceed "
                "intervention_count"
            )
        steps = _nonnegative_float(row["steps"], field="steps", row_number=row_number)

        identity = (benchmark, task_id)
        previous_name = task_names_by_identity.setdefault(identity, task_name)
        if previous_name != task_name:
            raise ValueError(
                f"{context} row {row_number}: task_id {task_id!r} maps to both "
                f"{previous_name!r} and {task_name!r}"
            )

        parsed.append(
            _Episode(
                benchmark=benchmark,
                task_name=task_name,
                task_id=task_id,
                method=method,
                success=success,
                intervention_count=intervention_count,
                recovery_success=recovery_success,
                steps=steps,
                paired_episode_id=paired_episode_id,
            )
        )
    if not parsed:
        raise ValueError(f"cannot aggregate empty {context}")
    return parsed


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values, dtype=np.float64)) if values else None


def _task_sort_key(task_key: tuple[str, str]) -> tuple[int, int | str, str]:
    """Sort integer-like official task ids numerically, then textual ids."""

    task_id, task_name = task_key
    try:
        numeric_id = int(task_id)
    except ValueError:
        return 1, task_id, task_name
    if str(numeric_id) == task_id or task_id == f"+{numeric_id}":
        return 0, numeric_id, task_name
    return 1, task_id, task_name


def _per_task_row(
    rows: Sequence[_Episode],
    *,
    confidence: float,
) -> dict[str, Any]:
    successes = np.asarray([row.success for row in rows], dtype=np.float64)
    interventions = np.asarray(
        [row.intervention_count for row in rows], dtype=np.float64
    )
    recovery_successes = np.asarray(
        [row.recovery_success for row in rows], dtype=np.float64
    )
    steps = np.asarray([row.steps for row in rows], dtype=np.float64)
    intervened = interventions > 0.0

    episode_count = len(rows)
    success_count = int(successes.sum())
    success_ci = wilson_score_interval(
        success_count,
        episode_count,
        confidence=confidence,
    )
    intervention_count = int(interventions.sum())
    recovery_success_count = int(recovery_successes.sum())
    intervened_episode_count = int(intervened.sum())
    post_intervention_success_count = int(successes[intervened].sum())

    return {
        "benchmark": rows[0].benchmark,
        "method": rows[0].method,
        "task_id": rows[0].task_id,
        "task_name": rows[0].task_name,
        "episode_count": episode_count,
        "success_count": success_count,
        "success_rate": float(successes.mean()),
        "success_ci_lower": float(success_ci[0]),
        "success_ci_upper": float(success_ci[1]),
        "intervened_episode_count": intervened_episode_count,
        "intervention_episode_rate": float(intervened.mean()),
        "intervention_count": intervention_count,
        "interventions_per_episode": float(interventions.mean()),
        "recovery_success_count": recovery_success_count,
        "recovery_success_rate": (
            recovery_success_count / intervention_count
            if intervention_count > 0
            else None
        ),
        "post_intervention_success_count": post_intervention_success_count,
        "post_intervention_success_rate": (
            post_intervention_success_count / intervened_episode_count
            if intervened_episode_count > 0
            else None
        ),
        "mean_steps": float(steps.mean()),
        "median_steps": float(np.median(steps)),
    }


def aggregate_multitask_metrics(
    records: Iterable[Mapping[str, Any]],
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Aggregate one benchmark-method episode collection.

    ``success_rate_worst_quartile`` is the arithmetic mean of the lowest
    ``ceil(task_count / 4)`` per-task success rates.  This definition remains
    meaningful for small suites and is less unstable than selecting a sample
    quantile without reporting the underlying task count.

    Args:
        records: Episode mappings containing :data:`REQUIRED_FIELDS`.
        confidence: Confidence level for per-task Wilson success intervals.

    Returns:
        A dictionary with ``summary`` and sorted ``per_task`` records.

    Raises:
        ValueError: If records are empty, malformed, or mix benchmarks/methods.
    """

    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    rows = _parse_records(records)
    benchmarks = {row.benchmark for row in rows}
    methods = {row.method for row in rows}
    if len(benchmarks) != 1 or len(methods) != 1:
        raise ValueError(
            "aggregate_multitask_metrics requires exactly one benchmark and method; "
            "use aggregate_multitask_results for mixed records"
        )

    episode_keys: set[tuple[tuple[str, str], str]] = set()
    for row in rows:
        episode_key = (row.task_key, row.paired_episode_id)
        if episode_key in episode_keys:
            raise ValueError(
                f"duplicate paired_episode_id {row.paired_episode_id!r} "
                f"for task {row.task_name!r}"
            )
        episode_keys.add(episode_key)

    grouped: dict[tuple[str, str], list[_Episode]] = defaultdict(list)
    for row in rows:
        grouped[row.task_key].append(row)
    per_task = [
        _per_task_row(grouped[key], confidence=confidence)
        for key in sorted(grouped, key=_task_sort_key)
    ]

    task_success_rates = [float(row["success_rate"]) for row in per_task]
    task_intervention_rates = [
        float(row["intervention_episode_rate"]) for row in per_task
    ]
    task_interventions_per_episode = [
        float(row["interventions_per_episode"]) for row in per_task
    ]
    task_recovery_rates = [
        float(row["recovery_success_rate"])
        for row in per_task
        if row["recovery_success_rate"] is not None
    ]
    task_post_intervention_rates = [
        float(row["post_intervention_success_rate"])
        for row in per_task
        if row["post_intervention_success_rate"] is not None
    ]
    task_mean_steps = [float(row["mean_steps"]) for row in per_task]

    success_count = sum(int(row["success_count"]) for row in per_task)
    intervention_count = sum(int(row["intervention_count"]) for row in per_task)
    recovery_success_count = sum(
        int(row["recovery_success_count"]) for row in per_task
    )
    intervened_episode_count = sum(
        int(row["intervened_episode_count"]) for row in per_task
    )
    post_intervention_success_count = sum(
        int(row["post_intervention_success_count"]) for row in per_task
    )
    task_count = len(per_task)
    episode_count = len(rows)
    worst_task_count = max(1, math.ceil(task_count / 4))
    worst_rates = sorted(task_success_rates)[:worst_task_count]

    summary = {
        "benchmark": rows[0].benchmark,
        "method": rows[0].method,
        "task_count": task_count,
        "episode_count": episode_count,
        "success_count": success_count,
        "success_rate_micro": success_count / episode_count,
        "success_rate_task_macro": float(np.mean(task_success_rates)),
        "success_rate_task_median": float(np.median(task_success_rates)),
        "success_rate_worst_quartile": float(np.mean(worst_rates)),
        "worst_quartile_task_count": worst_task_count,
        "intervened_episode_count": intervened_episode_count,
        "intervention_episode_rate_micro": intervened_episode_count / episode_count,
        "intervention_episode_rate_task_macro": float(
            np.mean(task_intervention_rates)
        ),
        "intervention_count": intervention_count,
        "interventions_per_episode_micro": intervention_count / episode_count,
        "interventions_per_episode_task_macro": float(
            np.mean(task_interventions_per_episode)
        ),
        "recovery_success_count": recovery_success_count,
        "recovery_success_rate_micro": (
            recovery_success_count / intervention_count
            if intervention_count > 0
            else None
        ),
        "recovery_success_rate_task_macro": _mean_or_none(task_recovery_rates),
        "recovery_eligible_task_count": len(task_recovery_rates),
        "post_intervention_success_count": post_intervention_success_count,
        "post_intervention_success_rate_micro": (
            post_intervention_success_count / intervened_episode_count
            if intervened_episode_count > 0
            else None
        ),
        "post_intervention_success_rate_task_macro": _mean_or_none(
            task_post_intervention_rates
        ),
        "post_intervention_eligible_task_count": len(
            task_post_intervention_rates
        ),
        "mean_steps_micro": float(np.mean([row.steps for row in rows])),
        "mean_steps_task_macro": float(np.mean(task_mean_steps)),
        "median_task_mean_steps": float(np.median(task_mean_steps)),
    }
    return {"summary": summary, "per_task": per_task}


def aggregate_multitask_results(
    records: Iterable[Mapping[str, Any]],
    *,
    confidence: float = 0.95,
) -> list[dict[str, Any]]:
    """Group mixed records by benchmark and method, then aggregate each group."""

    rows = list(records)
    if not rows:
        raise ValueError("cannot aggregate empty records")
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise TypeError(f"records row {row_number} is not a mapping")
        missing = {"benchmark", "method"} - set(row)
        if missing:
            raise ValueError(
                f"records row {row_number}: missing fields {sorted(missing)}"
            )
        key = (
            _required_text(row["benchmark"], field="benchmark", row_number=row_number),
            _required_text(row["method"], field="method", row_number=row_number),
        )
        groups[key].append(row)
    return [
        aggregate_multitask_metrics(groups[key], confidence=confidence)
        for key in sorted(groups)
    ]


def _metric_value(row: _Episode, metric: str) -> float:
    if metric == "success":
        return float(row.success)
    if metric == "intervention":
        return float(row.intervention_count > 0)
    if metric == "intervention_count":
        return float(row.intervention_count)
    if metric == "recovery_success":
        return float(row.recovery_success)
    if metric == "steps":
        return row.steps
    raise ValueError(
        f"unsupported metric {metric!r}; expected one of {sorted(_BOOTSTRAP_METRICS)}"
    )


def _paired_index(
    rows: Sequence[_Episode],
    *,
    context: str,
) -> dict[tuple[tuple[str, str], str], _Episode]:
    indexed: dict[tuple[tuple[str, str], str], _Episode] = {}
    for row in rows:
        key = (row.task_key, row.paired_episode_id)
        if key in indexed:
            raise ValueError(
                f"{context}: duplicate paired_episode_id {row.paired_episode_id!r} "
                f"for task {row.task_name!r}"
            )
        indexed[key] = row
    return indexed


def paired_task_stratified_bootstrap_delta(
    reference_records: Iterable[Mapping[str, Any]],
    candidate_records: Iterable[Mapping[str, Any]],
    *,
    metric: str = "success",
    confidence: float = 0.95,
    n_bootstrap: int = 2_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Estimate a paired candidate-minus-reference task-macro difference.

    Episode pairs are matched exactly by ``(task_id, task_name,
    paired_episode_id)``.  Each bootstrap replicate resamples paired episode
    differences *within every task*, then gives every task equal weight.  Thus
    an easy task with more recorded episodes cannot dominate the result and
    paired common-random-number structure is retained.

    The official benchmark task set is treated as fixed; this interval captures
    held-out episode/configuration uncertainty within those task strata.  Model
    seed uncertainty should be handled one level above this function.

    Supported episode-level metrics are ``success``, ``intervention`` (whether
    an episode was intervened on), ``intervention_count``, ``recovery_success``,
    and ``steps``.
    """

    if metric not in _BOOTSTRAP_METRICS:
        raise ValueError(
            f"unsupported metric {metric!r}; expected one of "
            f"{sorted(_BOOTSTRAP_METRICS)}"
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if isinstance(n_bootstrap, bool) or int(n_bootstrap) != n_bootstrap:
        raise ValueError("n_bootstrap must be a positive integer")
    n_bootstrap = int(n_bootstrap)
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer")

    reference = _parse_records(reference_records, context="reference_records")
    candidate = _parse_records(candidate_records, context="candidate_records")
    reference_benchmarks = {row.benchmark for row in reference}
    candidate_benchmarks = {row.benchmark for row in candidate}
    reference_methods = {row.method for row in reference}
    candidate_methods = {row.method for row in candidate}
    if len(reference_benchmarks) != 1 or len(candidate_benchmarks) != 1:
        raise ValueError("each paired input must contain exactly one benchmark")
    if reference_benchmarks != candidate_benchmarks:
        raise ValueError("paired inputs must use the same benchmark")
    if len(reference_methods) != 1 or len(candidate_methods) != 1:
        raise ValueError("each paired input must contain exactly one method")

    reference_index = _paired_index(reference, context="reference_records")
    candidate_index = _paired_index(candidate, context="candidate_records")
    reference_keys = set(reference_index)
    candidate_keys = set(candidate_index)
    if reference_keys != candidate_keys:
        missing_candidate = len(reference_keys - candidate_keys)
        missing_reference = len(candidate_keys - reference_keys)
        raise ValueError(
            "paired inputs must have identical task/paired_episode_id keys "
            f"(missing candidate={missing_candidate}, "
            f"missing reference={missing_reference})"
        )

    task_differences: dict[tuple[str, str], list[float]] = defaultdict(list)
    task_reference_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    task_candidate_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for key in sorted(
        reference_keys,
        key=lambda item: (_task_sort_key(item[0]), item[1]),
    ):
        task_key, _ = key
        reference_value = _metric_value(reference_index[key], metric)
        candidate_value = _metric_value(candidate_index[key], metric)
        task_reference_values[task_key].append(reference_value)
        task_candidate_values[task_key].append(candidate_value)
        task_differences[task_key].append(candidate_value - reference_value)

    ordered_tasks = sorted(task_differences, key=_task_sort_key)
    per_task: list[dict[str, Any]] = []
    for task_id, task_name in ordered_tasks:
        task_key = (task_id, task_name)
        reference_values = np.asarray(
            task_reference_values[task_key], dtype=np.float64
        )
        candidate_values = np.asarray(
            task_candidate_values[task_key], dtype=np.float64
        )
        per_task.append(
            {
                "task_id": task_id,
                "task_name": task_name,
                "pair_count": int(reference_values.size),
                "reference_mean": float(reference_values.mean()),
                "candidate_mean": float(candidate_values.mean()),
                "delta": float((candidate_values - reference_values).mean()),
            }
        )

    reference_task_macro = float(
        np.mean([row["reference_mean"] for row in per_task])
    )
    candidate_task_macro = float(
        np.mean([row["candidate_mean"] for row in per_task])
    )
    point_delta = candidate_task_macro - reference_task_macro

    if n_bootstrap == 1:
        estimates = np.asarray([point_delta], dtype=np.float64)
    else:
        rng = np.random.default_rng(seed)
        estimates = np.empty(n_bootstrap, dtype=np.float64)
        chunk_size = min(512, n_bootstrap)
        for start in range(0, n_bootstrap, chunk_size):
            stop = min(start + chunk_size, n_bootstrap)
            chunk_estimates = np.zeros(stop - start, dtype=np.float64)
            for task_key in ordered_tasks:
                differences = np.asarray(
                    task_differences[task_key], dtype=np.float64
                )
                sample_ids = rng.integers(
                    0,
                    differences.size,
                    size=(stop - start, differences.size),
                )
                chunk_estimates += differences[sample_ids].mean(axis=1)
            estimates[start:stop] = chunk_estimates / len(ordered_tasks)

    alpha = (1.0 - confidence) / 2.0
    ci_lower, ci_upper = np.quantile(estimates, [alpha, 1.0 - alpha])
    return {
        "benchmark": next(iter(reference_benchmarks)),
        "reference_method": next(iter(reference_methods)),
        "candidate_method": next(iter(candidate_methods)),
        "metric": metric,
        "estimand": "candidate_minus_reference_task_macro",
        "stratification": "paired_episodes_resampled_within_fixed_tasks",
        "task_count": len(ordered_tasks),
        "pair_count": len(reference_keys),
        "reference_task_macro": reference_task_macro,
        "candidate_task_macro": candidate_task_macro,
        "delta": point_delta,
        "confidence": confidence,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "bootstrap_standard_error": (
            float(estimates.std(ddof=1)) if estimates.size > 1 else 0.0
        ),
        "n_bootstrap": n_bootstrap,
        "seed": int(seed),
        "per_task": per_task,
    }


# Concise alias for callers that use the statistical operation before the
# implementation detail in the function name.
stratified_paired_bootstrap_delta = paired_task_stratified_bootstrap_delta


__all__ = [
    "REQUIRED_FIELDS",
    "aggregate_multitask_metrics",
    "aggregate_multitask_results",
    "paired_task_stratified_bootstrap_delta",
    "stratified_paired_bootstrap_delta",
]
