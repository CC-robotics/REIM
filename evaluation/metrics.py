"""Metrics shared by REIM evaluations and experiments.

The project stores rates as fractions in ``[0, 1]``.  Paper tables and plots
convert them to percentages only at presentation time.  Binary task success
uses Wilson score intervals; continuous and ratio metrics use percentile
bootstrap intervals over episodes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import NormalDist
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


@dataclass(slots=True)
class EpisodeMetrics:
    """One evaluation episode.

    ``recovery_attempts`` counts controller interventions.  For learned
    recovery options, ``recovery_successes`` follows ``recovery_definition``;
    for the published task-completion hold protocol it counts task completion
    while recovery has control, and for Random Reset it counts post-reset
    completion.  The aggregate output records this definition explicitly and
    also reports the common post-intervention episode outcome.
    """

    method: str
    episode: int
    seed: int
    backend: str
    success: bool
    steps: int
    elapsed_seconds: float
    recovery_attempts: int = 0
    recovery_successes: int = 0
    detector_triggers: int = 0
    recovery_steps: int = 0
    failure_probability_max: float = 0.0
    recovery_definition: str = ""
    terminated: bool = False
    truncated: bool = False
    failure_reason: str = ""
    episode_specification_sha256: str = ""
    episode_bank_sha256: str = ""
    metaworld_task_sha256: str = ""
    retry_specification_sha256s: str = ""
    retry_task_sha256s: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_1d(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 2_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap confidence interval.

    Empty input returns ``(nan, nan)`` and a singleton input returns the point
    estimate twice.  Resampling is performed in chunks to bound memory use for
    full 1,000-episode evaluations.
    """

    data = _finite_1d(values)
    if data.size == 0:
        return math.nan, math.nan
    if data.size == 1 or n_bootstrap <= 1:
        point = float(statistic(data))
        return point, point
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    rng = np.random.default_rng(seed)
    estimates = np.empty(int(n_bootstrap), dtype=np.float64)
    chunk_size = min(512, int(n_bootstrap))
    for start in range(0, int(n_bootstrap), chunk_size):
        stop = min(start + chunk_size, int(n_bootstrap))
        sample_ids = rng.integers(0, data.size, size=(stop - start, data.size))
        for offset, sample in enumerate(data[sample_ids]):
            estimates[start + offset] = float(statistic(sample))

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def bootstrap_ratio_ci(
    numerators: Sequence[float] | np.ndarray,
    denominators: Sequence[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 2_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap a ratio of episode-level sums.

    This is used for recovery rate so episodes with multiple interventions are
    weighted by their number of interventions.  A dataset with no attempted
    recoveries has a defined rate and interval of zero.
    """

    num = np.asarray(numerators, dtype=np.float64).reshape(-1)
    den = np.asarray(denominators, dtype=np.float64).reshape(-1)
    if num.shape != den.shape:
        raise ValueError("numerators and denominators must have equal shape")
    valid = np.isfinite(num) & np.isfinite(den)
    num, den = num[valid], den[valid]
    if num.size == 0 or float(den.sum()) <= 0.0:
        return 0.0, 0.0
    if num.size == 1 or n_bootstrap <= 1:
        point = float(num.sum() / den.sum())
        return point, point

    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(int(n_bootstrap)):
        ids = rng.integers(0, num.size, size=num.size)
        denominator = float(den[ids].sum())
        if denominator > 0.0:
            estimates.append(float(num[ids].sum() / denominator))
    if not estimates:
        return 0.0, 0.0
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def wilson_score_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion."""

    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials and trials > 0")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _get(record: EpisodeMetrics | Mapping[str, Any], key: str, default: Any = 0) -> Any:
    if isinstance(record, EpisodeMetrics):
        return getattr(record, key, default)
    return record.get(key, default)


def _truth(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def aggregate_episode_metrics(
    records: Iterable[EpisodeMetrics | Mapping[str, Any]],
    *,
    method: str | None = None,
    confidence: float = 0.95,
    n_bootstrap: int = 2_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Aggregate episode records into one publication-ready result row."""

    rows = list(records)
    if not rows:
        raise ValueError("cannot aggregate an empty episode collection")

    successes = np.asarray(
        [_truth(_get(row, "success")) for row in rows], dtype=np.float64
    )
    steps = np.asarray([float(_get(row, "steps")) for row in rows], dtype=np.float64)
    elapsed = np.asarray(
        [float(_get(row, "elapsed_seconds", 0.0)) for row in rows], dtype=np.float64
    )
    attempts = np.asarray(
        [float(_get(row, "recovery_attempts", 0)) for row in rows], dtype=np.float64
    )
    recovery_successes = np.asarray(
        [float(_get(row, "recovery_successes", 0)) for row in rows], dtype=np.float64
    )
    triggers = np.asarray(
        [float(_get(row, "detector_triggers", 0)) for row in rows], dtype=np.float64
    )

    success_ci = wilson_score_interval(
        int(successes.sum()),
        len(successes),
        confidence=confidence,
    )
    steps_ci = bootstrap_ci(
        steps,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        seed=seed + 1,
    )
    recovery_ci = bootstrap_ratio_ci(
        recovery_successes,
        attempts,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        seed=seed + 2,
    )
    recovery_rate = (
        float(recovery_successes.sum() / attempts.sum()) if attempts.sum() > 0 else 0.0
    )
    resolved_method = method or str(_get(rows[0], "method", "unknown"))
    intervened = attempts > 0
    intervened_episodes = int(intervened.sum())
    intervened_successes = int(successes[intervened].sum())
    intervention_task_rate = (
        intervened_successes / intervened_episodes
        if intervened_episodes
        else None
    )
    intervention_task_ci = (
        wilson_score_interval(
            intervened_successes,
            intervened_episodes,
            confidence=confidence,
        )
        if intervened_episodes
        else (None, None)
    )
    explicit_definitions = {
        str(_get(row, "recovery_definition", "")).strip()
        for row in rows
        if str(_get(row, "recovery_definition", "")).strip()
    }
    if attempts.sum() <= 0:
        recovery_definition = "not_applicable"
    elif len(explicit_definitions) == 1:
        recovery_definition = explicit_definitions.pop()
    elif len(explicit_definitions) > 1:
        recovery_definition = "mixed_intervention_outcomes"
    elif "Random Reset" in resolved_method:
        recovery_definition = "post_reset_task_success_per_reset"
    else:
        recovery_definition = "risk_clear_return_to_ACT_per_intervention"
    backends = sorted(
        {
            str(_get(row, "backend", "unknown"))
            for row in rows
            if str(_get(row, "backend", "unknown"))
        }
    )
    backend = backends[0] if len(backends) == 1 else "mixed"
    return {
        "Method": resolved_method,
        "Success Rate": float(successes.mean()),
        "Success CI Lower": success_ci[0],
        "Success CI Upper": success_ci[1],
        "Recovery Rate": recovery_rate,
        "Recovery CI Lower": recovery_ci[0],
        "Recovery CI Upper": recovery_ci[1],
        "Recovery Definition": recovery_definition,
        "Post-Intervention Task Success Rate": intervention_task_rate,
        "Post-Intervention Task Success CI Lower": intervention_task_ci[0],
        "Post-Intervention Task Success CI Upper": intervention_task_ci[1],
        "Average Steps": float(steps.mean()),
        "Steps CI Lower": steps_ci[0],
        "Steps CI Upper": steps_ci[1],
        "Average Time (s)": float(elapsed.mean()),
        "Episodes": int(len(rows)),
        "Successes": int(successes.sum()),
        "Recovery Attempts": int(attempts.sum()),
        "Recovery Successes": int(recovery_successes.sum()),
        "Detector Triggers": int(triggers.sum()),
        "Intervened Episodes": intervened_episodes,
        "Post-Intervention Task Successes": intervened_successes,
        "Backend": backend,
        # Only the CLI layer knows whether the full episode protocol was used.
        # It may promote this flag after checking backend *and* profile.
        "Benchmark Eligible": False,
    }


def binary_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
) -> dict[str, float | int | list[list[int]]]:
    """Compute detector metrics without depending on scikit-learn."""

    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if truth.shape != pred.shape or truth.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and have equal shape")
    if not np.isin(truth, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
        raise ValueError("binary metrics only accept labels 0 and 1")

    tn = int(np.sum((truth == 0) & (pred == 0)))
    fp = int(np.sum((truth == 0) & (pred == 1)))
    fn = int(np.sum((truth == 1) & (pred == 0)))
    tp = int(np.sum((truth == 1) & (pred == 1)))
    total = truth.size
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
