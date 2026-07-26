"""Tests for publication-facing evaluation metric semantics."""

from __future__ import annotations

import numpy as np

from evaluation.metrics import (
    EpisodeMetrics,
    aggregate_episode_metrics,
    binary_classification_metrics,
    bootstrap_ci,
    wilson_score_interval,
)


def test_binary_classification_metrics_match_known_confusion_matrix() -> None:
    result = binary_classification_metrics(
        [0, 0, 0, 1, 1, 1],
        [0, 1, 0, 1, 0, 1],
    )
    assert result["confusion_matrix"] == [[2, 1], [1, 2]]
    assert result["accuracy"] == 4 / 6
    assert result["precision"] == 2 / 3
    assert result["recall"] == 2 / 3
    assert result["f1"] == 2 / 3


def test_episode_aggregation_uses_micro_recovery_rate() -> None:
    records = [
        EpisodeMetrics(
            method="REIM",
            episode=0,
            seed=10,
            backend="toy",
            success=True,
            steps=20,
            elapsed_seconds=0.1,
            recovery_attempts=1,
            recovery_successes=1,
            detector_triggers=1,
        ),
        EpisodeMetrics(
            method="REIM",
            episode=1,
            seed=11,
            backend="toy",
            success=False,
            steps=40,
            elapsed_seconds=0.3,
            recovery_attempts=3,
            recovery_successes=1,
            detector_triggers=3,
        ),
    ]
    result = aggregate_episode_metrics(records, n_bootstrap=20, seed=4)
    assert result["Method"] == "REIM"
    assert result["Success Rate"] == 0.5
    assert result["Recovery Rate"] == 0.5  # 2 successful / 4 attempted
    assert result["Average Steps"] == 30.0
    assert result["Recovery Attempts"] == 4
    assert result["Detector Triggers"] == 4
    assert result["Backend"] == "toy"
    assert result["Benchmark Eligible"] is False


def test_bootstrap_interval_is_seed_deterministic_and_contains_point_estimate() -> None:
    values = np.asarray([0.0, 0.0, 1.0, 1.0])
    first = bootstrap_ci(values, n_bootstrap=200, seed=123)
    second = bootstrap_ci(values, n_bootstrap=200, seed=123)
    assert first == second
    assert first[0] <= values.mean() <= first[1]


def test_wilson_interval_is_informative_at_boundary() -> None:
    lower, upper = wilson_score_interval(200, 200)
    assert 0.98 < lower < 1.0
    assert upper == 1.0


def test_aggregate_csv_boolean_strings_are_parsed() -> None:
    rows = [
        {
            "method": "ACT",
            "success": "False",
            "steps": "10",
            "elapsed_seconds": "0.1",
            "backend": "metaworld",
        },
        {
            "method": "ACT",
            "success": "True",
            "steps": "20",
            "elapsed_seconds": "0.2",
            "backend": "metaworld",
        },
    ]
    result = aggregate_episode_metrics(rows, n_bootstrap=20)
    assert result["Success Rate"] == 0.5
