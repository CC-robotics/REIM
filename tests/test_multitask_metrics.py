"""Tests for task-balanced multi-task evaluation metrics."""

from __future__ import annotations

import pytest

from evaluation.multitask_metrics import (
    aggregate_multitask_metrics,
    aggregate_multitask_results,
    paired_task_stratified_bootstrap_delta,
)


def _row(
    task_id: int,
    task_name: str,
    episode: int,
    *,
    method: str = "MT-REIM",
    success: bool = False,
    interventions: int = 0,
    recoveries: int = 0,
    steps: int = 100,
) -> dict[str, object]:
    return {
        "benchmark": "MT10",
        "task_name": task_name,
        "task_id": task_id,
        "method": method,
        "success": success,
        "intervention_count": interventions,
        "recovery_success": recoveries,
        "steps": steps,
        "paired_episode_id": f"{task_name}-{episode}",
    }


def test_aggregation_separates_micro_macro_median_and_worst_quartile() -> None:
    rows = [
        # Unequal task episode counts deliberately make micro != task-macro.
        _row(0, "a", 0, success=True),
        _row(0, "a", 1, success=True, interventions=1, recoveries=1),
        _row(0, "a", 2, success=True, interventions=2, recoveries=1),
        _row(0, "a", 3, success=True),
        _row(1, "b", 0, success=False, interventions=1),
        _row(1, "b", 1, success=False),
        _row(2, "c", 0, success=False),
        _row(2, "c", 1, success=True, interventions=1, recoveries=1),
        _row(3, "d", 0, success=True),
        _row(3, "d", 1, success=True),
    ]

    result = aggregate_multitask_metrics(rows)
    summary = result["summary"]

    assert summary["task_count"] == 4
    assert summary["episode_count"] == 10
    assert summary["success_rate_micro"] == pytest.approx(0.7)
    assert summary["success_rate_task_macro"] == pytest.approx(0.625)
    assert summary["success_rate_task_median"] == pytest.approx(0.75)
    assert summary["worst_quartile_task_count"] == 1
    assert summary["success_rate_worst_quartile"] == pytest.approx(0.0)

    assert summary["intervention_episode_rate_micro"] == pytest.approx(0.4)
    assert summary["intervention_episode_rate_task_macro"] == pytest.approx(0.375)
    assert summary["interventions_per_episode_micro"] == pytest.approx(0.5)
    assert summary["interventions_per_episode_task_macro"] == pytest.approx(0.4375)
    assert summary["recovery_success_rate_micro"] == pytest.approx(3 / 5)
    assert summary["recovery_success_rate_task_macro"] == pytest.approx(
        (2 / 3 + 0 + 1) / 3
    )
    assert summary["recovery_eligible_task_count"] == 3
    assert summary["post_intervention_success_rate_micro"] == pytest.approx(3 / 4)
    assert summary["post_intervention_success_rate_task_macro"] == pytest.approx(
        2 / 3
    )

    assert [row["task_name"] for row in result["per_task"]] == ["a", "b", "c", "d"]
    assert result["per_task"][3]["recovery_success_rate"] is None
    assert result["per_task"][0]["success_ci_lower"] < 1.0


def test_no_intervention_reports_undefined_recovery_denominators() -> None:
    result = aggregate_multitask_metrics(
        [_row(0, "a", 0, success=True), _row(1, "b", 0, success=False)]
    )
    summary = result["summary"]
    assert summary["intervention_count"] == 0
    assert summary["recovery_success_rate_micro"] is None
    assert summary["recovery_success_rate_task_macro"] is None
    assert summary["recovery_eligible_task_count"] == 0
    assert summary["post_intervention_success_rate_micro"] is None


def test_mixed_records_are_grouped_in_stable_order() -> None:
    rows = [
        _row(0, "a", 0, method="MT-REIM", success=True),
        _row(0, "a", 0, method="MT-ACT", success=False),
    ]
    results = aggregate_multitask_results(rows)
    assert [result["summary"]["method"] for result in results] == [
        "MT-ACT",
        "MT-REIM",
    ]
    with pytest.raises(ValueError, match="exactly one benchmark and method"):
        aggregate_multitask_metrics(rows)


def test_task_stratified_paired_bootstrap_is_macro_and_deterministic() -> None:
    reference = [
        _row(0, "many", episode, method="MT-ACT", success=False)
        for episode in range(4)
    ] + [_row(1, "few", 0, method="MT-ACT", success=True)]
    candidate = [
        _row(0, "many", episode, method="MT-REIM", success=True)
        for episode in range(4)
    ] + [_row(1, "few", 0, method="MT-REIM", success=False)]

    first = paired_task_stratified_bootstrap_delta(
        reference,
        candidate,
        n_bootstrap=500,
        seed=123,
    )
    second = paired_task_stratified_bootstrap_delta(
        reference,
        candidate,
        n_bootstrap=500,
        seed=123,
    )

    # Episode-micro delta would be +0.6. Equal task weighting gives zero.
    assert first == second
    assert first["delta"] == pytest.approx(0.0)
    assert first["ci_lower"] == pytest.approx(0.0)
    assert first["ci_upper"] == pytest.approx(0.0)
    assert first["pair_count"] == 5
    assert [row["delta"] for row in first["per_task"]] == [1.0, -1.0]


def test_paired_bootstrap_retains_nonconstant_episode_pairing() -> None:
    reference = [
        _row(0, "a", 0, method="MT-ACT", success=False),
        _row(0, "a", 1, method="MT-ACT", success=True),
        _row(1, "b", 0, method="MT-ACT", success=False),
        _row(1, "b", 1, method="MT-ACT", success=False),
    ]
    candidate = [
        _row(0, "a", 0, method="MT-REIM", success=True),
        _row(0, "a", 1, method="MT-REIM", success=True),
        _row(1, "b", 0, method="MT-REIM", success=False),
        _row(1, "b", 1, method="MT-REIM", success=True),
    ]
    first = paired_task_stratified_bootstrap_delta(
        reference, candidate, n_bootstrap=300, seed=9
    )
    second = paired_task_stratified_bootstrap_delta(
        reference, candidate, n_bootstrap=300, seed=9
    )
    assert first == second
    assert first["delta"] == pytest.approx(0.5)
    assert first["ci_lower"] <= first["delta"] <= first["ci_upper"]


def test_validation_rejects_invalid_recovery_and_incomplete_pairs() -> None:
    bad = _row(0, "a", 0, interventions=0, recoveries=1)
    with pytest.raises(ValueError, match="cannot exceed"):
        aggregate_multitask_metrics([bad])

    reference = [_row(0, "a", 0, method="MT-ACT")]
    candidate = [_row(0, "a", 1, method="MT-REIM")]
    with pytest.raises(ValueError, match="identical task/paired_episode_id"):
        paired_task_stratified_bootstrap_delta(reference, candidate)


def test_duplicate_pair_ids_fail_closed() -> None:
    duplicate = [
        _row(0, "a", 0, method="MT-ACT"),
        _row(0, "a", 0, method="MT-ACT"),
    ]
    candidate = [_row(0, "a", 0, method="MT-REIM")]
    with pytest.raises(ValueError, match="duplicate paired_episode_id"):
        aggregate_multitask_metrics(duplicate)
    with pytest.raises(ValueError, match="duplicate paired_episode_id"):
        paired_task_stratified_bootstrap_delta(duplicate, candidate)


def test_integer_task_ids_are_sorted_numerically() -> None:
    result = aggregate_multitask_metrics(
        [_row(10, "ten", 0), _row(2, "two", 0), _row(1, "one", 0)]
    )
    assert [row["task_id"] for row in result["per_task"]] == ["1", "2", "10"]
