"""Core statistical guards for the strict matched-gate audit."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.audit_matched_gate import (
    exact_two_sided_mcnemar_binomial_p,
    paired_binary_comparison,
    paired_bootstrap_delta_ci,
)


def test_exact_two_sided_mcnemar_binomial_p_known_counts() -> None:
    assert exact_two_sided_mcnemar_binomial_p(0, 0) == 1.0
    assert exact_two_sided_mcnemar_binomial_p(5, 0) == pytest.approx(0.0625)
    assert exact_two_sided_mcnemar_binomial_p(4, 1) == pytest.approx(0.375)
    assert exact_two_sided_mcnemar_binomial_p(1, 4) == pytest.approx(0.375)


def test_paired_bootstrap_is_deterministic_and_preserves_constant_delta() -> None:
    reference = np.zeros(20, dtype=np.bool_)
    candidate = np.ones(20, dtype=np.bool_)
    first = paired_bootstrap_delta_ci(
        reference,
        candidate,
        samples=500,
        seed=123,
    )
    second = paired_bootstrap_delta_ci(
        reference,
        candidate,
        samples=500,
        seed=123,
    )
    assert first == second == (1.0, 1.0)


def test_paired_comparison_reports_candidate_wins_and_losses() -> None:
    reference = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.bool_)
    candidate = np.asarray([1, 1, 0, 0, 1, 1], dtype=np.bool_)
    result = paired_binary_comparison(
        reference,
        candidate,
        reference_name="reference",
        candidate_name="candidate",
        bootstrap_samples=1_000,
        bootstrap_seed=77,
    )
    assert result["wins"] == 2
    assert result["losses"] == 1
    assert result["both_success"] == 2
    assert result["both_failure"] == 1
    assert result["paired_delta"] == pytest.approx(1.0 / 6.0)
    assert result["exact_two_sided_mcnemar_binomial_p"] == 1.0


def test_paired_statistics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same non-empty shape"):
        paired_binary_comparison(
            [True],
            [True, False],
            reference_name="reference",
            candidate_name="candidate",
            bootstrap_samples=100,
            bootstrap_seed=0,
        )
