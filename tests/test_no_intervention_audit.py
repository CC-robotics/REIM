"""Unit guards for the strict no-intervention audit."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.audit_no_intervention_equivalence import (
    ForbiddenRecoveryPolicy,
    ZeroFailureDetector,
    array_sha256,
    compare_episode_traces,
)


def _trace(*, action: float = 0.25) -> dict[str, object]:
    return {
        "states": [[0.0, 1.0], [2.0, 3.0]],
        "actions": [[action]],
        "sources": ["bc"],
        "success": True,
    }


def test_complete_equal_trajectories_pass_and_have_stable_hashes() -> None:
    trace = _trace()
    evidence, passed = compare_episode_traces(
        act_trace=trace,
        reim_trace=trace,
        state_dim=2,
        action_dim=1,
    )
    expected_states = np.asarray(trace["states"], dtype=np.float32)

    assert passed
    assert all(evidence["checks"].values())
    assert evidence["act"]["state_sha256"] == array_sha256(expected_states)
    assert (
        evidence["act"]["commanded_action_sha256"]
        == evidence["reim_no_intervention"]["commanded_action_sha256"]
    )


def test_single_action_bit_difference_fails_with_location_and_bits() -> None:
    evidence, passed = compare_episode_traces(
        act_trace=_trace(action=0.25),
        reim_trace=_trace(action=0.5),
        state_dim=2,
        action_dim=1,
    )

    assert not passed
    assert not evidence["checks"]["commanded_actions_bitwise_equal"]
    assert evidence["first_commanded_action_mismatch"] == {
        "kind": "value",
        "index": [0, 0],
        "act_value": 0.25,
        "reim_value": 0.5,
        "act_float32_bits": "0x3e800000",
        "reim_float32_bits": "0x3f000000",
    }


def test_detector_is_exactly_zero_and_recovery_is_forbidden() -> None:
    detector = ZeroFailureDetector()
    recovery = ForbiddenRecoveryPolicy()

    assert detector(np.zeros((10, 2), dtype=np.float32), 1) == 0.0
    assert detector.calls == 1
    with pytest.raises(AssertionError, match="no-intervention"):
        recovery(np.zeros(2, dtype=np.float32))
    assert recovery.calls == 1
