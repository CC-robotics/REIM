"""Closed-loop controller tests that guard baseline trigger semantics."""

from __future__ import annotations

import numpy as np

from evaluation.evaluate_reim import ControllerConfig, REIMController


def _zero_policy(_state: np.ndarray) -> np.ndarray:
    return np.zeros(4, dtype=np.float32)


def test_heuristic_recovery_does_not_trigger_before_object_interaction() -> None:
    config = ControllerConfig(
        heuristic_window=3,
        heuristic_min_steps=1,
        failure_threshold=0.8,
    )
    controller = REIMController(
        "act_rl_recovery",
        _zero_policy,
        recovery_policy=_zero_policy,
        config=config,
    )
    state = np.zeros(21, dtype=np.float32)
    controller.reset_observation(state)
    initial = np.asarray([0.0, 0.6, 0.025], dtype=np.float32)
    common = {
        "object_position": initial,
        "distance_to_goal": 0.30,
        "failure": False,
    }
    for _ in range(4):
        controller.observe_transition(state, 0.0, common, success=False)
    decision = controller.act(state, common)
    assert decision.source == "bc"
    assert controller.recovery_attempts == 0

    moved = {**common, "object_position": initial + [0.03, 0.0, 0.0]}
    for _ in range(3):
        controller.observe_transition(state, 0.0, moved, success=False)
    decision = controller.act(state, moved)
    assert decision.source == "recovery"
    assert controller.recovery_attempts == 1


def test_act_temporal_ensemble_is_not_advanced_during_recovery() -> None:
    class CountingACT:
        def __init__(self) -> None:
            self.calls = 0
            self.resets = 0

        def __call__(self, _state: np.ndarray) -> np.ndarray:
            self.calls += 1
            return np.zeros(4, dtype=np.float32)

        def reset(self) -> None:
            self.resets += 1

    class OneShotDetector:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, _sequence: np.ndarray) -> float:
            self.calls += 1
            return 1.0 if self.calls == 1 else 0.0

    act = CountingACT()
    controller = REIMController(
        "reim",
        act,
        detector=OneShotDetector(),
        recovery_policy=_zero_policy,
        config=ControllerConfig(
            recovery_min_steps=1,
            recovery_clear_steps=1,
            recovery_exit_threshold=0.35,
        ),
    )
    state = np.zeros(21, dtype=np.float32)
    controller.reset_observation(state)
    controller.observe_transition(state, 0.0, {}, success=False)
    assert controller.act(state, {}).source == "recovery"
    assert act.calls == 0

    controller.observe_transition(state, 0.0, {}, success=False)
    assert not controller.recovery_active
    # One reset initializes the episode and one clears non-executed chunks at
    # the recovery-to-ACT hand-off.
    assert act.resets == 2
    assert controller.act(state, {}).source == "bc"
    assert act.calls == 1


def test_detector_deployment_uses_training_compatible_causal_padding() -> None:
    class CapturingDetector:
        def __init__(self) -> None:
            self.calls: list[tuple[np.ndarray, int]] = []

        def __call__(self, sequence: np.ndarray, length: int) -> float:
            self.calls.append((sequence.copy(), length))
            return 0.0

    detector = CapturingDetector()
    controller = REIMController(
        "reim",
        _zero_policy,
        detector=detector,
        recovery_policy=_zero_policy,
        config=ControllerConfig(sequence_length=4),
    )
    first = np.arange(3, dtype=np.float32)
    second = first + 10.0
    third = first + 20.0
    controller.reset_observation(first)
    controller.act(first, {})
    assert not detector.calls

    controller.observe_transition(second, 0.0, {}, success=False)
    controller.act(second, {})
    sequence, length = detector.calls[-1]
    assert length == 1
    np.testing.assert_array_equal(sequence[0], second)
    np.testing.assert_array_equal(sequence[1:], 0.0)

    controller.observe_transition(third, 0.0, {}, success=False)
    controller.act(third, {})
    sequence, length = detector.calls[-1]
    assert length == 2
    np.testing.assert_array_equal(sequence[:2], np.stack((second, third)))
    np.testing.assert_array_equal(sequence[2:], 0.0)
