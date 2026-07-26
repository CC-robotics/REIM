"""Regression tests for shard-safe common-random-number evaluation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from env.metaworld_pickplace import REIMPickPlaceEnv
from evaluation.episode_bank import (
    build_episode_bank,
    load_episode_bank,
    runtime_episode_specifications,
    save_episode_bank,
    validate_episode_bank,
)
from evaluation.evaluate_reim import (
    ControlDecision,
    ControllerConfig,
    REIMController,
    run_episode,
)


def _toy_bank(*, episodes: int = 6, retries: int = 1) -> dict:
    return build_episode_bank(
        backend="toy",
        env_name="pick-place-v3",
        task_bank_seed=7001,
        episode_seed_start=9001,
        episodes=episodes,
        max_steps=12,
        action_noise_std=0.08,
        observation_noise_std=0.005,
        object_noise_probability=1.0,
        object_noise_std=0.02,
        object_noise_magnitude=0.03,
        retries_per_episode=retries,
    )


def _toy_env(constructor_seed: int) -> REIMPickPlaceEnv:
    return REIMPickPlaceEnv(
        backend="toy",
        state_mode="raw",
        seed=constructor_seed,
        max_episode_steps=12,
        action_noise_std=0.08,
        observation_noise_std=0.005,
        object_noise_probability=1.0,
        object_noise_std=0.02,
        object_noise_magnitude=0.03,
    )


def _rollout(specification: dict, constructor_seed: int) -> list[np.ndarray]:
    env = _toy_env(constructor_seed)
    try:
        observation, _ = env.reset(
            seed=specification["episode_seed"],
            options={"reim_episode_spec": specification},
        )
        sequence = [observation.copy()]
        action = np.asarray([0.1, -0.05, 0.02, 1.0], dtype=np.float32)
        for _ in range(6):
            observation, _, _, _, _ = env.step(action)
            sequence.append(observation.copy())
        return sequence
    finally:
        env.close()


def test_bank_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    payload = _toy_bank(retries=2)
    path = save_episode_bank(payload, tmp_path / "bank.json")
    assert load_episode_bank(path) == payload
    assert len(payload["retry_specifications"]) == payload["episodes"]
    assert all(
        len(retries) == 2 for retries in payload["retry_specifications"]
    )

    tampered = deepcopy(payload)
    tampered["episode_specifications"][0]["reset_seed"] += 1
    with pytest.raises(ValueError, match="SHA256"):
        validate_episode_bank(tampered)


def test_same_spec_is_exact_across_constructor_seeds_and_shards() -> None:
    payload = _toy_bank()
    monolithic = runtime_episode_specifications(payload)
    sharded = (
        runtime_episode_specifications(payload, offset=0, count=2)
        + runtime_episode_specifications(payload, offset=2, count=2)
        + runtime_episode_specifications(payload, offset=4, count=2)
    )
    assert [
        spec["specification_sha256"] for spec in monolithic
    ] == [
        spec["specification_sha256"] for spec in sharded
    ]

    for index, specification in enumerate(sharded):
        first = _rollout(specification, constructor_seed=11)
        second = _rollout(specification, constructor_seed=80_000 + index)
        assert len(first) == len(second)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)


class _ResetOnceController:
    method = "bc_random_reset"

    def __init__(self) -> None:
        self.config = SimpleNamespace(max_random_resets=1)
        self.recovery_attempts = 0
        self.recovery_successes = 0
        self.detector_triggers = 0
        self.recovery_steps_total = 0
        self.failure_probability_max = 1.0
        self._requested = False

    def reset_observation(
        self, _observation: np.ndarray, preserve_statistics: bool = False
    ) -> None:
        del preserve_statistics

    def act(self, _observation: np.ndarray, _info: dict) -> ControlDecision:
        if not self._requested:
            self._requested = True
            return ControlDecision(
                np.zeros(4, dtype=np.float32),
                "random_reset",
                1.0,
                request_reset=True,
            )
        return ControlDecision(np.zeros(4, dtype=np.float32), "bc", 0.0)

    def register_external_reset(self) -> None:
        self.recovery_attempts += 1
        self.detector_triggers += 1

    def observe_transition(
        self,
        _next_observation: np.ndarray,
        _reward: float,
        _info: dict,
        *,
        success: bool,
    ) -> None:
        del success

    def finalize(self, _success: bool) -> None:
        return None


def test_random_reset_uses_hashed_retry_specification() -> None:
    specification = runtime_episode_specifications(
        _toy_bank(retries=1),
        count=1,
    )[0]
    retry = specification["retry_specifications"][0]
    env = _toy_env(constructor_seed=123)
    try:
        metrics, trace = run_episode(
            env,
            _ResetOnceController(),  # type: ignore[arg-type]
            episode=0,
            seed=specification["episode_seed"],
            max_steps=2,
            capture_trace=True,
            episode_specification=specification,
        )
    finally:
        env.close()
    assert metrics.retry_specification_sha256s == retry["specification_sha256"]
    assert metrics.retry_task_sha256s == retry["task_sha256"]
    assert trace is not None
    assert trace["random_reset_events"] == [
        {
            "retry_index": 0,
            "seed": retry["episode_seed"],
            "episode_specification_sha256": retry["specification_sha256"],
            "metaworld_task_sha256": retry["task_sha256"],
        }
    ]


def test_no_intervention_reim_is_strictly_identical_to_act() -> None:
    specification = runtime_episode_specifications(
        _toy_bank(retries=1),
        count=1,
    )[0]

    def act_policy(_state: np.ndarray) -> np.ndarray:
        return np.asarray([0.1, -0.05, 0.02, 1.0], dtype=np.float32)

    def zero_detector(_sequence: np.ndarray, _length: int) -> float:
        return 0.0

    def forbidden_recovery(_state: np.ndarray) -> np.ndarray:
        raise AssertionError("recovery must not be called")

    traces = []
    metrics = []
    for constructor_seed, method in ((1, "bc"), (999_999, "reim")):
        env = _toy_env(constructor_seed)
        controller = REIMController(
            method,
            act_policy,
            detector=zero_detector if method == "reim" else None,
            recovery_policy=forbidden_recovery if method == "reim" else None,
            config=ControllerConfig(
                failure_threshold=0.8,
                recovery_exit_threshold=0.7,
            ),
        )
        try:
            record, trace = run_episode(
                env,
                controller,
                episode=0,
                seed=specification["episode_seed"],
                max_steps=8,
                capture_trace=True,
                episode_specification=specification,
            )
        finally:
            env.close()
        metrics.append(record)
        assert trace is not None
        traces.append(trace)

    assert metrics[0].success == metrics[1].success
    assert metrics[0].steps == metrics[1].steps
    assert traces[0]["actions"] == traces[1]["actions"]
    assert traces[0]["states"] == traces[1]["states"]
    assert set(traces[1]["sources"]) == {"bc"}
