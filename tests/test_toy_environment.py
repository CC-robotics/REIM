"""Unit and integration tests for the explicit toy PickPlace backend."""

from __future__ import annotations

import numpy as np

from env.metaworld_pickplace import (
    ACTION_DIM,
    RAW_OBSERVATION_DIM,
    SEMANTIC_STATE_DIM,
    REIMPickPlaceEnv,
    make_scripted_expert,
)


def test_semantic_reset_and_step_follow_gymnasium_contract() -> None:
    env = REIMPickPlaceEnv(
        backend="toy",
        state_mode="semantic",
        seed=11,
        max_episode_steps=10,
    )
    try:
        observation, info = env.reset(seed=11)
        assert observation.shape == (SEMANTIC_STATE_DIM,)
        assert observation.dtype == np.float32
        assert env.observation_space.contains(observation)
        assert env.action_space.shape == (ACTION_DIM,)
        assert info["backend"] == "toy"
        assert info["toy_ci_backend"] is True
        assert env.state_metadata["layout"]["object_position"]["slice"] == [14, 17]

        next_observation, reward, terminated, truncated, step_info = env.step(
            np.zeros(ACTION_DIM, dtype=np.float32)
        )
        assert next_observation.shape == (SEMANTIC_STATE_DIM,)
        assert np.isfinite(next_observation).all()
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert step_info["commanded_action"].shape == (ACTION_DIM,)
        assert step_info["executed_action"].shape == (ACTION_DIM,)
        assert step_info["step"] == 1
    finally:
        env.close()


def test_raw_state_and_rgb_render_have_documented_shapes() -> None:
    env = REIMPickPlaceEnv(
        backend="toy",
        state_mode="raw",
        render_mode="rgb_array",
        seed=3,
    )
    try:
        observation, _ = env.reset(seed=3)
        assert observation.shape == (RAW_OBSERVATION_DIM,)
        np.testing.assert_array_equal(observation, env.raw_observation)
        image = env.render()
        assert image.shape == (256, 256, 3)
        assert image.dtype == np.uint8
    finally:
        env.close()


def test_noise_is_seed_reproducible_and_actions_are_clipped() -> None:
    kwargs = {
        "backend": "toy",
        "seed": 29,
        "action_noise_std": 0.4,
        "observation_noise_std": 0.2,
    }
    first = REIMPickPlaceEnv(**kwargs)
    second = REIMPickPlaceEnv(**kwargs)
    try:
        first.reset(seed=29)
        second.reset(seed=29)
        command = np.asarray([0.9, -0.9, 0.1, 0.0], dtype=np.float32)
        noisy_first = first.apply_action_noise(command)
        noisy_second = second.apply_action_noise(command)
        np.testing.assert_allclose(noisy_first, noisy_second)
        assert np.all(noisy_first >= -1.0)
        assert np.all(noisy_first <= 1.0)

        state = np.zeros(SEMANTIC_STATE_DIM, dtype=np.float32)
        np.testing.assert_allclose(
            first.apply_observation_noise(state),
            second.apply_observation_noise(state),
        )
    finally:
        first.close()
        second.close()


def test_explicit_object_disturbance_updates_physical_and_observed_state() -> None:
    env = REIMPickPlaceEnv(backend="toy", seed=5)
    try:
        env.reset(seed=5)
        before = env.get_state_components()["object_position"]
        requested = np.asarray([0.025, -0.015, 0.01], dtype=np.float32)
        applied = env.apply_object_noise(delta=requested)
        after = env.get_state_components()["object_position"]

        np.testing.assert_allclose(applied, requested, atol=1e-7)
        np.testing.assert_allclose(after - before, requested, atol=1e-7)
        np.testing.assert_allclose(env.get_state()[14:17], after)
    finally:
        env.close()


def test_time_limit_is_reported_as_truncation_and_failure() -> None:
    env = REIMPickPlaceEnv(backend="toy", seed=17, max_episode_steps=1)
    try:
        env.reset(seed=17)
        _, _, terminated, truncated, info = env.step(
            np.zeros(ACTION_DIM, dtype=np.float32)
        )
        assert not terminated
        assert truncated
        assert info["failure"] is True
        assert info["failure_reason"] == "timeout"
    finally:
        env.close()


def test_scripted_expert_completes_toy_pick_place() -> None:
    env = REIMPickPlaceEnv(backend="toy", seed=41, max_episode_steps=100)
    expert = make_scripted_expert(env)
    try:
        observation, _ = env.reset(seed=41)
        expert.reset()
        final_info: dict[str, object] = {}
        for _ in range(100):
            action = expert.act(observation)
            observation, _, terminated, truncated, final_info = env.step(action)
            if terminated or truncated:
                break
        assert final_info["success"] is True
        assert not final_info["failure"]
    finally:
        env.close()

