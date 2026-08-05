"""Integration tests for the strict Meta-World 3.1.1 MT10/MT50 layer."""

from __future__ import annotations

import importlib.metadata

import numpy as np
import pytest


pytest.importorskip("gymnasium")
pytest.importorskip("metaworld")

from env.metaworld_multitask import (  # noqa: E402
    ACTION_DIM,
    ENV_POLICY_MAP,
    OFFICIAL_MAX_EPISODE_STEPS,
    OFFICIAL_VARIANTS_PER_TASK,
    RAW_OBSERVATION_DIM,
    REIMMetaWorldMultiTaskEnv,
    SUPPORTED_METAWORLD_VERSION,
)


MT10_ORDER = (
    "reach-v3",
    "push-v3",
    "pick-place-v3",
    "door-open-v3",
    "drawer-open-v3",
    "drawer-close-v3",
    "button-press-topdown-v3",
    "peg-insert-side-v3",
    "window-open-v3",
    "window-close-v3",
)

MT50_ORDER = (
    "assembly-v3",
    "basketball-v3",
    "bin-picking-v3",
    "box-close-v3",
    "button-press-topdown-v3",
    "button-press-topdown-wall-v3",
    "button-press-v3",
    "button-press-wall-v3",
    "coffee-button-v3",
    "coffee-pull-v3",
    "coffee-push-v3",
    "dial-turn-v3",
    "disassemble-v3",
    "door-close-v3",
    "door-lock-v3",
    "door-open-v3",
    "door-unlock-v3",
    "hand-insert-v3",
    "drawer-close-v3",
    "drawer-open-v3",
    "faucet-open-v3",
    "faucet-close-v3",
    "hammer-v3",
    "handle-press-side-v3",
    "handle-press-v3",
    "handle-pull-side-v3",
    "handle-pull-v3",
    "lever-pull-v3",
    "pick-place-wall-v3",
    "pick-out-of-hole-v3",
    "pick-place-v3",
    "plate-slide-v3",
    "plate-slide-side-v3",
    "plate-slide-back-v3",
    "plate-slide-back-side-v3",
    "peg-insert-side-v3",
    "peg-unplug-side-v3",
    "soccer-v3",
    "stick-push-v3",
    "stick-pull-v3",
    "push-v3",
    "push-wall-v3",
    "push-back-v3",
    "reach-v3",
    "reach-wall-v3",
    "shelf-place-v3",
    "sweep-into-v3",
    "sweep-v3",
    "window-open-v3",
    "window-close-v3",
)


@pytest.fixture(scope="module")
def mt10() -> REIMMetaWorldMultiTaskEnv:
    env = REIMMetaWorldMultiTaskEnv(
        "MT10", task_id="pick-place-v3", variant_id=17, seed=41
    )
    yield env
    env.close()


def test_installed_version_is_the_audited_release() -> None:
    assert importlib.metadata.version("metaworld") == SUPPORTED_METAWORLD_VERSION


def test_mt10_official_order_spaces_metadata_and_one_hot(
    mt10: REIMMetaWorldMultiTaskEnv,
) -> None:
    observation, info = mt10.reset(seed=101)

    assert mt10.task_names == MT10_ORDER
    assert mt10.task_id == 2
    assert mt10.variant_id == 17
    assert mt10.observation_space.shape == (49,)
    assert mt10.action_space.shape == (ACTION_DIM,)
    assert observation.shape == (49,)
    assert observation.dtype == np.float64
    np.testing.assert_array_equal(observation[RAW_OBSERVATION_DIM:], mt10.task_one_hot)
    assert observation[RAW_OBSERVATION_DIM + 2] == 1.0
    assert np.count_nonzero(observation[RAW_OBSERVATION_DIM:]) == 1

    metadata = mt10.task_metadata
    assert metadata["benchmark"] == "MT10"
    assert metadata["task_name"] == "pick-place-v3"
    assert metadata["variant_id"] == 17
    assert metadata["num_variants"] == OFFICIAL_VARIANTS_PER_TASK
    assert metadata["max_episode_steps"] == OFFICIAL_MAX_EPISODE_STEPS
    assert len(metadata["task_sha256"]) == 64
    assert info["task_metadata"] == metadata
    assert info["success"] is False


def test_noise_never_modifies_task_one_hot() -> None:
    env = REIMMetaWorldMultiTaskEnv(
        "MT10",
        task_id=4,
        variant_id=8,
        seed=72,
        observation_noise_std=5.0,
        action_noise_std=0.2,
    )
    try:
        observation, _ = env.reset(seed=902)
        np.testing.assert_array_equal(observation[39:], env.task_one_hot)
        np.testing.assert_array_equal(env.get_state(noisy=True)[39:], env.task_one_hot)
        np.testing.assert_array_equal(env.get_state(noisy=False)[39:], env.task_one_hot)
        np.testing.assert_array_equal(env.get_state(noisy=False)[:39], env.raw_observation)
        assert not np.array_equal(observation[:39], env.raw_observation)

        synthetic = np.concatenate([np.zeros(39), env.task_one_hot])
        perturbed = env.apply_observation_noise(synthetic, std=10.0)
        np.testing.assert_array_equal(perturbed[39:], synthetic[39:])

        next_observation, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        np.testing.assert_array_equal(next_observation[39:], env.task_one_hot)
        assert info["executed_action"].shape == (4,)
        assert info["action_noise"].shape == (4,)
    finally:
        env.close()


def test_seed_reproduces_backend_and_both_noise_streams() -> None:
    kwargs = dict(
        benchmark="MT10",
        task_id="door-open-v3",
        variant_id=6,
        seed=314,
        action_noise_std=0.08,
        observation_noise_std=0.03,
    )
    first = REIMMetaWorldMultiTaskEnv(**kwargs)
    second = REIMMetaWorldMultiTaskEnv(**kwargs)
    try:
        first_obs, first_info = first.reset(seed=2718)
        second_obs, second_info = second.reset(seed=2718)
        np.testing.assert_allclose(first_obs, second_obs, atol=0.0, rtol=0.0)
        assert first_info["variant_id"] == second_info["variant_id"] == 6

        command = np.asarray([0.2, -0.1, 0.05, 0.6], dtype=np.float32)
        first_step = first.step(command)
        second_step = second.step(command)
        np.testing.assert_allclose(first_step[0], second_step[0], atol=0.0, rtol=0.0)
        assert first_step[1:4] == second_step[1:4]
        np.testing.assert_allclose(
            first_step[4]["executed_action"],
            second_step[4]["executed_action"],
            atol=0.0,
            rtol=0.0,
        )
    finally:
        first.close()
        second.close()


def test_select_variant_and_replay_official_task_object(
    mt10: REIMMetaWorldMultiTaskEnv,
) -> None:
    mt10.select_task("reach-v3", 49)
    selected_task = mt10.current_task
    selected_hash = mt10.task_metadata["task_sha256"]
    observation, info = mt10.reset(seed=13)
    assert observation[39] == 1.0
    assert info["variant_id"] == 49

    mt10.select_task("push-v3", 1)
    mt10.set_task(selected_task)
    replay, replay_info = mt10.reset(seed=13)
    assert mt10.task_name == "reach-v3"
    assert mt10.variant_id == 49
    assert mt10.task_metadata["task_sha256"] == selected_hash
    assert replay_info["variant_id"] == 49
    np.testing.assert_array_equal(replay[39:], mt10.task_one_hot)


def test_default_variant_sampling_is_seed_deterministic() -> None:
    first = REIMMetaWorldMultiTaskEnv("MT10", task_id=0, variant_id=None, seed=88)
    second = REIMMetaWorldMultiTaskEnv("MT10", task_id=0, variant_id=None, seed=88)
    try:
        sequence_a = []
        sequence_b = []
        for episode_seed in (5, 6, 7):
            first.reset(seed=episode_seed)
            second.reset(seed=episode_seed)
            sequence_a.append(first.variant_id)
            sequence_b.append(second.variant_id)
        assert sequence_a == sequence_b
        assert all(0 <= value < 50 for value in sequence_a)

        replay_task = first.current_task
        replay_variant = first.variant_id
        first.set_task(replay_task)
        first.reset(seed=99)
        assert first.variant_id == replay_variant
        first.enable_random_variant_sampling()
        first.reset(seed=99)
        assert 0 <= first.variant_id < 50
    finally:
        first.close()
        second.close()


def test_official_expert_and_success_info(mt10: REIMMetaWorldMultiTaskEnv) -> None:
    mt10.select_task("pick-place-v3", 3)
    observation, _ = mt10.reset(seed=33)
    assert mt10.task_name in ENV_POLICY_MAP
    action = mt10.get_expert_action(observation)
    assert action.shape == (4,)
    assert action.dtype == np.float32
    assert mt10.action_space.contains(action)

    _, _, _, _, info = mt10.step(action)
    assert "success" in info
    assert info["official_success"] == bool(info["success"])
    assert info["is_success"] == bool(info["success"])


def test_mt50_official_order_and_dimensions() -> None:
    env = REIMMetaWorldMultiTaskEnv(
        "MT50", task_id="sweep-into-v3", variant_id=49, seed=19
    )
    try:
        observation, info = env.reset(seed=23)
        assert env.task_names == MT50_ORDER
        assert env.task_id == MT50_ORDER.index("sweep-into-v3")
        assert env.observation_space.shape == (89,)
        assert env.action_space.shape == (4,)
        assert observation.shape == (89,)
        np.testing.assert_array_equal(observation[39:], env.task_one_hot)
        assert info["task_name"] == "sweep-into-v3"
        assert info["variant_id"] == 49
    finally:
        env.close()


def test_official_500_step_horizon(mt10: REIMMetaWorldMultiTaskEnv) -> None:
    mt10.select_task("reach-v3", 0)
    mt10.reset(seed=812)
    assert mt10.backend_env.max_path_length == OFFICIAL_MAX_EPISODE_STEPS
    action = np.zeros(4, dtype=np.float32)
    for step in range(OFFICIAL_MAX_EPISODE_STEPS):
        _, _, terminated, truncated, _ = mt10.step(action)
        assert terminated is False
        assert truncated is (step == OFFICIAL_MAX_EPISODE_STEPS - 1)
    with pytest.raises(RuntimeError, match="reset"):
        mt10.step(action)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"benchmark": "ML10"}, ValueError),
        ({"benchmark": "MT10", "task_id": 10}, ValueError),
        ({"benchmark": "MT10", "variant_id": 50}, ValueError),
        ({"benchmark": "MT10", "action_noise_std": -0.1}, ValueError),
    ],
)
def test_invalid_configuration_fails_loudly(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        REIMMetaWorldMultiTaskEnv(**kwargs)
