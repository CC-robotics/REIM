"""Recovery reward tests using both controllable synthetic and toy states."""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np

from env.metaworld_pickplace import REIMPickPlaceEnv
from models.recovery_policy import RecoveryRewardWrapper
from trainers.train_recovery import (
    _isolate_resumed_tensorboard_log,
    _latest_ppo_checkpoint,
)


class ScriptedRecoveryEnv(gym.Env):
    """Minimal deterministic environment that emits prescribed recovery info."""

    metadata: dict[str, object] = {}

    def __init__(self, transitions: Sequence[dict[str, object]]) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(3,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.transitions = list(transitions)
        self.index = 0
        self.object_position = np.asarray([0.0, 0.0, 0.02], dtype=np.float32)
        self.goal_position = np.asarray([1.0, 0.0, 0.02], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.index = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        del action
        transition = dict(self.transitions[min(self.index, len(self.transitions) - 1)])
        self.index += 1
        distance = float(transition.pop("distance_to_goal", 1.0))
        transition.setdefault("object_position", self.object_position.copy())
        transition.setdefault("goal_position", self.goal_position.copy())
        transition.setdefault("hand_object_distance", 1.0)
        transition.setdefault("success", False)
        transition.setdefault("failure", False)
        transition["distance_to_goal"] = distance
        return (
            np.full(3, self.index, dtype=np.float32),
            0.0,
            bool(transition.pop("terminated", False)),
            bool(transition.pop("truncated", False)),
            transition,
        )

    def get_state_components(self):
        return {
            "object_position": self.object_position.copy(),
            "goal_position": self.goal_position.copy(),
        }


def test_recovery_bonus_is_paid_once_after_failure_signal_clears() -> None:
    env = RecoveryRewardWrapper(
        ScriptedRecoveryEnv(
            [
                {"failure": True, "distance_to_goal": 1.0},
                {"recovered": True, "distance_to_goal": 0.9},
                {"recovered": True, "distance_to_goal": 0.8},
            ]
        ),
        initialize_with_failure_states=False,
        distance_progress_scale=0.0,
        max_recovery_steps=10,
        terminate_on_recovery=False,
    )
    env.reset()
    _, first_reward, _, _, first_info = env.step(np.zeros(4, dtype=np.float32))
    _, second_reward, _, _, second_info = env.step(np.zeros(4, dtype=np.float32))
    _, third_reward, _, _, third_info = env.step(np.zeros(4, dtype=np.float32))

    # An online failure signal activates recovery but is not itself terminal:
    # the -10 penalty is reserved for an unsuccessful MDP termination/timeout.
    assert first_reward == -0.01
    assert first_info["recovery_failure"] is False
    assert first_info["recovery_active"] is True
    assert second_reward == 4.99
    assert second_info["recovery_success"] is True
    assert third_reward == -0.01
    assert third_info["recovery_success"] is False


def test_recovery_timeout_penalizes_and_truncates_without_false_success() -> None:
    env = RecoveryRewardWrapper(
        ScriptedRecoveryEnv([{}, {}]),
        initialize_with_failure_states=False,
        distance_progress_scale=0.0,
        max_recovery_steps=2,
    )
    env.reset()
    _, first_reward, _, first_truncated, first_info = env.step(
        np.zeros(4, dtype=np.float32)
    )
    _, second_reward, _, second_truncated, second_info = env.step(
        np.zeros(4, dtype=np.float32)
    )

    assert first_reward == -0.01
    assert not first_truncated
    assert first_info["recovery_success"] is False
    assert second_reward == -10.01
    assert second_truncated
    assert second_info["recovery_timeout"] is True
    assert second_info["recovery_failure"] is True
    assert second_info["recovery_success"] is False


def test_toy_recovery_reset_creates_disturbed_post_interaction_state() -> None:
    base = REIMPickPlaceEnv(backend="toy", seed=31, max_episode_steps=100)
    env = RecoveryRewardWrapper(
        base,
        initialize_with_failure_states=True,
        initialization_disturbance=0.03,
        warmup_min_steps=20,
        warmup_max_steps=20,
        max_recovery_steps=5,
    )
    try:
        observation, info = env.reset(seed=31)
        assert observation.shape == base.observation_space.shape
        assert info["recovery_initialization"] is True
        assert info["recovery_warmup_steps"] > 0
        assert info["recovery_initialization_stage"] in {
            "approach",
            "grasped",
            "lifted",
        }
        assert np.linalg.norm(info["recovery_initialization_delta"]) > 0.0

        _, reward, _, _, step_info = env.step(np.zeros(4, dtype=np.float32))
        # Merely being initialized near the object must not award +5 recovery.
        assert step_info["recovery_success"] is False
        assert reward < 1.0
    finally:
        env.close()


def test_ppo_resume_prefers_training_endpoint_over_validation_best(tmp_path) -> None:
    output = tmp_path / "recovery_policy.zip"
    endpoint = tmp_path / "recovery_policy_final.zip"
    snapshot_dir = tmp_path / "ppo"
    snapshot_dir.mkdir()
    output.touch()
    (snapshot_dir / "recovery_500000_steps.zip").touch()
    endpoint.touch()

    assert _latest_ppo_checkpoint(snapshot_dir, output) == endpoint

    endpoint.unlink()
    assert _latest_ppo_checkpoint(snapshot_dir, output) == (
        snapshot_dir / "recovery_500000_steps.zip"
    )


def test_ppo_resume_redirects_tensorboard_away_from_parent_run(tmp_path) -> None:
    class DummyModel:
        tensorboard_log = "results/logs/ppo_trigger_seed44_v2"

    model = DummyModel()
    isolated_log = tmp_path / "ppo_trigger_seed44_full"

    _isolate_resumed_tensorboard_log(model, isolated_log)

    assert model.tensorboard_log == str(isolated_log)
