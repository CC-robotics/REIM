"""Thin, deployment-friendly wrapper around Stable-Baselines3 PPO."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # The policy loader remains importable before setup.
    gym = None


if gym is not None:

    class RecoveryRewardWrapper(gym.Wrapper):
        """Shape PickPlace transitions into a recovery-learning objective."""

        def __init__(
            self,
            env: gym.Env,
            *,
            task_success_reward: float = 10.0,
            recovery_reward: float = 5.0,
            failure_penalty: float = -10.0,
            time_penalty: float = -0.01,
            base_reward_scale: float = 0.0,
            distance_progress_scale: float = 2.0,
            reach_progress_scale: float = 0.0,
            lift_progress_scale: float = 0.0,
            initialize_with_failure_states: bool = True,
            initialization_disturbance: float = 0.04,
            recovered_distance: float = 0.08,
            max_recovery_steps: int = 50,
            warmup_min_steps: int = 45,
            warmup_max_steps: int = 45,
            relift_height: float = 0.03,
            terminate_on_recovery: bool = True,
            recovery_start_dataset: str | Path | None = None,
            successful_starts_only: bool = True,
        ) -> None:
            super().__init__(env)
            self.task_success_reward = float(task_success_reward)
            self.recovery_reward = float(recovery_reward)
            self.failure_penalty = float(failure_penalty)
            self.time_penalty = float(time_penalty)
            self.base_reward_scale = float(base_reward_scale)
            self.distance_progress_scale = float(distance_progress_scale)
            self.reach_progress_scale = float(reach_progress_scale)
            self.lift_progress_scale = float(lift_progress_scale)
            self.initialize_with_failure_states = bool(initialize_with_failure_states)
            self.initialization_disturbance = float(initialization_disturbance)
            self.recovered_distance = float(recovered_distance)
            self.max_recovery_steps = int(max_recovery_steps)
            self.warmup_min_steps = int(warmup_min_steps)
            self.warmup_max_steps = int(warmup_max_steps)
            self.relift_height = float(relift_height)
            self.terminate_on_recovery = bool(terminate_on_recovery)
            self.recovery_start_dataset = (
                None
                if recovery_start_dataset in (None, "")
                else Path(recovery_start_dataset).expanduser().resolve()
            )
            self.successful_starts_only = bool(successful_starts_only)
            if self.warmup_min_steps < 0:
                raise ValueError("warmup_min_steps cannot be negative.")
            if self.warmup_max_steps < self.warmup_min_steps:
                raise ValueError("warmup_max_steps must be >= warmup_min_steps.")
            self._failure_active = False
            self._recovery_reward_paid = False
            self._previous_distance = np.inf
            self._previous_hand_object_distance = np.inf
            self._previous_object_height = 0.0
            self._initial_distance = np.inf
            self._recovery_steps = 0
            self._post_disturbance_object_height = 0.0
            self._warmup_expert: Any | None = None
            self._snapshot_arrays: dict[str, np.ndarray] = {}
            self._snapshot_indices = np.empty(0, dtype=np.int64)
            self._active_snapshot_index = -1
            self._active_recovery_limit = self.max_recovery_steps
            if self.recovery_start_dataset is not None:
                self._load_recovery_starts(self.recovery_start_dataset)

        def _load_recovery_starts(self, path: Path) -> None:
            if not path.is_file():
                raise FileNotFoundError(f"Recovery-start dataset not found: {path}")
            with np.load(path, allow_pickle=False) as archive:
                schema = str(np.asarray(archive["schema_version"]).reshape(-1)[0])
                if schema != "reim-recovery-starts-v1":
                    raise ValueError(
                        f"Unsupported recovery-start schema {schema!r} in {path}"
                    )
                count = int(np.asarray(archive["episode_seed"]).shape[0])
                if count <= 0:
                    raise ValueError(f"Recovery-start dataset is empty: {path}")
                self._snapshot_arrays = {
                    key[len("snapshot_") :]: np.asarray(archive[key]).copy()
                    for key in archive.files
                    if key.startswith("snapshot_")
                }
                invalid = [
                    key
                    for key, values in self._snapshot_arrays.items()
                    if values.shape[0] != count
                ]
                if invalid:
                    raise ValueError(
                        f"Snapshot fields have inconsistent leading dimensions: {invalid}"
                    )
                self._snapshot_arrays.update(
                    {
                        "_episode_seed": np.asarray(
                            archive["episode_seed"], dtype=np.int64
                        ).copy(),
                        "_trigger_step": np.asarray(
                            archive["trigger_step"], dtype=np.int64
                        ).copy(),
                        "_trigger_probability": np.asarray(
                            archive["trigger_probability"], dtype=np.float32
                        ).copy(),
                        "_expert_success": np.asarray(
                            archive["expert_success"], dtype=np.bool_
                        ).copy(),
                    }
                )
            mask = np.ones(count, dtype=bool)
            if self.successful_starts_only:
                mask &= self._snapshot_arrays["_expert_success"]
            self._snapshot_indices = np.flatnonzero(mask).astype(np.int64)
            if not self._snapshot_indices.size:
                raise ValueError(
                    f"No eligible recovery starts remain after filtering {path}"
                )

        def _restore_recovery_start(
            self,
            observation: np.ndarray,
            info: Mapping[str, Any],
        ) -> tuple[np.ndarray, dict[str, Any], int, int, str]:
            base = self.env.unwrapped
            restore = getattr(base, "restore_sim_state", None)
            if not callable(restore):
                raise RuntimeError(
                    "Recovery-start datasets require an environment exposing "
                    "restore_sim_state()."
                )
            generator = getattr(base, "np_random", np.random.default_rng())
            index = int(generator.choice(self._snapshot_indices))
            snapshot = {
                key: values[index]
                for key, values in self._snapshot_arrays.items()
                if not key.startswith("_")
            }
            observation, restored_info = restore(snapshot)
            trigger_step = int(self._snapshot_arrays["_trigger_step"][index])
            episode_seed = int(self._snapshot_arrays["_episode_seed"][index])
            probability = float(
                self._snapshot_arrays["_trigger_probability"][index]
            )
            self._active_snapshot_index = index
            remaining_task_steps = max(
                int(getattr(base, "max_episode_steps", self.max_recovery_steps))
                - trigger_step,
                1,
            )
            self._active_recovery_limit = (
                min(self.max_recovery_steps, remaining_task_steps)
                if self.max_recovery_steps > 0
                else remaining_task_steps
            )
            restored_info = dict(restored_info)
            restored_info.update(
                {
                    "recovery_initialization": True,
                    "recovery_initialization_stage": "online_act_trigger",
                    "recovery_start_dataset": str(self.recovery_start_dataset),
                    "recovery_start_index": index,
                    "recovery_source_episode_seed": episode_seed,
                    "recovery_trigger_step": trigger_step,
                    "recovery_trigger_probability": probability,
                }
            )
            return (
                np.asarray(observation, dtype=np.float32),
                restored_info,
                trigger_step,
                trigger_step,
                "online_act_trigger",
            )

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[np.ndarray, dict[str, Any]]:
            observation, info = self.env.reset(seed=seed, options=options)
            self._failure_active = self.initialize_with_failure_states
            self._recovery_reward_paid = False
            self._recovery_steps = 0
            self._active_snapshot_index = -1
            self._active_recovery_limit = self.max_recovery_steps
            base = self.env.unwrapped
            warmup_steps = 0
            initialization_stage = "task_start"
            warmup_target = 0
            if self._snapshot_indices.size:
                observation, info, warmup_steps, warmup_target, initialization_stage = (
                    self._restore_recovery_start(observation, info)
                )
            elif self.initialize_with_failure_states:
                observation, info, warmup_steps, warmup_target, initialization_stage = (
                    self._warm_start_with_expert(observation, info)
                )
            if (
                self.initialize_with_failure_states
                and not self._snapshot_indices.size
                and hasattr(base, "apply_object_noise")
            ):
                components_before = (
                    base.get_state_components()
                    if hasattr(base, "get_state_components")
                    else {}
                )
                applied = base.apply_object_noise(
                    magnitude=self.initialization_disturbance
                )
                if hasattr(base, "get_state"):
                    observation = np.asarray(base.get_state(), dtype=np.float32)
                info = dict(info)
                info["recovery_initialization"] = True
                info["recovery_initialization_stage"] = initialization_stage
                info["recovery_warmup_steps"] = warmup_steps
                info["recovery_warmup_target"] = warmup_target
                info["recovery_initialization_delta"] = np.asarray(
                    applied, dtype=np.float32
                )
                if components_before:
                    info["recovery_pre_disturbance_object_position"] = np.asarray(
                        components_before["object_position"], dtype=np.float32
                    )
                standardize = getattr(base, "_standardize_info", None)
                if callable(standardize):
                    info = standardize(info, terminated=False, truncated=False)
            components_after = (
                base.get_state_components()
                if hasattr(base, "get_state_components")
                else {}
            )
            if components_after:
                object_position = np.asarray(
                    components_after["object_position"], dtype=np.float32
                )
                self._post_disturbance_object_height = float(object_position[2])
                info["recovery_post_disturbance_object_position"] = object_position
            # Recompute after the optional physical object perturbation; reset
            # info was produced before that perturbation.
            self._previous_distance = self._distance_from_env()
            self._initial_distance = self._previous_distance
            self._previous_hand_object_distance = self._hand_object_distance_from_env()
            self._previous_object_height = self._object_height_from_env()
            return observation, info

        def _warm_start_with_expert(
            self,
            observation: np.ndarray,
            info: dict[str, Any],
        ) -> tuple[np.ndarray, dict[str, Any], int, int, str]:
            """Advance to an interacted grasp/lift state before inducing failure."""
            base = self.env.unwrapped
            try:
                from env.metaworld_pickplace import make_scripted_expert
            except ImportError:
                return observation, info, 0, 0, "expert_unavailable"
            if self._warmup_expert is None:
                self._warmup_expert = make_scripted_expert(base)
            if hasattr(self._warmup_expert, "reset"):
                self._warmup_expert.reset()
            generator = getattr(base, "np_random", np.random.default_rng())
            if self.warmup_max_steps == self.warmup_min_steps:
                target_steps = self.warmup_min_steps
            else:
                target_steps = int(
                    generator.integers(
                        self.warmup_min_steps, self.warmup_max_steps + 1
                    )
                )
            initial_height = 0.0
            if hasattr(base, "get_state_components"):
                initial_height = float(
                    base.get_state_components()["object_position"][2]
                )
            stage = "approach"
            completed = 0
            for step_index in range(target_steps):
                action = self._warmup_expert.act(observation)
                observation, _, terminated, truncated, info = self.env.step(action)
                completed = step_index + 1
                object_height = initial_height
                if hasattr(base, "get_state_components"):
                    object_height = float(
                        base.get_state_components()["object_position"][2]
                    )
                grasped = bool(
                    info.get(
                        "grasped",
                        info.get("grasp_success", info.get("grasp_successful", False)),
                    )
                )
                lifted = bool(object_height >= initial_height + 0.025)
                if lifted:
                    stage = "lifted"
                    break
                if grasped:
                    stage = "grasped"
                if terminated or truncated:
                    # A completed/failed task is not a valid recovery start.
                    observation, info = self.env.reset()
                    if hasattr(self._warmup_expert, "reset"):
                        self._warmup_expert.reset()
                    completed = 0
                    stage = "task_restart"
                    break
            return observation, dict(info), completed, target_steps, stage

        def _distance_from_env(self) -> float:
            base = self.env.unwrapped
            if hasattr(base, "get_state_components"):
                components = base.get_state_components()
                return float(
                    np.linalg.norm(
                        np.asarray(components["object_position"])
                        - np.asarray(components["goal_position"])
                    )
                )
            return 0.0

        def _hand_object_distance_from_env(self) -> float:
            base = self.env.unwrapped
            if hasattr(base, "get_state_components"):
                components = base.get_state_components()
                end_effector = components.get("end_effector_position")
                object_position = components.get("object_position")
                if end_effector is not None and object_position is not None:
                    return float(
                        np.linalg.norm(
                            np.asarray(end_effector)
                            - np.asarray(object_position)
                        )
                    )
            return np.inf

        def _object_height_from_env(self) -> float:
            base = self.env.unwrapped
            if hasattr(base, "get_state_components"):
                components = base.get_state_components()
                object_position = components.get("object_position")
                if object_position is not None:
                    values = np.asarray(object_position).reshape(-1)
                    if values.size >= 3:
                        return float(values[2])
            return 0.0

        def step(
            self, action: np.ndarray
        ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            observation, _base_reward, terminated, truncated, info = self.env.step(
                action
            )
            self._recovery_steps += 1
            info = dict(info)
            distance = float(
                info.get("distance_to_goal", info.get("distance", self._distance_from_env()))
            )
            progress = (
                self._previous_distance - distance
                if np.isfinite(self._previous_distance)
                else 0.0
            )
            success = bool(info.get("success", info.get("is_success", False)))
            failure = bool(info.get("failure", False))
            if failure:
                self._failure_active = True

            recovered_signal = bool(info.get("recovered", False))
            object_position = np.asarray(
                info.get("object_position", [0.0, 0.0, self._post_disturbance_object_height]),
                dtype=np.float32,
            ).reshape(-1)
            object_height = (
                float(object_position[2])
                if object_position.size >= 3
                else self._post_disturbance_object_height
            )
            grasped = bool(
                info.get(
                    "grasped",
                    info.get("grasp_success", info.get("grasp_successful", False)),
                )
            )
            hand_object_distance = float(info.get("hand_object_distance", np.inf))
            reach_progress = (
                self._previous_hand_object_distance - hand_object_distance
                if np.isfinite(self._previous_hand_object_distance)
                and np.isfinite(hand_object_distance)
                else 0.0
            )
            lift_progress = object_height - self._previous_object_height
            relifted = bool(
                object_height
                >= self._post_disturbance_object_height + self.relift_height
            )
            regrasped_and_lifting = bool(
                self._recovery_steps >= 2
                and (
                    grasped
                    or (
                        hand_object_distance <= 0.055
                        and object_height
                        >= self._post_disturbance_object_height
                        + 0.5 * self.relift_height
                    )
                )
                and relifted
            )
            goal_progress_recovery = bool(
                self._initial_distance - distance >= self.recovered_distance
            )
            recovered_signal = recovered_signal or success
            recovered_signal = recovered_signal or (
                self._failure_active
                and (regrasped_and_lifting or goal_progress_recovery)
            )
            recovered = bool(recovered_signal and not self._recovery_reward_paid)
            if recovered:
                self._recovery_reward_paid = True
                self._failure_active = False
            recovery_terminated = bool(
                recovered and self.terminate_on_recovery and not success
            )
            if recovery_terminated:
                terminated = True

            recovery_timeout = bool(
                self._active_recovery_limit > 0
                and self._recovery_steps >= self._active_recovery_limit
                and not terminated
                and not success
            )
            if recovery_timeout:
                truncated = True
            # ``info["failure"]`` is an online state/event signal from the
            # base task (for example, a dropped object or excessive
            # deviation).  Recovery deliberately starts in such states, so
            # charging the terminal -10 penalty every time that signal is
            # present both changes the stated reward and overwhelms PPO's
            # value targets.  It still activates recovery above; the penalty
            # is paid exactly once only when the recovery MDP actually ends
            # unsuccessfully.
            base_terminal_failure = bool(
                (terminated or truncated)
                and not success
                and not recovered
                and not recovery_timeout
            )
            terminal_recovery_failure = bool(
                (recovery_timeout and not recovered) or base_terminal_failure
            )
            shaped_reward = self.time_penalty
            shaped_reward += self.base_reward_scale * float(_base_reward)
            shaped_reward += self.distance_progress_scale * progress
            # Reaching and lifting are necessary precursors to object-goal
            # progress after a failed grasp/drop.  Potential-difference
            # shaping supplies PPO with a causal signal before the object can
            # move toward the target, while still penalizing moving away or
            # dropping it.
            shaped_reward += self.reach_progress_scale * reach_progress
            shaped_reward += self.lift_progress_scale * lift_progress
            if success:
                shaped_reward += self.task_success_reward
            if recovered:
                shaped_reward += self.recovery_reward
            if terminal_recovery_failure:
                shaped_reward += self.failure_penalty

            info.update(
                {
                    "failure": terminal_recovery_failure,
                    "recovery_success": recovered,
                    "recovery_failure": terminal_recovery_failure,
                    "recovery_timeout": recovery_timeout,
                    "recovery_terminated": recovery_terminated,
                    "recovery_active": self._failure_active,
                    "recovery_step": self._recovery_steps,
                    "recovery_step_limit": self._active_recovery_limit,
                    "recovery_start_index": self._active_snapshot_index,
                    "recovery_base_reward": float(_base_reward),
                    "recovery_progress": float(progress),
                    "recovery_reach_progress": float(reach_progress),
                    "recovery_lift_progress": float(lift_progress),
                    "recovery_relifted": relifted,
                    "recovery_regrasped": regrasped_and_lifting,
                    "recovery_goal_progress": goal_progress_recovery,
                    "recovery_shaped_reward": float(shaped_reward),
                }
            )
            self._previous_distance = distance
            self._previous_hand_object_distance = hand_object_distance
            self._previous_object_height = object_height
            return (
                observation,
                float(shaped_reward),
                bool(terminated),
                bool(truncated),
                info,
            )

else:  # pragma: no cover - only used before dependencies are installed.

    class RecoveryRewardWrapper:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ImportError("gymnasium is required for RecoveryRewardWrapper.")


class RecoveryPolicy:
    """Wrap an SB3 PPO model behind the same ``act`` interface as BCPolicy."""

    def __init__(self, model: Any) -> None:
        if not hasattr(model, "predict"):
            raise TypeError("RecoveryPolicy expects an object exposing predict().")
        self.model = model

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        env: Any | None = None,
        device: str = "auto",
        custom_objects: dict[str, Any] | None = None,
    ) -> "RecoveryPolicy":
        try:
            from stable_baselines3 import PPO
        except ImportError as error:
            raise ImportError(
                "stable-baselines3 is required to load the recovery policy. "
                "Install the dependencies with ./setup.sh."
            ) from error
        model = PPO.load(
            str(path),
            env=env,
            device=device,
            custom_objects=custom_objects,
        )
        return cls(model)

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
        state: Any | None = None,
        episode_start: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, Any]:
        action, recurrent_state = self.model.predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        if state is None and episode_start is None:
            return np.asarray(action)
        return np.asarray(action), recurrent_state

    def act(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> np.ndarray:
        prediction = self.predict(observation, deterministic=deterministic)
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        return np.asarray(prediction)

    def save(self, path: str | Path) -> None:
        self.model.save(str(path))

    @property
    def num_timesteps(self) -> int:
        return int(getattr(self.model, "num_timesteps", 0))
