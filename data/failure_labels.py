"""Strictly causal failure labels for REIM trajectory histories."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


def build_causal_windows(
    states: np.ndarray, sequence_length: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Build chronological, right-padded histories ending at each sample.

    For timestep ``t``, only states from ``max(0, t-L+1)`` through ``t`` are
    used. Valid frames occupy ``window[:valid_length]`` and the trailing rows
    are zero. This layout works directly with ``pack_padded_sequence`` and
    contains no future information.
    """

    values = np.asarray(states, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"states must have shape [time, state_dim], got {values.shape}")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    windows = np.zeros(
        (len(values), sequence_length, values.shape[1]), dtype=np.float32
    )
    valid_lengths = np.empty(len(values), dtype=np.int16)
    for timestep in range(len(values)):
        start = max(0, timestep - sequence_length + 1)
        history = values[start : timestep + 1]
        windows[timestep, : len(history)] = history
        valid_lengths[timestep] = len(history)
    return windows, valid_lengths


def build_prospective_failure_targets(
    event_labels: np.ndarray,
    event_reasons: list[str] | np.ndarray,
    horizon: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forecast whether an observed failure event occurs within ``horizon`` steps.

    The target at time ``t`` is positive iff an online event is recorded in the
    inclusive interval ``[t, t + horizon]``. This is an outcome target, not an
    input feature: paired causal windows still end at ``t`` and therefore
    contain no future-state leakage.

    Returns
    -------
    targets:
        Binary prospective labels.
    first_event_offsets:
        Number of steps to the first event, or ``-1`` for a negative target.
    first_event_reasons:
        Reason associated with that first event, or ``"normal"``.
    """

    events = np.asarray(event_labels, dtype=np.uint8).reshape(-1)
    reasons = np.asarray(event_reasons, dtype="<U32").reshape(-1)
    if len(events) != len(reasons):
        raise ValueError("event_labels and event_reasons must have the same length.")
    if horizon < 0:
        raise ValueError("horizon must be non-negative.")
    targets = np.zeros(len(events), dtype=np.uint8)
    offsets = np.full(len(events), -1, dtype=np.int16)
    target_reasons = np.full(len(events), "normal", dtype="<U32")

    # Track the closest event at or after each timestep in one reverse pass.
    next_event = -1
    for timestep in range(len(events) - 1, -1, -1):
        if events[timestep]:
            next_event = timestep
        if next_event >= timestep and next_event - timestep <= horizon:
            targets[timestep] = 1
            offsets[timestep] = next_event - timestep
            target_reasons[timestep] = reasons[next_event]
    return targets, offsets, target_reasons


@dataclass
class CausalFailureLabeler:
    """Online failure/risk annotator using current and past measurements only.

    This class emits online event labels only. A separate prospective-target
    pass may associate each causal history with a future event outcome. The
    rules capture:

    * an observed drop or invalid workspace state,
    * a failed grasp after previously reaching the object,
    * post-contact progress stalling,
    * a large post-contact deviation from the initial object-to-goal distance,
    * an unsuccessful terminal/timeout state.
    """

    progress_window: int = 10
    minimum_progress: float = 0.004
    deviation_margin: float = 0.10
    grasp_distance: float = 0.055
    failed_grasp_distance: float = 0.12
    lift_delta: float = 0.035
    warmup_steps: int = 8
    _initial_object: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32), init=False
    )
    _initial_distance: float = field(default=0.0, init=False)
    _distances: deque[float] = field(default_factory=deque, init=False)
    _ever_near_object: bool = field(default=False, init=False)
    _ever_interacted: bool = field(default=False, init=False)
    _ever_lifted: bool = field(default=False, init=False)
    _step: int = field(default=0, init=False)

    def reset(self, info: Mapping[str, Any]) -> None:
        self._initial_object = self._vector(info, "object_position")
        self._initial_distance = float(
            info.get(
                "distance_to_goal",
                np.linalg.norm(
                    self._initial_object - self._vector(info, "goal_position")
                ),
            )
        )
        self._distances = deque(maxlen=max(2, int(self.progress_window)))
        self._ever_near_object = False
        self._ever_interacted = False
        self._ever_lifted = False
        self._step = 0

    @staticmethod
    def _vector(info: Mapping[str, Any], key: str) -> np.ndarray:
        value = np.asarray(info.get(key, np.zeros(3)), dtype=np.float32).reshape(-1)
        if value.size < 3:
            raise ValueError(f"info[{key!r}] must expose XYZ coordinates.")
        return value[:3].copy()

    def update(
        self,
        info: Mapping[str, Any],
        *,
        terminated: bool,
        truncated: bool,
    ) -> tuple[int, str]:
        self._step += 1
        object_position = self._vector(info, "object_position")
        goal_position = self._vector(info, "goal_position")
        hand_object_distance = float(
            info.get(
                "hand_object_distance",
                np.linalg.norm(
                    self._vector(info, "ee_position") - object_position
                ),
            )
        )
        goal_distance = float(
            info.get(
                "distance_to_goal",
                np.linalg.norm(object_position - goal_position),
            )
        )
        success = bool(info.get("success", info.get("is_success", False)))
        self._distances.append(goal_distance)
        self._ever_near_object |= hand_object_distance <= self.grasp_distance
        object_displacement = float(
            np.linalg.norm(object_position - self._initial_object)
        )
        lifted = bool(
            object_position[2] >= self._initial_object[2] + self.lift_delta
        )
        self._ever_lifted |= lifted
        self._ever_interacted |= self._ever_lifted or object_displacement >= 0.020

        # Terminal signals and observed physical faults have highest priority.
        if success:
            return 0, "normal"
        if bool(info.get("object_dropped", False)):
            return 1, "object_dropped"
        if bool(info.get("workspace_violation", False)):
            return 1, "workspace_violation"
        if truncated:
            return 1, "timeout"
        if terminated:
            return 1, str(info.get("failure_reason") or "terminal_failure")

        # A grasp attempt is known to have failed only after the gripper has
        # already reached the object and subsequently separated without lift.
        gripper = float(info.get("gripper_state", 0.0))
        if (
            self._step >= self.warmup_steps
            and self._ever_near_object
            and not self._ever_lifted
            and hand_object_distance >= self.failed_grasp_distance
            and gripper > 0.10
        ):
            return 1, "grasp_failed"

        # Progress/deviation rules are gated by observed object interaction so
        # normal pre-grasp reaching is never mislabeled as stalled.
        if self._ever_interacted:
            if goal_distance >= self._initial_distance + self.deviation_margin:
                return 1, "trajectory_deviation"
            if (
                len(self._distances) == self._distances.maxlen
                and max(self._distances) - min(self._distances)
                <= self.minimum_progress
                and goal_distance > 0.10
            ):
                return 1, "no_progress"
        return 0, "normal"
