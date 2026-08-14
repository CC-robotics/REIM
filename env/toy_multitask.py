"""Deterministic toy MT10/MT50 benchmark for dependency-light CI smoke runs.

This module mirrors the exact Meta-World 3.1.1 interface surface that the REIM
multi-task scripts consume, so every pipeline stage can run end-to-end without
MuJoCo:

* ``MT10`` / ``MT50`` benchmark classes with ``train_classes``, ``train_tasks``
  (50 variants per class), and empty ``test_tasks`` / ``test_classes``;
* per-task environment classes accepting ``render_mode=None``, exposing
  ``observation_space.shape == (39,)``, ``action_space.shape == (4,)``, a
  settable ``max_path_length``, ``set_task``, ``reset(seed=...)``, the
  Gymnasium five-value ``step`` contract with ``info["success"]``, ``seed()``,
  and ``close()``;
* an ``ENV_POLICY_MAP`` of stateless scripted experts whose ``get_action``
  consumes the raw 39D observation and returns a 4D action in ``[-1, 1]``.

The toy dynamics are a deliberately simple reactive pick-and-place world.
They exist so CI can verify wiring, provenance, determinism, and file
isolation.  Toy outputs are engineering artifacts only and are never benchmark
evidence; the runner writes them exclusively under ``datasets/smoke``,
``checkpoints/smoke``, and ``results/smoke``.  Like the PickPlace toy backend,
nothing ever falls back to this module silently: every entry point requires an
explicit ``--backend toy``.

Raw 39D observation layout (float64)::

    [0:3]   end-effector position
    [3]     gripper command state in {-1, +1}
    [4:7]   object position
    [7:11]  object orientation (fixed identity quaternion)
    [11:14] previous end-effector position
    [14:18] zeros (reserved)
    [18:21] goal position
    [21:39] zeros (reserved)
"""

from __future__ import annotations

from collections import OrderedDict
from collections import namedtuple
import hashlib
from typing import Any, Mapping

import numpy as np


TOY_VERSION = "toy-ci-1.0"
RAW_OBSERVATION_DIM = 39
ACTION_DIM = 4
VARIANTS_PER_TASK = 50
DEFAULT_MAX_PATH_LENGTH = 500

MT10_TASKS: tuple[str, ...] = (
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

MT50_TASKS: tuple[str, ...] = (
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

_BENCHMARK_TASKS = {"MT10": MT10_TASKS, "MT50": MT50_TASKS}

# Task payloads are stable bytes, exactly like Meta-World's pickled Task.data,
# so payload-hash provenance and bank-separation audits work unchanged.
Task = namedtuple("Task", ("env_name", "data"))

# Workspace and dynamics constants.  All positions live in a bounded region so
# the scripted expert always finishes well inside the 500-step horizon, while
# noisy or weak learned policies can still miss the grasp or knock the object
# out of the reachable workspace, producing genuine recoverable and
# unrecoverable failure states.
_STEP_SIZE = 0.05
_TABLE_HEIGHT = 0.03
_APPROACH_LIFT = 0.12
_GRASP_RADIUS = 0.06
_NEAR_MISS_RADIUS = 0.15
_KNOCK_DISTANCE = 0.30
_WORKSPACE_RADIUS = 1.0
_SUCCESS_THRESHOLD = 0.05
_RELEASE_HEIGHT = 0.06


def _stable_uint64(*parts: str) -> int:
    material = "|".join(parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def _task_center(env_name: str, kind: str) -> np.ndarray:
    """Deterministic per-task anchor point inside the bounded workspace."""

    rng = np.random.default_rng(_stable_uint64("reim-toy-center", env_name, kind))
    xy = rng.uniform(-0.35, 0.35, size=2)
    # Goals sit on the table plane so a released object can actually reach
    # them; hand anchors float above the workspace.
    z = _TABLE_HEIGHT if kind != "hand" else rng.uniform(0.25, 0.35)
    return np.array([xy[0], xy[1], z], dtype=np.float64)


class _ToySpace:
    """Minimal stand-in for ``gymnasium.spaces.Box`` (shape/low/high/seed)."""

    def __init__(self, shape: tuple[int, ...], low: float, high: float) -> None:
        self.shape = shape
        self.low = np.full(shape, low, dtype=np.float64)
        self.high = np.full(shape, high, dtype=np.float64)

    def seed(self, seed: int | None = None) -> None:  # pragma: no cover - trivial
        return None


class _ToySawyerEnv:
    """One toy task environment bound to a specific task class."""

    _env_name: str = ""

    def __init__(self, render_mode: str | None = None) -> None:
        if not self._env_name:
            raise TypeError("_ToySawyerEnv must be subclassed per task.")
        self.render_mode = render_mode
        self.observation_space = _ToySpace((RAW_OBSERVATION_DIM,), -np.inf, np.inf)
        self.action_space = _ToySpace((ACTION_DIM,), -1.0, 1.0)
        self.max_path_length = DEFAULT_MAX_PATH_LENGTH
        self._object_center = _task_center(self._env_name, "object")
        self._goal_center = _task_center(self._env_name, "goal")
        self._hand_center = _task_center(self._env_name, "hand") + np.array(
            [0.0, 0.0, 0.25]
        )
        self._rng = np.random.default_rng()
        self._task: Task | None = None
        self._hand = np.zeros(3)
        self._prev_hand = np.zeros(3)
        self._object = np.zeros(3)
        self._goal = np.zeros(3)
        self._gripper = -1.0
        self._holding = False
        self._lost = False
        self._steps = 0
        self._needs_reset = True

    # -- Meta-World compatible surface -------------------------------------

    def seed(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def set_task(self, task: Task) -> None:
        if getattr(task, "env_name", None) != self._env_name:
            raise ValueError(
                f"{self._env_name} cannot accept task for {getattr(task, 'env_name', None)!r}."
            )
        bytes(task.data)  # fail closed on non-bytes-like payloads
        self._task = task
        self._needs_reset = True

    def reset(self, *, seed: int | None = None, options: Any = None):
        del options
        if self._task is None:
            raise RuntimeError("set_task must be called before reset().")
        episode_seed = 0 if seed is None else int(seed)
        self._rng = np.random.default_rng(
            _stable_uint64(
                "reim-toy-reset",
                self._task.env_name,
                self._task.data.hex(),
                str(episode_seed),
            )
        )
        jitter = lambda scale: self._rng.normal(0.0, scale, size=3)
        self._hand = self._hand_center + jitter(0.04)
        self._hand[2] = max(self._hand[2], 0.20)
        self._prev_hand = self._hand.copy()
        self._object = self._object_center + jitter(0.05)
        self._object[2] = _TABLE_HEIGHT
        self._goal = self._goal_center + jitter(0.05)
        self._goal[2] = _TABLE_HEIGHT + float(self._rng.uniform(0.0, 0.02))
        self._gripper = -1.0
        self._holding = False
        self._lost = False
        self._steps = 0
        self._needs_reset = False
        return self._observation(), {"success": False}

    def step(self, action: Any):
        if self._needs_reset:
            raise RuntimeError("reset() must be called before step().")
        command = np.asarray(action, dtype=np.float64).reshape(-1)
        if command.shape != (ACTION_DIM,) or not np.all(np.isfinite(command)):
            raise ValueError("action must be a finite array with shape (4,).")
        command = np.clip(command, -1.0, 1.0)

        self._prev_hand = self._hand.copy()
        self._hand = self._hand + _STEP_SIZE * command[:3]
        self._hand[2] = max(self._hand[2], _TABLE_HEIGHT)
        self._gripper = 1.0 if command[3] > 0.0 else -1.0

        distance = float(np.linalg.norm(self._hand - self._object))
        if self._gripper > 0.0 and not self._holding:
            if distance < _GRASP_RADIUS:
                self._holding = True
            elif distance < _NEAR_MISS_RADIUS:
                # A near-miss grasp knocks the object away from the hand.  The
                # knock can push it outside the reachable workspace, which is
                # the toy world's unrecoverable failure mode.
                direction = self._object - self._hand
                norm = float(np.linalg.norm(direction))
                if norm > 1e-9:
                    self._object = self._object + direction / norm * _KNOCK_DISTANCE
                self._object[2] = _TABLE_HEIGHT
        if self._holding:
            self._object = self._hand + np.array([0.0, 0.0, -0.02])
            if self._gripper < 0.0:
                self._holding = False
                self._object = np.array(
                    [self._object[0], self._object[1], _TABLE_HEIGHT]
                )
        if float(np.linalg.norm(self._object[:2])) > _WORKSPACE_RADIUS:
            self._lost = True

        self._steps += 1
        success = (
            not self._holding
            and not self._lost
            and float(np.linalg.norm(self._object - self._goal)) < _SUCCESS_THRESHOLD
        )
        # Reward shaped toward object-goal progress so the reward-window
        # heuristic sees genuine progress on success and a plateau when stuck.
        reward = float(10.0 * np.exp(-5.0 * np.linalg.norm(self._object - self._goal)))
        terminated = False
        truncated = self._steps >= int(self.max_path_length)
        if terminated or truncated:
            self._needs_reset = True
        info = {
            "success": bool(success),
            "grasped": bool(self._holding),
            "object_lost": bool(self._lost),
        }
        return self._observation(), reward, terminated, truncated, info

    def render(self) -> None:  # pragma: no cover - CI never renders.
        return None

    def close(self) -> None:  # pragma: no cover - trivial
        return None

    # -- internals ----------------------------------------------------------

    def _observation(self) -> np.ndarray:
        raw = np.zeros(RAW_OBSERVATION_DIM, dtype=np.float64)
        raw[0:3] = self._hand
        raw[3] = self._gripper
        raw[4:7] = self._object
        raw[10] = 1.0  # identity quaternion (w)
        raw[11:14] = self._prev_hand
        raw[18:21] = self._goal
        return raw


class _ToyScriptedExpert:
    """Stateless reactive expert reading only the raw 39D observation."""

    @staticmethod
    def _move(hand: np.ndarray, target: np.ndarray, gripper: float) -> np.ndarray:
        displacement = np.clip((target - hand) / _STEP_SIZE * 0.9, -1.0, 1.0)
        return np.concatenate([displacement, np.asarray([gripper])])

    def get_action(self, raw_observation: Any) -> np.ndarray:
        raw = np.asarray(raw_observation, dtype=np.float64).reshape(-1)
        if raw.shape != (RAW_OBSERVATION_DIM,):
            raise ValueError(
                f"toy expert expects a (39,) observation, got {raw.shape}."
            )
        hand = raw[0:3]
        gripper = raw[3]
        obj = raw[4:7]
        goal = raw[18:21]
        distance = float(np.linalg.norm(hand - obj))
        holding = gripper > 0.0 and distance < _GRASP_RADIUS

        if not holding:
            # Approach with an open gripper and only close it when already
            # inside the latch radius, so the expert never knocks the object.
            if distance < 0.035:
                return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            if np.linalg.norm(hand[:2] - obj[:2]) > 0.02:
                target = obj + np.array([0.0, 0.0, _APPROACH_LIFT])
            else:
                target = obj.copy()
            return self._move(hand, target, -1.0).astype(np.float32)

        # Holding: lift, transport above the goal, then descend and release.
        if np.linalg.norm(obj - goal) < _SUCCESS_THRESHOLD and (
            obj[2] <= goal[2] + _RELEASE_HEIGHT
        ):
            return np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        if np.linalg.norm(hand[:2] - goal[:2]) > 0.02:
            target = np.array([hand[0], hand[1], goal[2] + _APPROACH_LIFT])
            if np.linalg.norm(hand[:2] - goal[:2]) > 0.02 and (
                hand[2] >= goal[2] + _APPROACH_LIFT - 0.02
            ):
                target = np.array([goal[0], goal[1], goal[2] + _APPROACH_LIFT])
        else:
            # Descend until the held object nearly touches the goal point;
            # the object hangs 0.02 below the hand.
            target = goal + np.array([0.0, 0.0, 0.02])
        return self._move(hand, target, 1.0).astype(np.float32)


def _make_env_class(env_name: str) -> type:
    return type(
        f"ToySawyer{env_name.replace('-', ' ').title().replace(' ', '')}Env",
        (_ToySawyerEnv,),
        {"_env_name": env_name, "__module__": __name__},
    )


class _ToyBenchmark:
    """Seeded toy benchmark bank mirroring ``metaworld.MT10``/``MT50``."""

    _benchmark_name = ""
    _task_names: tuple[str, ...] = ()

    def __init__(self, seed: int = 0) -> None:
        self.benchmark_seed = int(seed)
        self.train_classes: Mapping[str, type] = OrderedDict(
            (name, _make_env_class(name)) for name in self._task_names
        )
        self.test_classes: Mapping[str, type] = OrderedDict()
        # Variant payloads are deterministic functions of the benchmark seed,
        # exactly like Meta-World's per-seed goal sampling: banks collected
        # with different registered seeds expose disjoint payload identities,
        # which the bank-separation audit requires.
        self.train_tasks = [
            Task(
                env_name=name,
                data=hashlib.sha256(
                    f"reim-toy-task|{self._benchmark_name}|{self.benchmark_seed}|"
                    f"{name}|{variant}".encode("utf-8")
                ).digest(),
            )
            for name in self._task_names
            for variant in range(VARIANTS_PER_TASK)
        ]
        self.test_tasks: list[Task] = []


class MT10(_ToyBenchmark):
    _benchmark_name = "MT10"
    _task_names = MT10_TASKS


class MT50(_ToyBenchmark):
    _benchmark_name = "MT50"
    _task_names = MT50_TASKS


def _make_expert_class(env_name: str) -> type:
    return type(
        f"Toy{env_name.replace('-', ' ').title().replace(' ', '')}Policy",
        (_ToyScriptedExpert,),
        {"__module__": __name__},
    )


ENV_POLICY_MAP: Mapping[str, type] = {
    name: _make_expert_class(name) for name in (*MT10_TASKS, *MT50_TASKS)
}


__all__ = [
    "ACTION_DIM",
    "DEFAULT_MAX_PATH_LENGTH",
    "ENV_POLICY_MAP",
    "MT10",
    "MT10_TASKS",
    "MT50",
    "MT50_TASKS",
    "RAW_OBSERVATION_DIM",
    "TOY_VERSION",
    "Task",
    "VARIANTS_PER_TASK",
]
