"""Strict Meta-World 3.1.1 MT10/MT50 single-slot environment wrapper.

The official Gymnasium registrations expose MT10 and MT50 as vector
environments.  This module provides one independently controllable vector slot
with equivalent benchmark semantics:

* task classes and the 50 variants per class come from ``metaworld.MT10`` or
  ``metaworld.MT50``;
* the task one-hot follows ``benchmark.train_classes`` insertion order, exactly
  as Meta-World's :class:`~metaworld.wrappers.OneHotWrapper` does;
* observations are the untouched 39D V3 state followed by a 10D/50D task
  one-hot; and
* the horizon and ``info["success"]`` signal remain the official 500-step
  Meta-World definitions.

Gaussian disturbances are deliberately implemented outside the benchmark.
Observation noise is applied only to the first 39 values, so it can never
corrupt the discrete task identity appended to the observation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import logging
import threading
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    _GymEnv = gym.Env
except ImportError:  # pragma: no cover - construction reports the dependency.
    gym = None
    spaces = None
    _GymEnv = object


LOGGER = logging.getLogger(__name__)

SUPPORTED_METAWORLD_VERSION = "3.1.1"
SUPPORTED_BENCHMARKS = ("MT10", "MT50")
RAW_OBSERVATION_DIM = 39
ACTION_DIM = 4
OFFICIAL_MAX_EPISODE_STEPS = 500
OFFICIAL_VARIANTS_PER_TASK = 50


# Populated eagerly when possible and refreshed by ``_load_metaworld``.  Keeping
# import failure non-fatal preserves the repository's dependency-light imports;
# constructing this benchmark wrapper still fails loudly with an actionable
# error when Meta-World is unavailable.
try:  # pragma: no cover - availability depends on the active Python env.
    from metaworld.policies import ENV_POLICY_MAP as ENV_POLICY_MAP
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    ENV_POLICY_MAP: Mapping[str, type[Any]] = {}


@dataclass(frozen=True)
class _BenchmarkBundle:
    """Cached immutable view of one official seeded benchmark task bank."""

    names: tuple[str, ...]
    classes: tuple[type[Any], ...]
    tasks: tuple[tuple[Any, ...], ...]


_BENCHMARK_CACHE: dict[tuple[int, str, int], _BenchmarkBundle] = {}
_BENCHMARK_CACHE_LOCK = threading.RLock()


def _load_metaworld() -> Any:
    """Import and version-check the exact benchmark implementation."""

    try:
        import metaworld
        from metaworld.policies import ENV_POLICY_MAP as official_policy_map
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Meta-World 3.1.1 is required for MT10/MT50. Run setup.sh first."
        ) from exc

    try:
        installed = importlib.metadata.version("metaworld")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Cannot determine the installed Meta-World version.") from exc
    if installed != SUPPORTED_METAWORLD_VERSION:
        raise RuntimeError(
            "REIMMetaWorldMultiTaskEnv requires Meta-World "
            f"{SUPPORTED_METAWORLD_VERSION}, found {installed}."
        )

    global ENV_POLICY_MAP
    ENV_POLICY_MAP = official_policy_map
    return metaworld


def _official_bundle(metaworld: Any, benchmark_name: str, seed: int) -> _BenchmarkBundle:
    """Return the official ordered classes and 50 variants for a benchmark."""

    key = (id(metaworld), benchmark_name, int(seed))
    with _BENCHMARK_CACHE_LOCK:
        cached = _BENCHMARK_CACHE.get(key)
        if cached is not None:
            return cached

        benchmark_cls = getattr(metaworld, benchmark_name)
        benchmark = benchmark_cls(seed=int(seed))
        if benchmark.test_tasks or benchmark.test_classes:
            raise RuntimeError(
                f"{benchmark_name} unexpectedly exposes a non-empty test split."
            )

        names = tuple(benchmark.train_classes.keys())
        expected_tasks = 10 if benchmark_name == "MT10" else 50
        if len(names) != expected_tasks:
            raise RuntimeError(
                f"{benchmark_name} contains {len(names)} classes, expected {expected_tasks}."
            )

        classes = tuple(benchmark.train_classes[name] for name in names)
        grouped = tuple(
            tuple(task for task in benchmark.train_tasks if task.env_name == name)
            for name in names
        )
        invalid = {
            name: len(variants)
            for name, variants in zip(names, grouped, strict=True)
            if len(variants) != OFFICIAL_VARIANTS_PER_TASK
        }
        if invalid:
            raise RuntimeError(
                "Official benchmark task bank does not contain 50 variants per "
                f"class: {invalid}"
            )

        bundle = _BenchmarkBundle(names=names, classes=classes, tasks=grouped)
        _BENCHMARK_CACHE[key] = bundle
        return bundle


def _nonnegative_std(value: float, *, name: str) -> float:
    std = float(value)
    if not np.isfinite(std) or std < 0.0:
        raise ValueError(f"{name} must be a finite non-negative value.")
    return std


class REIMMetaWorldMultiTaskEnv(_GymEnv):
    """One selectable MT10/MT50 task slot with official one-hot semantics.

    Parameters
    ----------
    benchmark:
        ``"MT10"`` or ``"MT50"``.
    task_id:
        Integer in official benchmark order, or an exact ``*-v3`` task name.
    variant_id:
        One of the 50 official task variants.  ``None`` (the default) samples a
        variant on every reset, matching the official ``RandomTaskSelectWrapper``.
    seed:
        Seeds benchmark task generation, variant selection, backend reset, and
        both independent Gaussian-noise streams.
    action_noise_std / observation_noise_std:
        Gaussian standard deviations. Observation noise affects only the raw
        39D prefix and never the task one-hot.
    render_mode:
        Any render mode accepted by Meta-World 3.1.1.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 80,
    }

    def __init__(
        self,
        benchmark: str = "MT10",
        *,
        task_id: int | str = 0,
        variant_id: int | None = None,
        seed: int = 0,
        action_noise_std: float = 0.0,
        observation_noise_std: float = 0.0,
        render_mode: str | None = None,
    ) -> None:
        if gym is None or spaces is None:
            raise RuntimeError(
                "Gymnasium and Meta-World 3.1.1 are required for the multitask env."
            )
        super().__init__()

        benchmark_name = str(benchmark).strip().upper()
        if benchmark_name not in SUPPORTED_BENCHMARKS:
            raise ValueError("benchmark must be 'MT10' or 'MT50'.")

        self.benchmark_name = benchmark_name
        self.benchmark_seed = int(seed)
        self.render_mode = render_mode
        self.max_episode_steps = OFFICIAL_MAX_EPISODE_STEPS
        self.action_noise_std = _nonnegative_std(
            action_noise_std, name="action_noise_std"
        )
        self.observation_noise_std = _nonnegative_std(
            observation_noise_std, name="observation_noise_std"
        )

        self._metaworld = _load_metaworld()
        self._bundle = _official_bundle(
            self._metaworld, self.benchmark_name, self.benchmark_seed
        )
        self._task_names = self._bundle.names
        self.num_tasks = len(self._task_names)
        self.num_variants = OFFICIAL_VARIANTS_PER_TASK
        self.state_dim = RAW_OBSERVATION_DIM + self.num_tasks
        self.action_dim = ACTION_DIM

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
        )
        # Meta-World's raw state and official OneHotWrapper are float64.
        low = np.concatenate(
            [
                np.full(RAW_OBSERVATION_DIM, -np.inf, dtype=np.float64),
                np.zeros(self.num_tasks, dtype=np.float64),
            ]
        )
        high = np.concatenate(
            [
                np.full(RAW_OBSERVATION_DIM, np.inf, dtype=np.float64),
                np.ones(self.num_tasks, dtype=np.float64),
            ]
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)

        self._seed = int(seed)
        self._variant_rng = np.random.default_rng()
        self._action_noise_rng = np.random.default_rng()
        self._observation_noise_rng = np.random.default_rng()
        self._reseed(self._seed)

        self._backend_env: Any | None = None
        self._task_id = -1
        self._variant_id = -1
        self._current_task: Any | None = None
        self._expert_policy: Any | None = None
        self._raw_observation = np.zeros(RAW_OBSERVATION_DIM, dtype=np.float64)
        self._last_clean_state = np.zeros(self.state_dim, dtype=np.float64)
        self._last_observed_state = np.zeros(self.state_dim, dtype=np.float64)
        self._step_count = 0
        self._episode_index = -1
        self._needs_reset = True
        self._closed = False

        randomize_variant = variant_id is None
        self._randomize_variant_on_reset = False
        initial_variant = 0 if variant_id is None else self._resolve_variant(variant_id)
        self.select_task(task_id, initial_variant)
        self._randomize_variant_on_reset = randomize_variant
        LOGGER.info(
            "Initialized %s task=%s[%d] variant=%d state=%d action=%d seed=%d",
            self.benchmark_name,
            self.task_name,
            self.task_id,
            self.variant_id,
            self.state_dim,
            self.action_dim,
            self._seed,
        )

    @property
    def task_names(self) -> tuple[str, ...]:
        """Official ordered task names used by the one-hot encoding."""

        return self._task_names

    @property
    def task_id(self) -> int:
        return self._task_id

    @property
    def task_name(self) -> str:
        return self._task_names[self._task_id]

    @property
    def variant_id(self) -> int:
        return self._variant_id

    @property
    def current_task(self) -> Any:
        """Current official :class:`metaworld.types.Task` object."""

        if self._current_task is None:  # pragma: no cover - constructor activates it.
            raise RuntimeError("No Meta-World task is active.")
        return self._current_task

    @property
    def backend_env(self) -> Any:
        """Underlying official Sawyer environment for advanced diagnostics."""

        if self._backend_env is None:
            raise RuntimeError("The backend environment is closed.")
        return self._backend_env

    @property
    def raw_observation(self) -> np.ndarray:
        """Latest clean, official 39D Meta-World observation."""

        return self._raw_observation.copy()

    @property
    def task_one_hot(self) -> np.ndarray:
        one_hot = np.zeros(self.num_tasks, dtype=np.float64)
        one_hot[self.task_id] = 1.0
        return one_hot

    @property
    def expert_policy(self) -> Any:
        """Official scripted policy instance matching the active task."""

        if self._expert_policy is None:  # pragma: no cover - coverage is audited.
            raise RuntimeError(f"No official scripted expert for {self.task_name}.")
        return self._expert_policy

    @property
    def task_metadata(self) -> dict[str, Any]:
        """Serializable identity and benchmark metadata for the active Task."""

        task = self.current_task
        return {
            "benchmark": self.benchmark_name,
            "metaworld_version": SUPPORTED_METAWORLD_VERSION,
            "benchmark_seed": self.benchmark_seed,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "variant_id": self.variant_id,
            "num_tasks": self.num_tasks,
            "num_variants": self.num_variants,
            "task_sha256": hashlib.sha256(task.data).hexdigest(),
            "raw_observation_dim": RAW_OBSERVATION_DIM,
            "task_one_hot_dim": self.num_tasks,
            "observation_dim": self.state_dim,
            "action_dim": ACTION_DIM,
            "max_episode_steps": OFFICIAL_MAX_EPISODE_STEPS,
        }

    def _reseed(self, seed: int) -> None:
        self._seed = int(seed)
        children = np.random.SeedSequence(self._seed).spawn(3)
        self._variant_rng = np.random.default_rng(children[0])
        self._action_noise_rng = np.random.default_rng(children[1])
        self._observation_noise_rng = np.random.default_rng(children[2])
        self.action_space.seed(self._seed)
        self.observation_space.seed(self._seed)

    def _resolve_task_id(self, task_id: int | str) -> int:
        if isinstance(task_id, str):
            name = task_id.strip().lower().replace("_", "-")
            try:
                return self._task_names.index(name)
            except ValueError as exc:
                raise ValueError(
                    f"{name!r} is not in {self.benchmark_name}: {self._task_names}"
                ) from exc
        if isinstance(task_id, (bool, np.bool_)):
            raise TypeError("task_id must be an integer index or task name, not bool.")
        index = int(task_id)
        if index != task_id or not 0 <= index < self.num_tasks:
            raise ValueError(f"task_id must lie in [0, {self.num_tasks - 1}].")
        return index

    def _resolve_variant(self, variant_id: int) -> int:
        if isinstance(variant_id, (bool, np.bool_)):
            raise TypeError("variant_id must be an integer, not bool.")
        index = int(variant_id)
        if index != variant_id or not 0 <= index < self.num_variants:
            raise ValueError(f"variant_id must lie in [0, {self.num_variants - 1}].")
        return index

    def _new_backend(self, task_id: int) -> Any:
        env_cls = self._bundle.classes[task_id]
        backend = env_cls(render_mode=self.render_mode)
        if tuple(backend.observation_space.shape) != (RAW_OBSERVATION_DIM,):
            backend.close()
            raise RuntimeError(
                f"{self._task_names[task_id]} observation space is not 39D."
            )
        if tuple(backend.action_space.shape) != (ACTION_DIM,):
            backend.close()
            raise RuntimeError(f"{self._task_names[task_id]} action space is not 4D.")
        backend.max_path_length = OFFICIAL_MAX_EPISODE_STEPS
        return backend

    def _activate(self, task_id: int, variant_id: int, task: Any | None = None) -> None:
        if self._closed:
            raise RuntimeError("Cannot select a task on a closed environment.")
        official_task = self._bundle.tasks[task_id][variant_id] if task is None else task
        if official_task.env_name != self._task_names[task_id]:
            raise ValueError("Task.env_name does not match the requested task_id.")

        if self._backend_env is None or task_id != self._task_id:
            if self._backend_env is not None:
                self._backend_env.close()
            self._backend_env = self._new_backend(task_id)

        self._backend_env.set_task(official_task)
        self._task_id = task_id
        self._variant_id = variant_id
        self._current_task = official_task
        policy_cls = ENV_POLICY_MAP.get(self.task_name)
        self._expert_policy = None if policy_cls is None else policy_cls()
        self._last_clean_state = self._clean_state(self._raw_observation)
        self._last_observed_state = self._last_clean_state.copy()
        self._needs_reset = True

    def select_task(self, task_id: int | str, variant_id: int) -> None:
        """Select one official task class and one of its 50 variants."""

        self._activate(self._resolve_task_id(task_id), self._resolve_variant(variant_id))
        self._randomize_variant_on_reset = False

    def enable_random_variant_sampling(self, enabled: bool = True) -> None:
        """Toggle official-style random selection among the 50 variants on reset."""

        self._randomize_variant_on_reset = bool(enabled)

    def set_task(self, task: Any) -> None:
        """Replay an official Task object from this seeded benchmark bank.

        This mirrors Meta-World's manual ``env.set_task(task)`` workflow while
        retaining a correct task one-hot and variant identifier.
        """

        try:
            task_id = self._task_names.index(str(task.env_name))
            variant_id = next(
                index
                for index, candidate in enumerate(self._bundle.tasks[task_id])
                if candidate.data == task.data
            )
        except (AttributeError, ValueError, StopIteration) as exc:
            raise ValueError(
                "Task is not one of this environment's official seeded variants."
            ) from exc
        self._activate(task_id, variant_id, task=task)
        self._randomize_variant_on_reset = False

    def _validate_raw(self, observation: Any) -> np.ndarray:
        raw = np.asarray(observation, dtype=np.float64).reshape(-1)
        if raw.shape != (RAW_OBSERVATION_DIM,):
            raise RuntimeError(
                f"Meta-World returned observation shape {raw.shape}, expected (39,)."
            )
        if not np.all(np.isfinite(raw)):
            raise RuntimeError("Meta-World returned NaN or infinity in observation.")
        return raw

    def _clean_state(self, raw: np.ndarray) -> np.ndarray:
        return np.concatenate([raw, self.task_one_hot]).astype(np.float64, copy=False)

    def apply_action_noise(
        self, action: Sequence[float] | np.ndarray, *, std: float | None = None
    ) -> np.ndarray:
        """Clip a 4D command, add Gaussian action noise, then clip again."""

        command = np.asarray(action, dtype=np.float32).reshape(-1)
        if command.shape != (ACTION_DIM,):
            raise ValueError(f"action must have shape (4,), got {command.shape}.")
        if not np.all(np.isfinite(command)):
            raise ValueError("action contains NaN or infinity.")
        sigma = self.action_noise_std if std is None else _nonnegative_std(
            std, name="action noise std"
        )
        command = np.clip(command, self.action_space.low, self.action_space.high)
        if sigma == 0.0:
            return command.astype(np.float32, copy=True)
        noise = self._action_noise_rng.normal(0.0, sigma, size=ACTION_DIM)
        return np.clip(command + noise, self.action_space.low, self.action_space.high).astype(
            np.float32
        )

    def apply_observation_noise(
        self, observation: Sequence[float] | np.ndarray, *, std: float | None = None
    ) -> np.ndarray:
        """Add Gaussian noise to a raw/full observation without touching one-hot."""

        value = np.asarray(observation, dtype=np.float64).reshape(-1)
        if value.shape not in {(RAW_OBSERVATION_DIM,), (self.state_dim,)}:
            raise ValueError(
                f"observation must have shape (39,) or ({self.state_dim},), "
                f"got {value.shape}."
            )
        if not np.all(np.isfinite(value)):
            raise ValueError("observation contains NaN or infinity.")
        sigma = self.observation_noise_std if std is None else _nonnegative_std(
            std, name="observation noise std"
        )
        result = value.copy()
        if sigma > 0.0:
            result[:RAW_OBSERVATION_DIM] += self._observation_noise_rng.normal(
                0.0, sigma, size=RAW_OBSERVATION_DIM
            )
        # Deliberately leave result[39:] bit-for-bit unchanged.
        return result

    def _record_observation(self, raw: Any) -> np.ndarray:
        self._raw_observation = self._validate_raw(raw).copy()
        self._last_clean_state = self._clean_state(self._raw_observation)
        self._last_observed_state = self.apply_observation_noise(
            self._last_clean_state
        )
        return self._last_observed_state.copy()

    def _info(self, backend_info: Mapping[str, Any] | None) -> dict[str, Any]:
        info = dict(backend_info or {})
        # Step info is the official task-specific success signal. Reset has no
        # official success field, for which False is the only valid initial value.
        info.setdefault("success", False)
        official_success = bool(np.asarray(info["success"]).reshape(-1)[0])
        info["is_success"] = official_success
        info["official_success"] = official_success
        info["benchmark"] = self.benchmark_name
        info["task_id"] = self.task_id
        info["task_name"] = self.task_name
        info["variant_id"] = self.variant_id
        info["episode_seed"] = self._seed
        info["step"] = self._step_count
        return info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset, optionally selecting ``task_id``/``variant_id``/``task``."""

        if self._closed:
            raise RuntimeError("Cannot reset a closed environment.")
        if seed is not None:
            super().reset(seed=int(seed))
            self._reseed(int(seed))

        opts = dict(options or {})
        unknown = set(opts) - {"task_id", "task_name", "variant_id", "task"}
        if unknown:
            raise ValueError(f"Unsupported reset options: {sorted(unknown)}")
        if "task_id" in opts and "task_name" in opts:
            raise ValueError("Specify only one of task_id and task_name.")
        if "task" in opts and ({"task_id", "task_name", "variant_id"} & set(opts)):
            raise ValueError("A Task object cannot be combined with task/variant indices.")

        if "task" in opts:
            self.set_task(opts["task"])
        else:
            selected_task: int | str = opts.get(
                "task_id", opts.get("task_name", self.task_id)
            )
            task_index = self._resolve_task_id(selected_task)
            if "variant_id" in opts:
                requested_variant = opts["variant_id"]
                if requested_variant is None:
                    variant_index = int(self._variant_rng.integers(self.num_variants))
                else:
                    variant_index = self._resolve_variant(requested_variant)
            elif self._randomize_variant_on_reset:
                variant_index = int(self._variant_rng.integers(self.num_variants))
            else:
                variant_index = self.variant_id
            self._activate(task_index, variant_index)

        backend_seed = self._seed if seed is None else int(seed)
        raw, backend_info = self.backend_env.reset(seed=backend_seed)
        self._step_count = 0
        self._episode_index += 1
        self._needs_reset = False
        observed = self._record_observation(raw)
        info = self._info(backend_info)
        info["episode_index"] = self._episode_index
        info["task_metadata"] = self.task_metadata
        return observed, info

    def step(
        self, action: Sequence[float] | np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one noisy 4D action and preserve official success semantics."""

        if self._closed:
            raise RuntimeError("Cannot step a closed environment.")
        if self._needs_reset:
            raise RuntimeError("Call reset() before step(), and after truncation.")
        commanded = np.asarray(action, dtype=np.float32).reshape(-1)
        if commanded.shape != (ACTION_DIM,) or not np.all(np.isfinite(commanded)):
            raise ValueError("action must be a finite array with shape (4,).")
        clipped_command = np.clip(
            commanded, self.action_space.low, self.action_space.high
        ).astype(np.float32)
        executed = self.apply_action_noise(clipped_command)
        raw, reward, terminated, truncated, backend_info = self.backend_env.step(executed)
        self._step_count += 1
        if self._step_count >= OFFICIAL_MAX_EPISODE_STEPS:
            truncated = True
        self._needs_reset = bool(terminated or truncated)

        observed = self._record_observation(raw)
        info = self._info(backend_info)
        info.update(
            {
                "commanded_action": clipped_command.copy(),
                "executed_action": executed.copy(),
                "action_noise": (executed - clipped_command).astype(np.float32),
            }
        )
        return observed, float(reward), bool(terminated), bool(truncated), info

    def get_state(self, *, noisy: bool = False) -> np.ndarray:
        """Return the full clean state, or the last noisy observation."""

        source = self._last_observed_state if noisy else self._last_clean_state
        return source.copy()

    def get_expert_action(
        self,
        observation: Sequence[float] | np.ndarray | None = None,
        *,
        clip: bool = True,
    ) -> np.ndarray:
        """Return a 4D action from the active official scripted expert.

        Official experts consume raw 39D observations, so a full 49D/89D state
        is accepted but its one-hot suffix is intentionally stripped.
        """

        if observation is None:
            raw = self.raw_observation
        else:
            value = np.asarray(observation, dtype=np.float64).reshape(-1)
            if value.shape == (self.state_dim,):
                raw = value[:RAW_OBSERVATION_DIM]
            elif value.shape == (RAW_OBSERVATION_DIM,):
                raw = value
            else:
                raise ValueError(
                    f"expert observation must have shape (39,) or ({self.state_dim},)."
                )
        action = np.asarray(self.expert_policy.get_action(raw), dtype=np.float32).reshape(-1)
        if action.shape != (ACTION_DIM,) or not np.all(np.isfinite(action)):
            raise RuntimeError("Official expert returned an invalid action.")
        if clip:
            action = np.clip(action, self.action_space.low, self.action_space.high)
        return action.astype(np.float32, copy=False)

    def render(self) -> Any:
        return self.backend_env.render()

    def close(self) -> None:
        if self._closed:
            return
        if self._backend_env is not None:
            self._backend_env.close()
            self._backend_env = None
        self._closed = True


# Short alias for callers that do not use the REIM package prefix.
MetaWorldMultiTaskEnv = REIMMetaWorldMultiTaskEnv


__all__ = [
    "ACTION_DIM",
    "ENV_POLICY_MAP",
    "MetaWorldMultiTaskEnv",
    "OFFICIAL_MAX_EPISODE_STEPS",
    "OFFICIAL_VARIANTS_PER_TASK",
    "RAW_OBSERVATION_DIM",
    "REIMMetaWorldMultiTaskEnv",
    "SUPPORTED_BENCHMARKS",
    "SUPPORTED_METAWORLD_VERSION",
]
