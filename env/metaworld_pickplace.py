"""Gymnasium-compatible PickPlace environment used throughout REIM.

The benchmark backend wraps the current Farama Meta-World ``MT1`` API and
retains a small explicit ``toy`` backend for deterministic CI.  The toy
backend is never selected implicitly: benchmark runs fail loudly when
Meta-World is unavailable.

Meta-World's 39-dimensional observation does *not* expose the Sawyer joint
positions.  REIM therefore uses a semantic 21-dimensional state by default:

``joint qpos (7) + TCP position/quaternion (7) + object xyz (3) +
goal xyz (3) + gripper (1)``.

For official benchmark comparisons, ``state_mode="raw"`` returns the
unmodified 39-dimensional Meta-World observation.  In both modes the latest
raw observation remains available through :attr:`raw_observation`.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import logging
import math
import warnings
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

try:  # Gymnasium is optional only so the explicit toy CI backend can bootstrap.
    import gymnasium as gym
    from gymnasium import spaces

    _GymEnv = gym.Env
except ImportError:  # pragma: no cover - exercised only before setup.sh.
    gym = None

    class _FallbackBox:
        """Small Box subset used by the dependency-light toy smoke test."""

        def __init__(
            self,
            low: float | np.ndarray,
            high: float | np.ndarray,
            shape: tuple[int, ...],
            dtype: np.dtype | type = np.float32,
        ) -> None:
            self.shape = tuple(shape)
            self.dtype = np.dtype(dtype)
            self.low = np.broadcast_to(low, self.shape).astype(self.dtype)
            self.high = np.broadcast_to(high, self.shape).astype(self.dtype)
            self._rng = np.random.default_rng()

        def seed(self, seed: int | None = None) -> list[int | None]:
            self._rng = np.random.default_rng(seed)
            return [seed]

        def sample(self) -> np.ndarray:
            return self._rng.uniform(self.low, self.high).astype(self.dtype)

        def contains(self, value: object) -> bool:
            array = np.asarray(value)
            return (
                array.shape == self.shape
                and np.all(np.isfinite(array))
                and np.all(array >= self.low)
                and np.all(array <= self.high)
            )

    class _FallbackSpaces:
        Box = _FallbackBox

    spaces = _FallbackSpaces()
    _GymEnv = object


LOGGER = logging.getLogger(__name__)

SEMANTIC_STATE_DIM = 21
RAW_OBSERVATION_DIM = 39
ACTION_DIM = 4

SEMANTIC_STATE_LAYOUT: dict[str, dict[str, Any]] = {
    "joint_positions": {
        "slice": [0, 7],
        "shape": [7],
        "description": "Sawyer joint qpos in MuJoCo model order.",
    },
    "end_effector_position": {
        "slice": [7, 10],
        "shape": [3],
        "description": "TCP/hand Cartesian XYZ.",
    },
    "end_effector_quaternion": {
        "slice": [10, 14],
        "shape": [4],
        "description": "TCP/hand quaternion in MuJoCo WXYZ order.",
    },
    "object_position": {
        "slice": [14, 17],
        "shape": [3],
        "description": "Manipulated object Cartesian XYZ.",
    },
    "goal_position": {
        "slice": [17, 20],
        "shape": [3],
        "description": "Task goal Cartesian XYZ.",
    },
    "gripper_state": {
        "slice": [20, 21],
        "shape": [1],
        "description": "Meta-World gripper opening scalar.",
    },
}

RAW_STATE_LAYOUT: dict[str, dict[str, Any]] = {
    "end_effector_position": {"slice": [0, 3], "shape": [3]},
    "gripper_state": {"slice": [3, 4], "shape": [1]},
    "object_1_position": {"slice": [4, 7], "shape": [3]},
    "object_1_quaternion": {"slice": [7, 11], "shape": [4]},
    "object_2_position": {"slice": [11, 14], "shape": [3]},
    "object_2_quaternion": {"slice": [14, 18], "shape": [4]},
    "previous_observation": {"slice": [18, 36], "shape": [18]},
    "goal_position": {"slice": [36, 39], "shape": [3]},
}

ACTION_LAYOUT: dict[str, dict[str, Any]] = {
    "delta_position": {
        "slice": [0, 3],
        "shape": [3],
        "range": [-1.0, 1.0],
    },
    "gripper_control": {
        "slice": [3, 4],
        "shape": [1],
        "range": [-1.0, 1.0],
    },
}

DEFAULT_CONFIG: dict[str, Any] = {
    "backend": "metaworld",
    "benchmark": "MT1",
    "env_name": "pick-place-v3",
    "state_mode": "semantic",
    "max_episode_steps": 200,
    "terminate_on_success": True,
    "render_mode": None,
    "action_noise_std": 0.0,
    "observation_noise_std": 0.0,
    "object_noise_probability": 0.0,
    "object_noise_magnitude": 0.03,
    "object_noise_std": 0.0,
    "clip_actions": True,
    "success_threshold": 0.07,
}


def _deep_merge(
    base: MutableMapping[str, Any], update: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    for key, value in update.items():
        if (
            key in base
            and isinstance(base[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_project_config(config: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
    """Load a project YAML/mapping without requiring a particular working directory."""

    if config is None:
        return {}
    if isinstance(config, Mapping):
        return copy.deepcopy(dict(config))
    path = Path(config).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Environment config does not exist: {path}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - installed by requirements.
        raise RuntimeError("PyYAML is required to load a YAML config.") from exc
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Expected a mapping in {path}, got {type(loaded).__name__}.")
    return dict(loaded)


def _flatten_environment_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract constructor fields from the repository's nested YAML schema."""

    result: dict[str, Any] = {}
    environment = config.get("environment", config.get("env", {}))
    if isinstance(environment, Mapping):
        result.update(environment)

    disturbance = config.get("disturbance", config.get("noise", {}))
    if isinstance(disturbance, Mapping):
        result.update(disturbance)

    project = config.get("project", {})
    if isinstance(project, Mapping) and "seed" in project:
        result.setdefault("seed", project["seed"])

    # A flat mapping is useful for programmatic construction and unit tests.
    known = set(DEFAULT_CONFIG) | {
        "seed",
        "task",
        "task_name",
        "requested_task",
        "object_noise_probability",
        "object_noise_magnitude",
    }
    for key, value in config.items():
        if key in known:
            result[key] = value
    return result


def _as_float_vector(
    value: Any, length: int, *, name: str, finite: bool = True
) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size != length:
        raise ValueError(f"{name} must have {length} values, got shape {vector.shape}.")
    if finite and not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains NaN or infinity.")
    return vector


def _normalized_quaternion(quaternion: Any) -> np.ndarray:
    quat = _as_float_vector(quaternion, 4, name="quaternion")
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (quat / norm).astype(np.float32)


class REIMPickPlaceEnv(_GymEnv):
    """REIM's unified Sawyer PickPlace environment.

    Parameters
    ----------
    config:
        Repository YAML path or a mapping. Both the nested repository schema
        and flat constructor dictionaries are accepted.
    seed:
        Reproducibility seed. Overrides the config value.
    backend:
        ``"metaworld"`` (default) or explicit ``"toy"``. ``"auto"`` is
        accepted for older callers but intentionally behaves as
        ``"metaworld"`` and never silently falls back.
    state_mode:
        ``"semantic"`` (21D) or ``"raw"`` (39D).
    **overrides:
        Environment/noise fields that override values loaded from ``config``.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20,
    }

    def __init__(
        self,
        config: str | Path | Mapping[str, Any] | None = None,
        *,
        seed: int | None = None,
        backend: str | None = None,
        render_mode: str | None = None,
        state_mode: str | None = None,
        **overrides: Any,
    ) -> None:
        if gym is not None:
            super().__init__()

        loaded = load_project_config(config)
        flattened = _flatten_environment_config(loaded)
        settings = copy.deepcopy(DEFAULT_CONFIG)
        settings.update(flattened)
        settings.update({key: value for key, value in overrides.items() if value is not None})
        if seed is not None:
            settings["seed"] = seed
        if backend is not None:
            settings["backend"] = backend
        if render_mode is not None:
            settings["render_mode"] = render_mode
        if state_mode is not None:
            settings["state_mode"] = state_mode

        backend_value = str(settings.get("backend", "metaworld")).lower()
        if backend_value == "auto":
            warnings.warn(
                "backend='auto' is deprecated and is treated as 'metaworld'; "
                "REIM never silently falls back to the toy backend.",
                stacklevel=2,
            )
            backend_value = "metaworld"
        if backend_value not in {"metaworld", "toy"}:
            raise ValueError("backend must be 'metaworld' or explicit 'toy'.")

        mode_value = str(settings.get("state_mode", "semantic")).lower()
        if mode_value not in {"semantic", "raw"}:
            raise ValueError("state_mode must be 'semantic' or 'raw'.")

        self.backend = backend_value
        self.state_mode = mode_value
        self.render_mode = settings.get("render_mode")
        self.env_name = self._canonical_env_name(
            str(settings.get("env_name", settings.get("task", "pick-place-v3")))
        )
        self.benchmark = str(settings.get("benchmark", "MT1"))
        self.max_episode_steps = int(settings.get("max_episode_steps", 200))
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive.")
        self.terminate_on_success = bool(settings.get("terminate_on_success", True))
        self.action_noise_std = max(0.0, float(settings.get("action_noise_std", 0.0)))
        self.observation_noise_std = max(
            0.0, float(settings.get("observation_noise_std", 0.0))
        )
        self.object_noise_probability = float(
            settings.get("object_noise_probability", 0.0)
        )
        if not 0.0 <= self.object_noise_probability <= 1.0:
            raise ValueError("object_noise_probability must lie in [0, 1].")
        self.object_noise_magnitude = max(
            0.0, float(settings.get("object_noise_magnitude", 0.03))
        )
        self.object_noise_std = max(0.0, float(settings.get("object_noise_std", 0.0)))
        if (
            "object_noise_std" in overrides
            and "object_noise_probability" not in overrides
            and self.object_noise_std > 0.0
        ):
            # Robustness experiments passing a std expect one displacement/episode.
            self.object_noise_probability = 1.0
        self.clip_actions = bool(settings.get("clip_actions", True))
        self.success_threshold = max(
            1e-6, float(settings.get("success_threshold", 0.07))
        )

        configured_seed = settings.get("seed", 42)
        self._seed = int(42 if configured_seed is None else configured_seed)
        self.np_random = np.random.default_rng(self._seed)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
        )
        state_dim = (
            SEMANTIC_STATE_DIM
            if self.state_mode == "semantic"
            else RAW_OBSERVATION_DIM
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32,
        )
        if hasattr(self.action_space, "seed"):
            self.action_space.seed(self._seed)

        self._backend_env: Any | None = None
        self._legacy_tasks: list[Any] = []
        self._raw_observation = np.zeros(RAW_OBSERVATION_DIM, dtype=np.float32)
        self._last_clean_state = np.zeros(state_dim, dtype=np.float32)
        self._last_observed_state = np.zeros(state_dim, dtype=np.float32)
        self._step_count = 0
        self._episode_index = -1
        self._object_disturbed = False
        self._last_disturbance_delta = np.zeros(3, dtype=np.float32)
        self._initial_object_position = np.zeros(3, dtype=np.float32)
        self._ever_lifted = False
        self._failure_latched = False
        self._last_distance_to_goal = math.inf
        self._episode_spec: dict[str, Any] | None = None
        self._closed = False

        # Toy dynamics state. These fields are harmless for benchmark runs.
        self._toy_joint_qpos = np.zeros(7, dtype=np.float32)
        self._toy_hand = np.zeros(3, dtype=np.float32)
        self._toy_hand_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._toy_object = np.zeros(3, dtype=np.float32)
        self._toy_goal = np.zeros(3, dtype=np.float32)
        self._toy_gripper = -1.0
        self._toy_grasped = False
        self._toy_previous_raw = np.zeros(18, dtype=np.float32)

        if self.backend == "metaworld":
            self._backend_env = self._make_metaworld_env()
        LOGGER.info(
            "Initialized REIMPickPlaceEnv backend=%s env=%s state_mode=%s dim=%d seed=%d",
            self.backend,
            self.env_name,
            self.state_mode,
            state_dim,
            self._seed,
        )

    @staticmethod
    def _canonical_env_name(name: str) -> str:
        normalized = name.strip().lower().replace("_", "-")
        aliases = {
            "pickplace-v2": "pick-place-v2",
            "pickplace-v3": "pick-place-v3",
            "pickplace": "pick-place-v3",
            "pick-place": "pick-place-v3",
        }
        return aliases.get(normalized, normalized)

    @property
    def state_dim(self) -> int:
        return int(self.observation_space.shape[0])

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    @property
    def raw_observation(self) -> np.ndarray:
        """Latest clean 39D Meta-World-format observation."""

        return self._raw_observation.copy()

    @property
    def state_metadata(self) -> dict[str, Any]:
        layout = (
            SEMANTIC_STATE_LAYOUT
            if self.state_mode == "semantic"
            else RAW_STATE_LAYOUT
        )
        return {
            "state_mode": self.state_mode,
            "state_dim": self.state_dim,
            "raw_observation_dim": RAW_OBSERVATION_DIM,
            "layout": copy.deepcopy(layout),
            "raw_layout": copy.deepcopy(RAW_STATE_LAYOUT),
            "action_layout": copy.deepcopy(ACTION_LAYOUT),
            "backend": self.backend,
            "env_name": self.env_name,
        }

    def _make_metaworld_env(self) -> Any:
        if gym is None:
            raise RuntimeError(
                "Meta-World backend requires gymnasium. Run setup.sh, or pass "
                "--backend toy explicitly for a dependency-light CI smoke test."
            )
        try:
            import metaworld  # noqa: F401 - import registers Gymnasium environments.
        except ImportError as exc:
            raise RuntimeError(
                "Meta-World is not installed. Run setup.sh; REIM will not silently "
                "substitute toy results for the public benchmark."
            ) from exc

        errors: list[str] = []
        current_name = (
            self.env_name[:-2] + "3"
            if self.env_name.endswith("v2")
            else self.env_name
        )
        make_kwargs: dict[str, Any] = {
            "env_name": current_name,
            "seed": self._seed,
            # Meta-World 3.1.1 declares an overly narrow observation Box even
            # though the returned MT1 observation is the documented 39D state.
            # Our own strict shape/finite checks below are the relevant checks.
            "disable_env_checker": True,
        }
        if self.render_mode is not None:
            make_kwargs["render_mode"] = self.render_mode
        try:
            env = gym.make("Meta-World/MT1", **make_kwargs)
            self.env_name = current_name
            LOGGER.info("Using current Meta-World Gymnasium MT1 API.")
            return env
        except Exception as exc:  # Current registration is absent in older releases.
            errors.append(f"Gymnasium MT1 API: {type(exc).__name__}: {exc}")

        # Compatibility for Meta-World 2.x. This is still a real benchmark and
        # therefore not a fallback to synthetic data.
        try:
            import metaworld

            legacy_name = (
                self.env_name[:-2] + "2"
                if self.env_name.endswith("v3")
                else self.env_name
            )
            benchmark_factory = getattr(metaworld, "MT1")
            try:
                benchmark = benchmark_factory(legacy_name, seed=self._seed)
            except TypeError:
                benchmark = benchmark_factory(legacy_name)
            env_class = benchmark.train_classes[legacy_name]
            try:
                env = env_class(render_mode=self.render_mode)
            except TypeError:
                env = env_class()
            self._legacy_tasks = list(benchmark.train_tasks)
            if self._legacy_tasks:
                env.set_task(self._legacy_tasks[0])
            self.env_name = legacy_name
            LOGGER.warning("Using legacy Meta-World benchmark API for %s.", legacy_name)
            return env
        except Exception as exc:
            errors.append(f"legacy MT1 API: {type(exc).__name__}: {exc}")

        joined = "\n  - ".join(errors)
        raise RuntimeError(
            "Unable to construct the Meta-World PickPlace benchmark. Attempted:\n"
            f"  - {joined}\n"
            "Install a supported Meta-World release or use --backend toy explicitly "
            "for CI only."
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Cannot reset a closed environment.")
        backend_options = dict(options or {})
        episode_spec = backend_options.pop("reim_episode_spec", None)
        if episode_spec is not None:
            self._activate_episode_spec(episode_spec, requested_seed=seed)
            effective_seed = int(self._episode_spec["reset_seed"])
        else:
            self._episode_spec = None
            effective_seed = seed
        if effective_seed is not None:
            self._seed = int(effective_seed)
            self.np_random = np.random.default_rng(self._seed)
            if hasattr(self.action_space, "seed"):
                self.action_space.seed(self._seed)
        self._episode_index += 1
        self._step_count = 0
        self._object_disturbed = False
        self._last_disturbance_delta.fill(0.0)
        self._ever_lifted = False
        self._failure_latched = False

        if self.backend == "toy":
            raw, backend_info = self._reset_toy()
        else:
            raw, backend_info = self._reset_metaworld(
                seed=effective_seed,
                options=backend_options or None,
            )
        self._set_raw_observation(raw)
        components = self.get_state_components()
        self._initial_object_position = components["object_position"].copy()
        self._last_distance_to_goal = float(
            np.linalg.norm(
                components["object_position"] - components["goal_position"]
            )
        )
        clean = self._compose_state()
        observed = self.apply_observation_noise(clean, schedule_index=0)
        self._last_clean_state = clean
        self._last_observed_state = observed

        info = self._standardize_info(backend_info, terminated=False, truncated=False)
        info.update(
            {
                "episode_seed": self._seed,
                "episode_index": self._episode_index,
                "state_mode": self.state_mode,
                "episode_specification_sha256": (
                    self._episode_spec["specification_sha256"]
                    if self._episode_spec is not None
                    else ""
                ),
                "episode_bank_sha256": (
                    self._episode_spec["bank_sha256"]
                    if self._episode_spec is not None
                    else ""
                ),
                "metaworld_task_sha256": (
                    self._episode_spec["task_sha256"]
                    if self._episode_spec is not None
                    else ""
                ),
            }
        )
        return observed.copy(), info

    @staticmethod
    def _stateless_rng(stream_seed: int, schedule_index: int) -> np.random.Generator:
        if schedule_index < 0:
            raise ValueError("schedule_index must be non-negative")
        seed = int(stream_seed)
        sequence = np.random.SeedSequence(
            [
                seed & 0xFFFFFFFF,
                (seed >> 32) & 0xFFFFFFFF,
                int(schedule_index),
            ]
        )
        return np.random.default_rng(sequence)

    def _activate_episode_spec(
        self,
        specification: Mapping[str, Any],
        *,
        requested_seed: int | None,
    ) -> None:
        from evaluation.episode_bank import (
            BANK_TYPE,
            SCHEMA_VERSION,
            payload_sha256,
        )

        spec = copy.deepcopy(dict(specification))
        if int(spec.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("unsupported REIM episode specification schema")
        if spec.get("bank_type") != BANK_TYPE:
            raise ValueError("unexpected REIM episode specification type")
        if str(spec.get("backend", "")).lower() != self.backend:
            raise ValueError("episode specification backend mismatch")
        if self._canonical_env_name(str(spec.get("env_name", ""))) != self.env_name:
            raise ValueError("episode specification environment mismatch")
        episode_seed = int(spec.get("episode_seed", -1))
        if requested_seed is not None and int(requested_seed) != episode_seed:
            raise ValueError(
                "reset seed and episode specification seed must be identical"
            )
        if int(spec.get("max_steps", -1)) != self.max_episode_steps:
            raise ValueError("episode specification max_steps mismatch")

        disturbance = spec.get("disturbance")
        if not isinstance(disturbance, Mapping):
            raise ValueError("episode specification has no disturbance protocol")
        configured = {
            "action_noise_std": self.action_noise_std,
            "observation_noise_std": self.observation_noise_std,
            "object_noise_probability": self.object_noise_probability,
            "object_noise_std": self.object_noise_std,
            "object_noise_magnitude": self.object_noise_magnitude,
        }
        for field, expected in configured.items():
            try:
                actual = float(disturbance[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"episode specification has invalid {field}"
                ) from exc
            if not np.isclose(actual, expected, rtol=0.0, atol=1e-12):
                raise ValueError(
                    f"episode specification {field}={actual} differs from "
                    f"environment value {expected}"
                )

        core_fields = (
            "episode_index",
            "episode_seed",
            "reset_seed",
            "task_index",
            "task_sha256",
            "action_noise_seed",
            "observation_noise_seed",
            "object_disturbance_seed",
            "object_disturbance_step",
            "object_disturbance_delta",
        )
        try:
            core = {field: spec[field] for field in core_fields}
        except KeyError as exc:
            raise ValueError(
                f"episode specification is missing {exc.args[0]}"
            ) from exc
        if payload_sha256(core) != spec.get("specification_sha256"):
            raise ValueError("episode specification SHA256 mismatch")

        delta = _as_float_vector(
            spec["object_disturbance_delta"],
            3,
            name="scheduled object disturbance",
        )
        spec["object_disturbance_delta"] = delta.tolist()
        disturbance_step = spec.get("object_disturbance_step")
        if disturbance_step is not None:
            disturbance_step = int(disturbance_step)
            if not 2 <= disturbance_step < self.max_episode_steps:
                raise ValueError("scheduled object disturbance step is invalid")
            spec["object_disturbance_step"] = disturbance_step
        for field in (
            "reset_seed",
            "action_noise_seed",
            "observation_noise_seed",
            "object_disturbance_seed",
        ):
            if int(spec[field]) < 0:
                raise ValueError(f"episode specification {field} must be non-negative")
            spec[field] = int(spec[field])

        task = spec.get("task")
        if self.backend == "metaworld":
            if not isinstance(task, Mapping):
                raise ValueError("Meta-World episode specification has no task")
            try:
                task_data = base64.b64decode(
                    str(task["data_base64"]),
                    validate=True,
                )
            except Exception as exc:
                raise ValueError("invalid serialized Meta-World task") from exc
            task_sha256 = hashlib.sha256(task_data).hexdigest()
            if task_sha256 != task.get("data_sha256"):
                raise ValueError("serialized Meta-World task SHA256 mismatch")
            if task_sha256 != spec.get("task_sha256"):
                raise ValueError("episode specification task SHA256 mismatch")
            if self._canonical_env_name(str(task.get("env_name", ""))) != self.env_name:
                raise ValueError("serialized Meta-World task environment mismatch")
        elif task is not None:
            raise ValueError("toy episode specification cannot contain a task")
        self._episode_spec = spec

    def _set_explicit_metaworld_task(self) -> None:
        if self._episode_spec is None:
            return
        task_record = self._episode_spec.get("task")
        if not isinstance(task_record, Mapping):
            raise ValueError("active Meta-World episode specification has no task")
        try:
            from metaworld.types import Task
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Meta-World Task type is unavailable") from exc
        data = base64.b64decode(str(task_record["data_base64"]), validate=True)
        task = Task(env_name=str(task_record["env_name"]), data=data)

        wrapper = self._backend_env
        visited: set[int] = set()
        while wrapper is not None and id(wrapper) not in visited:
            visited.add(id(wrapper))
            toggle = getattr(wrapper, "toggle_sample_tasks_on_reset", None)
            if callable(toggle):
                toggle(False)
            wrapper = getattr(wrapper, "env", None)
        base = self._base_metaworld_env()
        setter = getattr(base, "set_task", None)
        if not callable(setter):
            raise RuntimeError("Meta-World backend does not expose set_task()")
        setter(task)

    def _reset_metaworld(
        self, *, seed: int | None, options: dict[str, Any] | None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        assert self._backend_env is not None
        if self._episode_spec is not None:
            self._set_explicit_metaworld_task()
        elif self._legacy_tasks:
            task_index = int(self.np_random.integers(0, len(self._legacy_tasks)))
            self._backend_env.set_task(self._legacy_tasks[task_index])
        else:
            # A preceding explicit CRN reset disables Meta-World's wrapper-level
            # task sampling. Restore the legacy behavior only for an explicitly
            # non-bank reset; benchmark-eligible runs always supply a spec.
            wrapper = self._backend_env
            visited: set[int] = set()
            while wrapper is not None and id(wrapper) not in visited:
                visited.add(id(wrapper))
                toggle = getattr(wrapper, "toggle_sample_tasks_on_reset", None)
                if callable(toggle):
                    toggle(True)
                wrapper = getattr(wrapper, "env", None)
        try:
            result = self._backend_env.reset(seed=seed, options=options)
        except TypeError:
            if seed is not None and hasattr(self._backend_env, "seed"):
                self._backend_env.seed(seed)
            result = self._backend_env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            raw, info = result
        else:
            raw, info = result, {}
        return self._validate_raw(raw), dict(info or {})

    def _reset_toy(self) -> tuple[np.ndarray, dict[str, Any]]:
        self._toy_joint_qpos = np.asarray(
            [0.0, -0.75, 0.0, 1.6, 0.0, 0.8, 0.0], dtype=np.float32
        )
        self._toy_hand = np.asarray(
            [
                self.np_random.uniform(-0.04, 0.04),
                self.np_random.uniform(0.48, 0.53),
                self.np_random.uniform(0.20, 0.24),
            ],
            dtype=np.float32,
        )
        self._toy_hand_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._toy_object = np.asarray(
            [
                self.np_random.uniform(-0.08, 0.02),
                self.np_random.uniform(0.58, 0.66),
                0.025,
            ],
            dtype=np.float32,
        )
        self._toy_goal = np.asarray(
            [
                self.np_random.uniform(0.08, 0.16),
                self.np_random.uniform(0.72, 0.80),
                self.np_random.uniform(0.12, 0.18),
            ],
            dtype=np.float32,
        )
        self._toy_gripper = -1.0
        self._toy_grasped = False
        current = self._toy_current_raw()
        self._toy_previous_raw = current[:18].copy()
        return self._toy_current_raw(), {"toy_ci_backend": True}

    def step(
        self, action: Sequence[float] | np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Cannot step a closed environment.")
        commanded = _as_float_vector(action, ACTION_DIM, name="action")
        executed = self.apply_action_noise(commanded)

        disturbed_this_step = self._maybe_apply_object_disturbance()
        if self.backend == "toy":
            raw, reward, terminated, truncated, backend_info = self._step_toy(executed)
        else:
            raw, reward, terminated, truncated, backend_info = self._step_metaworld(
                executed
            )
        self._step_count += 1
        if self._step_count >= self.max_episode_steps and not terminated:
            truncated = True

        self._set_raw_observation(raw)
        clean = self._compose_state()
        observed = self.apply_observation_noise(
            clean,
            schedule_index=self._step_count,
        )
        self._last_clean_state = clean
        self._last_observed_state = observed
        components = self.get_state_components()
        backend_success_signal = backend_info.get(
            "success", backend_info.get("is_success", False)
        )
        backend_success = bool(np.asarray(backend_success_signal).reshape(-1)[0])
        geometric_success = (
            float(
                np.linalg.norm(
                    components["object_position"] - components["goal_position"]
                )
            )
            <= self.success_threshold
        )
        if (backend_success or geometric_success) and self.terminate_on_success:
            terminated = True
            truncated = False
        info = self._standardize_info(
            backend_info, terminated=terminated, truncated=truncated
        )
        info.update(
            {
                "commanded_action": commanded.copy(),
                "executed_action": executed.copy(),
                "action_noise": (executed - commanded).astype(np.float32),
                "object_disturbance_applied": disturbed_this_step,
                "object_disturbance_delta": self._last_disturbance_delta.copy()
                if disturbed_this_step
                else np.zeros(3, dtype=np.float32),
                "step": self._step_count,
                "state_mode": self.state_mode,
                "episode_specification_sha256": (
                    self._episode_spec["specification_sha256"]
                    if self._episode_spec is not None
                    else ""
                ),
            }
        )
        return observed.copy(), float(reward), bool(terminated), bool(truncated), info

    def _step_metaworld(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._backend_env is not None
        result = self._backend_env.step(action)
        if not isinstance(result, tuple):
            raise RuntimeError("Meta-World step did not return a tuple.")
        if len(result) == 5:
            raw, reward, terminated, truncated, info = result
        elif len(result) == 4:  # Meta-World 2.x/Gym compatibility.
            raw, reward, done, info = result
            truncated = bool(
                (info or {}).get("TimeLimit.truncated", False)
                or (info or {}).get("truncated", False)
            )
            terminated = bool(done and not truncated)
        else:
            raise RuntimeError(
                f"Unsupported Meta-World step tuple with {len(result)} entries."
            )
        return (
            self._validate_raw(raw),
            float(reward),
            bool(terminated),
            bool(truncated),
            dict(info or {}),
        )

    def _step_toy(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        previous_raw = self._toy_current_raw()[:18].copy()
        delta = action[:3] * 0.025
        self._toy_hand = np.clip(
            self._toy_hand + delta,
            np.asarray([-0.45, 0.35, 0.015], dtype=np.float32),
            np.asarray([0.45, 1.05, 0.55], dtype=np.float32),
        )
        self._toy_joint_qpos[:3] = np.asarray(
            [
                self._toy_hand[0] * 2.0,
                (self._toy_hand[1] - 0.65) * 2.0,
                (self._toy_hand[2] - 0.20) * 2.0,
            ],
            dtype=np.float32,
        )
        self._toy_gripper = float(
            np.clip(0.65 * self._toy_gripper + 0.35 * action[3], -1.0, 1.0)
        )

        hand_object_distance = float(np.linalg.norm(self._toy_hand - self._toy_object))
        if (
            not self._toy_grasped
            and hand_object_distance < 0.048
            and self._toy_gripper > 0.20
        ):
            self._toy_grasped = True
        if self._toy_grasped and self._toy_gripper < -0.25:
            self._toy_grasped = False
        if self._toy_grasped:
            self._toy_object = (
                self._toy_hand - np.asarray([0.0, 0.0, 0.025], dtype=np.float32)
            )
        elif self._toy_object[2] > 0.025:
            self._toy_object[2] = max(0.025, float(self._toy_object[2] - 0.012))

        self._toy_previous_raw = previous_raw
        distance = float(np.linalg.norm(self._toy_object - self._toy_goal))
        reach = float(np.linalg.norm(self._toy_hand - self._toy_object))
        success = distance <= self.success_threshold
        reward = (
            1.0 - math.tanh(8.0 * distance)
            + 0.25 * (1.0 - math.tanh(8.0 * reach))
            + (10.0 if success else 0.0)
            - 0.01
        )
        info = {
            "success": float(success),
            "grasped": self._toy_grasped,
            "toy_ci_backend": True,
        }
        return self._toy_current_raw(), reward, success, False, info

    def _validate_raw(self, raw: Any) -> np.ndarray:
        observation = np.asarray(raw, dtype=np.float32).reshape(-1)
        if observation.size != RAW_OBSERVATION_DIM:
            raise ValueError(
                "REIM expects the single-task goal-observable Meta-World state "
                f"with {RAW_OBSERVATION_DIM} values, but {self.env_name} returned "
                f"{observation.size}. Do not use MT10/MT50 observations with task IDs."
            )
        if not np.all(np.isfinite(observation)):
            raise RuntimeError("Meta-World returned a non-finite observation.")
        return observation

    def _set_raw_observation(self, raw: Any) -> None:
        self._raw_observation = self._validate_raw(raw).copy()

    def _toy_current_raw(self) -> np.ndarray:
        object_quaternion = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        current = np.concatenate(
            [
                self._toy_hand,
                np.asarray([self._toy_gripper], dtype=np.float32),
                self._toy_object,
                object_quaternion,
                np.zeros(3, dtype=np.float32),
                np.zeros(4, dtype=np.float32),
            ]
        ).astype(np.float32)
        return np.concatenate([current, self._toy_previous_raw, self._toy_goal]).astype(
            np.float32
        )

    def _base_metaworld_env(self) -> Any:
        if self._backend_env is None:
            return None
        return getattr(self._backend_env, "unwrapped", self._backend_env)

    @staticmethod
    def _body_pose(base: Any, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        data = getattr(base, "data", None)
        if data is None:
            raise AttributeError("environment has no MuJoCo data")
        for name in names:
            try:
                body = data.body(name)
                return (
                    np.asarray(body.xpos, dtype=np.float32).copy(),
                    _normalized_quaternion(body.xquat),
                )
            except (KeyError, ValueError, AttributeError):
                continue
        raise AttributeError(f"none of the MuJoCo bodies exist: {tuple(names)}")

    def get_state_components(self) -> dict[str, np.ndarray]:
        """Return clean named physical components independent of ``state_mode``."""

        if self.backend == "toy":
            return {
                "joint_positions": self._toy_joint_qpos.copy(),
                "end_effector_position": self._toy_hand.copy(),
                "end_effector_quaternion": self._toy_hand_quat.copy(),
                "object_position": self._toy_object.copy(),
                "goal_position": self._toy_goal.copy(),
                "gripper_state": np.asarray(
                    [self._toy_gripper], dtype=np.float32
                ),
            }

        base = self._base_metaworld_env()
        raw = self._raw_observation
        data = getattr(base, "data", None)
        qpos = np.asarray(getattr(data, "qpos", []), dtype=np.float32).reshape(-1)
        if qpos.size < 7:
            raise RuntimeError(
                "Meta-World MuJoCo data does not expose the seven Sawyer qpos values."
            )
        joint_positions = qpos[:7].copy()
        try:
            hand_position, hand_quaternion = self._body_pose(
                base, ("hand", "right_hand", "gripper")
            )
        except AttributeError:
            hand_position = raw[0:3].copy()
            # Meta-World uses a fixed end-effector orientation for Cartesian control.
            hand_quaternion = np.asarray(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float32
            )

        try:
            object_position = np.asarray(
                base._get_pos_objects(), dtype=np.float32
            ).reshape(-1)[:3]
            if object_position.size != 3:
                raise ValueError
        except (AttributeError, TypeError, ValueError):
            object_position = raw[4:7].copy()
        try:
            goal_position = _as_float_vector(
                base._target_pos, 3, name="Meta-World target"
            )
        except (AttributeError, ValueError):
            goal_position = raw[36:39].copy()

        return {
            "joint_positions": joint_positions.astype(np.float32),
            "end_effector_position": hand_position.astype(np.float32),
            "end_effector_quaternion": hand_quaternion.astype(np.float32),
            "object_position": object_position.astype(np.float32),
            "goal_position": goal_position.astype(np.float32),
            "gripper_state": raw[3:4].copy().astype(np.float32),
        }

    def _compose_state(self) -> np.ndarray:
        if self.state_mode == "raw":
            return self._raw_observation.copy()
        components = self.get_state_components()
        state = np.concatenate(
            [
                components["joint_positions"],
                components["end_effector_position"],
                components["end_effector_quaternion"],
                components["object_position"],
                components["goal_position"],
                components["gripper_state"],
            ]
        ).astype(np.float32)
        if state.shape != (SEMANTIC_STATE_DIM,):
            raise RuntimeError(f"Internal semantic state has invalid shape {state.shape}.")
        return state

    def get_state(self, *, noisy: bool = False) -> np.ndarray:
        """Return the latest state.

        The default is the clean physical state. ``noisy=True`` returns the
        exact observation most recently delivered by ``reset``/``step`` rather
        than sampling a second independent perturbation.
        """

        source = self._last_observed_state if noisy else self._last_clean_state
        return source.copy()

    def capture_sim_state(self) -> dict[str, np.ndarray]:
        """Capture the physical state needed to restart recovery training.

        The snapshot deliberately contains only numeric arrays so it can be
        stored safely in an ``npz`` file.  It is more complete than the public
        policy observation: MuJoCo positions/velocities, Cartesian mocap
        targets, the task goal, frame-stack state, and REIM's episode bookkeeping
        are all retained.  This lets PPO start from *measured online ACT trigger
        states* instead of a hand-designed approximation.
        """

        if self.backend != "metaworld":
            raise RuntimeError(
                "Simulator snapshots are available only for the Meta-World backend."
            )
        base = self._base_metaworld_env()
        if base is None or not hasattr(base, "get_env_state"):
            raise RuntimeError("Meta-World backend does not expose get_env_state().")
        qpos, qvel = base.get_env_state()
        data = getattr(base, "data", None)
        if data is None:
            raise RuntimeError("Meta-World backend does not expose MuJoCo data.")

        def array_attr(name: str, shape: tuple[int, ...] = (0,)) -> np.ndarray:
            value = getattr(data, name, None)
            if value is None:
                return np.zeros(shape, dtype=np.float64)
            return np.asarray(value, dtype=np.float64).copy()

        snapshot = {
            "qpos": np.asarray(qpos, dtype=np.float64).copy(),
            "qvel": np.asarray(qvel, dtype=np.float64).copy(),
            "mocap_pos": array_attr("mocap_pos"),
            "mocap_quat": array_attr("mocap_quat"),
            "ctrl": array_attr("ctrl"),
            "act": array_attr("act"),
            "time": np.asarray([float(getattr(data, "time", 0.0))], dtype=np.float64),
            "target_pos": np.asarray(
                getattr(base, "_target_pos", self._raw_observation[36:39]),
                dtype=np.float64,
            ).copy(),
            "previous_observation": np.asarray(
                getattr(base, "_prev_obs", self._raw_observation[:18]),
                dtype=np.float64,
            ).copy(),
            "raw_observation": self._raw_observation.astype(np.float64, copy=True),
            "initial_object_position": self._initial_object_position.astype(
                np.float64, copy=True
            ),
            "last_distance_to_goal": np.asarray(
                [self._last_distance_to_goal], dtype=np.float64
            ),
            "step_count": np.asarray([self._step_count], dtype=np.int64),
            "ever_lifted": np.asarray([self._ever_lifted], dtype=np.bool_),
            "failure_latched": np.asarray([self._failure_latched], dtype=np.bool_),
            "object_disturbed": np.asarray([self._object_disturbed], dtype=np.bool_),
            "last_disturbance_delta": self._last_disturbance_delta.astype(
                np.float64, copy=True
            ),
        }
        # PickPlace's dense reward uses reset-time reference geometry in
        # addition to the instantaneous MuJoCo state.  Persist the small set of
        # numeric task fields so reward and termination are identical after a
        # cross-episode or cross-process restore.
        for name in (
            "_last_rand_vec",
            "obj_init_pos",
            "init_tcp",
            "init_left_pad",
            "init_right_pad",
            "hand_init_pos",
            "goal",
            "objHeight",
            "heightTarget",
            "curr_path_length",
        ):
            if hasattr(base, name):
                snapshot[f"task_{name}"] = np.asarray(
                    getattr(base, name)
                ).copy()
        return snapshot

    def restore_sim_state(
        self, snapshot: Mapping[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Restore a state produced by :meth:`capture_sim_state`.

        Call :meth:`reset` once before restoring so Gymnasium/Meta-World wrapper
        episode statistics are initialized.  The returned observation is clean;
        subsequent calls to :meth:`step` apply the environment's configured
        observation noise normally.
        """

        if self.backend != "metaworld":
            raise RuntimeError(
                "Simulator snapshots are available only for the Meta-World backend."
            )
        required = {
            "qpos",
            "qvel",
            "mocap_pos",
            "mocap_quat",
            "target_pos",
            "previous_observation",
            "raw_observation",
        }
        missing = sorted(required.difference(snapshot))
        if missing:
            raise ValueError(f"Simulator snapshot is missing fields: {missing}")

        base = self._base_metaworld_env()
        if base is None or not hasattr(base, "set_env_state"):
            raise RuntimeError("Meta-World backend does not expose set_env_state().")
        data = getattr(base, "data", None)
        if data is None:
            raise RuntimeError("Meta-World backend does not expose MuJoCo data.")

        base._target_pos = _as_float_vector(
            snapshot["target_pos"], 3, name="snapshot target"
        ).astype(np.float64)
        for key, value in snapshot.items():
            if not key.startswith("task_"):
                continue
            name = key[len("task_") :]
            restored_value = np.asarray(value).copy()
            if restored_value.ndim == 0:
                restored_value = restored_value.item()
            setattr(base, name, restored_value)
        base.set_env_state(
            (
                np.asarray(snapshot["qpos"], dtype=np.float64).copy(),
                np.asarray(snapshot["qvel"], dtype=np.float64).copy(),
            )
        )

        for name in ("mocap_pos", "mocap_quat", "ctrl", "act"):
            destination = getattr(data, name, None)
            source = np.asarray(snapshot.get(name, []), dtype=np.float64)
            if destination is not None and source.shape == np.asarray(destination).shape:
                destination[...] = source
        time_value = np.asarray(snapshot.get("time", [0.0]), dtype=np.float64)
        if time_value.size:
            data.time = float(time_value.reshape(-1)[0])

        try:
            import mujoco

            mujoco.mj_forward(base.model, data)
        except (ImportError, AttributeError) as exc:  # pragma: no cover
            raise RuntimeError("MuJoCo forward dynamics are required for restore.") from exc

        base._prev_obs = np.asarray(
            snapshot["previous_observation"], dtype=np.float64
        ).copy()
        raw = np.asarray(snapshot["raw_observation"], dtype=np.float32).copy()
        self._set_raw_observation(raw)
        self._initial_object_position = np.asarray(
            snapshot.get(
                "initial_object_position",
                self.get_state_components()["object_position"],
            ),
            dtype=np.float32,
        ).copy()
        self._last_distance_to_goal = float(
            np.asarray(
                snapshot.get(
                    "last_distance_to_goal",
                    [
                        np.linalg.norm(
                            self.get_state_components()["object_position"]
                            - self.get_state_components()["goal_position"]
                        )
                    ],
                )
            ).reshape(-1)[0]
        )
        self._step_count = int(
            np.asarray(snapshot.get("step_count", [0])).reshape(-1)[0]
        )
        self._ever_lifted = bool(
            np.asarray(snapshot.get("ever_lifted", [False])).reshape(-1)[0]
        )
        self._failure_latched = bool(
            np.asarray(snapshot.get("failure_latched", [False])).reshape(-1)[0]
        )
        self._object_disturbed = bool(
            np.asarray(snapshot.get("object_disturbed", [False])).reshape(-1)[0]
        )
        self._last_disturbance_delta = np.asarray(
            snapshot.get("last_disturbance_delta", np.zeros(3)),
            dtype=np.float32,
        ).reshape(3)

        clean = self._compose_state()
        self._last_clean_state = clean.copy()
        self._last_observed_state = clean.copy()
        info = self._standardize_info({}, terminated=False, truncated=False)
        info.update(
            {
                "episode_seed": self._seed,
                "episode_index": self._episode_index,
                "state_mode": self.state_mode,
                "restored_recovery_start": True,
            }
        )
        return clean.copy(), info

    def apply_action_noise(
        self, action: Sequence[float] | np.ndarray, std: float | None = None
    ) -> np.ndarray:
        """Add reproducible Gaussian action noise and enforce action bounds."""

        result = _as_float_vector(action, ACTION_DIM, name="action").copy()
        sigma = self.action_noise_std if std is None else max(0.0, float(std))
        if sigma > 0.0:
            if self._episode_spec is not None:
                rng = self._stateless_rng(
                    int(self._episode_spec["action_noise_seed"]),
                    self._step_count,
                )
                noise = rng.normal(0.0, sigma, size=ACTION_DIM)
            else:
                noise = self.np_random.normal(0.0, sigma, size=ACTION_DIM)
            result += np.asarray(noise, dtype=np.float32)
        if self.clip_actions:
            result = np.clip(result, -1.0, 1.0)
        return result.astype(np.float32)

    def apply_observation_noise(
        self,
        observation: Sequence[float] | np.ndarray,
        std: float | None = None,
        *,
        schedule_index: int | None = None,
    ) -> np.ndarray:
        """Add reproducible Gaussian noise to a state observation."""

        result = _as_float_vector(
            observation, self.state_dim, name="observation"
        ).copy()
        sigma = (
            self.observation_noise_std if std is None else max(0.0, float(std))
        )
        if sigma > 0.0:
            if self._episode_spec is not None:
                resolved_index = (
                    self._step_count
                    if schedule_index is None
                    else int(schedule_index)
                )
                rng = self._stateless_rng(
                    int(self._episode_spec["observation_noise_seed"]),
                    resolved_index,
                )
                noise = rng.normal(0.0, sigma, size=self.state_dim)
            else:
                noise = self.np_random.normal(0.0, sigma, size=self.state_dim)
            result += np.asarray(noise, dtype=np.float32)
        return result.astype(np.float32)

    def _maybe_apply_object_disturbance(self) -> bool:
        if self._object_disturbed or self._step_count < 2:
            return False
        if self.object_noise_probability <= 0.0:
            return False
        if self._episode_spec is not None:
            scheduled_step = self._episode_spec.get("object_disturbance_step")
            if scheduled_step is None or self._step_count != int(scheduled_step):
                return False
            self.apply_object_noise(
                delta=self._episode_spec["object_disturbance_delta"]
            )
            return True
        if self.np_random.random() >= self.object_noise_probability:
            return False
        self.apply_object_noise()
        return True

    def apply_object_noise(
        self,
        std: float | None = None,
        *,
        magnitude: float | None = None,
        delta: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply one instantaneous XYZ position displacement to the object.

        ``delta`` gives an exact reproducible perturbation. Otherwise Gaussian
        noise is used when a positive std is configured/passed; the repository
        config's ``object_noise_magnitude`` selects a bounded uniform XY
        displacement. This operation uses Meta-World's object-position setter;
        it is not a force or contact impulse.
        At most one automatically scheduled disturbance is applied per episode,
        while explicit calls remain available to experiments.
        """

        if delta is not None:
            offset = _as_float_vector(delta, 3, name="object disturbance")
        else:
            sigma = self.object_noise_std if std is None else max(0.0, float(std))
            bound = (
                self.object_noise_magnitude
                if magnitude is None
                else max(0.0, float(magnitude))
            )
            if sigma > 0.0:
                offset = self.np_random.normal(0.0, sigma, size=3).astype(np.float32)
            elif bound > 0.0:
                offset = np.asarray(
                    [
                        self.np_random.uniform(-bound, bound),
                        self.np_random.uniform(-bound, bound),
                        self.np_random.uniform(-0.20 * bound, 0.20 * bound),
                    ],
                    dtype=np.float32,
                )
            else:
                offset = np.zeros(3, dtype=np.float32)

        current = self.get_state_components()["object_position"]
        desired = current + offset
        desired = np.clip(
            desired,
            np.asarray([-0.45, 0.35, 0.015], dtype=np.float32),
            np.asarray([0.45, 1.05, 0.50], dtype=np.float32),
        ).astype(np.float32)
        applied = desired - current
        if self.backend == "toy":
            self._toy_object = desired
            if self._toy_grasped:
                self._toy_grasped = False
            self._set_raw_observation(self._toy_current_raw())
        else:
            base = self._base_metaworld_env()
            setter = getattr(base, "_set_obj_xyz", None)
            if setter is None:
                raise RuntimeError(
                    "This Meta-World release does not expose _set_obj_xyz; "
                    "object-position disturbances cannot be applied safely."
                )
            setter(desired)
            # Refresh the wrapper's raw observation without stepping physics.
            get_obs = getattr(base, "_get_obs", None)
            if callable(get_obs):
                self._set_raw_observation(get_obs())
        # Explicit callers (notably the recovery reset wrapper) expect
        # get_state() to reflect the position displacement immediately.
        clean = self._compose_state()
        self._last_clean_state = clean
        self._last_observed_state = self.apply_observation_noise(
            clean,
            schedule_index=self._step_count,
        )
        self._object_disturbed = True
        self._last_disturbance_delta = applied.astype(np.float32)
        LOGGER.debug("Applied object disturbance delta=%s", applied.tolist())
        return applied.copy()

    def _standardize_info(
        self,
        backend_info: Mapping[str, Any] | None,
        *,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        info = dict(backend_info or {})
        components = self.get_state_components()
        object_position = components["object_position"]
        goal_position = components["goal_position"]
        ee_position = components["end_effector_position"]
        distance = float(np.linalg.norm(object_position - goal_position))
        hand_object_distance = float(np.linalg.norm(ee_position - object_position))
        success_signal = info.get("success", info.get("is_success", False))
        success = bool(np.asarray(success_signal).reshape(-1)[0])
        success = success or distance <= self.success_threshold

        lift_threshold = float(self._initial_object_position[2] + 0.045)
        if object_position[2] > lift_threshold:
            self._ever_lifted = True
        dropped = bool(
            self._ever_lifted
            and object_position[2] <= self._initial_object_position[2] + 0.018
            and hand_object_distance > 0.065
            and not success
        )
        workspace_violation = bool(
            object_position[0] < -0.46
            or object_position[0] > 0.46
            or object_position[1] < 0.33
            or object_position[1] > 1.07
            or object_position[2] < 0.005
        )
        failure_reason = ""
        if dropped:
            failure_reason = "object_dropped"
        elif workspace_violation:
            failure_reason = "workspace_violation"
        elif truncated and not success:
            failure_reason = "timeout"
        elif terminated and not success:
            failure_reason = "terminal_failure"
        failure = bool(failure_reason)
        previously_failed = self._failure_latched
        recovered = bool(
            previously_failed
            and not failure
            and (
                hand_object_distance < 0.08
                or distance < self._last_distance_to_goal - 0.01
            )
        )
        if success or recovered:
            self._failure_latched = False
        else:
            self._failure_latched = bool(self._failure_latched or failure)
        self._last_distance_to_goal = distance

        info.update(
            {
                "success": success,
                "is_success": success,
                "failure": failure,
                "failure_reason": failure_reason,
                "object_dropped": dropped,
                "workspace_violation": workspace_violation,
                "recovered": recovered,
                "distance": distance,
                "distance_to_goal": distance,
                "hand_object_distance": hand_object_distance,
                "joint_positions": components["joint_positions"].copy(),
                "ee_position": ee_position.copy(),
                "tcp_position": ee_position.copy(),
                "tcp_quaternion": components[
                    "end_effector_quaternion"
                ].copy(),
                "object_position": object_position.copy(),
                "goal_position": goal_position.copy(),
                "gripper_state": float(components["gripper_state"][0]),
                "backend": self.backend,
                "env_name": self.env_name,
            }
        )
        return info

    def render(self) -> Any:
        if self.backend == "metaworld":
            assert self._backend_env is not None
            return self._backend_env.render()
        if self.render_mode == "rgb_array":
            return self._render_toy_rgb()
        if self.render_mode == "human":
            LOGGER.info(
                "toy state hand=%s object=%s goal=%s",
                np.round(self._toy_hand, 3),
                np.round(self._toy_object, 3),
                np.round(self._toy_goal, 3),
            )
        return None

    def _render_toy_rgb(self) -> np.ndarray:
        image = np.full((256, 256, 3), 248, dtype=np.uint8)

        def project(position: np.ndarray) -> tuple[int, int]:
            x = int(np.clip((position[0] + 0.45) / 0.9 * 255, 0, 255))
            y = int(np.clip((1.05 - position[1]) / 0.70 * 255, 0, 255))
            return x, y

        def square(position: np.ndarray, color: tuple[int, int, int], radius: int) -> None:
            x, y = project(position)
            image[
                max(0, y - radius) : min(256, y + radius + 1),
                max(0, x - radius) : min(256, x + radius + 1),
            ] = color

        square(self._toy_goal, (60, 180, 90), 8)
        square(self._toy_object, (230, 130, 45), 6)
        square(self._toy_hand, (55, 95, 205), 5)
        return image

    def close(self) -> None:
        if self._closed:
            return
        if self._backend_env is not None and hasattr(self._backend_env, "close"):
            self._backend_env.close()
        self._closed = True


class ScriptedPickPlaceExpert:
    """Reproducible scripted Sawyer expert for Meta-World and toy CI.

    Meta-World runs use its maintained ``SawyerPickPlaceV3Policy`` (or V2 for
    an older installed benchmark). The deterministic local state machine is
    used only by the explicit toy backend.
    """

    def __init__(self, env: REIMPickPlaceEnv) -> None:
        self.env = env
        self._phase = "approach"
        self._grasp_steps = 0
        self._policy: Any | None = None
        self.algorithm = "Sawyer scripted"
        if env.backend == "metaworld":
            self._policy = self._make_metaworld_policy()

    @staticmethod
    def _make_metaworld_policy() -> Any:
        errors: list[str] = []
        try:
            from metaworld.policies import SawyerPickPlaceV3Policy

            return SawyerPickPlaceV3Policy()
        except (ImportError, AttributeError) as exc:
            errors.append(f"V3: {exc}")
        try:
            from metaworld.policies import SawyerPickPlaceV2Policy

            return SawyerPickPlaceV2Policy()
        except (ImportError, AttributeError) as exc:
            errors.append(f"V2: {exc}")
        raise RuntimeError(
            "The installed Meta-World package has no PickPlace scripted policy: "
            + "; ".join(errors)
        )

    def reset(self) -> None:
        self._phase = "approach"
        self._grasp_steps = 0
        if self.env.backend == "metaworld":
            # Official scripted policies are stateless, but reconstructing also
            # protects against future versions adding internal episode state.
            self._policy = self._make_metaworld_policy()

    def act(self, state: np.ndarray | None = None) -> np.ndarray:
        if self._policy is not None:
            action = self._policy.get_action(self.env.raw_observation)
            return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        return self._toy_action()

    @staticmethod
    def _move(hand: np.ndarray, target: np.ndarray, gripper: float) -> np.ndarray:
        displacement = np.clip((target - hand) * 12.0, -1.0, 1.0)
        return np.concatenate(
            [displacement, np.asarray([gripper], dtype=np.float32)]
        ).astype(np.float32)

    def _toy_action(self) -> np.ndarray:
        parts = self.env.get_state_components()
        hand = parts["end_effector_position"]
        obj = parts["object_position"]
        goal = parts["goal_position"]

        if self._phase == "approach":
            target = obj + np.asarray([0.0, 0.0, 0.075], dtype=np.float32)
            if np.linalg.norm(hand - target) < 0.025:
                self._phase = "descend"
            return self._move(hand, target, -1.0)

        if self._phase == "descend":
            target = obj + np.asarray([0.0, 0.0, 0.022], dtype=np.float32)
            if np.linalg.norm(hand - target) < 0.018:
                self._phase = "grasp"
                self._grasp_steps = 0
            return self._move(hand, target, -1.0)

        if self._phase == "grasp":
            self._grasp_steps += 1
            if self._grasp_steps >= 6:
                self._phase = "lift"
            return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        if self._phase == "lift":
            target = np.asarray(
                [obj[0], obj[1], max(0.25, float(goal[2] + 0.10))],
                dtype=np.float32,
            )
            if hand[2] >= target[2] - 0.025:
                self._phase = "transport"
            return self._move(hand, target, 1.0)

        if self._phase == "transport":
            target = goal + np.asarray([0.0, 0.0, 0.09], dtype=np.float32)
            if np.linalg.norm(hand - target) < 0.030:
                self._phase = "place"
            return self._move(hand, target, 1.0)

        if self._phase == "place":
            target = goal + np.asarray([0.0, 0.0, 0.025], dtype=np.float32)
            if np.linalg.norm(obj - goal) <= self.env.success_threshold:
                self._phase = "release"
            return self._move(hand, target, 1.0)

        return np.asarray([0.0, 0.0, 0.0, -1.0], dtype=np.float32)


def make_scripted_expert(env: REIMPickPlaceEnv) -> ScriptedPickPlaceExpert:
    """Factory kept stable for collection/evaluation scripts."""

    return ScriptedPickPlaceExpert(env)
