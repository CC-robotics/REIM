"""Unified closed-loop evaluation for ACT, recovery baselines, and REIM.

The formal trigger-state deployment uses:

``ACT -> causal LSTM failure risk -> task-completing imitation recovery``.

The generic controller retains configurable hysteretic return-to-ACT support
for legacy ablations, while the scientific CLI defaults hold recovery control
until task success or the full recovery budget.

No model or result is synthesized when a checkpoint is absent.  The command
fails with an actionable error instead.  ``--backend toy`` is an explicit,
deterministic integration-test backend; ``auto`` never silently substitutes it
for Meta-World.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import inspect
import json
import logging
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:  # Support ``python evaluation/evaluate_reim.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import EpisodeMetrics, aggregate_episode_metrics

LOGGER = logging.getLogger("reim.evaluation")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ``noise_level`` is a dimensionless experimental fraction, not a raw standard
# deviation shared by quantities with different units.  At the requested 40%
# endpoint these calibrated scales reproduce the disturbed-rollout protocol:
# action sigma~=0.16, observation sigma~=0.01, and one object
# position displacement~=0.04 m.
ROBUSTNESS_ACTION_STD_SCALE = 0.40
ROBUSTNESS_OBSERVATION_STD_SCALE = 0.025
ROBUSTNESS_OBJECT_STD_SCALE = 0.10

# Deployment defaults for the trigger-state recovery curriculum.  The risk
# gate keeps the online state inside the corrective policy's support.  The
# recovery option then retains control until task success or its full 150-step
# horizon; setting
# ``recovery_min_steps == recovery_budget`` prevents a weak intermediate
# "recovered" signal from handing control back to ACT prematurely.
DEFAULT_FAILURE_THRESHOLD = 0.2
DEFAULT_RECOVERY_EXIT_THRESHOLD = 0.0
DEFAULT_RECOVERY_BUDGET = 150
DEFAULT_RECOVERY_MIN_STEPS = 150
DEFAULT_RECOVERY_CLEAR_STEPS = 200
DEFAULT_RECOVERY_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "imitation_recovery.pt"
)

METHOD_LABELS: dict[str, str] = {
    "bc": "ACT",
    "bc_random_reset": "ACT + Random Reset",
    "bc_detector": "ACT + Detector",
    "bc_rl_recovery": "ACT + Heuristic Recovery",
    "reim": "REIM (ACT + Detector + Recovery)",
}
METHOD_ALIASES: dict[str, str] = {
    "bc": "bc",
    "act": "bc",
    "random": "bc_random_reset",
    "random_reset": "bc_random_reset",
    "bc_random_reset": "bc_random_reset",
    "act_random_reset": "bc_random_reset",
    "detector": "bc_detector",
    "bc_detector": "bc_detector",
    "act_detector": "bc_detector",
    "recovery": "bc_rl_recovery",
    "bc_recovery": "bc_rl_recovery",
    "bc_rl_recovery": "bc_rl_recovery",
    "act_recovery": "bc_rl_recovery",
    "act_rl_recovery": "bc_rl_recovery",
    "reim": "reim",
}
CLI_METHOD_CHOICES = tuple(sorted(METHOD_ALIASES))


def canonical_method(method: str) -> str:
    key = method.strip().lower().replace("+", "_").replace(" ", "_")
    key = "_".join(piece for piece in key.split("_") if piece)
    if key not in METHOD_ALIASES:
        choices = ", ".join(sorted(METHOD_LABELS))
        raise ValueError(f"unknown method {method!r}; choose one of: {choices}")
    return METHOD_ALIASES[key]


@dataclass(slots=True)
class ControllerConfig:
    # Backwards-compatible library defaults. Scientific CLI entry points pass
    # the trigger-state deployment constants explicitly.
    failure_threshold: float = 0.8
    recovery_exit_threshold: float = 0.7
    sequence_length: int = 10
    recovery_budget: int = 50
    recovery_min_steps: int = 3
    recovery_clear_steps: int = 2
    heuristic_window: int = 12
    heuristic_min_steps: int = 20
    stagnation_tolerance: float = 1e-3
    max_random_resets: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.recovery_exit_threshold < self.failure_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= exit < failure <= 1; got "
                f"{self.recovery_exit_threshold}, {self.failure_threshold}"
            )
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.recovery_budget <= 0:
            raise ValueError("recovery_budget must be positive")
        if not 0 < self.recovery_min_steps <= self.recovery_budget:
            raise ValueError(
                "recovery_min_steps must lie in [1, recovery_budget]; got "
                f"{self.recovery_min_steps} and {self.recovery_budget}"
            )
        if self.recovery_clear_steps <= 0:
            raise ValueError("recovery_clear_steps must be positive")


@dataclass(slots=True)
class ControlDecision:
    action: np.ndarray
    source: str
    failure_probability: float
    request_reset: bool = False


class BCAdapter:
    """Normalize common PyTorch/SB3-style inference APIs to one callable."""

    def __init__(
        self,
        policy: Any,
        *,
        state_mean: np.ndarray | None = None,
        state_std: np.ndarray | None = None,
        device: str = "cpu",
    ) -> None:
        self.policy = policy
        self.state_mean = state_mean
        self.state_std = state_std
        self.device = device

    def reset(self) -> None:
        """Clear temporal action chunks for ACT-style policies."""

        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        if self.state_mean is not None and self.state_std is not None:
            obs = (obs - self.state_mean) / np.maximum(self.state_std, 1e-6)

        if hasattr(self.policy, "predict"):
            try:
                result = self.policy.predict(obs, deterministic=True)
            except TypeError:
                result = self.policy.predict(obs)
            if isinstance(result, tuple):
                result = result[0]
            return np.asarray(result, dtype=np.float32).reshape(-1)

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("PyTorch is required to run the ACT checkpoint") from exc
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            result = self.policy(tensor)
        if isinstance(result, (tuple, list)):
            result = result[0]
        return np.asarray(result.detach().cpu(), dtype=np.float32).reshape(-1)


class DetectorAdapter:
    """Failure probability inference with explicit sigmoid handling."""

    def __init__(self, detector: Any, *, device: str = "cpu") -> None:
        self.detector = detector
        self.device = device

    def __call__(
        self, state_sequence: np.ndarray, valid_length: int | None = None
    ) -> float:
        sequence = np.asarray(state_sequence, dtype=np.float32)
        if sequence.ndim != 2:
            raise ValueError(f"detector expects [T, D], got {sequence.shape}")
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyTorch is required to run the detector") from exc
        tensor = torch.as_tensor(
            sequence, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        length_tensor = (
            None
            if valid_length is None
            else torch.as_tensor(
                [valid_length], dtype=torch.long, device=self.device
            )
        )
        with torch.no_grad():
            if hasattr(self.detector, "predict_proba"):
                try:
                    output = self.detector.predict_proba(tensor, length_tensor)
                except (TypeError, AttributeError):
                    numpy_lengths = (
                        None
                        if valid_length is None
                        else np.asarray([valid_length], dtype=np.int64)
                    )
                    try:
                        output = self.detector.predict_proba(
                            sequence[None, ...], numpy_lengths
                        )
                    except TypeError:
                        output = self.detector.predict_proba(sequence[None, ...])
                probability = output
            else:
                try:
                    logits = self.detector(tensor, length_tensor)
                except TypeError:
                    logits = self.detector(tensor)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                probability = torch.sigmoid(logits)
        if hasattr(probability, "detach"):
            probability = probability.detach().cpu().numpy()
        value = float(np.asarray(probability).reshape(-1)[0])
        if not np.isfinite(value):
            raise RuntimeError("failure detector produced a non-finite probability")
        return float(np.clip(value, 0.0, 1.0))


class RecoveryAdapter:
    def __init__(self, policy: Any) -> None:
        self.policy = policy

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        if hasattr(self.policy, "predict"):
            try:
                result = self.policy.predict(observation, deterministic=True)
            except TypeError:
                result = self.policy.predict(observation)
        else:
            result = self.policy(observation)
        if isinstance(result, tuple):
            result = result[0]
        if hasattr(result, "detach"):
            result = result.detach().cpu().numpy()
        return np.asarray(result, dtype=np.float32).reshape(-1)


def _extract_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping):
        for key in (
            "model_state_dict",
            "state_dict",
            "policy_state_dict",
            "detector_state_dict",
        ):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        if checkpoint and all(hasattr(value, "shape") for value in checkpoint.values()):
            return checkpoint
    raise ValueError(
        "checkpoint does not contain model_state_dict/state_dict/policy_state_dict"
    )


def _strip_uniform_prefix(
    state_dict: Mapping[str, Any], prefixes: Sequence[str] = ("module.", "_orig_mod.")
) -> dict[str, Any]:
    result = dict(state_dict)
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if result and all(key.startswith(prefix) for key in result):
                result = {key[len(prefix) :]: value for key, value in result.items()}
                changed = True
    return result


def _load_torch_checkpoint(path: Path, device: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(
            f"required checkpoint not found: {path}. Run the corresponding trainer first."
        )
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required to load checkpoints") from exc
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch < 2.0.
        return torch.load(path, map_location=device)


def _checkpoint_metadata(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, Mapping):
        return {}
    metadata: dict[str, Any] = {}
    for key in ("model_config", "config", "metadata", "hparams", "hyperparameters"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            metadata.update(value)
    for key in (
        "state_dim",
        "input_dim",
        "action_dim",
        "hidden_dim",
        "hidden_dims",
        "num_layers",
    ):
        if key in checkpoint:
            metadata[key] = checkpoint[key]
    return metadata


def _normalization_arrays(
    checkpoint: Any, state_dim: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not isinstance(checkpoint, Mapping):
        return None, None
    pairs = (
        ("state_mean", "state_std"),
        ("obs_mean", "obs_std"),
        ("observation_mean", "observation_std"),
    )
    for mean_key, std_key in pairs:
        if mean_key in checkpoint and std_key in checkpoint:
            mean = np.asarray(checkpoint[mean_key], dtype=np.float32).reshape(-1)
            std = np.asarray(checkpoint[std_key], dtype=np.float32).reshape(-1)
            if mean.size != state_dim or std.size != state_dim:
                raise ValueError(
                    f"normalization in checkpoint has dimension {mean.size}/{std.size}, "
                    f"expected {state_dim}"
                )
            return mean, std
    return None, None


def load_bc_policy(
    checkpoint_path: str | Path,
    *,
    state_dim: int,
    action_dim: int,
    device: str = "cpu",
) -> BCAdapter:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"required ACT checkpoint not found: {checkpoint_path}. "
            "Run the imitation-policy trainer first."
        )
    from models.bc_policy import BCPolicy

    if hasattr(BCPolicy, "from_checkpoint"):
        policy = BCPolicy.from_checkpoint(
            checkpoint_path,
            map_location=device,
            state_dim=state_dim,
            action_dim=action_dim,
        )
    else:  # Compatibility with minimal third-party BC implementations.
        checkpoint = _load_torch_checkpoint(checkpoint_path, device)
        metadata = _checkpoint_metadata(checkpoint)
        hidden_dims = metadata.get("hidden_dims", (256, 256))
        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims, hidden_dims)
        policy = BCPolicy(
            state_dim=int(metadata.get("state_dim", state_dim)),
            action_dim=int(metadata.get("action_dim", action_dim)),
            hidden_dims=tuple(int(value) for value in hidden_dims),
        )
        state_dict = _strip_uniform_prefix(_extract_state_dict(checkpoint))
        try:
            policy.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"ACT checkpoint architecture does not match {checkpoint_path}: {exc}"
            ) from exc
        policy.to(device)
        policy.eval()
    LOGGER.info("Loaded ACT policy from %s", checkpoint_path)
    # BCPolicy persists normalization as module buffers.  External arrays are
    # intentionally not applied here, avoiding accidental double normalization.
    return BCAdapter(policy, device=device)


def load_failure_detector(
    checkpoint_path: str | Path,
    *,
    state_dim: int,
    device: str = "cpu",
) -> DetectorAdapter:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"required detector checkpoint not found: {checkpoint_path}. "
            "Run the failure-detector trainer first."
        )
    from models.failure_detector import FailureDetector

    if hasattr(FailureDetector, "from_checkpoint"):
        detector = FailureDetector.from_checkpoint(
            checkpoint_path, map_location=device, state_dim=state_dim
        )
    else:
        checkpoint = _load_torch_checkpoint(checkpoint_path, device)
        metadata = _checkpoint_metadata(checkpoint)
        signature = inspect.signature(FailureDetector)
        candidates = {
            "state_dim": int(metadata.get("state_dim", state_dim)),
            "input_dim": int(
                metadata.get("input_dim", metadata.get("state_dim", state_dim))
            ),
            "hidden_dim": int(metadata.get("hidden_dim", 128)),
            "num_layers": int(metadata.get("num_layers", 1)),
        }
        kwargs = {
            key: value for key, value in candidates.items() if key in signature.parameters
        }
        detector = FailureDetector(**kwargs)
        state_dict = _strip_uniform_prefix(_extract_state_dict(checkpoint))
        try:
            detector.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"detector checkpoint architecture does not match {checkpoint_path}: {exc}"
            ) from exc
        detector.to(device)
        detector.eval()
    LOGGER.info("Loaded failure detector from %s", checkpoint_path)
    return DetectorAdapter(detector, device=device)


def load_recovery_policy(
    checkpoint_path: str | Path,
    *,
    env: Any = None,
    device: str = "auto",
) -> RecoveryAdapter:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        # Stable-Baselines3 accepts "foo" for "foo.zip"; make the validation
        # similarly ergonomic while still rejecting absent checkpoints.
        zip_candidate = checkpoint_path.with_suffix(".zip")
        if zip_candidate.is_file():
            checkpoint_path = zip_candidate
        else:
            raise FileNotFoundError(
                f"required recovery checkpoint not found: {checkpoint_path}"
            )
    # The published controller uses a standalone deterministic imitation actor.
    # Legacy SB3 archives remain loadable for the PPO fine-tuning ablation.
    if checkpoint_path.suffix.lower() == ".pt":
        from models.imitation_recovery_policy import ImitationRecoveryPolicy

        standalone_device = (
            "cpu"
            if str(device).lower() == "auto"
            or str(device).lower().startswith("cuda")
            else device
        )
        policy = ImitationRecoveryPolicy.load(
            checkpoint_path, device=standalone_device
        )
        LOGGER.info(
            "Loaded trigger-aligned imitation recovery policy from %s",
            checkpoint_path,
        )
        return RecoveryAdapter(policy)

    # SB3's small MLP ActorCriticPolicy is faster on CPU and explicitly emits
    # a warning when placed on CUDA. ACT/LSTM may still use the requested GPU;
    # keeping recovery inference on CPU also avoids tiny-kernel contention.
    recovery_device = (
        "cpu" if str(device).lower().startswith("cuda") else device
    )
    try:
        from models.recovery_policy import RecoveryPolicy

        policy = RecoveryPolicy.load(
            str(checkpoint_path), env=env, device=recovery_device
        )
    except (ImportError, AttributeError, TypeError):
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "stable-baselines3 is required to load the recovery checkpoint"
            ) from exc
        policy = PPO.load(str(checkpoint_path), env=env, device=recovery_device)
    LOGGER.info("Loaded recovery policy from %s", checkpoint_path)
    return RecoveryAdapter(policy)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def effective_profile(
    requested: str,
    episodes: int,
    *,
    full_episodes: int,
    smoke_episodes: int,
) -> str:
    """Return ``full``/``smoke`` only when the episode budget matches protocol."""

    expected = full_episodes if requested == "full" else smoke_episodes
    return requested if int(episodes) == expected else "custom"


def load_environment_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"environment config not found: {config_path}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to read environment config") from exc
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"environment config must be a mapping: {config_path}")
    return data


def make_env(
    *,
    backend: str,
    seed: int,
    env_config: str | Path | None = None,
    noise_level: float = 0.0,
    render_mode: str | None = None,
) -> Any:
    """Construct a configured environment with calibrated mixed disturbances."""

    if noise_level < 0.0:
        raise ValueError("noise_level must be non-negative")
    from env.metaworld_pickplace import REIMPickPlaceEnv

    config = load_environment_config(env_config)
    try:
        return REIMPickPlaceEnv(
            config=config,
            seed=seed,
            backend=backend,
            render_mode=render_mode,
            action_noise_std=(
                float(noise_level) * ROBUSTNESS_ACTION_STD_SCALE
            ),
            observation_noise_std=(
                float(noise_level) * ROBUSTNESS_OBSERVATION_STD_SCALE
            ),
            object_noise_std=(
                float(noise_level) * ROBUSTNESS_OBJECT_STD_SCALE
            ),
        )
    except Exception as exc:
        if backend in {"metaworld", "auto"}:
            raise RuntimeError(
                "Meta-World environment construction failed. Install the project "
                "dependencies and MuJoCo, or explicitly use --backend toy for an "
                f"integration test. Original error: {exc}"
            ) from exc
        raise


def _space_dim(space: Any, name: str) -> int:
    shape = getattr(space, "shape", None)
    if not shape:
        raise ValueError(f"environment {name}_space must expose a finite shape")
    return int(np.prod(shape))


def _reset_env(
    env: Any,
    seed: int | None = None,
    *,
    episode_specification: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    options = (
        None
        if episode_specification is None
        else {"reim_episode_spec": dict(episode_specification)}
    )
    if seed is not None:
        result = env.reset(seed=seed, options=options)
    elif options is not None:
        result = env.reset(options=options)
    else:
        result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        observation, info = result
    else:  # Legacy Gym.
        observation, info = result, {}
    return np.asarray(observation, dtype=np.float32).reshape(-1), dict(info or {})


def _step_env(
    env: Any, action: np.ndarray
) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    result = env.step(action)
    if not isinstance(result, tuple):
        raise TypeError("environment step() must return a tuple")
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
    elif len(result) == 4:  # Legacy Gym.
        observation, reward, done, info = result
        terminated, truncated = bool(done), False
    else:
        raise ValueError(f"environment step() returned {len(result)} values")
    return (
        np.asarray(observation, dtype=np.float32).reshape(-1),
        float(reward),
        bool(terminated),
        bool(truncated),
        dict(info or {}),
    )


def _is_success(info: Mapping[str, Any], reward: float | None = None) -> bool:
    del reward  # Success must be explicit; reward scales differ across backends.
    return bool(info.get("success", info.get("is_success", False)))


def _failure_reason(info: Mapping[str, Any]) -> str:
    value = info.get("failure_reason", "")
    if value:
        return str(value)
    if info.get("object_dropped"):
        return "object_dropped"
    if info.get("failure"):
        return "failure"
    return ""


class REIMController:
    """Stateful controller with detector history and recovery hysteresis."""

    def __init__(
        self,
        method: str,
        bc_policy: Callable[[np.ndarray], np.ndarray],
        *,
        detector: Callable[[np.ndarray], float] | None = None,
        recovery_policy: Callable[[np.ndarray], np.ndarray] | None = None,
        config: ControllerConfig | None = None,
    ) -> None:
        self.method = canonical_method(method)
        self.bc_policy = bc_policy
        self.detector = detector
        self.recovery_policy = recovery_policy
        self.config = config or ControllerConfig()
        if self.method in {"reim", "bc_detector"} and detector is None:
            raise ValueError(f"{METHOD_LABELS[self.method]} requires a failure detector")
        if self.method in {"reim", "bc_rl_recovery"} and recovery_policy is None:
            raise ValueError(f"{METHOD_LABELS[self.method]} requires a recovery policy")
        self.history: deque[np.ndarray] = deque(maxlen=self.config.sequence_length)
        self.distance_history: deque[float] = deque(
            maxlen=self.config.heuristic_window
        )
        self.reset_statistics()

    def reset_statistics(self) -> None:
        self.recovery_active = False
        self.recovery_steps_current = 0
        self.recovery_steps_total = 0
        self.recovery_attempts = 0
        self.recovery_successes = 0
        self.detector_triggers = 0
        self.failure_probability_max = 0.0
        self._clear_count = 0
        self._risk_latched = False
        self._external_resets_pending = 0
        self._episode_steps = 0
        self._initial_object_position: np.ndarray | None = None
        self._interaction_observed = False

    def reset_observation(
        self, observation: np.ndarray, *, preserve_statistics: bool = False
    ) -> None:
        if not preserve_statistics:
            self.reset_statistics()
        self.history.clear()
        if hasattr(self.bc_policy, "reset"):
            self.bc_policy.reset()
        # The detector was trained with a causal valid prefix and right padding.
        # Its first feature is the observation *after* the first transition,
        # so the reset observation is intentionally excluded.
        self.distance_history.clear()
        # A task reset starts a new physical interaction phase even when
        # intervention counters are intentionally preserved.
        self._initial_object_position = None
        self._interaction_observed = False
        self.recovery_active = False
        self.recovery_steps_current = 0
        self._clear_count = 0
        self._risk_latched = False

    def _detector_probability(self) -> float:
        if self.detector is None:
            return 0.0
        if not self.history:
            return 0.0
        observed = np.stack(tuple(self.history), axis=0)
        sequence = np.zeros(
            (self.config.sequence_length, observed.shape[-1]), dtype=np.float32
        )
        sequence[: len(observed)] = observed
        try:
            probability = self.detector(sequence, len(observed))
        except TypeError:
            # Lightweight third-party/test callables may expose only the
            # historical one-argument interface.
            probability = self.detector(sequence)
        return float(np.clip(probability, 0.0, 1.0))

    def _update_interaction_state(self, info: Mapping[str, Any]) -> None:
        value = info.get("object_position")
        if value is not None:
            try:
                position = np.asarray(value, dtype=np.float32).reshape(-1)[:3]
                if position.size == 3 and np.isfinite(position).all():
                    if self._initial_object_position is None:
                        self._initial_object_position = position.copy()
                    elif (
                        np.linalg.norm(position - self._initial_object_position)
                        >= 0.02
                    ):
                        self._interaction_observed = True
            except (TypeError, ValueError):
                pass
        self._interaction_observed |= bool(
            info.get(
                "grasped",
                info.get("grasp_success", info.get("grasp_successful", False)),
            )
        )

    def _heuristic_probability(self, info: Mapping[str, Any]) -> float:
        self._update_interaction_state(info)
        if info.get("failure") or info.get("object_dropped"):
            return 1.0
        if bool(info.get("recovered", False)):
            return 0.0
        if (
            self._episode_steps >= self.config.heuristic_min_steps
            and self._interaction_observed
            and len(self.distance_history) == self.distance_history.maxlen
        ):
            distances = np.asarray(self.distance_history, dtype=np.float64)
            if np.isfinite(distances).all():
                improvement = float(distances[0] - np.min(distances[1:]))
                worsening = float(distances[-1] - np.min(distances[:-1]))
                if worsening > 5.0 * self.config.stagnation_tolerance:
                    return 0.9
                if improvement < self.config.stagnation_tolerance:
                    return 0.85
        return 0.0

    def _risk(self, info: Mapping[str, Any]) -> float:
        if self.method in {"reim", "bc_detector"}:
            value = self._detector_probability()
        else:
            value = self._heuristic_probability(info)
        self.failure_probability_max = max(self.failure_probability_max, value)
        return value

    def _enter_recovery(self) -> None:
        self.recovery_active = True
        self.recovery_steps_current = 0
        self._clear_count = 0
        self.recovery_attempts += 1
        self.detector_triggers += 1
        self._risk_latched = True

    def _close_recovery(self, successful: bool) -> None:
        if self.recovery_active and successful:
            self.recovery_successes += 1
        self.recovery_active = False
        self.recovery_steps_current = 0
        self._clear_count = 0
        # ACT temporal ensembles contain predictions for actions that were not
        # executed while recovery held control. Clear them before handing
        # control back, otherwise stale failure-state chunks can immediately
        # undo a physically successful recovery.
        if hasattr(self.bc_policy, "reset"):
            self.bc_policy.reset()

    def register_external_reset(self) -> None:
        self.recovery_attempts += 1
        self.detector_triggers += 1
        self._external_resets_pending += 1

    def finalize(self, success: bool) -> None:
        if self.recovery_active:
            self._close_recovery(successful=success)
        if success and self._external_resets_pending:
            # Each random reset is an intervention; only the final intervention
            # leading to task completion is credited as recovered.
            self.recovery_successes += 1
        self._external_resets_pending = 0

    def act(
        self, observation: np.ndarray, info: Mapping[str, Any] | None = None
    ) -> ControlDecision:
        state = np.asarray(observation, dtype=np.float32).reshape(-1)
        info = info or {}

        if self.method == "bc":
            bc_action = np.asarray(self.bc_policy(state), dtype=np.float32).reshape(-1)
            return ControlDecision(bc_action, "bc", 0.0)

        risk = self._risk(info)
        if risk < self.config.recovery_exit_threshold:
            self._risk_latched = False

        if self.recovery_active:
            if self.recovery_policy is None:  # Defensive invariant.
                raise RuntimeError("recovery is active but no recovery policy is loaded")
            return ControlDecision(
                np.asarray(self.recovery_policy(state), dtype=np.float32).reshape(-1),
                "recovery",
                risk,
            )

        triggered = risk >= self.config.failure_threshold
        if triggered and self.method in {"reim", "bc_rl_recovery"}:
            if not self._risk_latched:
                self._enter_recovery()
                if self.recovery_policy is None:  # Defensive invariant.
                    raise RuntimeError("failure triggered without a recovery policy")
                return ControlDecision(
                    np.asarray(self.recovery_policy(state), dtype=np.float32).reshape(-1),
                    "recovery",
                    risk,
                )

        # Query ACT only when its action may actually be used.  Besides saving
        # inference, this prevents non-executed chunks from entering the
        # temporal ensemble during a recovery intervention.
        bc_action = np.asarray(self.bc_policy(state), dtype=np.float32).reshape(-1)
        if self.method == "bc_random_reset" and triggered and not self._risk_latched:
            self._risk_latched = True
            return ControlDecision(
                bc_action, "random_reset", risk, request_reset=True
            )

        if self.method == "bc_detector":
            if triggered:
                if not self._risk_latched:
                    self.detector_triggers += 1
                self._risk_latched = True
                # Detector-only ablation applies a conservative safety hold.
                # It can avoid compounding an error but cannot actively recover.
                safe_action = 0.2 * bc_action
                if safe_action.size:
                    safe_action[-1] = bc_action[-1]
                return ControlDecision(safe_action, "safety_hold", risk)
            return ControlDecision(bc_action, "bc", risk)

        return ControlDecision(bc_action, "bc", risk)

    def observe_transition(
        self,
        next_observation: np.ndarray,
        reward: float,
        info: Mapping[str, Any],
        *,
        success: bool,
    ) -> None:
        del reward
        self._episode_steps += 1
        self._update_interaction_state(info)
        self.history.append(
            np.asarray(next_observation, dtype=np.float32).reshape(-1).copy()
        )
        distance = info.get("distance_to_goal", info.get("distance"))
        if distance is not None:
            try:
                self.distance_history.append(float(distance))
            except (TypeError, ValueError):
                pass

        if not self.recovery_active:
            return
        self.recovery_steps_current += 1
        self.recovery_steps_total += 1
        if success:
            self._close_recovery(successful=True)
            return

        risk = self._risk(info)
        explicitly_recovered = bool(info.get("recovered", False))
        if explicitly_recovered or risk <= self.config.recovery_exit_threshold:
            self._clear_count += 1
        else:
            self._clear_count = 0
        min_steps_done = (
            self.recovery_steps_current >= self.config.recovery_min_steps
        )
        if min_steps_done and self._clear_count >= self.config.recovery_clear_steps:
            self._close_recovery(successful=True)
        elif self.recovery_steps_current >= self.config.recovery_budget:
            self._close_recovery(successful=False)


def _trace_info(info: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "backend",
        "ee_position",
        "object_position",
        "goal_position",
        "distance_to_goal",
        "distance",
        "failure",
        "failure_reason",
        "success",
        "is_success",
    ):
        value = info.get(key)
        if value is None:
            continue
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, np.generic):
            result[key] = value.item()
        elif isinstance(value, (str, int, float, bool, list, tuple)):
            result[key] = value
    return result


def run_episode(
    env: Any,
    controller: REIMController,
    *,
    episode: int,
    seed: int,
    max_steps: int,
    render: bool = False,
    capture_trace: bool = False,
    episode_specification: Mapping[str, Any] | None = None,
) -> tuple[EpisodeMetrics, dict[str, Any] | None]:
    """Run one logical task episode, including any random-reset intervention."""

    observation, info = _reset_env(
        env,
        seed=seed,
        episode_specification=episode_specification,
    )
    episode_specification_sha256 = str(
        info.get("episode_specification_sha256", "")
    )
    episode_bank_sha256 = str(info.get("episode_bank_sha256", ""))
    metaworld_task_sha256 = str(info.get("metaworld_task_sha256", ""))
    retry_episode_specifications = (
        []
        if episode_specification is None
        else list(episode_specification.get("retry_specifications", []))
    )
    if (
        episode_specification is not None
        and controller.method == "bc_random_reset"
        and controller.config.max_random_resets > len(retry_episode_specifications)
    ):
        raise ValueError(
            "episode bank provides fewer deterministic retry specifications "
            "than max_random_resets"
        )
    used_retry_specification_sha256s: list[str] = []
    used_retry_task_sha256s: list[str] = []
    controller.reset_observation(observation)
    action_space = getattr(env, "action_space", None)
    start = time.perf_counter()
    success = False
    terminated = False
    truncated = False
    reason = ""
    random_resets = 0
    trace: dict[str, Any] | None = (
        {
            "episode": episode,
            "seed": seed,
            "method": METHOD_LABELS[controller.method],
            "states": [observation.tolist()],
            "actions": [],
            "sources": [],
            "failure_probabilities": [],
            "info": [_trace_info(info)],
            "episode_specification_sha256": str(
                episode_specification_sha256
            ),
            "episode_bank_sha256": episode_bank_sha256,
            "metaworld_task_sha256": metaworld_task_sha256,
            "random_reset_events": [],
        }
        if capture_trace
        else None
    )

    executed_steps = 0
    while executed_steps < max_steps:
        decision = controller.act(observation, info)
        if (
            decision.request_reset
            and random_resets < controller.config.max_random_resets
        ):
            controller.register_external_reset()
            retry_index = random_resets
            random_resets += 1
            retry_specification = (
                None
                if episode_specification is None
                else retry_episode_specifications[retry_index]
            )
            reset_seed = (
                seed + 100_003 * random_resets
                if retry_specification is None
                else int(retry_specification["episode_seed"])
            )
            observation, info = _reset_env(
                env,
                seed=reset_seed,
                episode_specification=retry_specification,
            )
            retry_spec_sha = str(
                info.get("episode_specification_sha256", "")
            )
            retry_task_sha = str(info.get("metaworld_task_sha256", ""))
            used_retry_specification_sha256s.append(retry_spec_sha)
            used_retry_task_sha256s.append(retry_task_sha)
            controller.reset_observation(observation, preserve_statistics=True)
            if trace is not None:
                trace["random_reset_events"].append(
                    {
                        "retry_index": retry_index,
                        "seed": reset_seed,
                        "episode_specification_sha256": retry_spec_sha,
                        "metaworld_task_sha256": retry_task_sha,
                    }
                )
                trace["sources"].append("random_reset")
                trace["actions"].append(
                    np.zeros(_space_dim(action_space, "action"), dtype=np.float32).tolist()
                )
                trace["failure_probabilities"].append(
                    float(decision.failure_probability)
                )
                trace["states"].append(observation.tolist())
                trace["info"].append(_trace_info(info))
            continue

        action = np.asarray(decision.action, dtype=np.float32).reshape(-1)
        if action_space is not None:
            expected_dim = _space_dim(action_space, "action")
            if action.size != expected_dim:
                raise ValueError(
                    f"policy produced {action.size} actions, environment expects {expected_dim}"
                )
            low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
            high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
            action = np.clip(action, low, high)

        next_observation, reward, terminated, truncated, info = _step_env(env, action)
        executed_steps += 1
        success = _is_success(info, reward)
        reason = _failure_reason(info) or reason
        controller.observe_transition(
            next_observation, reward, info, success=success
        )
        if trace is not None:
            trace["actions"].append(action.tolist())
            trace["sources"].append(decision.source)
            trace["failure_probabilities"].append(
                float(decision.failure_probability)
            )
            trace["states"].append(next_observation.tolist())
            trace["info"].append(_trace_info(info))
        observation = next_observation

        if render:
            env.render()
        if success:
            break
        if terminated or truncated:
            if (
                controller.method == "bc_random_reset"
                and random_resets < controller.config.max_random_resets
            ):
                controller.register_external_reset()
                retry_index = random_resets
                random_resets += 1
                retry_specification = (
                    None
                    if episode_specification is None
                    else retry_episode_specifications[retry_index]
                )
                reset_seed = (
                    seed + 100_003 * random_resets
                    if retry_specification is None
                    else int(retry_specification["episode_seed"])
                )
                observation, info = _reset_env(
                    env,
                    seed=reset_seed,
                    episode_specification=retry_specification,
                )
                retry_spec_sha = str(
                    info.get("episode_specification_sha256", "")
                )
                retry_task_sha = str(info.get("metaworld_task_sha256", ""))
                used_retry_specification_sha256s.append(retry_spec_sha)
                used_retry_task_sha256s.append(retry_task_sha)
                controller.reset_observation(observation, preserve_statistics=True)
                terminated = truncated = False
                if trace is not None:
                    trace["random_reset_events"].append(
                        {
                            "retry_index": retry_index,
                            "seed": reset_seed,
                            "episode_specification_sha256": retry_spec_sha,
                            "metaworld_task_sha256": retry_task_sha,
                        }
                    )
                    trace["sources"].append("random_reset")
                    trace["actions"].append(
                        np.zeros(
                            _space_dim(action_space, "action"), dtype=np.float32
                        ).tolist()
                    )
                    trace["failure_probabilities"].append(1.0)
                    trace["states"].append(observation.tolist())
                    trace["info"].append(_trace_info(info))
                continue
            break

    controller.finalize(success)
    elapsed = time.perf_counter() - start
    if controller.method == "bc_random_reset":
        recovery_definition = "post_reset_task_success_per_reset"
    elif (
        controller.method in {"reim", "bc_rl_recovery"}
        and controller.config.recovery_min_steps
        >= controller.config.recovery_budget
        and controller.config.recovery_clear_steps
        > controller.config.recovery_budget
    ):
        recovery_definition = (
            "task_success_while_recovery_active_per_intervention"
        )
    elif controller.method in {"reim", "bc_rl_recovery"}:
        recovery_definition = "risk_clear_or_task_success_per_intervention"
    else:
        recovery_definition = ""
    metrics = EpisodeMetrics(
        method=METHOD_LABELS[controller.method],
        episode=episode,
        seed=seed,
        backend=str(
            info.get(
                "backend",
                getattr(env, "backend_name", getattr(env, "backend", "unknown")),
            )
        ),
        success=success,
        steps=executed_steps,
        elapsed_seconds=elapsed,
        recovery_attempts=controller.recovery_attempts,
        recovery_successes=controller.recovery_successes,
        detector_triggers=controller.detector_triggers,
        recovery_steps=controller.recovery_steps_total,
        failure_probability_max=controller.failure_probability_max,
        recovery_definition=recovery_definition,
        terminated=terminated,
        truncated=truncated,
        failure_reason="" if success else reason,
        episode_specification_sha256=episode_specification_sha256,
        episode_bank_sha256=episode_bank_sha256,
        metaworld_task_sha256=metaworld_task_sha256,
        retry_specification_sha256s=";".join(
            used_retry_specification_sha256s
        ),
        retry_task_sha256s=";".join(used_retry_task_sha256s),
    )
    if trace is not None:
        trace["success"] = success
        trace["recovery_attempts"] = controller.recovery_attempts
        trace["recovery_successes"] = controller.recovery_successes
    return metrics, trace


def evaluate_controller(
    env: Any,
    controller_factory: Callable[[], REIMController],
    *,
    episodes: int,
    seed: int,
    max_steps: int,
    render: bool = False,
    capture_traces: int = 0,
    episode_specifications: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[EpisodeMetrics], list[dict[str, Any]]]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if episode_specifications is not None and len(episode_specifications) != episodes:
        raise ValueError(
            "episode_specifications length must equal the episode count"
        )
    records: list[EpisodeMetrics] = []
    traces: list[dict[str, Any]] = []
    try:
        from tqdm import trange

        iterator: Iterable[int] = trange(episodes, desc="Evaluating", leave=False)
    except ImportError:
        iterator = range(episodes)

    for episode in iterator:
        episode_specification = (
            None
            if episode_specifications is None
            else episode_specifications[episode]
        )
        logical_episode = (
            episode
            if episode_specification is None
            else int(episode_specification["episode_index"])
        )
        episode_seed = (
            seed + episode
            if episode_specification is None
            else int(episode_specification["episode_seed"])
        )
        controller = controller_factory()
        record, trace = run_episode(
            env,
            controller,
            episode=logical_episode,
            seed=episode_seed,
            max_steps=max_steps,
            render=render,
            capture_trace=len(traces) < capture_traces,
            episode_specification=episode_specification,
        )
        records.append(record)
        if trace is not None and record.recovery_attempts > 0:
            traces.append(trace)
    return records, traces[:capture_traces]


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0].keys())
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_evaluation(
    output_path: str | Path,
    records: Sequence[EpisodeMetrics],
    *,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output_path)
    summary = aggregate_episode_metrics(
        records,
        n_bootstrap=bootstrap_samples,
        seed=bootstrap_seed,
    )
    if metadata:
        summary.update(metadata)
    _atomic_write_csv(output, [summary])
    raw_path = output.with_name(f"{output.stem}_episodes.csv")
    raw_rows = []
    for record in records:
        row = record.to_dict()
        if metadata:
            row.update(metadata)
        raw_rows.append(row)
    _atomic_write_csv(raw_path, raw_rows)
    LOGGER.info("Saved summary to %s and episode metrics to %s", output, raw_path)
    return summary


def save_traces(path: str | Path, traces: Sequence[Mapping[str, Any]]) -> None:
    if not traces:
        LOGGER.warning("No traces captured; not writing %s", path)
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(list(traces), handle, indent=2)
    temporary.replace(output)
    LOGGER.info("Saved %d controller traces to %s", len(traces), output)


def build_controller_factory(
    *,
    method: str,
    env: Any,
    bc_checkpoint: str | Path,
    detector_checkpoint: str | Path,
    recovery_checkpoint: str | Path,
    controller_config: ControllerConfig,
    device: str,
) -> Callable[[], REIMController]:
    canonical = canonical_method(method)
    state_dim = _space_dim(env.observation_space, "observation")
    action_dim = _space_dim(env.action_space, "action")
    bc = load_bc_policy(
        bc_checkpoint, state_dim=state_dim, action_dim=action_dim, device=device
    )
    detector = (
        load_failure_detector(
            detector_checkpoint, state_dim=state_dim, device=device
        )
        if canonical in {"reim", "bc_detector"}
        else None
    )
    recovery = (
        load_recovery_policy(recovery_checkpoint, env=env, device=device)
        if canonical in {"reim", "bc_rl_recovery"}
        else None
    )

    def factory() -> REIMController:
        return REIMController(
            canonical,
            bc,
            detector=detector,
            recovery_policy=recovery,
            config=controller_config,
        )

    return factory


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        default="reim",
        choices=CLI_METHOD_CHOICES,
        help="controller to evaluate",
    )
    parser.add_argument(
        "--profile",
        choices=("full", "smoke"),
        default="full",
        help="full defaults to 1000 episodes; smoke defaults to 5",
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backend",
        choices=("metaworld", "auto", "toy"),
        default="metaworld",
    )
    parser.add_argument(
        "--env-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "environment.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        "--act-checkpoint",
        "--bc-checkpoint",
        dest="bc_checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "bc_policy.pt",
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "failure_detector.pt",
    )
    parser.add_argument(
        "--recovery-checkpoint",
        type=Path,
        default=DEFAULT_RECOVERY_CHECKPOINT,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help=(
            "PyTorch intra-op CPU threads used by ACT/LSTM inference. "
            "A small value avoids severe oversubscription in parallel sweeps."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--failure-threshold", type=float, default=DEFAULT_FAILURE_THRESHOLD
    )
    parser.add_argument(
        "--recovery-exit-threshold",
        type=float,
        default=DEFAULT_RECOVERY_EXIT_THRESHOLD,
    )
    parser.add_argument(
        "--recovery-budget", type=int, default=DEFAULT_RECOVERY_BUDGET
    )
    parser.add_argument(
        "--recovery-min-steps", type=int, default=DEFAULT_RECOVERY_MIN_STEPS
    )
    parser.add_argument(
        "--recovery-clear-steps", type=int, default=DEFAULT_RECOVERY_CLEAR_STEPS
    )
    parser.add_argument("--max-random-resets", type=int, default=1)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "tables" / "reim_evaluation.csv",
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=PROJECT_ROOT / "results" / "recovery_traces.json",
    )
    parser.add_argument("--capture-traces", type=int, default=4)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if args.torch_threads <= 0:
        raise ValueError("--torch-threads must be positive")
    try:
        import torch

        torch.set_num_threads(args.torch_threads)
    except ImportError:
        if args.device != "cpu":
            raise
    episodes = args.episodes
    if episodes is None:
        episodes = 1_000 if args.profile == "full" else 5
    profile = effective_profile(
        args.profile, episodes, full_episodes=1_000, smoke_episodes=5
    )
    seed_everything(args.seed)
    config = ControllerConfig(
        failure_threshold=args.failure_threshold,
        recovery_exit_threshold=args.recovery_exit_threshold,
        recovery_budget=args.recovery_budget,
        recovery_min_steps=args.recovery_min_steps,
        recovery_clear_steps=args.recovery_clear_steps,
        max_random_resets=args.max_random_resets,
    )
    env = make_env(
        backend=args.backend,
        seed=args.seed,
        env_config=args.env_config,
        noise_level=args.noise_level,
        render_mode="human" if args.render else None,
    )
    try:
        factory = build_controller_factory(
            method=args.method,
            env=env,
            bc_checkpoint=args.bc_checkpoint,
            detector_checkpoint=args.detector_checkpoint,
            recovery_checkpoint=args.recovery_checkpoint,
            controller_config=config,
            device=args.device,
        )
        records, traces = evaluate_controller(
            env,
            factory,
            episodes=episodes,
            seed=args.seed,
            max_steps=args.max_steps,
            render=args.render,
            capture_traces=args.capture_traces,
        )
    finally:
        if hasattr(env, "close"):
            env.close()
    summary = save_evaluation(
        args.output,
        records,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.seed,
        metadata={
            "Profile": profile,
            "Noise Level": args.noise_level,
            "Benchmark Eligible": bool(
                profile == "full" and args.backend in {"metaworld", "auto"}
            ),
        },
    )
    for trace in traces:
        trace["profile"] = profile
        trace["evaluation_episodes"] = args.episodes
        trace["benchmark_eligible"] = bool(
            profile == "full" and args.backend == "metaworld"
        )
    save_traces(args.trace_output, traces)
    LOGGER.info("Evaluation summary: %s", json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
