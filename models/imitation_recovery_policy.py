"""Standalone supervised recovery actor used by the pivoted REIM controller.

The original recovery actor was trained through a Smooth-L1 imitation loss on
risk-triggered expert continuations, but stored inside a Stable-Baselines3 PPO
checkpoint.  This module contains only the deterministic actor required at
deployment: observation normalization, the policy MLP, the action head, and
action bounds.  It has no critic, stochastic log standard deviation, optimizer,
rollout buffer, or Stable-Baselines3 dependency.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


SCHEMA_VERSION = "reim-imitation-recovery-v1"
POLICY_TYPE = "TriggerAlignedImitationRecovery"


def _torch_load(path: str | Path, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location=map_location)


def _activation(name: str) -> type[nn.Module]:
    normalized = name.strip().lower()
    if normalized == "tanh":
        return nn.Tanh
    if normalized == "relu":
        return nn.ReLU
    raise ValueError(f"Unsupported recovery activation {name!r}.")


class ImitationRecoveryPolicy(nn.Module):
    """Deterministic MLP distilled from the zero-PPO-step recovery actor.

    :meth:`mean_action` reproduces the unbounded Gaussian mean emitted by the
    source SB3 actor. :meth:`forward`, :meth:`predict`, and :meth:`act` clip that
    mean to the action-space bounds, matching ``PPO.predict(...,
    deterministic=True)``.
    """

    schema_version = SCHEMA_VERSION
    policy_type = POLICY_TYPE

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "tanh",
        observation_mean: Sequence[float] | np.ndarray | Tensor | None = None,
        observation_std: Sequence[float] | np.ndarray | Tensor | None = None,
        action_low: Sequence[float] | np.ndarray | Tensor | None = None,
        action_high: Sequence[float] | np.ndarray | Tensor | None = None,
        normalization_epsilon: float = 1e-8,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if int(state_dim) <= 0 or int(action_dim) <= 0:
            raise ValueError("state_dim and action_dim must be positive.")
        dimensions = tuple(int(value) for value in hidden_dims)
        if not dimensions or any(value <= 0 for value in dimensions):
            raise ValueError("hidden_dims must contain positive dimensions.")
        if (
            not math.isfinite(float(normalization_epsilon))
            or float(normalization_epsilon) <= 0
        ):
            raise ValueError("normalization_epsilon must be positive.")

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = dimensions
        self.activation_name = activation.strip().lower()
        self.normalization_epsilon = float(normalization_epsilon)
        activation_type = _activation(self.activation_name)

        layers: list[nn.Module] = []
        input_dim = self.state_dim
        for hidden_dim in self.hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), activation_type()))
            input_dim = hidden_dim
        self.policy_net = nn.Sequential(*layers)
        self.action_net = nn.Linear(input_dim, self.action_dim)

        mean = (
            torch.zeros(self.state_dim, dtype=torch.float32)
            if observation_mean is None
            else torch.as_tensor(observation_mean, dtype=torch.float32).reshape(-1)
        )
        std = (
            torch.ones(self.state_dim, dtype=torch.float32)
            if observation_std is None
            else torch.as_tensor(observation_std, dtype=torch.float32).reshape(-1)
        )
        low = (
            -torch.ones(self.action_dim, dtype=torch.float32)
            if action_low is None
            else torch.as_tensor(action_low, dtype=torch.float32).reshape(-1)
        )
        high = (
            torch.ones(self.action_dim, dtype=torch.float32)
            if action_high is None
            else torch.as_tensor(action_high, dtype=torch.float32).reshape(-1)
        )
        if mean.numel() != self.state_dim or std.numel() != self.state_dim:
            raise ValueError("Observation normalization has the wrong dimension.")
        if low.numel() != self.action_dim or high.numel() != self.action_dim:
            raise ValueError("Action bounds have the wrong dimension.")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("Observation normalization must be finite.")
        if not torch.isfinite(low).all() or not torch.isfinite(high).all():
            raise ValueError("Action bounds must be finite.")
        if torch.any(std <= 0):
            raise ValueError("Observation standard deviations must be positive.")
        if torch.any(low >= high):
            raise ValueError("Every action lower bound must be below its upper bound.")

        self.register_buffer("observation_mean", mean.clone())
        self.register_buffer("observation_std", std.clone())
        self.register_buffer("action_low", low.clone())
        self.register_buffer("action_high", high.clone())
        self.provenance: dict[str, Any] = copy.deepcopy(dict(provenance or {}))

    @property
    def model_config(self) -> dict[str, Any]:
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dims": list(self.hidden_dims),
            "activation": self.activation_name,
            "normalization_epsilon": self.normalization_epsilon,
            "observation_normalization": (
                "identity"
                if torch.equal(
                    self.observation_mean, torch.zeros_like(self.observation_mean)
                )
                and torch.equal(
                    self.observation_std, torch.ones_like(self.observation_std)
                )
                else "affine"
            ),
            "action_clipping": True,
        }

    @property
    def num_timesteps(self) -> int:
        """Compatibility metadata; supervised artifacts contain no PPO steps."""

        return int(self.provenance.get("source_training", {}).get("num_timesteps", 0))

    def _tensor(self, observation: Tensor | np.ndarray | Sequence[float]) -> Tensor:
        parameter = self.action_net.weight
        value = torch.as_tensor(
            observation, dtype=parameter.dtype, device=parameter.device
        )
        if value.ndim not in (1, 2) or value.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected observation shape ({self.state_dim},) or "
                f"(N, {self.state_dim}), got {tuple(value.shape)}."
            )
        if not torch.isfinite(value).all():
            raise ValueError("Recovery observations must be finite.")
        return value

    def mean_action(self, observation: Tensor | np.ndarray | Sequence[float]) -> Tensor:
        """Return the source actor's raw deterministic Gaussian mean."""

        value = self._tensor(observation)
        normalized = (value - self.observation_mean) / torch.clamp_min(
            self.observation_std, self.normalization_epsilon
        )
        return self.action_net(self.policy_net(normalized))

    def forward(self, observation: Tensor | np.ndarray | Sequence[float]) -> Tensor:
        mean = self.mean_action(observation)
        return torch.maximum(torch.minimum(mean, self.action_high), self.action_low)

    @torch.inference_mode()
    def predict_mean(
        self, observation: Tensor | np.ndarray | Sequence[float]
    ) -> np.ndarray:
        return self.mean_action(observation).detach().cpu().numpy()

    @torch.inference_mode()
    def predict(
        self,
        observation: Tensor | np.ndarray | Sequence[float],
        *,
        deterministic: bool = True,
        state: Any | None = None,
        episode_start: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, Any]:
        if not deterministic:
            raise ValueError(
                "ImitationRecoveryPolicy is deterministic; stochastic prediction "
                "is not available."
            )
        action = self(observation).detach().cpu().numpy()
        if state is None and episode_start is None:
            return action
        return action, state

    def act(
        self,
        observation: Tensor | np.ndarray | Sequence[float],
        *,
        deterministic: bool = True,
    ) -> np.ndarray:
        result = self.predict(observation, deterministic=deterministic)
        return np.asarray(result[0] if isinstance(result, tuple) else result)

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_type": self.policy_type,
            "model_config": self.model_config,
            "model_state_dict": self.state_dict(),
            "provenance": copy.deepcopy(self.provenance),
        }

    def save(self, path: str | Path) -> None:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(self.checkpoint_payload(), temporary)
        temporary.replace(target)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "ImitationRecoveryPolicy":
        resolved_device: str | torch.device = device
        if str(device).lower() == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = _torch_load(path, map_location=resolved_device)
        if not isinstance(checkpoint, Mapping):
            raise TypeError("Imitation recovery checkpoint must be a mapping.")
        if checkpoint.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported imitation recovery schema "
                f"{checkpoint.get('schema_version')!r}."
            )
        if checkpoint.get("policy_type") != POLICY_TYPE:
            raise ValueError(
                f"Unsupported imitation recovery policy "
                f"{checkpoint.get('policy_type')!r}."
            )
        config = checkpoint.get("model_config")
        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(config, Mapping) or not isinstance(state_dict, Mapping):
            raise ValueError("Checkpoint is missing model config or state dict.")
        model = cls(
            state_dim=int(config["state_dim"]),
            action_dim=int(config["action_dim"]),
            hidden_dims=tuple(int(value) for value in config["hidden_dims"]),
            activation=str(config["activation"]),
            normalization_epsilon=float(config.get("normalization_epsilon", 1e-8)),
            # Exact arrays are restored from the state dict.
            provenance=(
                checkpoint.get("provenance")
                if isinstance(checkpoint.get("provenance"), Mapping)
                else None
            ),
        )
        model.load_state_dict(dict(state_dict), strict=True)
        model.to(resolved_device)
        model.eval()
        return model
