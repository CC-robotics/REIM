"""State-based ACT policy with CVAE action chunking.

The primary imitation policy is ACT (Action Chunking with Transformers).
``BCPolicy`` remains an alias for compatibility with the original project
layout. ``MLPBCPolicy`` is retained only as an explicit ablation baseline.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn


def _torch_load(path: str | Path, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location=map_location)


def _extract_state_dict(checkpoint: Any) -> dict[str, Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("A policy checkpoint must be a state-dict or mapping.")
    for key in ("model_state_dict", "state_dict", "policy_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    if checkpoint and all(isinstance(value, Tensor) for value in checkpoint.values()):
        return checkpoint
    raise KeyError("Checkpoint does not contain a policy state-dict.")


def _clean_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    prefixes = ("module.", "policy.", "model.")
    cleaned: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def _sinusoidal_encoding(length: int, dimension: int) -> Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dimension, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / dimension)
    )
    encoding = torch.zeros(length, dimension, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * div_term)
    if dimension > 1:
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
    return encoding


class ACTPolicy(nn.Module):
    """CVAE Transformer that predicts a fixed-size future action chunk.

    At training time the latent encoder conditions on the demonstrated future
    actions. At inference time the posterior is replaced by its standard-normal
    prior mean (zero), making deployment deterministic. Calling :meth:`act`
    performs temporal ensembling over overlapping chunks.
    """

    policy_type = "ACT"

    def __init__(
        self,
        state_dim: int | None = None,
        action_dim: int | None = None,
        *,
        input_dim: int | None = None,
        output_dim: int | None = None,
        chunk_size: int = 20,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        nheads: int = 8,
        encoder_layers: int = 2,
        decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        temporal_ensemble: bool = True,
        ensemble_decay: float = 0.01,
        action_scale: float = 1.0,
    ) -> None:
        super().__init__()
        state_dim = state_dim if state_dim is not None else input_dim
        action_dim = action_dim if action_dim is not None else output_dim
        if state_dim is None or action_dim is None:
            raise ValueError("Both state_dim and action_dim are required.")
        integer_values = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "chunk_size": chunk_size,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "nheads": nheads,
            "encoder_layers": encoder_layers,
            "decoder_layers": decoder_layers,
            "dim_feedforward": dim_feedforward,
        }
        if any(int(value) <= 0 for value in integer_values.values()):
            raise ValueError(f"All ACT dimensions must be positive: {integer_values}")
        if hidden_dim % nheads:
            raise ValueError("hidden_dim must be divisible by nheads.")

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.nheads = int(nheads)
        self.encoder_layers = int(encoder_layers)
        self.decoder_layers = int(decoder_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout_probability = float(dropout)
        self.temporal_ensemble = bool(temporal_ensemble)
        self.ensemble_decay = float(ensemble_decay)
        self.action_scale = float(action_scale)

        # CVAE posterior encoder: [CLS, current state, future actions].
        self.encoder_cls = nn.Parameter(torch.empty(1, 1, self.hidden_dim))
        self.encoder_state_projection = nn.Linear(self.state_dim, self.hidden_dim)
        self.encoder_action_projection = nn.Linear(self.action_dim, self.hidden_dim)
        self.register_buffer(
            "encoder_position",
            _sinusoidal_encoding(self.chunk_size + 2, self.hidden_dim).unsqueeze(0),
            persistent=True,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.nheads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_probability,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.latent_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.encoder_layers, enable_nested_tensor=False
        )
        self.latent_parameters = nn.Linear(self.hidden_dim, 2 * self.latent_dim)

        # Decoder memory contains current state and the sampled latent style.
        self.decoder_state_projection = nn.Linear(self.state_dim, self.hidden_dim)
        self.decoder_latent_projection = nn.Linear(self.latent_dim, self.hidden_dim)
        self.memory_type_embedding = nn.Parameter(torch.empty(1, 2, self.hidden_dim))
        self.action_queries = nn.Parameter(
            torch.empty(1, self.chunk_size, self.hidden_dim)
        )
        self.register_buffer(
            "query_position",
            _sinusoidal_encoding(self.chunk_size, self.hidden_dim).unsqueeze(0),
            persistent=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.nheads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_probability,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=self.decoder_layers
        )
        self.decoder_norm = nn.LayerNorm(self.hidden_dim)
        self.action_head = nn.Linear(self.hidden_dim, self.action_dim)

        self.register_buffer("state_mean", torch.zeros(self.state_dim))
        self.register_buffer("state_std", torch.ones(self.state_dim))
        self.reset_parameters()
        self.reset()

    @property
    def model_config(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "nheads": self.nheads,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout_probability,
            "temporal_ensemble": self.temporal_ensemble,
            "ensemble_decay": self.ensemble_decay,
            "action_scale": self.action_scale,
        }

    def reset_parameters(self) -> None:
        nn.init.normal_(self.encoder_cls, std=0.02)
        nn.init.normal_(self.memory_type_embedding, std=0.02)
        nn.init.normal_(self.action_queries, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.action_head.weight, gain=0.01)

    @torch.no_grad()
    def set_normalization(
        self,
        mean: Tensor | np.ndarray,
        std: Tensor | np.ndarray,
        epsilon: float = 1e-6,
    ) -> None:
        mean_tensor = torch.as_tensor(mean, dtype=self.state_mean.dtype)
        std_tensor = torch.as_tensor(std, dtype=self.state_std.dtype)
        if mean_tensor.numel() != self.state_dim or std_tensor.numel() != self.state_dim:
            raise ValueError(f"Expected {self.state_dim} state normalization values.")
        self.state_mean.copy_(mean_tensor.reshape_as(self.state_mean))
        self.state_std.copy_(
            std_tensor.reshape_as(self.state_std).clamp_min(float(epsilon))
        )

    def _posterior(
        self,
        normalized_states: Tensor,
        action_chunks: Tensor,
        padding_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = normalized_states.shape[0]
        if action_chunks.shape != (batch_size, self.chunk_size, self.action_dim):
            raise ValueError(
                "Expected action chunks with shape "
                f"{(batch_size, self.chunk_size, self.action_dim)}, "
                f"got {tuple(action_chunks.shape)}."
            )
        cls_token = self.encoder_cls.expand(batch_size, -1, -1)
        state_token = self.encoder_state_projection(normalized_states).unsqueeze(1)
        action_tokens = self.encoder_action_projection(action_chunks)
        tokens = torch.cat((cls_token, state_token, action_tokens), dim=1)
        tokens = tokens + self.encoder_position.to(tokens.dtype)
        source_padding_mask = None
        if padding_mask is not None:
            padding_mask = padding_mask.to(dtype=torch.bool)
            if padding_mask.shape != (batch_size, self.chunk_size):
                raise ValueError(
                    f"Expected padding_mask {(batch_size, self.chunk_size)}, "
                    f"got {tuple(padding_mask.shape)}."
                )
            prefix_mask = torch.zeros(
                (batch_size, 2), dtype=torch.bool, device=padding_mask.device
            )
            source_padding_mask = torch.cat((prefix_mask, padding_mask), dim=1)
        encoded = self.latent_encoder(
            tokens, src_key_padding_mask=source_padding_mask
        )
        mean, log_variance = self.latent_parameters(encoded[:, 0]).chunk(2, dim=-1)
        log_variance = log_variance.clamp(-10.0, 10.0)
        if self.training:
            latent = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
        else:
            latent = mean
        return latent, mean, log_variance

    def _decode(self, normalized_states: Tensor, latent: Tensor) -> Tensor:
        batch_size = normalized_states.shape[0]
        state_memory = self.decoder_state_projection(normalized_states).unsqueeze(1)
        latent_memory = self.decoder_latent_projection(latent).unsqueeze(1)
        memory = torch.cat((state_memory, latent_memory), dim=1)
        memory = memory + self.memory_type_embedding
        queries = self.action_queries.expand(batch_size, -1, -1)
        queries = queries + self.query_position.to(queries.dtype)
        decoded = self.action_decoder(tgt=queries, memory=memory)
        decoded = self.decoder_norm(decoded)
        return torch.tanh(self.action_head(decoded)) * self.action_scale

    def forward(
        self,
        states: Tensor,
        action_chunks: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor | dict[str, Tensor]:
        was_unbatched = states.ndim == 1
        if was_unbatched:
            states = states.unsqueeze(0)
            if action_chunks is not None and action_chunks.ndim == 2:
                action_chunks = action_chunks.unsqueeze(0)
            if padding_mask is not None and padding_mask.ndim == 1:
                padding_mask = padding_mask.unsqueeze(0)
        if states.ndim != 2 or states.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected states [batch,{self.state_dim}], got {tuple(states.shape)}."
            )
        normalized_states = (states - self.state_mean) / self.state_std
        if action_chunks is None:
            latent = torch.zeros(
                (len(states), self.latent_dim),
                dtype=states.dtype,
                device=states.device,
            )
            predicted = self._decode(normalized_states, latent)
            return predicted.squeeze(0) if was_unbatched else predicted

        latent, mean, log_variance = self._posterior(
            normalized_states, action_chunks, padding_mask
        )
        predicted = self._decode(normalized_states, latent)
        return {
            "actions": predicted,
            "latent_mean": mean,
            "latent_log_variance": log_variance,
        }

    @staticmethod
    def loss(
        output: dict[str, Tensor],
        target_actions: Tensor,
        padding_mask: Tensor,
        *,
        kl_weight: float = 10.0,
    ) -> dict[str, Tensor]:
        valid = (~padding_mask.to(torch.bool)).unsqueeze(-1)
        absolute_error = torch.abs(output["actions"] - target_actions)
        l1 = (absolute_error * valid).sum() / (
            valid.sum().clamp_min(1) * target_actions.shape[-1]
        )
        mean = output["latent_mean"]
        log_variance = output["latent_log_variance"]
        kl = -0.5 * (1.0 + log_variance - mean.square() - log_variance.exp())
        kl = kl.sum(dim=-1).mean()
        total = l1 + float(kl_weight) * kl
        return {"loss": total, "l1": l1, "kl": kl}

    def reset(self) -> None:
        """Clear chunk history at the beginning of every environment episode."""
        self._inference_step = 0
        self._chunk_history: list[tuple[int, np.ndarray]] = []
        self._open_loop_chunk: np.ndarray | None = None
        self._open_loop_index = 0

    @torch.inference_mode()
    def predict_chunk(self, state: Tensor | np.ndarray) -> np.ndarray:
        device = next(self.parameters()).device
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        predicted = self(state_tensor)
        if not isinstance(predicted, Tensor):
            raise RuntimeError("ACT inference unexpectedly returned posterior outputs.")
        return predicted.detach().cpu().numpy()

    @torch.inference_mode()
    def act(
        self,
        state: Tensor | np.ndarray,
        *,
        deterministic: bool = True,
    ) -> np.ndarray:
        del deterministic  # ACT uses the prior mean for deterministic deployment.
        array = np.asarray(state)
        if array.ndim > 1:
            chunks = self.predict_chunk(array)
            return np.asarray(chunks[:, 0])

        if not self.temporal_ensemble:
            if (
                self._open_loop_chunk is None
                or self._open_loop_index >= len(self._open_loop_chunk)
            ):
                self._open_loop_chunk = np.asarray(self.predict_chunk(array))
                self._open_loop_index = 0
            action = self._open_loop_chunk[self._open_loop_index]
            self._open_loop_index += 1
            self._inference_step += 1
            return np.asarray(action)

        chunk = np.asarray(self.predict_chunk(array))
        self._chunk_history.append((self._inference_step, chunk))
        proposals: list[np.ndarray] = []
        weights: list[float] = []
        retained: list[tuple[int, np.ndarray]] = []
        for start_step, previous_chunk in self._chunk_history:
            age = self._inference_step - start_step
            if 0 <= age < len(previous_chunk):
                retained.append((start_step, previous_chunk))
                proposals.append(previous_chunk[age])
                # ACT Algorithm 2 orders overlapping predictions oldest→newest
                # and assigns w_i = exp(-m i), hence the oldest valid chunk is
                # deliberately weighted most strongly.
                prediction_index = len(proposals) - 1
                weights.append(
                    math.exp(-self.ensemble_decay * prediction_index)
                )
        self._chunk_history = retained
        normalized_weights = np.asarray(weights, dtype=np.float64)
        normalized_weights /= normalized_weights.sum()
        action = np.sum(
            np.stack(proposals) * normalized_weights[:, None], axis=0
        ).astype(np.float32)
        self._inference_step += 1
        return action

    predict = act

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        state_dim: int | None = None,
        action_dim: int | None = None,
    ) -> "ACTPolicy":
        checkpoint = _torch_load(path, map_location)
        metadata = checkpoint if isinstance(checkpoint, dict) else {}
        policy_type = str(metadata.get("policy_type", "ACT")).upper()
        if policy_type not in {"ACT", "ACTPOLICY"}:
            raise ValueError(
                f"Checkpoint policy_type={policy_type!r} is not an ACT checkpoint."
            )
        state_dict = _clean_state_dict(_extract_state_dict(checkpoint))
        state_projection = state_dict.get("encoder_state_projection.weight")
        action_projection = state_dict.get("encoder_action_projection.weight")
        state_dim = int(
            metadata.get(
                "state_dim",
                state_dim
                or (state_projection.shape[1] if state_projection is not None else 0),
            )
        )
        action_dim = int(
            metadata.get(
                "action_dim",
                action_dim
                or (action_projection.shape[1] if action_projection is not None else 0),
            )
        )
        model_config = dict(metadata.get("model_config", {}))
        for key in (
            "chunk_size",
            "hidden_dim",
            "latent_dim",
            "nheads",
            "encoder_layers",
            "decoder_layers",
            "dim_feedforward",
            "dropout",
            "temporal_ensemble",
            "ensemble_decay",
            "action_scale",
        ):
            if key in metadata and key not in model_config:
                model_config[key] = metadata[key]
        model = cls(state_dim=state_dim, action_dim=action_dim, **model_config)
        model.provenance = {
            key: metadata[key]
            for key in (
                "benchmark",
                "task_vocabulary",
                "task_vocabulary_sha256",
                "data_manifest_sha256",
                "seed",
            )
            if key in metadata
        }
        incompatible = model.load_state_dict(state_dict, strict=False)
        allowed_missing = {"state_mean", "state_std"}
        missing = set(incompatible.missing_keys) - allowed_missing
        unexpected = set(incompatible.unexpected_keys)
        if missing or unexpected:
            raise RuntimeError(
                f"Incompatible ACT checkpoint. Missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        model.to(map_location)
        model.eval()
        model.reset()
        return model


class MLPBCPolicy(nn.Module):
    """Legacy two-layer behavior-cloning baseline, for ablations only."""

    policy_type = "MLP_BC"

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Iterable[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = tuple(int(width) for width in hidden_dims)
        layers: list[nn.Module] = []
        previous = self.state_dim
        for width in self.hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        layers.extend((nn.Linear(previous, self.action_dim), nn.Tanh()))
        self.network = nn.Sequential(*layers)
        self.register_buffer("state_mean", torch.zeros(self.state_dim))
        self.register_buffer("state_std", torch.ones(self.state_dim))

    def forward(self, states: Tensor) -> Tensor:
        return self.network((states - self.state_mean) / self.state_std)

    @torch.no_grad()
    def set_normalization(self, mean: Tensor | np.ndarray, std: Tensor | np.ndarray) -> None:
        self.state_mean.copy_(torch.as_tensor(mean).reshape_as(self.state_mean))
        self.state_std.copy_(
            torch.as_tensor(std).reshape_as(self.state_std).clamp_min(1e-6)
        )

    def reset(self) -> None:
        return None

    @torch.inference_mode()
    def act(self, state: Tensor | np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        del deterministic
        device = next(self.parameters()).device
        tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        return self(tensor).detach().cpu().numpy()

    predict = act


# Backward-compatible import name; ACT is intentionally the default.
BCPolicy = ACTPolicy
BehaviorCloningPolicy = ACTPolicy


def load_bc_policy(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> ACTPolicy:
    """Load the primary imitation policy from its metadata-rich checkpoint."""
    return ACTPolicy.from_checkpoint(path, map_location=map_location)
