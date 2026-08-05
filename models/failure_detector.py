"""Causal LSTM failure predictor for REIM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence


def _torch_load(path: str | Path, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _extract_state_dict(checkpoint: Any) -> dict[str, Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("A detector checkpoint must be a checkpoint mapping.")
    for key in ("model_state_dict", "state_dict", "detector_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    if checkpoint and all(isinstance(value, Tensor) for value in checkpoint.values()):
        return checkpoint
    raise KeyError("Checkpoint does not contain a detector state-dict.")


def _clean_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    cleaned: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "detector.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


class FailureDetector(nn.Module):
    """Predict whether an execution is entering a failure state.

    ``forward`` returns logits for numerically stable BCE training.
    ``predict_proba`` applies the sigmoid needed at deployment.
    """

    def __init__(
        self,
        state_dim: int | None = None,
        hidden_dim: int = 128,
        num_layers: int = 1,
        mlp_hidden: int = 64,
        dropout: float = 0.1,
        sequence_length: int = 10,
        *,
        input_dim: int | None = None,
        hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        state_dim = state_dim if state_dim is not None else input_dim
        hidden_dim = int(hidden_size if hidden_size is not None else hidden_dim)
        if state_dim is None or int(state_dim) <= 0:
            raise ValueError("state_dim must be a positive integer.")
        if hidden_dim <= 0 or num_layers <= 0 or mlp_hidden <= 0:
            raise ValueError("LSTM and MLP dimensions must be positive.")

        self.state_dim = int(state_dim)
        self.hidden_dim = hidden_dim
        self.hidden_size = hidden_dim
        self.num_layers = int(num_layers)
        self.mlp_hidden = int(mlp_hidden)
        self.dropout_probability = float(dropout)
        self.sequence_length = int(sequence_length)

        recurrent_dropout = self.dropout_probability if self.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.state_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.mlp_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(self.mlp_hidden, 1),
        )
        self.register_buffer("state_mean", torch.zeros(self.state_dim))
        self.register_buffer("state_std", torch.ones(self.state_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, parameter in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(parameter)
            elif "weight_hh" in name:
                nn.init.orthogonal_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)
                # A positive forget-gate bias helps gradients in short histories.
                parameter.data[self.hidden_dim : 2 * self.hidden_dim].fill_(1.0)
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

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
            raise ValueError(f"Expected {self.state_dim} normalization values.")
        self.state_mean.copy_(mean_tensor.reshape_as(self.state_mean))
        self.state_std.copy_(
            std_tensor.reshape_as(self.state_std).clamp_min(float(epsilon))
        )

    def forward(
        self,
        state_sequences: Tensor,
        lengths: Tensor | None = None,
    ) -> Tensor:
        was_unbatched = state_sequences.ndim == 2
        if was_unbatched:
            state_sequences = state_sequences.unsqueeze(0)
            if lengths is not None and lengths.ndim == 0:
                lengths = lengths.unsqueeze(0)
        if state_sequences.ndim != 3:
            raise ValueError(
                "Expected state_sequences with shape [batch, time, state_dim] "
                f"or [time, state_dim], got {tuple(state_sequences.shape)}."
            )
        if state_sequences.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected state dimension {self.state_dim}, "
                f"got {state_sequences.shape[-1]}."
            )

        normalized = (state_sequences - self.state_mean) / self.state_std
        if lengths is None:
            _, (hidden, _) = self.lstm(normalized)
        else:
            lengths = lengths.to(dtype=torch.long).clamp(
                min=1, max=state_sequences.shape[1]
            )
            packed = pack_padded_sequence(
                normalized,
                lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (hidden, _) = self.lstm(packed)
        logits = self.classifier(hidden[-1]).squeeze(-1)
        return logits.squeeze(0) if was_unbatched else logits

    @torch.no_grad()
    def predict_proba(
        self,
        state_sequences: Tensor | np.ndarray,
        lengths: Tensor | np.ndarray | None = None,
    ) -> Tensor:
        device = next(self.parameters()).device
        sequences = torch.as_tensor(
            state_sequences, dtype=torch.float32, device=device
        )
        length_tensor = (
            None
            if lengths is None
            else torch.as_tensor(lengths, dtype=torch.long, device=device)
        )
        return torch.sigmoid(self(sequences, length_tensor))

    @torch.no_grad()
    def predict(
        self,
        state_sequences: Tensor | np.ndarray,
        lengths: Tensor | np.ndarray | None = None,
        *,
        threshold: float = 0.5,
    ) -> np.ndarray:
        probabilities = self.predict_proba(state_sequences, lengths)
        return (probabilities >= threshold).to(torch.uint8).cpu().numpy()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        state_dim: int | None = None,
    ) -> "FailureDetector":
        checkpoint = _torch_load(path, map_location)
        state_dict = _clean_state_dict(_extract_state_dict(checkpoint))
        metadata = checkpoint if isinstance(checkpoint, dict) else {}
        input_weight = state_dict.get("lstm.weight_ih_l0")
        if state_dim is None and input_weight is not None:
            state_dim = int(input_weight.shape[1])
        state_dim = int(metadata.get("state_dim", state_dim or 0))
        hidden_dim = int(
            metadata.get(
                "hidden_dim",
                metadata.get(
                    "hidden_size",
                    int(input_weight.shape[0] // 4) if input_weight is not None else 128,
                ),
            )
        )
        inferred_layers = max(
            [
                int(key.rsplit("_l", 1)[1].split(".", 1)[0]) + 1
                for key in state_dict
                if key.startswith("lstm.weight_ih_l")
            ]
            or [1]
        )
        classifier_weight = state_dict.get("classifier.0.weight")
        model = cls(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=int(metadata.get("num_layers", inferred_layers)),
            mlp_hidden=int(
                metadata.get(
                    "mlp_hidden",
                    classifier_weight.shape[0]
                    if classifier_weight is not None
                    else 64,
                )
            ),
            dropout=float(metadata.get("dropout", 0.1)),
            sequence_length=int(metadata.get("sequence_length", 10)),
        )
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
                f"Incompatible detector checkpoint. Missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        model.to(map_location)
        model.eval()
        return model
