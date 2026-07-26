"""Mask-aware temporal action-consistency losses.

The losses operate on ``[batch, time, action]`` tensors. A Boolean validity
mask prevents padded or cross-trajectory transitions from contributing.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _validate(actions: Tensor, valid_mask: Tensor | None) -> Tensor:
    if actions.ndim != 3:
        raise ValueError(
            "actions must have shape [batch, time, action], "
            f"received {tuple(actions.shape)}"
        )
    if valid_mask is None:
        return torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device)
    if valid_mask.shape != actions.shape[:2]:
        raise ValueError(
            f"valid_mask must have shape {tuple(actions.shape[:2])}, "
            f"received {tuple(valid_mask.shape)}"
        )
    return valid_mask.to(device=actions.device, dtype=torch.bool)


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    if values.shape[:2] != mask.shape:
        raise ValueError("values and temporal mask are misaligned")
    expanded = mask.unsqueeze(-1).expand_as(values)
    count = expanded.sum()
    if int(count) == 0:
        return values.sum() * 0.0
    return values.masked_select(expanded).mean()


def velocity_consistency_loss(
    actions: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Return mean ``L1`` first-order action variation."""

    valid = _validate(actions, valid_mask)
    if actions.shape[1] < 2:
        return actions.sum() * 0.0
    differences = (actions[:, 1:] - actions[:, :-1]).abs()
    pair_mask = valid[:, 1:] & valid[:, :-1]
    return _masked_mean(differences, pair_mask)


def acceleration_consistency_loss(
    actions: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Return mean squared second-order action variation."""

    valid = _validate(actions, valid_mask)
    if actions.shape[1] < 3:
        return actions.sum() * 0.0
    acceleration = actions[:, 2:] - 2.0 * actions[:, 1:-1] + actions[:, :-2]
    triple_mask = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
    return _masked_mean(acceleration.square(), triple_mask)


def temporal_consistency_loss(
    actions: Tensor,
    valid_mask: Tensor | None = None,
    *,
    velocity_weight: float = 1.0,
    acceleration_weight: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return weighted temporal loss and its unweighted components."""

    if velocity_weight < 0.0 or acceleration_weight < 0.0:
        raise ValueError("temporal loss weights must be non-negative")
    velocity = velocity_consistency_loss(actions, valid_mask)
    acceleration = acceleration_consistency_loss(actions, valid_mask)
    total = velocity_weight * velocity + acceleration_weight * acceleration
    return total, velocity, acceleration
