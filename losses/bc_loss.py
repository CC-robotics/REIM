"""Behavior-cloning and failure-aware policy objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .smooth_loss import acceleration_consistency_loss, velocity_consistency_loss


def behavior_cloning_loss(predicted: Tensor, expert: Tensor) -> Tensor:
    """Mean-squared expert-action regression loss."""

    if predicted.shape != expert.shape:
        raise ValueError(
            f"predicted/expert action shapes differ: {predicted.shape} vs {expert.shape}"
        )
    return torch.mean((predicted - expert).square())


def failure_risk_loss(
    predicted: Tensor,
    safe_actions: Tensor,
    failure_risk: Tensor,
) -> Tensor:
    """Risk-weighted distance to a safe corrective action.

    ``failure_risk`` can contain binary labels or calibrated probabilities. It
    is detached intentionally: policy optimization must not alter the detector.
    """

    if predicted.shape != safe_actions.shape:
        raise ValueError(
            "predicted/safe action shapes differ: "
            f"{predicted.shape} vs {safe_actions.shape}"
        )
    risk = failure_risk.reshape(-1).to(predicted.device, predicted.dtype).detach()
    if predicted.ndim != 2 or risk.shape[0] != predicted.shape[0]:
        raise ValueError("failure risk must provide one value per action")
    if torch.any((risk < 0.0) | (risk > 1.0)):
        raise ValueError("failure risk values must lie in [0, 1]")
    per_sample = (predicted - safe_actions).square().mean(dim=-1)
    denominator = risk.sum().clamp_min(1.0)
    return torch.sum(risk * per_sample) / denominator


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    total: Tensor
    bc: Tensor
    velocity: Tensor
    acceleration: Tensor
    failure: Tensor


class FailureAwareBCLoss(nn.Module):
    """Composite REIM policy loss with explicit, auditable coefficients."""

    def __init__(
        self,
        *,
        velocity_weight: float = 0.05,
        acceleration_weight: float = 0.02,
        failure_weight: float = 0.10,
    ) -> None:
        super().__init__()
        weights = (velocity_weight, acceleration_weight, failure_weight)
        if any(weight < 0.0 for weight in weights):
            raise ValueError("loss coefficients must be non-negative")
        self.velocity_weight = float(velocity_weight)
        self.acceleration_weight = float(acceleration_weight)
        self.failure_weight = float(failure_weight)

    def forward(
        self,
        predicted_sequence: Tensor,
        expert_sequence: Tensor,
        *,
        valid_mask: Tensor | None = None,
        risky_predicted: Tensor | None = None,
        safe_actions: Tensor | None = None,
        failure_risk: Tensor | None = None,
    ) -> LossBreakdown:
        if predicted_sequence.ndim != 3:
            raise ValueError("policy sequences must have shape [batch, time, action]")
        bc = behavior_cloning_loss(predicted_sequence, expert_sequence)
        velocity = velocity_consistency_loss(predicted_sequence, valid_mask)
        acceleration = acceleration_consistency_loss(predicted_sequence, valid_mask)
        supplied = (risky_predicted, safe_actions, failure_risk)
        if all(value is None for value in supplied):
            failure = predicted_sequence.sum() * 0.0
        elif any(value is None for value in supplied):
            raise ValueError(
                "risky_predicted, safe_actions, and failure_risk are all required"
            )
        else:
            failure = failure_risk_loss(
                risky_predicted,
                safe_actions,
                failure_risk,
            )
        total = (
            bc
            + self.velocity_weight * velocity
            + self.acceleration_weight * acceleration
            + self.failure_weight * failure
        )
        return LossBreakdown(total, bc, velocity, acceleration, failure)
