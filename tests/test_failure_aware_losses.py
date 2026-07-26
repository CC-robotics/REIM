from __future__ import annotations

import pytest
import torch

from losses.bc_loss import FailureAwareBCLoss, failure_risk_loss
from losses.smooth_loss import (
    acceleration_consistency_loss,
    velocity_consistency_loss,
)


def test_temporal_losses_are_zero_for_constant_actions() -> None:
    actions = torch.ones(2, 5, 4)
    assert velocity_consistency_loss(actions).item() == pytest.approx(0.0)
    assert acceleration_consistency_loss(actions).item() == pytest.approx(0.0)


def test_velocity_and_acceleration_match_known_sequence() -> None:
    actions = torch.tensor([[[0.0], [1.0], [3.0]]])
    assert velocity_consistency_loss(actions).item() == pytest.approx(1.5)
    assert acceleration_consistency_loss(actions).item() == pytest.approx(1.0)


def test_mask_excludes_padded_transitions() -> None:
    actions = torch.tensor([[[0.0], [1.0], [100.0], [200.0]]])
    mask = torch.tensor([[True, True, False, False]])
    assert velocity_consistency_loss(actions, mask).item() == pytest.approx(1.0)
    assert acceleration_consistency_loss(actions, mask).item() == pytest.approx(0.0)


def test_failure_loss_uses_only_positive_risk() -> None:
    predicted = torch.tensor([[2.0], [3.0]])
    safe = torch.zeros_like(predicted)
    risk = torch.tensor([1.0, 0.0])
    assert failure_risk_loss(predicted, safe, risk).item() == pytest.approx(4.0)


def test_composite_loss_backpropagates() -> None:
    predicted = torch.randn(3, 4, 2, requires_grad=True)
    expert = torch.zeros_like(predicted)
    risky = predicted[:, 0]
    loss = FailureAwareBCLoss(
        velocity_weight=0.1,
        acceleration_weight=0.1,
        failure_weight=0.2,
    )(
        predicted,
        expert,
        risky_predicted=risky,
        safe_actions=torch.zeros_like(risky),
        failure_risk=torch.ones(3),
    )
    loss.total.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
