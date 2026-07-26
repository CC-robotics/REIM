"""Tests for ACT action chunking, CVAE loss, and deployment behavior."""

from __future__ import annotations

from types import MethodType

import numpy as np
import torch

from models.bc_policy import ACTPolicy, BCPolicy


def make_tiny_act(*, temporal_ensemble: bool = True) -> ACTPolicy:
    model = ACTPolicy(
        state_dim=6,
        action_dim=4,
        chunk_size=4,
        hidden_dim=16,
        latent_dim=4,
        nheads=4,
        encoder_layers=1,
        decoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        temporal_ensemble=temporal_ensemble,
        ensemble_decay=0.5,
    )
    model.eval()
    return model


def test_bc_policy_alias_points_to_act() -> None:
    assert BCPolicy is ACTPolicy


def test_act_forward_shapes_bounds_and_deterministic_prior() -> None:
    model = make_tiny_act()
    states = torch.randn(3, 6)
    with torch.inference_mode():
        first = model(states)
        second = model(states)
    assert first.shape == (3, 4, 4)
    assert torch.equal(first, second)
    assert torch.max(torch.abs(first)).item() <= 1.0


def test_act_training_loss_ignores_padded_action_targets() -> None:
    model = make_tiny_act()
    model.eval()  # deterministic posterior mean makes the comparison exact
    states = torch.randn(2, 6)
    targets = torch.randn(2, 4, 4)
    padding = torch.tensor(
        [[False, False, True, True], [False, False, False, True]]
    )
    changed = targets.clone()
    changed[padding] = 10_000.0

    output = model(states, targets, padding)
    changed_output = model(states, changed, padding)
    # The posterior encoder also masks padded actions, so both predictions and
    # the masked reconstruction objective must be unchanged.
    assert isinstance(output, dict)
    assert isinstance(changed_output, dict)
    original_loss = model.loss(output, targets, padding, kl_weight=0.1)
    changed_loss = model.loss(changed_output, changed, padding, kl_weight=0.1)
    for key in ("loss", "l1", "kl"):
        torch.testing.assert_close(original_loss[key], changed_loss[key])
        assert torch.isfinite(original_loss[key])


def test_temporal_ensemble_uses_all_valid_overlapping_chunks() -> None:
    model = make_tiny_act(temporal_ensemble=True)
    chunks = [
        np.asarray([[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0]], dtype=np.float32),
        np.asarray([[10, 0, 0, 0], [20, 0, 0, 0], [30, 0, 0, 0], [40, 0, 0, 0]], dtype=np.float32),
    ]

    def fake_predict_chunk(_self: ACTPolicy, _state: np.ndarray) -> np.ndarray:
        return chunks.pop(0)

    model.predict_chunk = MethodType(fake_predict_chunk, model)
    state = np.zeros(6, dtype=np.float32)
    np.testing.assert_allclose(model.act(state), [1, 0, 0, 0])
    second = model.act(state)
    expected = (2.0 + np.exp(-0.5) * 10.0) / (1.0 + np.exp(-0.5))
    np.testing.assert_allclose(second, [expected, 0, 0, 0], atol=1e-6)


def test_act_checkpoint_round_trip_preserves_predictions(tmp_path) -> None:
    model = make_tiny_act()
    model.set_normalization(np.arange(6, dtype=np.float32), np.full(6, 2.0))
    state = np.linspace(-1.0, 1.0, 6, dtype=np.float32)
    expected = model.predict_chunk(state)
    checkpoint = tmp_path / "tiny_act.pt"
    torch.save(
        {
            "policy_type": "ACT",
            "state_dim": model.state_dim,
            "action_dim": model.action_dim,
            "model_config": model.model_config,
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )

    restored = ACTPolicy.from_checkpoint(checkpoint)
    np.testing.assert_allclose(restored.predict_chunk(state), expected, atol=1e-7)
    np.testing.assert_array_equal(
        restored.state_mean.cpu().numpy(), model.state_mean.cpu().numpy()
    )

