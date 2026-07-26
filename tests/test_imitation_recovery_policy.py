"""Tests for the standalone trigger-aligned imitation recovery actor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from models.imitation_recovery_policy import ImitationRecoveryPolicy


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_imitation_recovery_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(7)
    policy = ImitationRecoveryPolicy(
        state_dim=5,
        action_dim=2,
        hidden_dims=(8, 6),
        activation="tanh",
        observation_mean=np.arange(5, dtype=np.float32),
        observation_std=np.full(5, 2.0, dtype=np.float32),
        action_low=[-0.4, -0.2],
        action_high=[0.3, 0.5],
        provenance={"source_training": {"num_timesteps": 0}},
    )
    policy.eval()
    states = np.linspace(-2.0, 3.0, 20, dtype=np.float32).reshape(4, 5)
    expected_mean = policy.predict_mean(states)
    expected_action = policy.predict(states)

    checkpoint = tmp_path / "imitation_recovery.pt"
    policy.save(checkpoint)
    restored = ImitationRecoveryPolicy.load(checkpoint)

    np.testing.assert_array_equal(restored.predict_mean(states), expected_mean)
    np.testing.assert_array_equal(restored.predict(states), expected_action)
    assert restored.num_timesteps == 0
    assert restored.provenance == policy.provenance
    assert not any(
        token in key
        for key in restored.state_dict()
        for token in ("value_net", "critic", "log_std", "optimizer")
    )


def test_imitation_recovery_is_deterministic_and_clips_actions() -> None:
    policy = ImitationRecoveryPolicy(
        state_dim=3,
        action_dim=2,
        hidden_dims=(4,),
        action_low=[-0.1, -0.2],
        action_high=[0.1, 0.2],
    )
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        policy.action_net.bias.copy_(torch.tensor([4.0, -5.0]))
    state = np.zeros(3, dtype=np.float32)

    np.testing.assert_array_equal(
        policy.predict_mean(state), np.asarray([4.0, -5.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        policy.act(state), np.asarray([0.1, -0.2], dtype=np.float32)
    )
    with pytest.raises(ValueError, match="deterministic"):
        policy.predict(state, deterministic=False)


def test_repository_imitation_artifact_provenance_and_sb3_equivalence() -> None:
    from stable_baselines3 import PPO

    artifact_path = ROOT / "checkpoints/imitation_recovery.pt"
    audit_path = ROOT / "checkpoints/imitation_recovery.audit.json"
    source_path = ROOT / "checkpoints/recovery_ablation_warmstart_only.zip"
    assert artifact_path.is_file()
    assert audit_path.is_file()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["audit_result"] == "pass"
    assert audit["artifact"]["sha256"] == _sha256(artifact_path)
    assert audit["source"]["checkpoint_sha256"] == _sha256(source_path)
    assert audit["source"]["num_timesteps"] == 0
    assert audit["source"]["n_updates"] == 0
    assert audit["source"]["optimizer_state_entries"] == 0
    assert audit["equivalence"]["total_states"] == 60_598
    assert audit["equivalence"]["maximum_abs_error"] == 0.0

    standalone = ImitationRecoveryPolicy.load(artifact_path)
    source = PPO.load(source_path, device="cpu")
    assert source.num_timesteps == 0
    assert getattr(source, "_n_updates", -1) == 0
    assert len(source.policy.optimizer.state) == 0
    assert standalone.num_timesteps == 0
    assert sum(parameter.numel() for parameter in standalone.parameters()) == 72_452

    state_keys = set(standalone.state_dict())
    assert state_keys == {
        "observation_mean",
        "observation_std",
        "action_low",
        "action_high",
        "policy_net.0.weight",
        "policy_net.0.bias",
        "policy_net.2.weight",
        "policy_net.2.bias",
        "action_net.weight",
        "action_net.bias",
    }

    with np.load(
        ROOT / "datasets/recovery_starts/train.npz", allow_pickle=False
    ) as archive:
        train = np.asarray(archive["demo_states"], dtype=np.float32)
    with np.load(
        ROOT / "datasets/recovery_starts/validation.npz", allow_pickle=False
    ) as archive:
        validation = np.asarray(archive["demo_states"], dtype=np.float32)
    indices_train = np.linspace(0, len(train) - 1, 257, dtype=np.int64)
    indices_validation = np.linspace(
        0, len(validation) - 1, 257, dtype=np.int64
    )
    states = np.concatenate(
        (train[indices_train], validation[indices_validation]), axis=0
    )

    with torch.inference_mode():
        tensor = torch.as_tensor(states)
        source_mean = (
            source.policy.get_distribution(tensor)
            .distribution.mean.detach()
            .cpu()
            .numpy()
        )
    source_action, _ = source.predict(states, deterministic=True)
    np.testing.assert_array_equal(standalone.predict_mean(states), source_mean)
    np.testing.assert_array_equal(standalone.predict(states), source_action)
