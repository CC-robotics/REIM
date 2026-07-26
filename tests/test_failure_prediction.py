"""Causality and model-interface tests for failure prediction."""

from __future__ import annotations

import numpy as np
import torch

from data.failure_labels import (
    CausalFailureLabeler,
    build_causal_windows,
    build_prospective_failure_targets,
)
from models.failure_detector import FailureDetector


def test_causal_windows_never_include_future_states() -> None:
    states = np.arange(15, dtype=np.float32).reshape(5, 3)
    windows, lengths = build_causal_windows(states, sequence_length=3)
    assert windows.shape == (5, 3, 3)
    np.testing.assert_array_equal(lengths, [1, 2, 3, 3, 3])
    np.testing.assert_array_equal(windows[0, 0], states[0])
    np.testing.assert_array_equal(windows[0, 1:], 0.0)
    np.testing.assert_array_equal(windows[3], states[1:4])

    changed_future = states.copy()
    changed_future[4] = -999
    changed_windows, _ = build_causal_windows(changed_future, sequence_length=3)
    np.testing.assert_array_equal(changed_windows[:4], windows[:4])


def test_prospective_targets_use_inclusive_horizon_and_first_event_reason() -> None:
    events = np.asarray([0, 0, 0, 1, 0, 0, 1, 0], dtype=np.uint8)
    reasons = np.asarray(
        ["normal", "normal", "normal", "drop", "normal", "normal", "timeout", "normal"]
    )
    targets, offsets, target_reasons = build_prospective_failure_targets(
        events, reasons, horizon=2
    )
    np.testing.assert_array_equal(targets, [0, 1, 1, 1, 1, 1, 1, 0])
    np.testing.assert_array_equal(offsets, [-1, 2, 1, 0, 2, 1, 0, -1])
    assert target_reasons.tolist() == [
        "normal",
        "drop",
        "drop",
        "drop",
        "timeout",
        "timeout",
        "timeout",
        "normal",
    ]


def test_online_labeler_prioritizes_success_and_observed_faults() -> None:
    labeler = CausalFailureLabeler()
    base = {
        "object_position": np.asarray([0.0, 0.6, 0.025]),
        "goal_position": np.asarray([0.1, 0.75, 0.15]),
        "ee_position": np.asarray([0.0, 0.5, 0.2]),
        "distance_to_goal": 0.22,
        "hand_object_distance": 0.20,
        "gripper_state": -1.0,
    }
    labeler.reset(base)

    success_info = {**base, "success": True, "object_dropped": True}
    assert labeler.update(success_info, terminated=True, truncated=False) == (
        0,
        "normal",
    )
    drop_info = {**base, "success": False, "object_dropped": True}
    assert labeler.update(drop_info, terminated=False, truncated=False) == (
        1,
        "object_dropped",
    )


def test_detector_padding_is_ignored_when_lengths_are_supplied() -> None:
    detector = FailureDetector(
        state_dim=3,
        hidden_dim=8,
        num_layers=1,
        mlp_hidden=4,
        dropout=0.0,
        sequence_length=5,
    ).eval()
    valid_prefix = torch.randn(1, 2, 3)
    first = torch.cat([valid_prefix, torch.zeros(1, 3, 3)], dim=1)
    second = torch.cat([valid_prefix, torch.full((1, 3, 3), 999.0)], dim=1)
    lengths = torch.tensor([2])

    with torch.inference_mode():
        first_logit = detector(first, lengths)
        second_logit = detector(second, lengths)
    torch.testing.assert_close(first_logit, second_logit)


def test_detector_probability_prediction_and_checkpoint_round_trip(tmp_path) -> None:
    detector = FailureDetector(
        state_dim=3,
        hidden_dim=8,
        num_layers=1,
        mlp_hidden=4,
        dropout=0.0,
        sequence_length=5,
    ).eval()
    detector.set_normalization(np.ones(3), np.full(3, 2.0))
    sequences = np.zeros((2, 5, 3), dtype=np.float32)
    probabilities = detector.predict_proba(sequences, np.asarray([1, 5]))
    labels = detector.predict(sequences, np.asarray([1, 5]), threshold=0.5)
    assert probabilities.shape == (2,)
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert labels.shape == (2,)
    assert set(labels.tolist()) <= {0, 1}

    checkpoint = tmp_path / "detector.pt"
    torch.save(
        {
            "state_dim": 3,
            "hidden_dim": 8,
            "num_layers": 1,
            "mlp_hidden": 4,
            "dropout": 0.0,
            "sequence_length": 5,
            "model_state_dict": detector.state_dict(),
        },
        checkpoint,
    )
    restored = FailureDetector.from_checkpoint(checkpoint)
    restored_probabilities = restored.predict_proba(sequences, np.asarray([1, 5]))
    torch.testing.assert_close(restored_probabilities, probabilities)

