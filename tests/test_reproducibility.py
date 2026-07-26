"""Tests for resumable-training reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np
import torch

from utils.common import capture_rng_state, restore_rng_state, seed_everything


def test_rng_state_round_trip_restores_all_generators() -> None:
    seed_everything(123)
    snapshot = capture_rng_state()
    expected = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
    )

    seed_everything(999)
    restore_rng_state(snapshot)
    actual = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
    )
    assert actual == expected
