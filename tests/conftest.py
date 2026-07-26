"""Shared pytest configuration for deterministic, fast CPU tests."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch


# Small Transformer/LSTM tests are faster and more reproducible with one BLAS
# thread.  This affects only the pytest process, never the training scripts.
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def deterministic_test_seed() -> None:
    """Start every test from the same independent random state."""

    np.random.seed(7)
    torch.manual_seed(7)

