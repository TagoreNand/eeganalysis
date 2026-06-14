"""Shared fixtures. Synthetic data keeps unit tests fast and dependency-light."""

import numpy as np
import pytest


@pytest.fixture
def synthetic_eeg():
    """(X, y, groups): 60 trials, 8 channels, 256 samples, 4 subjects, 2 classes."""
    rng = np.random.default_rng(0)
    n, c, t = 60, 8, 256
    X = rng.standard_normal((n, c, t)).astype("float32")
    y = rng.integers(0, 2, n)
    groups = np.repeat(np.arange(4), n // 4)  # 4 subjects, 15 trials each
    return X, y, groups
