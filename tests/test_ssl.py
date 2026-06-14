"""Relative-Positioning self-supervised dataset tests."""

import numpy as np
import pytest


def test_rp_pairs_stay_within_recording_and_are_binary():
    pytest.importorskip("torch")
    from eegpipe.models.ssl import RelativePositioningDataset

    X = np.random.randn(60, 2, 100).astype("float32")
    rec = np.array([0] * 30 + [1] * 30)
    ds = RelativePositioningDataset(X, rec, tau_pos=2, tau_neg=10, n_samples=300, seed=0)
    assert len(ds) > 0
    for a, o in zip(ds.a, ds.o, strict=False):
        assert rec[a] == rec[o]  # never cross a recording boundary
    labels = ds.y.numpy()
    assert set(np.unique(labels)).issubset({0, 1})
    assert 0.0 < labels.mean() < 1.0  # both positives and negatives present


def test_rp_item_shapes():
    pytest.importorskip("torch")
    from eegpipe.models.ssl import RelativePositioningDataset

    X = np.random.randn(50, 3, 128).astype("float32")
    ds = RelativePositioningDataset(X, np.zeros(50, dtype=int), n_samples=40)
    x1, x2, y = ds[0]
    assert tuple(x1.shape) == (3, 128) and tuple(x2.shape) == (3, 128)
    assert y.item() in (0, 1)
