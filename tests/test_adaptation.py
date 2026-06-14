"""Euclidean Alignment domain adaptation (runnable: numpy/scipy only)."""

import numpy as np
import pytest

from eegpipe.adaptation import EuclideanAlignment, mean_covariance


def test_euclidean_alignment_whitens_each_subject_to_identity():
    rng = np.random.default_rng(0)
    # subject 0 normal scale; subject 1 amplified 3x (a classic cross-site gain shift)
    X0 = rng.standard_normal((20, 4, 200))
    X1 = 3.0 * rng.standard_normal((20, 4, 200))
    X = np.concatenate([X0, X1]).astype("float32")
    groups = np.array([0] * 20 + [1] * 20)

    Xa = EuclideanAlignment().fit_transform(X, groups=groups)
    for g in (0, 1):
        cov = mean_covariance(Xa[groups == g])
        assert np.allclose(cov, np.eye(4), atol=1e-4)  # aligned mean cov == identity


def test_alignment_preserves_shape_and_new_subject_path():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((10, 3, 128)).astype("float32")
    # groups=None => align an unseen subject by its own statistics (unsupervised)
    Xa = EuclideanAlignment().fit(X, groups=np.zeros(10)).transform(X)
    assert Xa.shape == X.shape
    assert np.allclose(mean_covariance(Xa), np.eye(3), atol=1e-4)


def test_riemannian_recentering_optional():
    pytest.importorskip("pyriemann")
    from eegpipe.adaptation import RiemannianRecentering

    rng = np.random.default_rng(2)
    A = rng.standard_normal((15, 4, 4))
    C = np.einsum("nij,nkj->nik", A, A) + np.eye(4)  # SPD covariances
    out = RiemannianRecentering().fit_transform(C, groups=np.zeros(15))
    assert out.shape == C.shape
