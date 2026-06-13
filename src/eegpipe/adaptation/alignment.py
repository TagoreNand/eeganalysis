"""Cross-site / cross-subject domain adaptation by covariance alignment.

EEG distributions shift across subjects, sessions and headsets, so a model trained on one
site degrades on another. These transforms *re-centre* each domain so the model sees a
consistent input distribution. Both are **unsupervised** (use no labels), which is what makes
them safe and usable on a brand-new test subject: align that subject by its own statistics.

* :class:`EuclideanAlignment` (He & Wu, 2020) — whiten raw trials by the inverse square root
  of the domain's mean spatial covariance. Drop-in before any pipeline.
* :class:`RiemannianRecentering` (Zanini et al., 2018) — recentre covariance matrices to the
  identity on the SPD manifold. Pairs with the Riemannian tangent-space classifier.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def _sym_inv_sqrt(R: np.ndarray) -> np.ndarray:
    """Symmetric inverse square root of an SPD matrix via eigendecomposition."""
    vals, vecs = np.linalg.eigh(R)
    vals = np.clip(vals, 1e-12, None)
    return (vecs * vals**-0.5) @ vecs.T


def mean_covariance(X: np.ndarray) -> np.ndarray:
    """Arithmetic mean spatial covariance over trials. X: (n_trials, C, T) -> (C, C)."""
    covs = np.einsum("nct,nkt->nck", X, X) / X.shape[-1]
    return covs.mean(axis=0)


class EuclideanAlignment(BaseEstimator, TransformerMixin):
    """Whiten trials by the domain reference R^{-1/2}; after alignment each domain's mean
    covariance is the identity, so subjects/sites become comparable.

    Unsupervised: ``transform`` with ``groups=None`` aligns the passed trials by *their own*
    reference — exactly the recipe for a new, unlabelled test subject.
    """

    def __init__(self):
        self.refs_: dict = {}
        self.global_ref_: np.ndarray | None = None

    def fit(self, X, y=None, groups=None):
        groups = np.zeros(len(X)) if groups is None else np.asarray(groups)
        for g in np.unique(groups):
            self.refs_[g] = _sym_inv_sqrt(mean_covariance(X[groups == g]))
        self.global_ref_ = _sym_inv_sqrt(mean_covariance(X))
        return self

    def _apply(self, X, R):
        return np.einsum("ij,njt->nit", R, X)

    def transform(self, X, groups=None):
        X = np.asarray(X, dtype="float64")
        if groups is None:  # treat X as one new domain -> align by its own reference
            return self._apply(X, _sym_inv_sqrt(mean_covariance(X)))
        groups = np.asarray(groups)
        out = np.empty_like(X)
        for g in np.unique(groups):
            R = self.refs_.get(g, self.global_ref_)
            out[groups == g] = self._apply(X[groups == g], R)
        return out

    def fit_transform(self, X, y=None, groups=None):
        return self.fit(X, groups=groups).transform(X, groups=groups)


class RiemannianRecentering(BaseEstimator, TransformerMixin):
    """Recentre SPD covariance matrices to the identity per domain (Riemannian mean).

    Input/*output* are covariances (n_trials, C, C) — slot between ``pyriemann`` Covariances
    and TangentSpace. Requires ``pyriemann``.
    """

    def __init__(self):
        self.whiteners_: dict = {}
        self.global_whitener_: np.ndarray | None = None

    def _whitener(self, C):
        from pyriemann.utils.mean import mean_riemann

        return _sym_inv_sqrt(mean_riemann(C))

    def fit(self, C, y=None, groups=None):
        groups = np.zeros(len(C)) if groups is None else np.asarray(groups)
        for g in np.unique(groups):
            self.whiteners_[g] = self._whitener(C[groups == g])
        self.global_whitener_ = self._whitener(C)
        return self

    def transform(self, C, groups=None):
        C = np.asarray(C, dtype="float64")
        if groups is None:
            W = self._whitener(C)
            return W @ C @ W
        groups = np.asarray(groups)
        out = np.empty_like(C)
        for g in np.unique(groups):
            W = self.whiteners_.get(g, self.global_whitener_)
            out[groups == g] = W @ C[groups == g] @ W
        return out

    def fit_transform(self, C, y=None, groups=None):
        return self.fit(C, groups=groups).transform(C, groups=groups)
