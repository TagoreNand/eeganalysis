"""Riemannian-geometry features — the strongest *classical* baseline for EEG/BCI.

Trial covariance matrices are Symmetric Positive Definite (SPD) and live on a Riemannian
manifold, not a Euclidean space. Projecting them to the tangent space at the geometric
mean linearises them so an ordinary linear classifier becomes extremely effective and is
near state-of-the-art on small motor-imagery datasets — often beating deep nets when
trials are scarce. Backed by ``pyriemann``.
"""
from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin


class RiemannianTangentSpace(BaseEstimator, TransformerMixin):
    """Covariances -> tangent-space projection. Input ``X``: (n_trials, n_channels, n_times)."""

    def __init__(self, estimator: str = "oas", metric: str = "riemann"):
        self.estimator = estimator  # 'oas'/'lwf' shrinkage => robust covs for short trials
        self.metric = metric
        self._cov = None
        self._ts = None

    def fit(self, X, y=None):
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace

        self._cov = Covariances(estimator=self.estimator)
        self._ts = TangentSpace(metric=self.metric)
        C = self._cov.fit_transform(X)
        self._ts.fit(C, y)
        return self

    def transform(self, X):
        C = self._cov.transform(X)
        return self._ts.transform(C)


def build_riemann_classifier(C: float = 1.0):
    """A complete, leakage-safe sklearn pipeline: covs -> tangent space -> logistic reg.

    Drop-in replacement for the raw ``reshape -> StandardScaler -> LogisticRegression``
    baseline in the original 'temporal analysis' notebook, but typically far stronger.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    return make_pipeline(
        RiemannianTangentSpace(estimator="oas"),
        LogisticRegression(C=C, max_iter=1000),
    )
