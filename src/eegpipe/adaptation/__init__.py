"""Domain adaptation by covariance alignment (unsupervised, leakage-free)."""

from eegpipe.adaptation.alignment import (
    EuclideanAlignment,
    RiemannianRecentering,
    mean_covariance,
)

__all__ = ["EuclideanAlignment", "RiemannianRecentering", "mean_covariance"]
