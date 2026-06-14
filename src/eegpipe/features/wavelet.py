"""Wavelet features. DWT band energies capture non-stationary, transient EEG dynamics
that fixed-window Fourier features miss (e.g. ERP components, sleep spindles)."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class WaveletEnergyFeatures(BaseEstimator, TransformerMixin):
    """Per-channel discrete wavelet transform sub-band log-energy + entropy.

    Input ``X``: (n_trials, n_channels, n_times). Output: (n_trials, n_channels*(levels+1)*2).
    """

    def __init__(self, wavelet: str = "db4", level: int = 5):
        self.wavelet, self.level = wavelet, level

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        import pywt

        feats = []
        for trial in X:  # (n_channels, n_times)
            ch_feats = []
            for ch in trial:
                coeffs = pywt.wavedec(ch, self.wavelet, level=self.level)
                for c in coeffs:
                    energy = np.sum(c**2)
                    p = c**2 / (energy + 1e-12)
                    entropy = -np.sum(p * np.log(p + 1e-12))
                    ch_feats += [np.log(energy + 1e-12), entropy]
            feats.append(ch_feats)
        return np.asarray(feats, dtype="float32")
