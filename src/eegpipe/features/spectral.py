"""Spectral band-power features (delta/theta/alpha/beta/gamma) via Welch PSD."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


class BandPowerFeatures(BaseEstimator, TransformerMixin):
    """Relative band power per channel. Input ``X``: (n_trials, n_channels, n_times)."""

    def __init__(self, sfreq: float = 128.0, bands: dict | None = None, relative: bool = True):
        self.sfreq, self.relative = sfreq, relative
        self.bands = bands or BANDS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from scipy.integrate import trapezoid
        from scipy.signal import welch

        n_per_seg = min(256, X.shape[-1])
        freqs, psd = welch(X, fs=self.sfreq, nperseg=n_per_seg, axis=-1)
        total = trapezoid(psd, freqs, axis=-1) + 1e-12  # (n_trials, n_channels)
        out = []
        for lo, hi in self.bands.values():
            mask = (freqs >= lo) & (freqs < hi)
            bp = trapezoid(psd[..., mask], freqs[mask], axis=-1)
            out.append(bp / total if self.relative else bp)
        # stack -> (n_trials, n_channels, n_bands) -> flatten channels*bands
        feats = np.stack(out, axis=-1)
        return feats.reshape(feats.shape[0], -1).astype("float32")
