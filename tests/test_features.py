"""Feature extractors produce correct shapes / value ranges."""
import numpy as np
import pytest


def test_bandpower_shape_and_range(synthetic_eeg):
    from eegpipe.features.spectral import BandPowerFeatures

    X, _, _ = synthetic_eeg
    feats = BandPowerFeatures(sfreq=128.0, relative=True).fit_transform(X)
    assert feats.shape == (X.shape[0], X.shape[1] * 5)  # 5 bands
    assert np.isfinite(feats).all()


def test_riemann_pipeline_fits(synthetic_eeg):
    pytest.importorskip("pyriemann")
    from eegpipe.features import build_riemann_classifier

    X, y, _ = synthetic_eeg
    clf = build_riemann_classifier().fit(X, y)
    assert clf.predict(X).shape == y.shape


def test_wavelet_features(synthetic_eeg):
    pytest.importorskip("pywt")
    from eegpipe.features.wavelet import WaveletEnergyFeatures

    X, _, _ = synthetic_eeg
    feats = WaveletEnergyFeatures(level=3).fit_transform(X)
    assert feats.shape[0] == X.shape[0] and np.isfinite(feats).all()
