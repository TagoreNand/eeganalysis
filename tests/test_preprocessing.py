"""Preprocessing pipeline builds from config and runs on synthetic MNE data."""
import numpy as np
import pytest

mne = pytest.importorskip("mne")


def _raw():
    info = mne.create_info(["Fz", "Cz", "Pz", "Oz"], sfreq=256.0, ch_types="eeg")
    return mne.io.RawArray(np.random.randn(4, 2560) * 1e-6, info, verbose="error")


def test_build_and_apply_pipeline():
    from eegpipe.preprocessing import build_preprocessing

    cfg = {"steps": [{"name": "bandpass", "l_freq": 1.0, "h_freq": 40.0, "notch": None},
                     {"name": "resample", "sfreq": 128.0}]}
    pipe = build_preprocessing(cfg)
    out = pipe.fit_transform(_raw())
    assert out.info["sfreq"] == 128.0


def test_unknown_step_raises():
    from eegpipe.preprocessing import build_preprocessing

    with pytest.raises(KeyError):
        build_preprocessing({"steps": [{"name": "nope"}]})
