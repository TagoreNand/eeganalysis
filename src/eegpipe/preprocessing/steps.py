"""Concrete preprocessing steps. Each is configurable from Hydra and composable."""
from __future__ import annotations

from eegpipe.preprocessing.base import BaseTransform
from eegpipe.utils import get_logger

log = get_logger(__name__)


class SetMontage(BaseTransform):
    def __init__(self, montage: str = "standard_1020"):
        self.montage = montage

    def transform(self, inst):
        inst = inst.copy().set_montage(self.montage, on_missing="warn")
        return inst


class BandpassFilter(BaseTransform):
    """Zero-phase FIR band-pass — the workhorse from the original 'Mini-project' notebook."""

    def __init__(self, l_freq: float = 0.5, h_freq: float = 40.0, notch: float | None = 50.0):
        self.l_freq, self.h_freq, self.notch = l_freq, h_freq, notch

    def transform(self, inst):
        inst = inst.copy()
        if self.notch:
            inst.notch_filter(self.notch, verbose="error")
        inst.filter(self.l_freq, self.h_freq, fir_design="firwin", verbose="error")
        return inst


class Resample(BaseTransform):
    def __init__(self, sfreq: float = 128.0):
        self.sfreq = sfreq

    def transform(self, inst):
        return inst.copy().resample(self.sfreq, verbose="error")


class Rereference(BaseTransform):
    def __init__(self, ref: str = "average"):
        self.ref = ref

    def transform(self, inst):
        return inst.copy().set_eeg_reference(self.ref, verbose="error")


class ICAArtifactRemoval(BaseTransform):
    """Automated ICA cleaning.

    Fits ICA on the training signal, then auto-labels components with **ICLabel**
    (via ``mne-icalabel``) and drops eye/muscle/heart/line-noise components above a
    probability threshold. This automates the manual component-picking that the original
    'cleaning epoched data' notebook did by eye.
    """

    stateless = False
    ARTIFACT_LABELS = ("eye blink", "muscle artifact", "heart beat", "line noise", "channel noise")

    def __init__(self, n_components: float = 0.99, method: str = "infomax",
                 threshold: float = 0.8, random_state: int = 42):
        self.n_components, self.method = n_components, method
        self.threshold, self.random_state = threshold, random_state
        self._ica = None
        self._exclude: list[int] = []

    def fit(self, inst, y=None):
        import mne

        # ICLabel expects an infomax-fit ICA on 1-100Hz, common-average referenced data.
        self._ica = mne.preprocessing.ICA(
            n_components=self.n_components, method=self.method,
            fit_params=dict(extended=True), random_state=self.random_state,
        )
        self._ica.fit(inst, verbose="error")
        try:
            from mne_icalabel import label_components

            labels = label_components(inst, self._ica, method="iclabel")
            probs, names = labels["y_pred_proba"], labels["labels"]
            self._exclude = [
                i for i, (p, n) in enumerate(zip(probs, names))
                if n in self.ARTIFACT_LABELS and p >= self.threshold
            ]
            log.info("ICLabel flagged %d/%d components as artefacts", len(self._exclude), len(names))
        except ImportError:
            log.warning("mne-icalabel not installed; falling back to EOG correlation.")
            eog_idx, _ = self._ica.find_bads_eog(inst, verbose="error")
            self._exclude = eog_idx
        return self

    def transform(self, inst):
        if self._ica is None:
            raise RuntimeError("Call fit() before transform().")
        out = inst.copy()
        self._ica.exclude = self._exclude
        self._ica.apply(out, verbose="error")
        return out


class AutoRejectStep(BaseTransform):
    """Learn per-channel peak-to-peak rejection thresholds with the ``autoreject`` library.

    Replaces hand-tuned ``reject=dict(eeg=...)`` with a data-driven, cross-validated
    threshold. Operates on Epochs only.
    """

    stateless = False

    def __init__(self, n_interpolate=(1, 4, 8), random_state: int = 42):
        self.n_interpolate, self.random_state = n_interpolate, random_state
        self._ar = None

    def fit(self, epochs, y=None):
        from autoreject import AutoReject

        self._ar = AutoReject(n_interpolate=list(self.n_interpolate),
                              random_state=self.random_state, verbose=False)
        self._ar.fit(epochs)
        return self

    def transform(self, epochs):
        if self._ar is None:
            raise RuntimeError("Call fit() before transform().")
        return self._ar.transform(epochs)
