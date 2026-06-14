"""Compose preprocessing steps and build them from config (Builder pattern)."""

from __future__ import annotations

from eegpipe.preprocessing import steps as _steps
from eegpipe.preprocessing.base import BaseTransform
from eegpipe.utils import get_logger

log = get_logger(__name__)

# name -> class, so configs can list steps declaratively
STEP_REGISTRY = {
    "set_montage": _steps.SetMontage,
    "bandpass": _steps.BandpassFilter,
    "resample": _steps.Resample,
    "rereference": _steps.Rereference,
    "ica": _steps.ICAArtifactRemoval,
    "autoreject": _steps.AutoRejectStep,
}


class PreprocessingPipeline(BaseTransform):
    """Ordered chain of :class:`BaseTransform`. Stateful steps fit on train data only."""

    def __init__(self, steps: list[BaseTransform]):
        self.steps = steps

    def fit(self, inst, y=None):
        cur = inst
        for step in self.steps:
            cur = step.fit_transform(cur, y) if not step.stateless else step.transform(cur)
        return self

    def transform(self, inst):
        cur = inst
        for step in self.steps:
            cur = step.transform(cur)
        return cur

    def fit_transform(self, inst, y=None):
        cur = inst
        for step in self.steps:
            log.info("preprocess: %s", step)
            cur = step.fit_transform(cur, y)
        return cur

    def __repr__(self) -> str:
        return "PreprocessingPipeline([\n  " + ",\n  ".join(map(repr, self.steps)) + "\n])"


def build_preprocessing(cfg) -> PreprocessingPipeline:
    """Build a pipeline from a config list like::

    preprocess:
      steps:
        - {name: bandpass, l_freq: 8, h_freq: 32}
        - {name: resample, sfreq: 128}
        - {name: ica, threshold: 0.8}
    """
    steps = []
    for spec in cfg["steps"]:
        spec = dict(spec)
        name = spec.pop("name")
        if name not in STEP_REGISTRY:
            raise KeyError(f"Unknown step '{name}'. Known: {sorted(STEP_REGISTRY)}")
        steps.append(STEP_REGISTRY[name](**spec))
    return PreprocessingPipeline(steps)
