"""Preprocessing transform contract.

We adopt the scikit-learn ``fit``/``transform`` protocol but operate on **MNE objects**
(``Raw``/``Epochs``) rather than arrays. This keeps channel info, montage and annotations
intact through the whole chain, while still letting us compose steps like an sklearn
``Pipeline``. Stateful steps (ICA, AutoReject) learn parameters in ``fit`` on training
data only — never on the test fold.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTransform(ABC):
    """Base class for an MNE-aware, fit/transform-compatible preprocessing step."""

    #: set False for steps that must not see test data (ICA, AutoReject, scalers)
    stateless: bool = True

    def fit(self, inst, y=None) -> "BaseTransform":  # noqa: D401
        return self

    @abstractmethod
    def transform(self, inst):
        """Return a (usually copied) transformed MNE instance."""
        raise NotImplementedError

    def fit_transform(self, inst, y=None):
        return self.fit(inst, y).transform(inst)

    def __repr__(self) -> str:
        params = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        inner = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{type(self).__name__}({inner})"
