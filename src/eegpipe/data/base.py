"""Data-loading contracts.

Every dataset (BIDS, MOABB, Sleep-EDF, the MNE sample set) is hidden behind one
interface so the rest of the pipeline never knows which dataset it is training on.
This is the Strategy pattern: ``build_loader(cfg)`` returns a concrete strategy,
callers depend only on :class:`BaseDataLoader`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpochsBundle:
    """Standardised container returned by every loader.

    Keeping ``epochs`` (an MNE object) *and* the materialised arrays lets downstream
    code choose: MNE transforms operate on ``epochs``; torch/sklearn want ``X, y``.

    Attributes
    ----------
    epochs:     ``mne.BaseEpochs`` — preprocessed, epoched signal.
    subject_ids: shape ``(n_epochs,)`` group label per epoch. **This is what prevents
                 subject leakage** — it is threaded all the way into cross-validation.
    label_name:  human-readable name of the prediction target.
    metadata:    arbitrary provenance (dataset name, paradigm, montage, ...).
    """

    epochs: object  # mne.BaseEpochs (kept loose to avoid a hard import here)
    subject_ids: np.ndarray
    label_name: str = "target"
    metadata: dict = field(default_factory=dict)

    @property
    def X(self) -> np.ndarray:
        """(n_epochs, n_channels, n_times) float32 array."""
        return self.epochs.get_data(copy=False).astype("float32")

    @property
    def y(self) -> np.ndarray:
        """Integer class labels derived from the epochs' event codes."""
        return self.epochs.events[:, 2].astype("int64")

    @property
    def groups(self) -> np.ndarray:
        return np.asarray(self.subject_ids)

    @property
    def sfreq(self) -> float:
        return float(self.epochs.info["sfreq"])

    @property
    def ch_names(self) -> list[str]:
        return list(self.epochs.ch_names)

    @property
    def info(self):
        return self.epochs.info

    def __repr__(self) -> str:
        return (
            f"EpochsBundle(n_epochs={len(self.subject_ids)}, "
            f"n_subjects={len(np.unique(self.subject_ids))}, "
            f"n_channels={len(self.ch_names)}, sfreq={self.sfreq:g}Hz, "
            f"classes={np.unique(self.y).tolist()})"
        )


class BaseDataLoader(ABC):
    """Abstract dataset strategy. Subclass and implement :meth:`load`."""

    name: str = "base"

    def __init__(self, cache_dir: str | None = None) -> None:
        self.cache_dir = cache_dir

    @abstractmethod
    def load(self, subjects: Sequence[int | str] | None = None) -> EpochsBundle:
        """Return an :class:`EpochsBundle`. ``subjects=None`` loads all available."""
        raise NotImplementedError

    def describe(self) -> dict:
        """Lightweight metadata for logging without materialising the data."""
        return {"loader": self.name}
