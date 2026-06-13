"""Load a trained checkpoint and run inference on raw EEG windows.

Used by both the FastAPI server and the LSL streaming loop, so preprocessing and the
model live in exactly one place (no train/serve skew).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class Predictor:
    """Thin wrapper around a trained :class:`LitEEGClassifier` checkpoint."""

    def __init__(self, ckpt_path: str | Path, device: str = "cpu",
                 preprocessing=None, class_names: list[str] | None = None):
        import torch

        from eegpipe.models.base import LitEEGClassifier

        self.device = device
        self.model = LitEEGClassifier.load_from_checkpoint(ckpt_path, map_location=device)
        self.model.eval().to(device)
        self.preprocessing = preprocessing      # applied to MNE objects, optional
        self.class_names = class_names
        self._torch = torch

    @property
    def n_classes(self) -> int:
        return self.model.n_classes

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """X: (n_trials, n_channels, n_times) -> (n_trials, n_classes) probabilities."""
        x = self._torch.as_tensor(np.ascontiguousarray(X), dtype=self._torch.float32, device=self.device)
        with self._torch.no_grad():
            return self._torch.softmax(self.model(x), dim=1).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(1)
