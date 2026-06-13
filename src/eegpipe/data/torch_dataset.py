"""Bridge from MNE/NumPy to a PyTorch ``Dataset``.

Crops continuous epochs into fixed-length windows (the standard input contract for
EEGNet / Conformer) and applies optional on-the-fly augmentations.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset

    _TORCH = True
except ImportError:  # allow importing the package without the DL extra
    _TORCH = False
    Dataset = object  # type: ignore


class EEGWindowsDataset(Dataset):
    """Wrap ``(X, y)`` arrays as a torch dataset.

    Parameters
    ----------
    X: (n_epochs, n_channels, n_times) float array.
    y: (n_epochs,) integer labels.
    transform: optional callable applied to each ``(C, T)`` sample (e.g. augmentation).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        transform: Callable | None = None,
    ) -> None:
        if not _TORCH:
            raise ImportError("Install the DL extra: pip install 'eegpipe[dl]'")
        self.X = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32)
        self.y = torch.as_tensor(np.ascontiguousarray(y), dtype=torch.long)
        self.transform = transform

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[idx]
