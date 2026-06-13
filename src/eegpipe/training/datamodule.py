"""Lightning DataModule built from an :class:`EpochsBundle` and pre-computed split indices.

Splits are passed in (already subject-aware) rather than computed here, so the same
DataModule serves a single train run and each fold of cross-validation.
"""
from __future__ import annotations

import lightning as L
import numpy as np
from torch.utils.data import DataLoader

from eegpipe.data.torch_dataset import EEGWindowsDataset


class EEGDataModule(L.LightningDataModule):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray | None = None,
        batch_size: int = 64,
        num_workers: int = 4,
        train_transform=None,
    ):
        super().__init__()
        self.X, self.y = X, y
        self.train_idx, self.val_idx, self.test_idx = train_idx, val_idx, test_idx
        self.batch_size, self.num_workers = batch_size, num_workers
        self.train_transform = train_transform

    def setup(self, stage: str | None = None):
        self.train_ds = EEGWindowsDataset(self.X[self.train_idx], self.y[self.train_idx],
                                          transform=self.train_transform)
        self.val_ds = EEGWindowsDataset(self.X[self.val_idx], self.y[self.val_idx])
        if self.test_idx is not None:
            self.test_ds = EEGWindowsDataset(self.X[self.test_idx], self.y[self.test_idx])

    def _dl(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=True, drop_last=shuffle)

    def train_dataloader(self):
        return self._dl(self.train_ds, True)

    def val_dataloader(self):
        return self._dl(self.val_ds, False)

    def test_dataloader(self):
        return self._dl(self.test_ds, False)


class SequenceEEGDataModule(L.LightningDataModule):
    """DataModule for epoch-sequence models (sleep staging).

    Splits operate over *window indices* (already subject-aware) and all splits share the
    same underlying epoch arrays, so overlap and CV folds cost only index arrays.
    ``sampler_weights`` (length = n_train_windows) enables a WeightedRandomSampler to
    counter class imbalance.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        windows: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        sampler_weights: np.ndarray | None = None,
    ):
        super().__init__()
        self.X, self.y, self.windows = X, y, windows
        self.train_idx, self.val_idx, self.test_idx = train_idx, val_idx, test_idx
        self.batch_size, self.num_workers = batch_size, num_workers
        self.sampler_weights = sampler_weights

    def setup(self, stage: str | None = None):
        from eegpipe.data.sequence import SleepSequenceDataset

        self.train_ds = SleepSequenceDataset(self.X, self.y, self.windows[self.train_idx])
        self.val_ds = SleepSequenceDataset(self.X, self.y, self.windows[self.val_idx])
        if self.test_idx is not None:
            self.test_ds = SleepSequenceDataset(self.X, self.y, self.windows[self.test_idx])

    def train_dataloader(self):
        import torch
        from torch.utils.data import DataLoader, WeightedRandomSampler

        if self.sampler_weights is not None:
            sampler = WeightedRandomSampler(
                torch.as_tensor(self.sampler_weights, dtype=torch.double),
                num_samples=len(self.train_idx), replacement=True,
            )
            return DataLoader(self.train_ds, batch_size=self.batch_size, sampler=sampler,
                              num_workers=self.num_workers, pin_memory=True, drop_last=True)
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        from torch.utils.data import DataLoader

        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        from torch.utils.data import DataLoader

        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)
