"""Sequence windowing for epoch-sequence models (sleep staging).

A single 30-s sleep epoch is ambiguous in isolation; the surrounding epochs carry the
context that disambiguates it (this is why human scorers read a hypnogram, not one epoch).
So sleep models are *many-to-many*: input a sequence of L consecutive epochs, predict the
stage of each.

Two hard requirements, both handled here:
  1. **A sequence must never cross a recording/subject boundary.** Windows are built only
     within contiguous runs of the same recording id, so a sequence is always one night.
  2. **Memory.** Overlapping windows would duplicate the signal many times. Instead we keep
     the epoch array *once* and store only integer window indices; the Dataset gathers on
     access. Overlap then costs indices, not gigabytes.
"""

from __future__ import annotations

import warnings

import numpy as np


def make_sequence_windows(
    recording_ids: np.ndarray,
    seq_len: int,
    stride: int | None = None,
    pad_last: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sequence windows that never cross a recording boundary.

    Parameters
    ----------
    recording_ids: (n_epochs,) id marking which recording/night each epoch belongs to.
        Epochs are assumed time-ordered within each recording (MNE preserves event order).
    seq_len: number of consecutive epochs per sequence (L).
    stride: hop between window starts. Defaults to ``seq_len`` (non-overlapping, for
        training). Use 1 for dense inference.
    pad_last: if True, emit a final window per recording aligned to its end so every epoch
        is covered (used for full-night inference). The returned ``windows`` may then repeat
        a few trailing indices.

    Returns
    -------
    windows: (n_windows, seq_len) int array of epoch indices into the original arrays.
    seq_groups: (n_windows,) the recording id each window belongs to (use as CV groups).
    """
    stride = stride or seq_len
    rec = np.asarray(recording_ids)
    n = len(rec)
    # contiguous-run boundaries (where the recording id changes)
    change = np.flatnonzero(rec[1:] != rec[:-1]) + 1
    bounds = np.concatenate([[0], change, [n]])

    windows: list[np.ndarray] = []
    seq_groups: list = []
    for start, end in zip(bounds[:-1], bounds[1:], strict=False):
        run = np.arange(start, end)
        if len(run) < seq_len:
            warnings.warn(
                f"recording {rec[start]!r} has {len(run)} epochs < seq_len={seq_len}; skipped.",
                stacklevel=2,
            )
            continue
        starts = list(range(0, len(run) - seq_len + 1, stride))
        if pad_last and (len(run) - seq_len) % stride != 0:
            starts.append(len(run) - seq_len)  # tail window aligned to recording end
        for s in starts:
            windows.append(run[s : s + seq_len])
            seq_groups.append(rec[start])

    if not windows:
        raise ValueError("No sequences produced; is seq_len longer than every recording?")
    return np.asarray(windows, dtype=np.int64), np.asarray(seq_groups)


try:
    import torch
    from torch.utils.data import Dataset

    _TORCH = True
except ImportError:
    _TORCH = False
    Dataset = object  # type: ignore


class SleepSequenceDataset(Dataset):
    """Gathers L-epoch sequences on the fly from a shared epoch array.

    Parameters
    ----------
    X: (n_epochs, n_channels, n_times) full epoch array (kept once, shared across splits).
    y: (n_epochs,) per-epoch integer labels.
    windows: (n_windows, seq_len) index array from :func:`make_sequence_windows`.
    transform: optional per-epoch augmentation applied to each ``(C, T)`` slice.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, windows: np.ndarray, transform=None):
        if not _TORCH:
            raise ImportError("Install the DL extra: pip install 'eegpipe[dl]'")
        self.X = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32)
        self.y = torch.as_tensor(np.ascontiguousarray(y), dtype=torch.long)
        self.windows = torch.as_tensor(windows, dtype=torch.long)
        self.transform = transform

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        x = self.X[w]  # (L, C, T)
        if self.transform is not None:
            x = torch.stack([self.transform(e) for e in x])
        return x, self.y[w]  # (L, C, T), (L,)


def sequence_sampler_weights(y: np.ndarray, windows: np.ndarray, class_weights) -> np.ndarray:
    """Per-window sampling weight for a WeightedRandomSampler.

    Each window is weighted by the *mean* inverse-frequency weight of the epochs it contains,
    so sequences that include rare stages (N1, often <5% of a night) are sampled more often.
    Combats the N2-dominated imbalance without discarding majority data.
    """
    w = np.asarray(class_weights, dtype="float64")
    labels = np.asarray(y)[windows]  # (n_windows, seq_len)
    return w[labels].mean(axis=1)
