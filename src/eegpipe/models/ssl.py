"""Self-supervised pretraining for label-scarce EEG (Relative Positioning).

Pretext task (Banville et al., 2021): sample two epochs from the same recording. Label = 1
if they are temporally **close** (|i-j| <= tau_pos), 0 if **far** (|i-j| >= tau_neg). An
encoder that solves this learns physiologically meaningful structure from *unlabelled* nights;
we then transfer it into a sleep backbone and fine-tune on the few labelled recordings.

The encoder here is the same :class:`EpochEncoder` the sleep models use, so its weights drop
straight in via ``models.registry.load_pretrained_encoder`` (config: ``model.pretrained_encoder``).
"""

from __future__ import annotations

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativePositioningHead(nn.Module):
    """Given two epoch embeddings, predict 'temporally close?' from their absolute difference."""

    def __init__(self, emb_dim: int):
        super().__init__()
        self.clf = nn.Linear(emb_dim, 1)

    def forward(self, z1, z2):
        return self.clf(torch.abs(z1 - z2)).squeeze(-1)


class LitRelativePositioning(L.LightningModule):
    """Pretext trainer. ``encoder`` maps (B, C, T) -> (B, emb_dim)."""

    def __init__(self, encoder: nn.Module, emb_dim: int, lr: float = 1e-3):
        super().__init__()
        self.encoder = encoder
        self.head = RelativePositioningHead(emb_dim)
        self.lr = lr

    def _shared(self, batch, stage):
        x1, x2, y = batch
        logits = self.head(self.encoder(x1), self.encoder(x2))
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        acc = ((logits > 0).long() == y).float().mean()
        self.log(f"ssl/{stage}_loss", loss, prog_bar=True)
        self.log(f"ssl/{stage}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, _):
        return self._shared(batch, "train")

    def validation_step(self, batch, _):
        self._shared(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def transfer_encoder(
    pretrained: nn.Module, n_classes: int, emb_dim: int, freeze: bool = False
) -> nn.Module:
    """Attach a fresh linear head to a pretrained encoder (simple linear-probe / fine-tune)."""
    if freeze:
        for p in pretrained.parameters():
            p.requires_grad_(False)
    return nn.Sequential(pretrained, nn.Linear(emb_dim, n_classes))


try:
    from torch.utils.data import Dataset

    _TORCH = True
except ImportError:
    _TORCH = False
    Dataset = object  # type: ignore


class RelativePositioningDataset(Dataset):
    """Samples (anchor, other, label) epoch pairs within recording boundaries.

    Parameters
    ----------
    X: (n_epochs, C, T) epoch array (no labels needed — this is self-supervised).
    recording_ids: (n_epochs,) recording/subject id per epoch.
    tau_pos: max epoch distance for a *positive* (close) pair.
    tau_neg: min epoch distance for a *negative* (far) pair.
    n_samples: number of pairs to generate (balanced 50/50 positive/negative).
    """

    def __init__(
        self,
        X,
        recording_ids,
        tau_pos: int = 2,
        tau_neg: int = 10,
        n_samples: int = 10000,
        seed: int = 42,
    ):
        if not _TORCH:
            raise ImportError("Install the DL extra: pip install 'eegpipe[dl]'")
        self.X = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32)
        rec = np.asarray(recording_ids)
        rng = np.random.default_rng(seed)
        # map recording id -> sorted epoch indices
        groups = {r: np.flatnonzero(rec == r) for r in np.unique(rec)}
        usable = [r for r, idx in groups.items() if len(idx) > tau_neg + 1]
        if not usable:
            raise ValueError("No recording long enough for the chosen tau_neg.")

        a_list, o_list, y_list = [], [], []
        for k in range(n_samples):
            r = usable[rng.integers(len(usable))]
            idx = groups[r]
            i_pos = int(rng.integers(len(idx)))
            anchor = idx[i_pos]
            if k % 2 == 0:  # positive (close)
                lo, hi = max(0, i_pos - tau_pos), min(len(idx) - 1, i_pos + tau_pos)
                cand = [j for j in range(lo, hi + 1) if j != i_pos]
                label = 1
            else:  # negative (far)
                cand = [j for j in range(len(idx)) if abs(j - i_pos) >= tau_neg]
                label = 0
            if not cand:
                continue
            other = idx[cand[rng.integers(len(cand))]]
            a_list.append(anchor)
            o_list.append(other)
            y_list.append(label)
        self.a = np.asarray(a_list, dtype=np.int64)
        self.o = np.asarray(o_list, dtype=np.int64)
        self.y = torch.as_tensor(y_list, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[self.a[i]], self.X[self.o[i]], self.y[i]
