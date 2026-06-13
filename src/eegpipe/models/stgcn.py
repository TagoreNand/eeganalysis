"""Spatio-Temporal Graph Convolutional Network for EEG.

Electrodes form a graph: nodes = channels, edges = spatial proximity on the scalp.
ST-GCN alternates *graph* convolutions (mix information between neighbouring electrodes)
with *temporal* convolutions, explicitly modelling EEG's non-Euclidean sensor topology.

Starter implementation — functional forward pass. See ROADMAP §2 for extensions
(learnable adjacency, Chebyshev polynomials, edge attention).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def adjacency_from_positions(pos: np.ndarray, sigma: float = 0.05) -> torch.Tensor:
    """Gaussian kernel on 3D electrode coordinates -> symmetric normalised adjacency.

    ``pos``: (n_channels, 3) montage positions, e.g. from ``epochs.get_montage()``.
    Returns the symmetric-normalised \\hat{A} = D^{-1/2}(A+I)D^{-1/2}.
    """
    d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    A = np.exp(-d2 / (2 * sigma**2))
    A = A + np.eye(A.shape[0])
    deg = A.sum(1)
    Dinv = np.diag(1.0 / np.sqrt(deg))
    return torch.tensor(Dinv @ A @ Dinv, dtype=torch.float32)


class GraphConv(nn.Module):
    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.lin = nn.Linear(in_f, out_f)

    def forward(self, x, A):              # x: (B, C, F), A: (C, C)
        return self.lin(torch.einsum("ij,bjf->bif", A, x))


class STGCN(nn.Module):
    def __init__(self, n_channels: int, n_times: int, n_classes: int,
                 adjacency: torch.Tensor | None = None, hidden: int = 64):
        super().__init__()
        A = adjacency if adjacency is not None else torch.eye(n_channels)
        self.register_buffer("A", A)
        self.temporal1 = nn.Conv1d(n_channels, n_channels, kernel_size=9, padding=4, groups=n_channels)
        self.gconv1 = GraphConv(n_times, hidden)
        self.gconv2 = GraphConv(hidden, hidden)
        self.act = nn.ELU()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n_channels * hidden, n_classes))

    def forward(self, x):                 # x: (B, C, T)
        x = self.temporal1(x)             # temporal conv per electrode
        x = self.act(self.gconv1(x, self.A))
        x = self.act(self.gconv2(x, self.A))
        return self.head(x)
