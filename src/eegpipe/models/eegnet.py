"""EEGNet (Lawhern et al., 2018) — the canonical compact CNN for EEG.

Architecture (EEGNet-8,2): temporal conv -> depthwise spatial conv (learns per-frequency
spatial filters, analogous to CSP) -> separable conv -> classifier. ~2k params, strong on
small datasets, and the default baseline you should beat before reaching for anything bigger.
"""

from __future__ import annotations

import torch.nn as nn

from eegpipe.models.base import Conv2dWithConstraint


class EEGNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        F1: int = 8,
        D: int = 2,
        kernel_length: int = 64,  # ~half the sampling rate (set 32 for 128Hz)
        dropout: float = 0.5,
    ):
        super().__init__()
        F2 = F1 * D
        # Block 1: temporal conv then depthwise spatial conv across all electrodes
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding="same", bias=False),
            nn.BatchNorm2d(F1),
            Conv2dWithConstraint(F1, F1 * D, (n_channels, 1), groups=F1, bias=False, max_norm=1.0),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        # Block 2: separable conv = depthwise (temporal) + pointwise (mix feature maps)
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding="same", groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2 * (n_times // 32), n_classes),
        )

    def forward(self, x):
        # x: (B, C, T) -> add the conv "image" channel -> (B, 1, C, T)
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)
