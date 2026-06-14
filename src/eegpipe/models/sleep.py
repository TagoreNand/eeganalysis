"""Sleep-stage sequence models (TinySleepNet / SleepTransformer / U-Time).

All three share one **per-epoch encoder** and differ only in how they model context across
epochs (BiLSTM / self-attention / temporal U-Net). The encoder is *injectable*: pass
``encoder=`` to use an SSL-pretrained ``EpochEncoder`` or a wrapped foundation model
(:class:`eegpipe.models.foundation.FoundationEncoderAdapter`); otherwise a fresh
``EpochEncoder`` is built. Any encoder must expose ``.emb_dim`` and ``.encode_sequence``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class EpochEncoder(nn.Module):
    """(N, C, T) -> (N, emb_dim). One feature vector per 30-s epoch."""

    def __init__(self, n_channels: int, emb_dim: int = 128, sfreq: int = 100):
        super().__init__()
        k, s = max(2, sfreq // 2), max(1, sfreq // 16)
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 64, k, s, padding=k // 2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(8, 8),
            nn.Dropout(0.3),
            nn.Conv1d(64, 128, 8, padding=4),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 128, 8, padding=4),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(4, 4),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(128, emb_dim)
        self.emb_dim = emb_dim

    def forward(self, x):
        return self.proj(self.net(x).squeeze(-1))

    def encode_sequence(self, x):
        """(B, L, C, T) -> (B, L, emb_dim)."""
        b, length, c, t = x.shape
        return self.forward(x.reshape(b * length, c, t)).reshape(b, length, -1)


def _resolve_encoder(encoder, n_channels, emb_dim, sfreq):
    """Use an injected encoder (and adopt its emb_dim) or build a default EpochEncoder."""
    if encoder is not None:
        return encoder, encoder.emb_dim
    enc = EpochEncoder(n_channels, emb_dim, sfreq)
    return enc, emb_dim


class TinySleepNet(nn.Module):
    """(B, L, C, T) -> (B, L, n_classes) via BiLSTM with a residual skip from epoch features."""

    is_sequence = True

    def __init__(
        self,
        n_channels,
        n_times,
        n_classes,
        emb_dim=128,
        lstm_hidden=128,
        lstm_layers=1,
        dropout=0.5,
        sfreq=100,
        bidirectional=True,
        encoder=None,
    ):
        super().__init__()
        self.encoder, emb_dim = _resolve_encoder(encoder, n_channels, emb_dim, sfreq)
        self.lstm = nn.LSTM(
            emb_dim,
            lstm_hidden,
            lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.skip = nn.Linear(emb_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(out_dim, n_classes)

    def forward(self, x):
        e = self.encoder.encode_sequence(x)
        h, _ = self.lstm(e)
        h = self.dropout(h + self.skip(e))
        return self.head(h)


class _PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SleepTransformer(nn.Module):
    """Self-attention over the epoch sequence. (B, L, C, T) -> (B, L, n_classes)."""

    is_sequence = True

    def __init__(
        self,
        n_channels,
        n_times,
        n_classes,
        emb_dim=128,
        depth=4,
        n_heads=8,
        mlp_ratio=4,
        dropout=0.3,
        sfreq=100,
        max_len=512,
        encoder=None,
    ):
        super().__init__()
        self.encoder, emb_dim = _resolve_encoder(encoder, n_channels, emb_dim, sfreq)
        self.pos = _PositionalEncoding(emb_dim, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        e = self.pos(self.encoder.encode_sequence(x))
        return self.head(self.norm(self.transformer(e)))


class _DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, 3, padding=1),
            nn.BatchNorm1d(cout),
            nn.GELU(),
            nn.Conv1d(cout, cout, 3, padding=1),
            nn.BatchNorm1d(cout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UTime(nn.Module):
    """U-Time-style multi-scale temporal U-Net over epoch embeddings. (B,L,C,T)->(B,L,n_cls)."""

    is_sequence = True

    def __init__(
        self,
        n_channels,
        n_times,
        n_classes,
        emb_dim=128,
        base=64,
        depth=3,
        pool=2,
        sfreq=100,
        encoder=None,
    ):
        super().__init__()
        self.encoder, emb_dim = _resolve_encoder(encoder, n_channels, emb_dim, sfreq)
        self.depth, self.pool = depth, pool
        dims = [emb_dim] + [base * (2**i) for i in range(depth)]
        self.downs = nn.ModuleList(_DoubleConv(dims[i], dims[i + 1]) for i in range(depth))
        self.poolL = nn.MaxPool1d(pool)
        self.bottleneck = _DoubleConv(dims[-1], dims[-1] * 2)
        self.ups = nn.ModuleList(
            nn.ConvTranspose1d(
                dims[-1] * 2 if i == 0 else dims[depth - i + 1], dims[depth - i], pool, stride=pool
            )
            for i in range(depth)
        )
        self.dec = nn.ModuleList(
            _DoubleConv(dims[depth - i] * 2, dims[depth - i]) for i in range(depth)
        )
        self.head = nn.Conv1d(dims[1], n_classes, 1)

    def forward(self, x):
        e = self.encoder.encode_sequence(x).transpose(1, 2)  # (B, emb, L)
        L0 = e.size(-1)
        mult = self.pool**self.depth
        pad = (mult - L0 % mult) % mult
        if pad:
            e = nn.functional.pad(e, (0, pad))
        skips, h = [], e
        for down in self.downs:
            h = down(h)
            skips.append(h)
            h = self.poolL(h)
        h = self.bottleneck(h)
        for up, dec, skip in zip(self.ups, self.dec, reversed(skips), strict=False):
            h = up(h)
            h = dec(torch.cat([h, skip], dim=1))
        return self.head(h)[..., :L0].transpose(1, 2)
