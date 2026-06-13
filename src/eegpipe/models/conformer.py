"""EEG-Conformer (Song et al., 2022): a convolutional tokenizer feeding a Transformer.

The conv stem learns local temporal+spatial features (like EEGNet's first blocks); the
Transformer encoder then models long-range temporal dependencies via self-attention.
Stronger than EEGNet when you have more data, at higher compute cost.
"""
from __future__ import annotations

import torch.nn as nn
from einops import rearrange


class PatchEmbedding(nn.Module):
    """Conv stem: temporal conv -> spatial conv over all channels -> pooled tokens."""

    def __init__(self, n_channels: int, emb_dim: int = 40):
        super().__init__()
        self.tokenizer = nn.Sequential(
            nn.Conv2d(1, emb_dim, (1, 25), padding=(0, 12)),
            nn.Conv2d(emb_dim, emb_dim, (n_channels, 1)),
            nn.BatchNorm2d(emb_dim),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # temporal pooling -> sequence of tokens
            nn.Dropout(0.3),
        )
        self.proj = nn.Conv2d(emb_dim, emb_dim, (1, 1))

    def forward(self, x):
        x = x.unsqueeze(1)            # (B, 1, C, T)
        x = self.tokenizer(x)         # (B, E, 1, T')
        x = self.proj(x)
        return rearrange(x, "b e 1 t -> b t e")  # (B, seq_len, emb_dim)


class EEGConformer(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        emb_dim: int = 40,
        depth: int = 4,
        n_heads: int = 5,
        mlp_ratio: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embed = PatchEmbedding(n_channels, emb_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=emb_dim * mlp_ratio,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, n_classes)

    def forward(self, x):                 # x: (B, C, T)
        tokens = self.embed(x)            # (B, seq, emb)
        z = self.encoder(tokens)
        z = self.norm(z).mean(dim=1)      # global average pool over tokens
        return self.head(z)
