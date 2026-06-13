"""Adapter to slot a pretrained EEG **foundation model** into the EpochEncoder interface.

Models like **BENDR** (Kostas et al., 2021) and **LaBraM** (Jiang et al., 2024) are pretrained
on thousands of hours of EEG and transfer well to small labelled tasks. They expect a fixed
montage and emit a feature vector (or sequence) per window. :class:`FoundationEncoderAdapter`
normalises any such backbone to ``(N, C, T) -> (N, emb_dim)`` + ``encode_sequence`` so it drops
straight into ``TinySleepNet`` / ``SleepTransformer`` / ``UTime`` via ``model.encoder=foundation``.

Two pieces make arbitrary montages work:
  * a 1x1 **channel adapter** mapping your electrodes to the backbone's expected channels;
  * a linear **projection** from the backbone's feature dim to the pipeline's ``emb_dim``.

We can't bundle the (large, separately-licensed) weights, so ``build_foundation_encoder``
supports a runnable ``mock`` backbone for wiring/tests, and documents how to load real ones.
"""
from __future__ import annotations

import torch.nn as nn


class MockFoundationEncoder(nn.Module):
    """A tiny stand-in with the foundation-backbone interface (N, C, T) -> (N, out_dim).

    Lets the whole transfer path run end-to-end in tests/CI without downloading weights.
    """

    def __init__(self, in_channels: int, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_dim, 25, 8, padding=12), nn.GELU(),
            nn.AdaptiveAvgPool1d(1))
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FoundationEncoderAdapter(nn.Module):
    """Wrap a foundation ``backbone`` to the EpochEncoder contract.

    Parameters
    ----------
    backbone: module mapping (N, C_b, T) -> (N, backbone_dim) or (N, backbone_dim, T').
    backbone_dim: feature dim emitted by ``backbone``.
    emb_dim: target embedding dim for the downstream sequence model.
    in_channels / backbone_channels: if given and different, a 1x1 conv adapts montage->backbone.
    freeze: freeze backbone weights (linear-probe style); unfreeze later for full fine-tuning.
    """

    def __init__(self, backbone, backbone_dim, emb_dim=128, in_channels=None,
                 backbone_channels=None, freeze=True):
        super().__init__()
        self.backbone = backbone
        self.channel_adapter = (
            nn.Conv1d(in_channels, backbone_channels, 1)
            if in_channels and backbone_channels and in_channels != backbone_channels else None)
        self.proj = nn.Linear(backbone_dim, emb_dim)
        self.emb_dim = emb_dim
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, x):                       # (N, C, T)
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)
        feat = self.backbone(x)
        if feat.dim() == 3:                     # (N, dim, T') -> pool time
            feat = feat.mean(dim=-1)
        return self.proj(feat)

    def encode_sequence(self, x):               # (B, L, C, T) -> (B, L, emb_dim)
        b, l, c, t = x.shape
        return self.forward(x.reshape(b * l, c, t)).reshape(b, l, -1)


def build_foundation_encoder(name: str, in_channels: int, emb_dim: int = 128,
                             weights: str | None = None, freeze: bool = True):
    """Construct a :class:`FoundationEncoderAdapter` for ``name`` in {mock, bendr, labram}.

    ``mock`` runs anywhere. ``bendr``/``labram`` require the upstream package + weights:
    install the model's repo, load its encoder, and pass it here. We adapt channels/dim for you.
    """
    if name == "mock":
        bk = MockFoundationEncoder(in_channels)
        return FoundationEncoderAdapter(bk, bk.out_dim, emb_dim=emb_dim, freeze=freeze)

    if name == "bendr":
        try:
            # e.g. from dn3 / BENDR repo: encoder = BENDREncoder(...); encoder.load(weights)
            from bendr import BENDREncoder  # type: ignore
        except ImportError as e:
            raise ImportError(
                "BENDR not installed. Clone github.com/SPOClab-ca/BENDR, install it, then pass "
                "the loaded encoder via FoundationEncoderAdapter(backbone=...).") from e
        bk = BENDREncoder()
        if weights:
            bk.load(weights)
        return FoundationEncoderAdapter(bk, bk.encoder_h, emb_dim=emb_dim,
                                        in_channels=in_channels, backbone_channels=20, freeze=freeze)

    if name == "labram":
        raise ImportError(
            "LaBraM weights are separately licensed. Install the LaBraM repo "
            "(github.com/935963004/LaBraM), load its encoder, and wrap it with "
            "FoundationEncoderAdapter(backbone=<labram_encoder>, backbone_dim=200, "
            "in_channels=<n_ch>, backbone_channels=<labram_ch>).")

    raise KeyError(f"Unknown foundation encoder '{name}'. Known: mock, bendr, labram.")
