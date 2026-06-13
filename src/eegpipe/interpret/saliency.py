"""Input attribution for clinician trust.

Integrated Gradients (Sundararajan et al., 2017): attribute a prediction to input
channels x time-points by integrating gradients along a straight path from a baseline
(zeros) to the input. For EEG this answers "which electrodes and which moments drove the
call?" — the kind of evidence a clinician needs before trusting a model.

Works on per-trial models ``(B, C, T) -> (B, n_classes)``. No Captum dependency.
"""
from __future__ import annotations

import numpy as np


def _as_batch(x):
    import torch

    t = x if hasattr(x, "dim") else torch.as_tensor(np.asarray(x), dtype=torch.float32)
    return t.unsqueeze(0) if t.dim() == 2 else t


def integrated_gradients(model, x, target=None, steps: int = 32, baseline=None):
    """Return IG attributions with the same (C, T) shape as a single input trial."""
    import torch

    model.eval()
    x = _as_batch(x)
    baseline = torch.zeros_like(x) if baseline is None else baseline
    total = torch.zeros_like(x)
    for k in range(1, steps + 1):
        point = (baseline + (k / steps) * (x - baseline)).clone().requires_grad_(True)
        out = model(point)
        tgt = out.argmax(1) if target is None else torch.as_tensor(target).view(-1)
        score = out.gather(1, tgt.view(-1, 1)).sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        total = total + point.grad.detach()
    attr = (x - baseline) * (total / steps)
    return attr.squeeze(0).cpu().numpy()


def vanilla_saliency(model, x, target=None):
    """Single-pass |gradient| saliency — cheaper, noisier than IG."""
    import torch

    model.eval()
    x = _as_batch(x).clone().requires_grad_(True)
    out = model(x)
    tgt = out.argmax(1) if target is None else torch.as_tensor(target).view(-1)
    out.gather(1, tgt.view(-1, 1)).sum().backward()
    return x.grad.detach().abs().squeeze(0).cpu().numpy()


def channel_importance(attr: np.ndarray) -> np.ndarray:
    """Collapse a (C, T) attribution to a per-channel score (mean |attr| over time)."""
    return np.abs(attr).mean(axis=-1)
