"""Attention extraction for the SleepTransformer.

Rather than fight ``nn.TransformerEncoder``'s fused fast path (which bypasses Python and
hides attention weights), we re-run the encoder layers transparently, reproducing the exact
post-/pre-norm math while requesting attention weights. This yields the *real* per-layer
epoch x epoch attention, and ``attention_rollout`` (Abnar & Zuidema, 2020) composes them into
a single map showing which neighbouring epochs influenced each epoch's stage prediction.
"""

from __future__ import annotations


def _layer_with_attention(layer, x):
    """Replicate a torch TransformerEncoderLayer forward while capturing attention weights."""
    norm_first = getattr(layer, "norm_first", False)

    def ff(z):
        return layer.linear2(layer.dropout(layer.activation(layer.linear1(z))))

    if norm_first:
        n1 = layer.norm1(x)
        attn, w = layer.self_attn(n1, n1, n1, need_weights=True, average_attn_weights=True)
        x = x + layer.dropout1(attn)
        x = x + layer.dropout2(ff(layer.norm2(x)))
    else:
        attn, w = layer.self_attn(x, x, x, need_weights=True, average_attn_weights=True)
        x = layer.norm1(x + layer.dropout1(attn))
        x = layer.norm2(x + layer.dropout2(ff(x)))
    return x, w


def sequence_attention(model, x):
    """Return a list of (B, L, L) attention maps, one per encoder layer.

    ``model`` must be a :class:`eegpipe.models.sleep.SleepTransformer`; ``x`` is (B, L, C, T).
    """
    import torch

    model.eval()
    with torch.no_grad():
        h = model.pos(model.encoder.encode_sequence(x))
        weights = []
        for layer in model.transformer.layers:
            h, w = _layer_with_attention(layer, h)
            weights.append(w)
    return weights


def attention_rollout(attentions):
    """Compose per-layer attentions into one (B, L, L) map (adds residual, row-normalises)."""
    import torch

    rolled = None
    for a in attentions:
        a = a + torch.eye(a.size(-1), device=a.device).unsqueeze(0)
        a = a / a.sum(dim=-1, keepdim=True)
        rolled = a if rolled is None else torch.bmm(a, rolled)
    return rolled
