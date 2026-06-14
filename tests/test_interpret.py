"""Saliency + attention interpretability (torch/matplotlib tests skip if absent)."""

import numpy as np
import pytest


def test_integrated_gradients_shape_matches_input():
    pytest.importorskip("torch")
    from eegpipe.interpret import integrated_gradients
    from eegpipe.models import build_model

    model = build_model({"name": "eegnet"}, n_channels=8, n_times=128, n_classes=3)
    attr = integrated_gradients(model, np.random.randn(8, 128).astype("float32"), steps=8)
    assert attr.shape == (8, 128)


def test_sequence_attention_and_rollout():
    pytest.importorskip("torch")
    import torch

    from eegpipe.interpret import attention_rollout, sequence_attention
    from eegpipe.models import build_sequence_model

    lit = build_sequence_model(
        {"name": "sleep_transformer", "depth": 2, "n_heads": 4, "emb_dim": 32},
        n_channels=2,
        n_times=600,
        n_classes=5,
    )
    maps = sequence_attention(lit.backbone, torch.randn(1, 6, 2, 600))
    assert len(maps) == 2 and maps[0].shape == (1, 6, 6)
    assert torch.allclose(maps[0].sum(-1), torch.ones(1, 6), atol=1e-4)  # attention rows sum to 1
    assert attention_rollout(maps).shape == (1, 6, 6)


def test_confusion_matrix_plot():
    pytest.importorskip("matplotlib")
    from eegpipe.interpret import plot_confusion_matrix

    cm = np.array([[5, 1], [2, 4]])
    ax = plot_confusion_matrix(cm, ["A", "B"])
    assert ax is not None
