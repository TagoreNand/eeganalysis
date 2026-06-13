"""Foundation-encoder adapter + injection into sequence backbones."""
import pytest


def test_mock_foundation_encoder_interface():
    pytest.importorskip("torch")
    import torch

    from eegpipe.models import build_foundation_encoder

    enc = build_foundation_encoder("mock", in_channels=4, emb_dim=64)
    assert enc.emb_dim == 64
    assert enc(torch.randn(5, 4, 3000)).shape == (5, 64)              # (N, C, T) -> (N, emb)
    assert enc.encode_sequence(torch.randn(2, 7, 4, 3000)).shape == (2, 7, 64)


def test_foundation_encoder_injects_into_sequence_model():
    pytest.importorskip("torch")
    import torch

    from eegpipe.models import build_sequence_model

    cfg = {"name": "sleep_transformer", "encoder": "foundation", "foundation": "mock",
           "emb_dim": 64, "depth": 2, "n_heads": 4}
    lit = build_sequence_model(cfg, n_channels=4, n_times=3000, n_classes=5)
    assert lit(torch.randn(2, 8, 4, 3000)).shape == (2, 8, 5)


def test_labram_requires_weights():
    pytest.importorskip("torch")
    from eegpipe.models import build_foundation_encoder

    with pytest.raises(ImportError):
        build_foundation_encoder("labram", in_channels=4)
