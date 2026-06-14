"""Forward-pass shape tests for every backbone (skipped without torch)."""

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("name", ["eegnet", "conformer", "stgcn"])
def test_backbone_forward_shape(name):
    from eegpipe.models import build_model

    n_ch, n_t, n_cls, batch = 22, 256, 4, 5
    lit = build_model({"name": name}, n_channels=n_ch, n_times=n_t, n_classes=n_cls)
    x = torch.randn(batch, n_ch, n_t)
    out = lit(x)
    assert out.shape == (batch, n_cls)


def test_eegnet_param_count_is_small():
    from eegpipe.models import EEGNet

    net = EEGNet(n_channels=22, n_times=256, n_classes=4)
    n_params = sum(p.numel() for p in net.parameters())
    assert n_params < 50_000  # EEGNet is intentionally tiny
