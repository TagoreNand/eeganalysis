"""Tests for the sleep-staging sequence path.

The boundary tests are the most important: a sequence that crosses two nights would leak
future/different-subject context into a prediction. Torch-dependent shape tests skip
gracefully when the DL extra isn't installed.
"""
import numpy as np
import pytest

from eegpipe.data.sequence import make_sequence_windows, sequence_sampler_weights
from eegpipe.utils.metrics import sleep_metrics


def test_windows_never_cross_recording_boundary():
    rec = np.array([0, 0, 0, 0, 1, 1, 1])
    windows, groups = make_sequence_windows(rec, seq_len=2, stride=1)
    for w in windows:
        assert len(set(rec[w].tolist())) == 1          # one recording per window
    for w, g in zip(windows, groups):
        assert rec[w[0]] == g                           # group label is correct


def test_nonoverlapping_window_count():
    rec = np.array([0] * 4 + [1] * 3)                   # 4-epoch + 3-epoch nights
    windows, _ = make_sequence_windows(rec, seq_len=2)  # stride defaults to seq_len
    assert windows.shape == (3, 2)                      # 2 from first night + 1 from second


def test_pad_last_covers_every_epoch():
    rec = np.zeros(5, dtype=int)
    windows, _ = make_sequence_windows(rec, seq_len=2, stride=2, pad_last=True)
    covered = {int(i) for w in windows for i in w}
    assert covered == set(range(5))


def test_short_recording_is_skipped_with_warning():
    rec = np.array([0, 0, 1])                           # second night too short for seq_len=2
    with pytest.warns(UserWarning):
        _, groups = make_sequence_windows(rec, seq_len=2)
    assert set(groups.tolist()) == {0}


def test_sampler_weights_upweight_windows_with_rare_classes():
    y = np.array([2, 2, 2, 2, 1])                       # class 1 (N1) is rare
    windows = np.array([[0, 1], [3, 4]])
    weights = sequence_sampler_weights(y, windows, class_weights=[1.0, 5.0, 1.0])
    assert weights[1] > weights[0]                      # window containing class 1 sampled more


def test_sleep_metrics_perfect_prediction():
    y = np.array([0, 1, 2, 3, 4, 2, 2])
    m = sleep_metrics(y, y, ["W", "N1", "N2", "N3", "REM"])
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0 and m["kappa"] == 1.0
    assert set(m["per_class_f1"]) == {"W", "N1", "N2", "N3", "REM"}


def test_sleep_metrics_penalises_majority_only_prediction():
    # All-N2 predictor: high accuracy but macro-F1 should be poor.
    y = np.array([0, 1, 2, 2, 2, 3, 4])
    pred = np.full_like(y, 2)
    m = sleep_metrics(y, pred, ["W", "N1", "N2", "N3", "REM"])
    assert m["macro_f1"] < 0.3


def test_tinysleepnet_forward_shape():
    pytest.importorskip("torch")
    import torch

    from eegpipe.models.sleep import TinySleepNet

    model = TinySleepNet(n_channels=2, n_times=3000, n_classes=5,
                         emb_dim=32, lstm_hidden=32, sfreq=100)
    out = model(torch.randn(4, 20, 2, 3000))            # (B, L, C, T)
    assert out.shape == (4, 20, 5)                       # (B, L, n_classes)


def test_sequence_dataset_returns_correct_shapes():
    pytest.importorskip("torch")
    from eegpipe.data.sequence import SleepSequenceDataset

    X = np.random.randn(10, 2, 100).astype("float32")
    y = np.random.randint(0, 5, 10)
    windows, _ = make_sequence_windows(np.zeros(10, dtype=int), seq_len=4, stride=2)
    ds = SleepSequenceDataset(X, y, windows)
    xb, yb = ds[0]
    assert tuple(xb.shape) == (4, 2, 100) and tuple(yb.shape) == (4,)


@pytest.mark.parametrize("name", ["sleep_seqnet", "sleep_transformer", "utime"])
def test_sequence_backbones_forward_shape(name):
    pytest.importorskip("torch")
    import torch

    from eegpipe.models import build_sequence_model

    lit = build_sequence_model({"name": name}, n_channels=2, n_times=3000, n_classes=5)
    out = lit(torch.randn(2, 16, 2, 3000))      # (B, L, C, T)
    assert out.shape == (2, 16, 5)               # (B, L, n_classes)


def test_utime_handles_non_power_of_two_sequence_length():
    pytest.importorskip("torch")
    import torch

    from eegpipe.models import build_sequence_model

    lit = build_sequence_model({"name": "utime"}, n_channels=2, n_times=600, n_classes=5)
    out = lit(torch.randn(1, 13, 2, 600))        # 13 not divisible by pool**depth -> padded internally
    assert out.shape == (1, 13, 5)
