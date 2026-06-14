"""ONNX export + ONNX-runtime inference parity (skips without torch/onnx)."""

import numpy as np
import pytest


def test_onnx_export_and_runtime_inference(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from eegpipe.inference import OnnxPredictor
    from eegpipe.models import build_model
    from eegpipe.serving.export import export_module_onnx

    lit = build_model({"name": "eegnet"}, n_channels=8, n_times=128, n_classes=3)
    out = tmp_path / "eegnet.onnx"
    export_module_onnx(lit, str(out), kind="trial", n_channels=8, n_times=128)

    predictor = OnnxPredictor(str(out))
    proba = predictor.predict_proba(np.random.randn(4, 8, 128).astype("float32"))
    assert proba.shape == (4, 3)
    assert np.allclose(proba.sum(1), 1.0, atol=1e-4)  # valid softmax
