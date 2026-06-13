"""ONNX Runtime inference backend — same interface as :class:`Predictor`, no torch needed."""
from __future__ import annotations

import numpy as np


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class OnnxPredictor:
    """Drop-in replacement for ``Predictor`` backed by an exported ``.onnx`` model."""

    def __init__(self, onnx_path: str, class_names: list[str] | None = None,
                 providers: list[str] | None = None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            onnx_path, providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        out_shape = self.session.get_outputs()[0].shape
        self._n_classes = int(out_shape[-1]) if isinstance(out_shape[-1], int) else None
        self.class_names = class_names

    @property
    def n_classes(self):
        return self._n_classes

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.session.run(None, {self.input_name: np.ascontiguousarray(X, dtype="float32")})[0]
        return _softmax(logits, axis=-1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(-1)
