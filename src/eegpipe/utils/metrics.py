"""Metrics for imbalanced multi-class staging.

Accuracy is misleading when one class (N2) is ~50% of epochs — a model that predicts N2
forever scores well on accuracy and is useless. The sleep-staging literature reports
**macro-F1** and **Cohen's kappa**; this module computes both plus per-class F1 so you can
see exactly which stage (usually N1) the model struggles with.
"""
from __future__ import annotations

import numpy as np


def sleep_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str] | None = None
) -> dict:
    """Return accuracy, balanced accuracy, macro-F1, kappa and per-class F1."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        f1_score,
    )

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = int(max(y_true.max(), y_pred.max())) + 1 if len(y_true) else 0
    names = class_names or [f"class_{i}" for i in range(n)]
    per_class = f1_score(y_true, y_pred, labels=list(range(len(names))),
                         average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "per_class_f1": {n: float(f) for n, f in zip(names, per_class)},
    }


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    from sklearn.metrics import confusion_matrix

    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
