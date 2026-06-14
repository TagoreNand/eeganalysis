"""Matplotlib renderers for the interpretability artefacts (used by the dashboard & reports)."""

from __future__ import annotations

import numpy as np


def plot_confusion_matrix(cm, class_names, normalize=True, ax=None, out_path=None):
    """Row-normalised confusion matrix heatmap. Diagonal = per-class recall."""
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    if normalize:
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=(1.0 if normalize else None))
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix" + (" (recall)" if normalize else ""))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(
                j,
                i,
                val,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if cm[i, j] > 0.5 else "black",
            )
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if out_path:
        ax.figure.savefig(out_path, dpi=120, bbox_inches="tight")
    return ax


def plot_saliency(attr, ch_names=None, ax=None, out_path=None):
    """(C, T) attribution heatmap — bright = influential channel/time."""
    import matplotlib.pyplot as plt

    attr = np.abs(np.asarray(attr))
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(attr, aspect="auto", cmap="magma", origin="lower")
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Channel")
    if ch_names is not None:
        ax.set_yticks(range(len(ch_names)))
        ax.set_yticklabels(ch_names, fontsize=7)
    ax.set_title("Integrated-Gradients saliency")
    ax.figure.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    if out_path:
        ax.figure.savefig(out_path, dpi=120, bbox_inches="tight")
    return ax


def plot_attention(att, ax=None, out_path=None):
    """(L, L) epoch attention/rollout map — row i = what epoch i attends to."""
    import matplotlib.pyplot as plt

    att = np.asarray(att)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(att, cmap="viridis", origin="upper")
    ax.set_xlabel("Key epoch")
    ax.set_ylabel("Query epoch")
    ax.set_title("Epoch attention rollout")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if out_path:
        ax.figure.savefig(out_path, dpi=120, bbox_inches="tight")
    return ax
