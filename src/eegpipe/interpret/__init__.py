"""Interpretability: input saliency, attention rollout, and confusion-matrix plots."""

from eegpipe.interpret.plots import plot_attention, plot_confusion_matrix, plot_saliency

__all__ = [
    "integrated_gradients",
    "vanilla_saliency",
    "channel_importance",
    "sequence_attention",
    "attention_rollout",
    "plot_confusion_matrix",
    "plot_saliency",
    "plot_attention",
]


def __getattr__(name):  # lazy torch imports
    if name in ("integrated_gradients", "vanilla_saliency", "channel_importance"):
        import eegpipe.interpret.saliency as s

        return getattr(s, name)
    if name in ("sequence_attention", "attention_rollout"):
        import eegpipe.interpret.attention as a

        return getattr(a, name)
    raise AttributeError(name)
