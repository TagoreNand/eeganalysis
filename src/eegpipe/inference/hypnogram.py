"""Full-night hypnogram inference for sleep-sequence models.

A trained sequence model sees L epochs at a time. To label a whole night we slide dense,
overlapping windows (stride 1, end-padded) and **average the softmax** over all windows that
cover each epoch — this smooths boundary effects and is more robust than a single pass.
"""

from __future__ import annotations

import numpy as np


class HypnogramPredictor:
    STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]

    def __init__(
        self,
        ckpt_path: str,
        seq_len: int = 20,
        device: str = "cpu",
        stage_names: list[str] | None = None,
        batch: int = 64,
    ):
        import torch

        from eegpipe.models.base import LitSequenceClassifier

        self._torch = torch
        self.model = LitSequenceClassifier.load_from_checkpoint(ckpt_path, map_location=device)
        self.model.eval().to(device)
        self.seq_len, self.device, self.batch = seq_len, device, batch
        self.stage_names = (
            stage_names or getattr(self.model, "class_names", None) or self.STAGE_NAMES
        )

    def predict(self, X: np.ndarray, recording_ids: np.ndarray | None = None):
        """X: (n_epochs, C, T) for one or more nights -> (stages, probabilities)."""
        from eegpipe.data.sequence import make_sequence_windows

        torch = self._torch
        n, n_cls = X.shape[0], len(self.stage_names)
        rec = recording_ids if recording_ids is not None else np.zeros(n, dtype=int)
        windows, _ = make_sequence_windows(rec, self.seq_len, stride=1, pad_last=True)
        Xt = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=self.device)

        acc = np.zeros((n, n_cls), dtype="float64")
        cnt = np.zeros(n, dtype="float64")
        with torch.no_grad():
            for i in range(0, len(windows), self.batch):
                wb = windows[i : i + self.batch]
                xb = Xt[torch.as_tensor(wb, device=self.device)]  # (b, L, C, T)
                pb = torch.softmax(self.model(xb), dim=-1).cpu().numpy()  # (b, L, n_cls)
                for w, p in zip(wb, pb, strict=False):
                    acc[w] += p
                    cnt[w] += 1
        proba = acc / np.maximum(cnt[:, None], 1.0)
        return proba.argmax(1), proba


def plot_hypnogram(
    stages,
    stage_names: list[str] | None = None,
    epoch_sec: int = 30,
    out_path: str | None = None,
    ax=None,
):
    """Render a conventional hypnogram (W at top, time in hours). Returns the Axes."""
    import matplotlib.pyplot as plt

    names = stage_names or HypnogramPredictor.STAGE_NAMES
    display = [s for s in ["W", "REM", "N1", "N2", "N3"] if s in names]
    row = {names.index(s): i for i, s in enumerate(display)}
    ypos = np.array([row.get(int(s), 0) for s in stages])
    t = np.arange(len(stages)) * epoch_sec / 3600.0

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3))
    ax.step(t, ypos, where="post", lw=1.0, color="#3b6ea5")
    # shade REM spans for readability
    rem_row = row.get(names.index("REM")) if "REM" in names else None
    if rem_row is not None:
        ax.fill_between(
            t, ypos, rem_row, where=(ypos == rem_row), step="post", color="#d1495b", alpha=0.5
        )
    ax.set_yticks(range(len(display)))
    ax.set_yticklabels(display)
    ax.invert_yaxis()
    ax.set_xlabel("Time (hours)")
    ax.set_title("Predicted hypnogram")
    ax.margins(x=0)
    if out_path:
        ax.figure.savefig(out_path, dpi=120, bbox_inches="tight")
    return ax
