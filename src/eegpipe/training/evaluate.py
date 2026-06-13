"""Leakage-safe cross-validation & reporting.

Demonstrates the *correct* evaluation protocol the original notebook lacked:
  * subject-aware folds (LOSO or grouped k-fold),
  * preprocessing fit **inside** each fold (no peeking at the test subject),
  * multiple metrics — accuracy alone is misleading on imbalanced data.

The classical Riemannian pipeline here is fully runnable and is the baseline every deep
model must beat. For DL models, loop :func:`eegpipe.training.train.main` over folds.
"""
from __future__ import annotations

import hydra
import numpy as np
from omegaconf import DictConfig

from eegpipe.utils import get_logger, seed_everything

log = get_logger(__name__)


def cross_validate_classical(bundle, n_splits: int = 5, loso: bool = False,
                             align: bool = False) -> dict:
    """Subject-aware CV of the Riemannian baseline. Returns per-metric mean +/- std."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        f1_score,
    )

    from eegpipe.data.splits import (
        assert_no_subject_leakage,
        leave_one_subject_out,
        subject_kfold,
    )
    from eegpipe.features import build_riemann_classifier

    X, y, groups = bundle.X, bundle.y, bundle.groups
    if align:  # cross-site/subject domain adaptation (unsupervised, per-subject)
        from eegpipe.adaptation import EuclideanAlignment
        X = EuclideanAlignment().fit_transform(X, groups=groups).astype('float32')
    splitter = (leave_one_subject_out(X, y, groups) if loso
                else subject_kfold(X, y, groups, n_splits=n_splits))

    rows = {"accuracy": [], "balanced_acc": [], "kappa": [], "macro_f1": []}
    for k, (tr, te) in enumerate(splitter):
        assert_no_subject_leakage(groups[tr], groups[te])
        clf = build_riemann_classifier()
        clf.fit(X[tr], y[tr])               # covariance estimation fit on train only
        pred = clf.predict(X[te])
        rows["accuracy"].append(accuracy_score(y[te], pred))
        rows["balanced_acc"].append(balanced_accuracy_score(y[te], pred))
        rows["kappa"].append(cohen_kappa_score(y[te], pred))
        rows["macro_f1"].append(f1_score(y[te], pred, average="macro"))
        log.info("fold %d | acc=%.3f kappa=%.3f", k, rows["accuracy"][-1], rows["kappa"][-1])

    return {m: (float(np.mean(v)), float(np.std(v))) for m, v in rows.items()}


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from eegpipe.data import build_loader
    from eegpipe.preprocessing import build_preprocessing

    seed_everything(cfg.seed)
    bundle = build_loader(cfg.data).load()
    # Stateless preprocessing only here; stateful steps belong inside the CV loop.
    bundle.epochs = build_preprocessing(cfg.preprocess).fit_transform(bundle.epochs)
    results = cross_validate_classical(bundle, n_splits=cfg.training.n_splits,
                                       loso=cfg.get("loso", False),
                                       align=cfg.get("align", False))
    log.info("=== Riemannian baseline (subject-aware CV) ===")
    for metric, (mean, std) in results.items():
        log.info("%-14s %.3f +/- %.3f", metric, mean, std)


if __name__ == "__main__":
    main()
