"""Sleep-stage classification training (epoch-sequence model).

Distinct from the per-trial ``train.py`` in three ways that matter for staging:
  * builds **epoch sequences** that never cross a night boundary,
  * selects on **macro-F1** (not accuracy — N2 dominates),
  * counters imbalance via inverse-frequency class weights + an optional weighted sampler.

Run::

    eegpipe-sleep data=sleep_edf model=sleep_seqnet training=sleep preprocess=sleep
    eegpipe-sleep ... training.fast_dev_run=true     # quick smoke test
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from eegpipe.utils import get_logger, seed_everything

log = get_logger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    from eegpipe.data import (
        assert_no_subject_leakage,
        build_loader,
        make_sequence_windows,
        sequence_sampler_weights,
        subject_kfold,
    )
    from eegpipe.models import build_sequence_model
    from eegpipe.preprocessing import build_preprocessing
    from eegpipe.training.datamodule import SequenceEEGDataModule
    from eegpipe.training.train import _make_logger

    seed_everything(cfg.seed)
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    # 1) Load Sleep-EDF and apply sleep-appropriate preprocessing (keep delta band!)
    bundle = build_loader(cfg.data).load()
    log.info("Loaded %s", bundle)
    epochs = build_preprocessing(cfg.preprocess).fit_transform(bundle.epochs)
    X = epochs.get_data(copy=False).astype("float32")
    y, rec = bundle.y, bundle.groups
    stage_names = bundle.metadata.get("stage_names")
    n_classes = int(y.max()) + 1

    # 2) Build epoch sequences (never crossing a night) and split by subject over windows
    windows, seq_groups = make_sequence_windows(
        rec, seq_len=cfg.training.seq_len, stride=cfg.training.get("stride", None)
    )
    log.info(
        "Built %d sequences of length %d from %d epochs", len(windows), cfg.training.seq_len, len(y)
    )

    folds = list(
        subject_kfold(
            windows, seq_groups, seq_groups, n_splits=cfg.training.n_splits, stratified=False
        )
    )
    trainval, test_idx = folds[cfg.training.fold]
    inner = list(
        subject_kfold(
            windows[trainval],
            seq_groups[trainval],
            seq_groups[trainval],
            n_splits=4,
            stratified=False,
        )
    )
    tr_rel, val_rel = inner[0]
    train_idx, val_idx = trainval[tr_rel], trainval[val_rel]
    assert_no_subject_leakage(seq_groups[train_idx], seq_groups[test_idx])
    assert_no_subject_leakage(seq_groups[val_idx], seq_groups[test_idx])

    # 3) Inverse-frequency class weights from TRAIN epochs only
    train_labels = y[windows[train_idx]].ravel()
    classes, counts = np.unique(train_labels, return_counts=True)
    class_weights = np.ones(n_classes)
    class_weights[classes] = counts.sum() / (len(classes) * counts)
    class_weights = class_weights.tolist()
    log.info(
        "class weights: %s",
        {(stage_names or range(n_classes))[i]: round(w, 2) for i, w in enumerate(class_weights)},
    )
    sampler_w = (
        sequence_sampler_weights(y, windows[train_idx], class_weights)
        if cfg.training.weighted_sampler
        else None
    )

    dm = SequenceEEGDataModule(
        X,
        y,
        windows,
        train_idx,
        val_idx,
        test_idx,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        sampler_weights=sampler_w,
    )
    model = build_sequence_model(
        cfg.model,
        n_channels=X.shape[1],
        n_times=X.shape[2],
        n_classes=n_classes,
        class_weights=class_weights,
        class_names=stage_names,
    )

    ckpt = ModelCheckpoint(
        dirpath="models_store",
        monitor="val/macro_f1",
        mode="max",
        filename="sleep-{epoch}-{val/macro_f1:.3f}",
        save_top_k=1,
    )
    early = EarlyStopping(monitor="val/macro_f1", mode="max", patience=cfg.training.patience)
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        logger=_make_logger(cfg.training.logger, cfg),
        callbacks=[ckpt, early],
        gradient_clip_val=1.0,
        fast_dev_run=cfg.training.fast_dev_run,
        log_every_n_steps=10,
    )
    trainer.fit(model, dm)

    if cfg.training.fast_dev_run:
        return 0.0
    results = trainer.test(model, dm, ckpt_path="best")[0]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/sleep_metrics.json").write_text(json.dumps(results, indent=2))
    if getattr(model, "test_confusion", None) is not None:
        np.save("reports/confusion.npy", model.test_confusion)
        labels = stage_names or list(range(model.n_classes))
        Path("reports/confusion_labels.json").write_text(json.dumps(labels))
    log.info("Best checkpoint: %s", ckpt.best_model_path)
    log.info(
        "Test macro-F1=%.3f  kappa=%.3f",
        results.get("test/macro_f1", 0),
        results.get("test/kappa", 0),
    )
    return float(results.get("test/macro_f1", 0.0))


if __name__ == "__main__":
    main()
