"""Hydra-driven single training run with subject-aware train/val/test split.

Run from the repo root (editable install)::

    eegpipe-train data=bci_iv_2a model=eegnet training.logger=mlflow
    eegpipe-train data=mne_sample model=eegnet training.fast_dev_run=true
"""

from __future__ import annotations

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from eegpipe.utils import get_logger, seed_everything

log = get_logger(__name__)


def _make_logger(name: str, cfg: DictConfig):
    if name == "mlflow":
        from lightning.pytorch.loggers import MLFlowLogger

        return MLFlowLogger(
            experiment_name=cfg.experiment_name, tracking_uri=cfg.get("mlflow_uri", "file:./mlruns")
        )
    if name == "wandb":
        from lightning.pytorch.loggers import WandbLogger

        return WandbLogger(project=cfg.experiment_name)
    from lightning.pytorch.loggers import CSVLogger

    return CSVLogger(save_dir="lightning_logs")


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    from eegpipe.data import assert_no_subject_leakage, build_loader, subject_kfold
    from eegpipe.models import build_model
    from eegpipe.preprocessing import build_preprocessing
    from eegpipe.training.datamodule import EEGDataModule

    seed_everything(cfg.seed)
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    # 1) Load -> 2) preprocess (fit on full set here is acceptable for stateless steps;
    #    for ICA/AutoReject inside CV, fit per-fold — see evaluate.py).
    bundle = build_loader(cfg.data).load()
    log.info("Loaded %s", bundle)
    epochs = build_preprocessing(cfg.preprocess).fit_transform(bundle.epochs)
    X, y, groups = epochs.get_data(copy=False).astype("float32"), bundle.y, bundle.groups

    # 3) Subject-aware split: outer fold -> (train+val, test); inner -> (train, val)
    folds = list(subject_kfold(X, y, groups, n_splits=cfg.training.n_splits))
    trainval_idx, test_idx = folds[cfg.training.fold]
    inner = list(subject_kfold(X[trainval_idx], y[trainval_idx], groups[trainval_idx], n_splits=4))
    tr_rel, val_rel = inner[0]
    train_idx, val_idx = trainval_idx[tr_rel], trainval_idx[val_rel]
    assert_no_subject_leakage(groups[train_idx], groups[test_idx])
    assert_no_subject_leakage(groups[val_idx], groups[test_idx])

    # 4) Class weights (critical for imbalanced sleep staging)
    classes, counts = np.unique(y[train_idx], return_counts=True)
    class_weights = (counts.sum() / (len(classes) * counts)).tolist()

    dm = EEGDataModule(
        X,
        y,
        train_idx,
        val_idx,
        test_idx,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )
    model = build_model(
        cfg.model,
        n_channels=X.shape[1],
        n_times=X.shape[2],
        n_classes=int(len(classes)),
        class_weights=class_weights,
    )

    ckpt = ModelCheckpoint(
        dirpath="models_store",
        monitor="val/acc",
        mode="max",
        filename=f"{cfg.model.name}-{{epoch}}-{{val/acc:.3f}}",
        save_top_k=1,
    )
    early = EarlyStopping(monitor="val/acc", mode="max", patience=cfg.training.patience)
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        logger=_make_logger(cfg.training.logger, cfg),
        callbacks=[ckpt, early],
        fast_dev_run=cfg.training.fast_dev_run,
        log_every_n_steps=10,
    )
    trainer.fit(model, dm)
    results = trainer.test(model, dm, ckpt_path="best") if not cfg.training.fast_dev_run else [{}]
    log.info("Best checkpoint: %s", ckpt.best_model_path)
    return float(results[0].get("test/acc", 0.0))  # return value enables Optuna/Hydra sweeps


if __name__ == "__main__":
    main()
