"""Self-supervised pretraining of the shared epoch encoder (Relative Positioning).

Trains on *unlabelled* recordings, then saves the encoder weights so any sleep backbone can
warm-start from them via ``model.pretrained_encoder=...``. This is the lever for the
low-label regime: pretrain on many unlabelled nights, fine-tune on the few labelled ones.

Run::

    eegpipe-pretrain data=sleep_edf model=sleep_seqnet training=pretrain preprocess=sleep
    # then fine-tune:
    eegpipe-sleep data=sleep_edf model=sleep_seqnet training=sleep preprocess=sleep \
        model.pretrained_encoder=models_store/ssl_encoder.pt
"""
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from eegpipe.utils import get_logger, seed_everything

log = get_logger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    import lightning as L
    import torch
    from torch.utils.data import DataLoader

    from eegpipe.data import build_loader
    from eegpipe.models.sleep import EpochEncoder
    from eegpipe.models.ssl import LitRelativePositioning, RelativePositioningDataset
    from eegpipe.preprocessing import build_preprocessing
    from eegpipe.training.train import _make_logger

    seed_everything(cfg.seed)
    bundle = build_loader(cfg.data).load()
    epochs = build_preprocessing(cfg.preprocess).fit_transform(bundle.epochs)
    X = epochs.get_data(copy=False).astype("float32")
    rec = bundle.groups
    log.info("Pretraining encoder on %d unlabelled epochs from %d recordings",
             len(rec), len(set(rec.tolist())))

    ds = RelativePositioningDataset(
        X, rec, tau_pos=cfg.training.tau_pos, tau_neg=cfg.training.tau_neg,
        n_samples=cfg.training.n_samples, seed=cfg.seed)
    dl = DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=True,
                    num_workers=cfg.training.num_workers, drop_last=True)

    encoder = EpochEncoder(n_channels=X.shape[1], emb_dim=cfg.model.emb_dim,
                           sfreq=cfg.model.get("sfreq", 100))
    lit = LitRelativePositioning(encoder, emb_dim=cfg.model.emb_dim, lr=cfg.training.lr)
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs, accelerator="auto",
        logger=_make_logger(cfg.training.logger, cfg),
        fast_dev_run=cfg.training.fast_dev_run, log_every_n_steps=10)
    trainer.fit(lit, dl)

    out = cfg.training.get("out", "models_store/ssl_encoder.pt")
    Path("models_store").mkdir(exist_ok=True)
    torch.save({"encoder_state_dict": encoder.state_dict(),
                "emb_dim": cfg.model.emb_dim, "n_channels": X.shape[1]}, out)
    log.info("Saved pretrained encoder -> %s", out)


if __name__ == "__main__":
    main()
