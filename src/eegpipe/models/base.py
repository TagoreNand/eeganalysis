"""LightningModule wrapper shared by every architecture.

Backbones (EEGNet, Conformer, ST-GCN) are plain ``nn.Module`` objects returning logits.
This wrapper adds the training/val/test loops, class-weighted loss (vital for sleep
staging), optimiser + scheduler, and metric logging — so swapping architectures never
touches training code (Template-Method pattern).
"""

from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class LitEEGClassifier(L.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        n_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: list[float] | None = None,
        max_epochs: int = 200,
        model_spec: dict | None = None,
    ):
        super().__init__()
        # ignore=['backbone'] keeps the checkpoint hyper-params readable & picklable
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone
        w = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.register_buffer("_class_weights", w if w is not None else torch.empty(0))
        self.n_classes = n_classes

    @classmethod
    def load(cls, ckpt_path, map_location="cpu"):
        """Rebuild from a checkpoint produced by build_model / build_sequence_model.

        The backbone is excluded from hyper-parameters, so it is reconstructed from the saved
        ``model_spec`` before the state dict is loaded.
        """
        import torch

        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        hp = dict(ckpt["hyper_parameters"])
        spec = hp.get("model_spec")
        if spec is None:
            raise ValueError(
                "Checkpoint lacks 'model_spec'; build via eegpipe.models.build_model / "
                "build_sequence_model so it can be reloaded."
            )
        from eegpipe.models.registry import rebuild_backbone

        backbone = rebuild_backbone(spec)
        model = cls(backbone=backbone, **hp)
        model.load_state_dict(ckpt["state_dict"])
        return model.eval()

    def forward(self, x):  # x: (B, C, T)
        return self.backbone(x)

    def _loss(self, logits, y):
        w = self._class_weights if self._class_weights.numel() else None
        return F.cross_entropy(logits, y, weight=w)

    def _step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self._loss(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log(f"{stage}/loss", loss, prog_bar=(stage != "train"))
        self.log(f"{stage}/acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def test_step(self, batch, _):
        return self._step(batch, "test")

    def predict_step(self, batch, _):
        x = batch[0] if isinstance(batch, list | tuple) else batch
        return torch.softmax(self(x), dim=1)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sched}


class Conv2dWithConstraint(nn.Conv2d):
    """Conv2d with a max-norm constraint on its weights — used by EEGNet's depthwise conv."""

    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class LitSequenceClassifier(L.LightningModule):
    """Lightning wrapper for many-to-many epoch-sequence models (sleep staging).

    Input batches are ``(B, L, C, T)`` and logits ``(B, L, n_classes)``; loss is computed
    over all ``B*L`` epochs. Supports class-weighted CE *and* focal loss for imbalance, and
    reports macro-F1 / kappa at epoch end (not just accuracy) so model selection optimises
    the metric that actually matters for staging.
    """

    def __init__(
        self,
        backbone: nn.Module,
        n_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: list[float] | None = None,
        max_epochs: int = 100,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        class_names: list[str] | None = None,
        model_spec: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone
        self.n_classes = n_classes
        self.class_names = class_names or [f"class_{i}" for i in range(n_classes)]
        w = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.register_buffer("_cw", w if w is not None else torch.empty(0))
        self._buf: dict[str, list] = {"val": [], "test": []}

    @classmethod
    def load(cls, ckpt_path, map_location="cpu"):
        """Rebuild from a checkpoint produced by build_model / build_sequence_model.

        The backbone is excluded from hyper-parameters, so it is reconstructed from the saved
        ``model_spec`` before the state dict is loaded.
        """
        import torch

        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        hp = dict(ckpt["hyper_parameters"])
        spec = hp.get("model_spec")
        if spec is None:
            raise ValueError(
                "Checkpoint lacks 'model_spec'; build via eegpipe.models.build_model / "
                "build_sequence_model so it can be reloaded."
            )
        from eegpipe.models.registry import rebuild_backbone

        backbone = rebuild_backbone(spec)
        model = cls(backbone=backbone, **hp)
        model.load_state_dict(ckpt["state_dict"])
        return model.eval()

    def forward(self, x):
        return self.backbone(x)

    def _loss(self, logits, y):
        logits = logits.reshape(-1, self.n_classes)
        y = y.reshape(-1)
        w = self._cw if self._cw.numel() else None
        if self.hparams.use_focal:
            ce = F.cross_entropy(logits, y, weight=w, reduction="none")
            pt = torch.exp(-ce)
            return ((1 - pt) ** self.hparams.focal_gamma * ce).mean()
        return F.cross_entropy(logits, y, weight=w)

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self._loss(logits, y)
        acc = (logits.argmax(-1) == y).float().mean()
        self.log("train/loss", loss)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def _eval_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        self.log(f"{stage}/loss", self._loss(logits, y), prog_bar=True)
        self._buf[stage].append((logits.argmax(-1).flatten().cpu(), y.flatten().cpu()))

    def validation_step(self, batch, _):
        self._eval_step(batch, "val")

    def test_step(self, batch, _):
        self._eval_step(batch, "test")

    def _epoch_end(self, stage: str):
        from eegpipe.utils.metrics import sleep_metrics

        if not self._buf[stage]:
            return
        preds = torch.cat([p for p, _ in self._buf[stage]]).numpy()
        trues = torch.cat([t for _, t in self._buf[stage]]).numpy()
        m = sleep_metrics(trues, preds, self.class_names)
        self.log(f"{stage}/acc", m["accuracy"])
        self.log(f"{stage}/macro_f1", m["macro_f1"], prog_bar=True)
        self.log(f"{stage}/kappa", m["kappa"], prog_bar=True)
        for name, f1 in m["per_class_f1"].items():
            self.log(f"{stage}/f1_{name}", f1)
        if stage == "test":
            from eegpipe.utils.metrics import confusion

            self.test_confusion = confusion(trues, preds, self.n_classes)
            self.test_targets, self.test_preds = trues, preds
        self._buf[stage].clear()

    def on_validation_epoch_end(self):
        self._epoch_end("val")

    def on_test_epoch_end(self):
        self._epoch_end("test")

    def predict_step(self, batch, _):
        x = batch[0] if isinstance(batch, list | tuple) else batch
        return torch.softmax(self(x), dim=-1)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sched}
