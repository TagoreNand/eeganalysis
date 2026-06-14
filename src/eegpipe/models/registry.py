"""Model factory (Factory pattern). Builds a backbone and wraps it for Lightning training."""

from __future__ import annotations

from eegpipe.models.base import LitEEGClassifier, LitSequenceClassifier
from eegpipe.models.conformer import EEGConformer
from eegpipe.models.eegnet import EEGNet
from eegpipe.models.sleep import SleepTransformer, TinySleepNet, UTime
from eegpipe.models.stgcn import STGCN
from eegpipe.utils import get_logger

log = get_logger(__name__)

BACKBONES = {"eegnet": EEGNet, "conformer": EEGConformer, "stgcn": STGCN}
SEQUENCE_BACKBONES = {
    "sleep_seqnet": TinySleepNet,
    "sleep_transformer": SleepTransformer,
    "utime": UTime,
}


def build_model(
    cfg, n_channels: int, n_times: int, n_classes: int, class_weights=None
) -> LitEEGClassifier:
    """Instantiate a per-trial backbone from ``cfg`` and wrap in :class:`LitEEGClassifier`."""
    params = dict(cfg)
    name = params.pop("name")
    if name not in BACKBONES:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(BACKBONES)}")
    train_keys = {"lr", "weight_decay", "max_epochs"}
    train_kw = {k: params.pop(k) for k in list(params) if k in train_keys}
    backbone = BACKBONES[name](
        n_channels=n_channels, n_times=n_times, n_classes=n_classes, **params
    )
    return LitEEGClassifier(backbone, n_classes=n_classes, class_weights=class_weights, **train_kw)


def load_pretrained_encoder(backbone, ckpt_path: str) -> None:
    """Load an SSL-pretrained EpochEncoder's weights into ``backbone.encoder`` (transfer)."""
    import torch

    state = torch.load(ckpt_path, map_location="cpu")
    state = state.get("encoder_state_dict", state)
    missing, unexpected = backbone.encoder.load_state_dict(state, strict=False)
    log.info(
        "Loaded pretrained encoder from %s (missing=%d unexpected=%d)",
        ckpt_path,
        len(missing),
        len(unexpected),
    )


def build_sequence_model(
    cfg, n_channels: int, n_times: int, n_classes: int, class_weights=None, class_names=None
) -> LitSequenceClassifier:
    """Build a sequence backbone and wrap it in :class:`LitSequenceClassifier`.

    Encoder selection (mutually exclusive):
      * ``encoder: foundation`` + ``foundation: {mock|bendr|labram}`` -> wrapped foundation model
      * ``pretrained_encoder: <path>``                               -> SSL-pretrained EpochEncoder
      * (neither)                                                    -> a fresh EpochEncoder
    """
    params = dict(cfg)
    name = params.pop("name")
    if name not in SEQUENCE_BACKBONES:
        raise KeyError(f"Unknown sequence model '{name}'. Known: {sorted(SEQUENCE_BACKBONES)}")

    pretrained = params.pop("pretrained_encoder", None)
    enc_kind = params.pop("encoder", None)
    foundation_name = params.pop("foundation", "mock")
    freeze = params.pop("freeze_encoder", True)
    wrap_keys = {"lr", "weight_decay", "max_epochs", "use_focal", "focal_gamma"}
    wrap_kw = {k: params.pop(k) for k in list(params) if k in wrap_keys}

    encoder = None
    if enc_kind == "foundation":
        from eegpipe.models.foundation import build_foundation_encoder

        encoder = build_foundation_encoder(
            foundation_name,
            in_channels=n_channels,
            emb_dim=params.get("emb_dim", 128),
            freeze=freeze,
        )
        log.info("Using foundation encoder '%s' (frozen=%s)", foundation_name, freeze)

    backbone = SEQUENCE_BACKBONES[name](
        n_channels=n_channels, n_times=n_times, n_classes=n_classes, encoder=encoder, **params
    )
    if pretrained and enc_kind is None:
        load_pretrained_encoder(backbone, pretrained)
    return LitSequenceClassifier(
        backbone,
        n_classes=n_classes,
        class_weights=class_weights,
        class_names=class_names,
        **wrap_kw,
    )
