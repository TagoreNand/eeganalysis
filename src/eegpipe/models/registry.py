"""Model factory.

Builds a backbone, wraps it in a LightningModule, and can *rebuild* the backbone from a saved
``model_spec`` so checkpoints reload without any external information (the backbone is
intentionally excluded from saved hyper-parameters, so ``load_from_checkpoint`` alone cannot
reconstruct it — see ``LitEEGClassifier.load``).
"""

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
_WRAP_KEYS = {"lr", "weight_decay", "max_epochs", "use_focal", "focal_gamma"}


def _to_plain(cfg) -> dict:
    """Convert an OmegaConf node (or mapping) to a plain, picklable dict."""
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(cfg, DictConfig):
            return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass
    return dict(cfg)


def _make_backbone(cfg, n_channels, n_times, n_classes):
    params = _to_plain(cfg)
    name = params.pop("name")
    for k in _WRAP_KEYS:
        params.pop(k, None)
    if name not in BACKBONES:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(BACKBONES)}")
    return BACKBONES[name](n_channels=n_channels, n_times=n_times, n_classes=n_classes, **params)


def _make_sequence_backbone(cfg, n_channels, n_times, n_classes, load_pretrained=True):
    params = _to_plain(cfg)
    name = params.pop("name")
    pretrained = params.pop("pretrained_encoder", None)
    enc_kind = params.pop("encoder", None)
    foundation_name = params.pop("foundation", "mock")
    freeze = params.pop("freeze_encoder", True)
    for k in _WRAP_KEYS:
        params.pop(k, None)
    if name not in SEQUENCE_BACKBONES:
        raise KeyError(f"Unknown sequence model '{name}'. Known: {sorted(SEQUENCE_BACKBONES)}")
    encoder = None
    if enc_kind == "foundation":
        from eegpipe.models.foundation import build_foundation_encoder

        encoder = build_foundation_encoder(
            foundation_name,
            in_channels=n_channels,
            emb_dim=params.get("emb_dim", 128),
            freeze=freeze,
        )
    backbone = SEQUENCE_BACKBONES[name](
        n_channels=n_channels, n_times=n_times, n_classes=n_classes, encoder=encoder, **params
    )
    if pretrained and enc_kind is None and load_pretrained:
        load_pretrained_encoder(backbone, pretrained)
    return backbone


def load_pretrained_encoder(backbone, ckpt_path):
    """Load an SSL-pretrained EpochEncoder's weights into ``backbone.encoder`` (transfer)."""
    import torch

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = state.get("encoder_state_dict", state)
    missing, unexpected = backbone.encoder.load_state_dict(state, strict=False)
    log.info(
        "Loaded pretrained encoder from %s (missing=%d unexpected=%d)",
        ckpt_path,
        len(missing),
        len(unexpected),
    )


def rebuild_backbone(spec):
    """Reconstruct a backbone from a saved ``model_spec`` (used by ``LitX.load``)."""
    cfg = spec["cfg"]
    nc = spec["n_channels"]
    nt = spec["n_times"]
    ncl = spec["n_classes"]
    if spec["kind"] == "sequence":
        return _make_sequence_backbone(cfg, nc, nt, ncl, load_pretrained=False)
    return _make_backbone(cfg, nc, nt, ncl)


def _wrap_kwargs(cfg, allowed):
    params = _to_plain(cfg)
    return {k: params[k] for k in allowed if k in params}


def build_model(cfg, n_channels, n_times, n_classes, class_weights=None) -> LitEEGClassifier:
    """Build a per-trial backbone and wrap it in :class:`LitEEGClassifier`."""
    backbone = _make_backbone(cfg, n_channels, n_times, n_classes)
    spec = {
        "kind": "trial",
        "cfg": _to_plain(cfg),
        "n_channels": n_channels,
        "n_times": n_times,
        "n_classes": n_classes,
    }
    wrap = _wrap_kwargs(cfg, {"lr", "weight_decay", "max_epochs"})
    return LitEEGClassifier(
        backbone, n_classes=n_classes, class_weights=class_weights, model_spec=spec, **wrap
    )


def build_sequence_model(
    cfg, n_channels, n_times, n_classes, class_weights=None, class_names=None
) -> LitSequenceClassifier:
    """Build a sequence backbone and wrap it in :class:`LitSequenceClassifier`.

    Encoder selection: ``encoder=foundation`` (+ ``foundation=mock|bendr|labram``) uses a
    wrapped foundation model; ``pretrained_encoder=<path>`` warm-starts the EpochEncoder from
    SSL; otherwise a fresh EpochEncoder is built.
    """
    backbone = _make_sequence_backbone(cfg, n_channels, n_times, n_classes, load_pretrained=True)
    spec = {
        "kind": "sequence",
        "cfg": _to_plain(cfg),
        "n_channels": n_channels,
        "n_times": n_times,
        "n_classes": n_classes,
    }
    wrap = _wrap_kwargs(cfg, _WRAP_KEYS)
    return LitSequenceClassifier(
        backbone,
        n_classes=n_classes,
        class_weights=class_weights,
        class_names=class_names,
        model_spec=spec,
        **wrap,
    )
