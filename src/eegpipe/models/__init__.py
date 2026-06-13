"""Model zoo. Importing this subpackage requires the DL extra: pip install 'eegpipe[dl]'."""
try:
    import torch  # noqa: F401

    from eegpipe.models.base import LitEEGClassifier, LitSequenceClassifier
    from eegpipe.models.conformer import EEGConformer
    from eegpipe.models.eegnet import EEGNet
    from eegpipe.models.foundation import (
        FoundationEncoderAdapter,
        MockFoundationEncoder,
        build_foundation_encoder,
    )
    from eegpipe.models.registry import (
        BACKBONES,
        SEQUENCE_BACKBONES,
        build_model,
        build_sequence_model,
        load_pretrained_encoder,
    )
    from eegpipe.models.sleep import EpochEncoder, SleepTransformer, TinySleepNet, UTime
    from eegpipe.models.stgcn import STGCN, adjacency_from_positions

    __all__ = [
        "LitEEGClassifier", "LitSequenceClassifier", "EEGNet", "EEGConformer", "STGCN",
        "TinySleepNet", "SleepTransformer", "UTime", "EpochEncoder",
        "FoundationEncoderAdapter", "MockFoundationEncoder", "build_foundation_encoder",
        "adjacency_from_positions", "build_model", "build_sequence_model",
        "load_pretrained_encoder", "BACKBONES", "SEQUENCE_BACKBONES",
    ]
except ImportError as _e:
    _IMPORT_ERROR = _e

    def build_model(*_a, **_k):  # type: ignore
        raise ImportError("Deep-learning extra required: pip install 'eegpipe[dl]'") from _IMPORT_ERROR

    def build_sequence_model(*_a, **_k):  # type: ignore
        raise ImportError("Deep-learning extra required: pip install 'eegpipe[dl]'") from _IMPORT_ERROR

    __all__ = ["build_model", "build_sequence_model"]
