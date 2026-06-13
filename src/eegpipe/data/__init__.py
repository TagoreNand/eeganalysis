from eegpipe.data.base import BaseDataLoader, EpochsBundle
from eegpipe.data.registry import LOADER_REGISTRY, build_loader, register_loader

# Import concrete loaders so their @register_loader decorators run on package import.
from eegpipe.data import loaders  # noqa: E402,F401
from eegpipe.data.splits import (  # noqa: E402
    assert_no_subject_leakage,
    leave_one_subject_out,
    nested_subject_cv,
    subject_kfold,
)
from eegpipe.data.sequence import (  # noqa: E402
    make_sequence_windows,
    sequence_sampler_weights,
)

__all__ = [
    "BaseDataLoader",
    "EpochsBundle",
    "build_loader",
    "register_loader",
    "LOADER_REGISTRY",
    "subject_kfold",
    "leave_one_subject_out",
    "nested_subject_cv",
    "assert_no_subject_leakage",
    "make_sequence_windows",
    "sequence_sampler_weights",
]
