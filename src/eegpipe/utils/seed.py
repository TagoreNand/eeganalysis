"""Global determinism helpers.

Reproducibility in EEG matters doubly: small datasets mean a different seed can swing
cross-validation AUC by several points. Always log the seed alongside metrics.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Seed Python, NumPy and (if installed) PyTorch RNGs.

    Parameters
    ----------
    seed: base seed.
    deterministic: if True, force CuDNN into deterministic mode (slower, reproducible).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - intentionally seed the global legacy RNG
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed
