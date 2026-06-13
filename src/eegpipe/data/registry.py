"""Tiny registry + factory for data loaders (Factory pattern).

Decorate a loader class with ``@register_loader("name")`` and it becomes constructible
from a Hydra config via ``build_loader(cfg.data)``.
"""
from __future__ import annotations

from typing import Type

LOADER_REGISTRY: dict[str, Type] = {}


def register_loader(name: str):
    def _wrap(cls):
        if name in LOADER_REGISTRY:
            raise KeyError(f"Loader '{name}' already registered")
        LOADER_REGISTRY[name] = cls
        return cls

    return _wrap


def build_loader(cfg) -> "object":
    """Instantiate a loader from a config mapping with a ``name`` key.

    Accepts a dict or an OmegaConf node: ``{name: bci_iv_2a, fmin: 8, fmax: 32}``.
    """
    params = dict(cfg)
    name = params.pop("name")
    if name not in LOADER_REGISTRY:
        raise KeyError(f"Unknown loader '{name}'. Registered: {sorted(LOADER_REGISTRY)}")
    return LOADER_REGISTRY[name](**params)
