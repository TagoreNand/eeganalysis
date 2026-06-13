#!/usr/bin/env python
"""Convenience wrapper: ``python scripts/train.py model=conformer``.
Identical to the ``eegpipe-train`` console entry point."""
from eegpipe.training.train import main

if __name__ == "__main__":
    main()
