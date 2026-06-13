#!/usr/bin/env python
"""Convenience wrapper: ``python scripts/evaluate.py data=bci_iv_2a loso=true``."""
from eegpipe.training.evaluate import main

if __name__ == "__main__":
    main()
