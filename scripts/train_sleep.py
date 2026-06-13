#!/usr/bin/env python
"""Convenience wrapper for sleep-stage training:
    python scripts/train_sleep.py data=sleep_edf model=sleep_seqnet training=sleep preprocess=sleep
"""
from eegpipe.training.sleep import main

if __name__ == "__main__":
    main()
