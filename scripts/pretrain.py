#!/usr/bin/env python
"""Convenience wrapper:
    python scripts/pretrain.py data=sleep_edf model=sleep_seqnet training=pretrain preprocess=sleep
"""
from eegpipe.training.pretrain import main

if __name__ == "__main__":
    main()
