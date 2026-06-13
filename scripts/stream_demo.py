#!/usr/bin/env python
"""Real-time BCI demo: classify a live LSL stream with a trained checkpoint.

    python scripts/stream_demo.py path/to/model.ckpt

Tip: replay a recording to an LSL outlet first (e.g. with the `mne-lsl` player) to test
without hardware.
"""
import sys

from eegpipe.inference import LSLStreamProcessor, Predictor

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/stream_demo.py <checkpoint.ckpt>")
    LSLStreamProcessor(Predictor(sys.argv[1])).run(max_seconds=60)
