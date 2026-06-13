"""Real-time inference over a Lab Streaming Layer (LSL) source.

Consumes a live EEG stream (OpenBCI, g.tec, BrainFlow->LSL, or a replayed file), keeps a
sliding window, and emits a prediction every ``step`` seconds. This is the backbone of a
local closed-loop BCI. Marker/output predictions are pushed to a second LSL stream so a
game or neurofeedback UI can subscribe.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from eegpipe.utils import get_logger

log = get_logger(__name__)


class LSLStreamProcessor:
    def __init__(self, predictor, window_s: float = 4.0, step_s: float = 0.25,
                 stream_type: str = "EEG"):
        self.predictor = predictor
        self.window_s, self.step_s = window_s, step_s
        self.stream_type = stream_type

    def _connect(self):
        from pylsl import StreamInfo, StreamInlet, StreamOutlet, resolve_byprop

        log.info("Resolving LSL %s stream...", self.stream_type)
        streams = resolve_byprop("type", self.stream_type, timeout=10)
        if not streams:
            raise RuntimeError(f"No LSL stream of type '{self.stream_type}' found.")
        inlet = StreamInlet(streams[0], max_buflen=int(self.window_s) + 1)
        info = inlet.info()
        sfreq, n_ch = info.nominal_srate(), info.channel_count()
        out_info = StreamInfo("eegpipe_pred", "Markers", 1, sfreq, "float32", "eegpipe-out")
        return inlet, StreamOutlet(out_info), sfreq, n_ch

    def run(self, max_seconds: float | None = None):
        inlet, outlet, sfreq, n_ch = self._connect()
        win = int(self.window_s * sfreq)
        buf: deque[list[float]] = deque(maxlen=win)
        t0, last = time.time(), 0.0
        log.info("Streaming @ %g Hz, %d ch, window=%ds", sfreq, n_ch, self.window_s)
        while max_seconds is None or time.time() - t0 < max_seconds:
            sample, _ = inlet.pull_sample(timeout=1.0)
            if sample is not None:
                buf.append(sample)
            if len(buf) == win and (time.time() - last) >= self.step_s:
                X = np.asarray(buf, dtype="float32").T[None]  # (1, n_ch, win)
                proba = self.predictor.predict_proba(X)[0]
                pred = int(proba.argmax())
                outlet.push_sample([float(pred)])
                log.info("pred=%d  p=%s", pred, np.round(proba, 3).tolist())
                last = time.time()


# TODO(roadmap-§3): add online artefact gating (skip windows with high-amplitude artefacts),
#   exponential smoothing of consecutive predictions, and a calibration phase per session.
