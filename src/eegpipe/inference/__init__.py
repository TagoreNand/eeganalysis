from eegpipe.inference.predictor import Predictor

__all__ = [
    "Predictor",
    "OnnxPredictor",
    "LSLStreamProcessor",
    "HypnogramPredictor",
    "plot_hypnogram",
]


def __getattr__(name):  # lazy imports so heavy/optional deps load only when used
    if name == "OnnxPredictor":
        from eegpipe.inference.onnx_predictor import OnnxPredictor

        return OnnxPredictor
    if name == "LSLStreamProcessor":
        from eegpipe.inference.stream import LSLStreamProcessor

        return LSLStreamProcessor
    if name in ("HypnogramPredictor", "plot_hypnogram"):
        import eegpipe.inference.hypnogram as h

        return getattr(h, name)
    raise AttributeError(name)
