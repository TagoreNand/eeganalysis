"""Export a trained checkpoint to ONNX / TorchScript for low-latency, dependency-light serving.

ONNX Runtime is typically 2-5x faster than eager PyTorch on CPU and drops the Lightning /
Python-class dependency at inference time. Batch (and sequence length, for sequence models)
are exported as dynamic axes so one artefact serves any batch size.

CLI (entry point ``eegpipe-export``)::

    eegpipe-export model.ckpt model.onnx --kind trial --channels 22 --times 256
    eegpipe-export sleep.ckpt sleep.onnx --kind sequence --channels 2 --times 3000 --seq-len 20
"""

from __future__ import annotations

import argparse

from eegpipe.utils import get_logger

log = get_logger(__name__)


def _load(ckpt: str, kind: str, device: str):
    from eegpipe.models.base import LitEEGClassifier, LitSequenceClassifier

    cls = LitSequenceClassifier if kind == "sequence" else LitEEGClassifier
    return cls.load_from_checkpoint(ckpt, map_location=device).eval().to(device)


def _dummy(kind, n_channels, n_times, seq_len, device):
    import torch

    if kind == "sequence":
        return torch.randn(1, seq_len, n_channels, n_times, device=device)
    return torch.randn(1, n_channels, n_times, device=device)


def export_module_onnx(
    model, out, kind="trial", n_channels=22, n_times=256, seq_len=20, device="cpu", opset=17
):
    """Export an already-loaded ``nn.Module`` to ONNX (checkpoint-independent; unit-testable)."""
    import torch

    model = model.eval().to(device)
    dummy = _dummy(kind, n_channels, n_times, seq_len, device)
    if kind == "sequence":
        dyn = {"input": {0: "batch", 1: "seq"}, "output": {0: "batch", 1: "seq"}}
    else:
        dyn = {"input": {0: "batch"}, "output": {0: "batch"}}
    torch.onnx.export(
        model,
        dummy,
        out,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dyn,
        opset_version=opset,
    )
    log.info("Exported ONNX -> %s (kind=%s, opset=%d)", out, kind, opset)
    return out


def export_onnx(
    ckpt, out, kind="trial", n_channels=22, n_times=256, seq_len=20, device="cpu", opset=17
):
    return export_module_onnx(
        _load(ckpt, kind, device), out, kind, n_channels, n_times, seq_len, device, opset
    )


def export_torchscript(
    ckpt, out, kind="trial", n_channels=22, n_times=256, seq_len=20, device="cpu"
):
    import torch

    model = _load(ckpt, kind, device)
    with torch.no_grad():
        scripted = torch.jit.trace(model, _dummy(kind, n_channels, n_times, seq_len, device))
    scripted.save(out)
    log.info("Exported TorchScript -> %s", out)
    return out


def main():
    p = argparse.ArgumentParser(description="Export an eegpipe checkpoint.")
    p.add_argument("ckpt")
    p.add_argument("out")
    p.add_argument("--format", choices=["onnx", "torchscript"], default="onnx")
    p.add_argument("--kind", choices=["trial", "sequence"], default="trial")
    p.add_argument("--channels", type=int, default=22)
    p.add_argument("--times", type=int, default=256)
    p.add_argument("--seq-len", type=int, default=20)
    a = p.parse_args()
    fn = export_onnx if a.format == "onnx" else export_torchscript
    fn(a.ckpt, a.out, kind=a.kind, n_channels=a.channels, n_times=a.times, seq_len=a.seq_len)


if __name__ == "__main__":
    main()
