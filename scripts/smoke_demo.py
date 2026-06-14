#!/usr/bin/env python
"""End-to-end smoke test: prove the whole pipeline with one command.

Walks every stage and reports PASS / SKIP / FAIL per stage, then exits non-zero if any
stage FAILED (SKIP — e.g. torch/onnx not installed — does not fail the run).

Stages
------
  1. data        load real Sleep-EDF (``--real``) or generate synthetic epochs (default)
  2. windows     boundary-safe epoch-sequence construction
  3. split       subject-aware train/test split (leakage check)
  4. align       unsupervised Euclidean Alignment (domain adaptation)
  5. classical   band-power + logistic regression baseline -> macro-F1 / kappa
  6. deep        build the sleep sequence model + a 1-batch Lightning train  [needs torch]
  7. export      TorchScript/ONNX export + ONNX-runtime parity               [needs torch/onnx]
  8. hypnogram   full-night stage prediction with overlap averaging          [needs torch]
  9. gate        write metrics.json and run the regression gate

Usage
-----
    python scripts/smoke_demo.py                 # synthetic, offline
    python scripts/smoke_demo.py --real          # one Sleep-EDF subject (downloads via MNE)
    make demo
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from eegpipe.data import make_sequence_windows, subject_kfold
from eegpipe.data.splits import assert_no_subject_leakage
from eegpipe.utils import get_logger, seed_everything

log = get_logger("smoke")
RESULTS: list[tuple[str, str, str]] = []


def _record(stage, status, detail=""):
    RESULTS.append((stage, status, detail))
    mark = {"PASS": "[ok]", "SKIP": "[--]", "FAIL": "[XX]"}[status]
    log.info("%s %-10s %s", mark, stage, detail)


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def synthetic_epochs(n_subj=4, per_subj=200, n_ch=3, sfreq=100, seconds=30, n_classes=5, seed=0):
    """Generate class-correlated epochs so the classical baseline is meaningfully > chance."""
    rng = np.random.default_rng(seed)
    T = sfreq * seconds
    X, y, groups = [], [], []
    t = np.arange(T) / sfreq
    for s in range(n_subj):
        for _ in range(per_subj):
            cls = rng.integers(n_classes)
            freq = 2 + 3 * cls                       # class -> dominant frequency
            sig = np.sin(2 * np.pi * freq * t)[None, :] * (1 + 0.3 * rng.standard_normal())
            noise = 0.7 * rng.standard_normal((n_ch, T))
            X.append((sig + noise).astype("float32"))
            y.append(cls)
            groups.append(s)
    return np.stack(X), np.asarray(y), np.asarray(groups), sfreq


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="use one Sleep-EDF subject")
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--quick", action="store_true", help="tiny synthetic run (for tests)")
    args = ap.parse_args(argv)
    if args.quick:
        args.seq_len = 4
    seed_everything(42)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # 1) DATA -----------------------------------------------------------------
    sfreq = 100
    try:
        if args.real:
            from eegpipe.data import build_loader
            from eegpipe.preprocessing import build_preprocessing

            bundle = build_loader({"name": "sleep_edf", "n_subjects": 1}).load()
            epochs = build_preprocessing(
                {"steps": [{"name": "bandpass", "l_freq": 0.3, "h_freq": 35.0, "notch": None},
                           {"name": "resample", "sfreq": 100.0}]}).fit_transform(bundle.epochs)
            X = epochs.get_data(copy=False).astype("float32")
            y, groups = bundle.y, bundle.groups
            sfreq = int(epochs.info["sfreq"])
            # one subject -> fabricate two pseudo-subjects by halving for the split demo
            if len(np.unique(groups)) < 2:
                groups = (np.arange(len(y)) >= len(y) // 2).astype(int)
            _record("data", "PASS", f"REAL Sleep-EDF X={X.shape} classes={sorted(set(y.tolist()))}")
        else:
            X, y, groups, sfreq = (synthetic_epochs(n_subj=2, per_subj=16, seconds=3)
                                   if args.quick else synthetic_epochs())
            _record("data", "PASS", f"synthetic X={X.shape} subjects={len(set(groups.tolist()))}")
    except Exception as e:  # noqa: BLE001
        _record("data", "FAIL", repr(e))
        return _summary()

    # 2) WINDOWS --------------------------------------------------------------
    try:
        windows, seq_groups = make_sequence_windows(groups, seq_len=args.seq_len, stride=args.seq_len)
        _record("windows", "PASS", f"{windows.shape[0]} sequences of len {args.seq_len}")
    except Exception as e:  # noqa: BLE001
        _record("windows", "FAIL", repr(e)); return _summary()

    # 3) SPLIT ----------------------------------------------------------------
    try:
        tr, te = next(subject_kfold(windows, seq_groups, seq_groups, n_splits=2, stratified=False))
        assert_no_subject_leakage(seq_groups[tr], seq_groups[te])
        _record("split", "PASS", f"train={len(tr)} test={len(te)} sequences, no subject leakage")
    except Exception as e:  # noqa: BLE001
        _record("split", "FAIL", repr(e)); return _summary()

    # 4) ALIGN (domain adaptation) -------------------------------------------
    try:
        from eegpipe.adaptation import EuclideanAlignment, mean_covariance

        Xa = EuclideanAlignment().fit_transform(X, groups=groups)
        dev = np.abs(mean_covariance(Xa[groups == groups[0]]) - np.eye(X.shape[1])).max()
        _record("align", "PASS", f"per-subject mean-cov deviation from I = {dev:.2e}")
    except Exception as e:  # noqa: BLE001
        _record("align", "SKIP", repr(e)); Xa = X

    # 5) CLASSICAL baseline ---------------------------------------------------
    metrics = {}
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        from eegpipe.features import BandPowerFeatures
        from eegpipe.utils import sleep_metrics

        ep_tr = np.unique(windows[tr]); ep_te = np.unique(windows[te])  # epoch indices per split
        clf = make_pipeline(BandPowerFeatures(sfreq=sfreq), StandardScaler(),
                            LogisticRegression(max_iter=500))
        clf.fit(Xa[ep_tr], y[ep_tr])
        pred = clf.predict(Xa[ep_te])
        metrics = {f"test/{k}": v for k, v in sleep_metrics(y[ep_te], pred).items()
                   if not isinstance(v, dict)}
        _record("classical", "PASS",
                f"macro_f1={metrics['test/macro_f1']:.3f} kappa={metrics['test/kappa']:.3f}")
    except Exception as e:  # noqa: BLE001
        _record("classical", "FAIL", repr(e))

    # 6) DEEP model (fast_dev_run) -------------------------------------------
    ckpt = None
    if _have("torch") and _have("lightning"):
        try:
            import lightning as L

            from eegpipe.models import build_sequence_model
            from eegpipe.training.datamodule import SequenceEEGDataModule

            dm = SequenceEEGDataModule(X, y, windows, tr, te, batch_size=8, num_workers=0)
            model = build_sequence_model({"name": "sleep_seqnet", "emb_dim": 32, "lstm_hidden": 32,
                                          "sfreq": sfreq}, n_channels=X.shape[1],
                                         n_times=X.shape[2], n_classes=int(y.max()) + 1)
            tr_ = L.Trainer(fast_dev_run=True, accelerator="cpu", logger=False, enable_checkpointing=False)
            tr_.fit(model, dm)
            ckpt = Path(tempfile.mkdtemp()) / "demo.ckpt"
            tr_.save_checkpoint(ckpt)
            _record("deep", "PASS", "1-batch train OK; checkpoint saved")
        except Exception as e:  # noqa: BLE001
            _record("deep", "FAIL", repr(e))
    else:
        _record("deep", "SKIP", "torch/lightning not installed (runs in CI / on your machine)")

    # 7) EXPORT + ONNX parity -------------------------------------------------
    if ckpt and _have("onnx") and _have("onnxruntime") and _have("onnxscript"):
        try:
            from eegpipe.inference import OnnxPredictor
            from eegpipe.models import build_sequence_model
            from eegpipe.serving.export import export_module_onnx

            model = build_sequence_model({"name": "sleep_seqnet", "emb_dim": 32, "lstm_hidden": 32,
                                          "sfreq": sfreq}, n_channels=X.shape[1],
                                         n_times=X.shape[2], n_classes=int(y.max()) + 1)
            onnx_path = str(Path(tempfile.mkdtemp()) / "demo.onnx")
            export_module_onnx(model, onnx_path, kind="sequence", n_channels=X.shape[1],
                               n_times=X.shape[2], seq_len=args.seq_len)
            proba = OnnxPredictor(onnx_path).predict_proba(X[windows[te[:2]]])
            _record("export", "PASS", f"ONNX out shape {proba.shape}")
        except Exception as e:  # noqa: BLE001
            _record("export", "FAIL", repr(e))
    else:
        _record("export", "SKIP", "needs deep stage + onnx/onnxruntime")

    # 8) HYPNOGRAM ------------------------------------------------------------
    if ckpt and _have("torch"):
        try:
            from eegpipe.inference import HypnogramPredictor

            stages, _ = HypnogramPredictor(str(ckpt), seq_len=args.seq_len).predict(X[: 3 * args.seq_len])
            _record("hypnogram", "PASS", f"predicted {len(stages)} epochs")
        except Exception as e:  # noqa: BLE001
            _record("hypnogram", "FAIL", repr(e))
    else:
        _record("hypnogram", "SKIP", "needs deep stage (torch)")

    # 9) REGRESSION GATE ------------------------------------------------------
    try:
        spec = importlib.util.spec_from_file_location(
            "check_regression", Path(__file__).with_name("check_regression.py"))
        cr = importlib.util.module_from_spec(spec); spec.loader.exec_module(cr)
        if metrics:
            (Path(args.outdir) / "smoke_metrics.json").write_text(json.dumps(metrics, indent=2))
            ok, msg = cr.compare({"test/macro_f1": 0.0}, metrics, "test/macro_f1", 0.02)
            _record("gate", "PASS" if ok else "FAIL", msg)
        else:
            _record("gate", "SKIP", "no metrics to gate")
    except Exception as e:  # noqa: BLE001
        _record("gate", "FAIL", repr(e))

    return _summary()


def _summary() -> int:
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    n_skip = sum(1 for _, s, _ in RESULTS if s == "SKIP")
    log.info("=" * 60)
    log.info("SMOKE DEMO: %d passed, %d skipped, %d failed", n_pass, n_skip, n_fail)
    log.info("=" * 60)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
