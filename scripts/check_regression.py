#!/usr/bin/env python
"""Model-regression gate for CI.

Fails (exit 1) if a key metric in the latest run dropped more than ``--tol`` below the
committed baseline. Wire this after training writes ``reports/sleep_metrics.json`` (or any
metrics JSON). Promote a new baseline only via an explicit, reviewed commit.

    python scripts/check_regression.py --baseline tests/baseline_metrics.json \
        --current reports/sleep_metrics.json --metric test/macro_f1 --tol 0.02
"""
from __future__ import annotations

import argparse
import json
import sys


def compare(baseline: dict, current: dict, metric: str, tol: float) -> tuple[bool, str]:
    """Return (ok, message). ok=False if current is worse than baseline by more than tol."""
    if metric not in baseline:
        return False, f"metric '{metric}' missing from baseline"
    if metric not in current:
        return False, f"metric '{metric}' missing from current run"
    b, c = float(baseline[metric]), float(current[metric])
    ok = c >= b - tol
    verdict = "OK" if ok else "REGRESSION"
    return ok, f"[{verdict}] {metric}: baseline={b:.4f} current={c:.4f} (tol={tol}, Δ={c - b:+.4f})"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--current", required=True)
    p.add_argument("--metric", default="test/macro_f1")
    p.add_argument("--tol", type=float, default=0.02)
    a = p.parse_args()
    ok, msg = compare(json.load(open(a.baseline)), json.load(open(a.current)), a.metric, a.tol)
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
