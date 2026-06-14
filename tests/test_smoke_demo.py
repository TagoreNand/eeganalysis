"""The end-to-end smoke demo runs without FAILs on a tiny synthetic dataset.

Stages needing torch/onnx SKIP gracefully (and still run in CI where they're installed);
the classical stages (windows, split, align, band-power baseline, regression gate) execute
here on installed deps, so this guards the whole orchestration end-to-end.
"""

import importlib.util
import pathlib


def _load_demo():
    p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "smoke_demo.py"
    spec = importlib.util.spec_from_file_location("smoke_demo", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_smoke_demo_no_failures(tmp_path):
    demo = _load_demo()
    rc = demo.main(["--quick", "--outdir", str(tmp_path)])
    assert rc == 0, [r for r in demo.RESULTS if r[1] == "FAIL"]
    stages = {name: status for name, status, _ in demo.RESULTS}
    # the dependency-light stages must actually PASS (not skip) here
    for s in ("data", "windows", "split", "classical"):
        assert stages[s] == "PASS", (s, stages[s])
