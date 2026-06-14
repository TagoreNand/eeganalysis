"""Regression-gate logic and (optional) API smoke tests."""

import importlib.util
import pathlib

import pytest


def _load_check_regression():
    p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "check_regression.py"
    spec = importlib.util.spec_from_file_location("check_regression", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_regression_ok_when_metric_holds():
    cr = _load_check_regression()
    ok, _ = cr.compare({"test/macro_f1": 0.70}, {"test/macro_f1": 0.70}, "test/macro_f1", 0.02)
    assert ok


def test_regression_fails_on_drop_beyond_tol():
    cr = _load_check_regression()
    ok, msg = cr.compare({"test/macro_f1": 0.70}, {"test/macro_f1": 0.60}, "test/macro_f1", 0.02)
    assert not ok and "REGRESSION" in msg


def test_regression_within_tolerance_passes():
    cr = _load_check_regression()
    ok, _ = cr.compare({"test/macro_f1": 0.70}, {"test/macro_f1": 0.69}, "test/macro_f1", 0.02)
    assert ok


def test_api_health_and_readiness():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from eegpipe.serving.api import app

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503  # no model loaded in test env
        assert client.post("/predict", json={"data": [[0.0]], "sfreq": 100.0}).status_code == 503
