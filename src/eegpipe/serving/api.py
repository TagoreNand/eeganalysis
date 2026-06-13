"""Hardened FastAPI inference service.

Endpoints:
  GET  /health   -> liveness (always 200 once the process is up)
  GET  /ready    -> readiness (200 only when a model is loaded)
  GET  /version  -> model name/version/backend
  POST /predict  -> class probabilities for one or many trials

Backends, in priority order:
  * ``EEGPIPE_ONNX``  -> ONNX Runtime (fast, no torch)         [preferred for prod]
  * ``EEGPIPE_CKPT``  -> PyTorch Lightning checkpoint

Hardening: request-timing header, optional API-key auth (``EEGPIPE_API_KEY``), input-size
guard (``EEGPIPE_MAX_ELEMENTS``), structured logging, and clean 503/413/401 responses.
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from eegpipe.serving.schemas import (
    ClassProbability,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    VersionResponse,
)
from eegpipe.utils import get_logger

log = get_logger(__name__)

MAX_ELEMENTS = int(os.environ.get("EEGPIPE_MAX_ELEMENTS", 5_000_000))
_STATE: dict = {}


def _load_backend():
    """Return (predictor, backend_name) from env, preferring ONNX."""
    onnx = os.environ.get("EEGPIPE_ONNX")
    ckpt = os.environ.get("EEGPIPE_CKPT")
    if onnx and os.path.exists(onnx):
        from eegpipe.inference.onnx_predictor import OnnxPredictor

        return OnnxPredictor(onnx), "onnx"
    if ckpt and os.path.exists(ckpt):
        from eegpipe.inference import Predictor

        return Predictor(ckpt), "torch"
    return None, "none"


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor, backend = _load_backend()
    _STATE["predictor"] = predictor
    _STATE["backend"] = backend
    _STATE["name"] = os.environ.get("EEGPIPE_MODEL_NAME", "eegpipe-model")
    _STATE["version"] = os.environ.get("EEGPIPE_MODEL_VERSION", "0.1.0")
    if predictor is not None:
        n = predictor.n_classes or 0
        _STATE["labels"] = getattr(predictor, "class_names", None) or [f"class_{i}" for i in range(n)]
        log.info("Loaded %s backend (%s classes)", backend, n)
    else:
        _STATE["labels"] = []
        log.warning("No model loaded; set EEGPIPE_ONNX or EEGPIPE_CKPT. /predict will 503.")
    yield
    _STATE.clear()


app = FastAPI(title="eegpipe inference API", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def add_process_time(request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
    return response


def require_api_key(x_api_key: str | None = Header(default=None)):
    expected = os.environ.get("EEGPIPE_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(401, "Invalid or missing X-API-Key header.")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    p = _STATE.get("predictor")
    return HealthResponse(status="ok", model_loaded=p is not None,
                          n_classes=(p.n_classes if p else None))


@app.get("/ready")
def ready():
    if _STATE.get("predictor") is None:
        return JSONResponse(status_code=503, content={"status": "not ready"})
    return {"status": "ready"}


@app.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    p = _STATE.get("predictor")
    return VersionResponse(model_name=_STATE.get("name", ""), model_version=_STATE.get("version", ""),
                           backend=_STATE.get("backend", "none"), n_classes=(p.n_classes if p else None))


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
def predict(req: PredictRequest) -> PredictResponse:
    predictor = _STATE.get("predictor")
    if predictor is None:
        raise HTTPException(503, "Model not loaded. Set EEGPIPE_ONNX or EEGPIPE_CKPT and restart.")
    X = np.asarray(req.data, dtype="float32")
    if X.size > MAX_ELEMENTS:
        raise HTTPException(413, f"Input has {X.size} elements > limit {MAX_ELEMENTS}.")
    if X.ndim == 2:
        X = X[None]
    if X.ndim != 3:
        raise HTTPException(422, f"Expected 2D or 3D data, got shape {X.shape}.")

    t0 = time.perf_counter()
    proba = predictor.predict_proba(X)[0]
    dt = (time.perf_counter() - t0) * 1000
    labels = _STATE["labels"] or [f"class_{i}" for i in range(len(proba))]
    top = int(np.argmax(proba))
    return PredictResponse(
        prediction=top, label=labels[top],
        probabilities=[ClassProbability(label=l, probability=float(p)) for l, p in zip(labels, proba)],
        model_name=_STATE.get("name", ""), model_version=_STATE.get("version", ""),
        backend=_STATE.get("backend", "none"), inference_ms=round(dt, 2))


def run():  # console-script entry point: eegpipe-serve
    import uvicorn

    uvicorn.run("eegpipe.serving.api:app", host="0.0.0.0", port=8000, reload=False)
