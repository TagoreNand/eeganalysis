"""Pydantic request/response models — the typed contract for the inference API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    # (n_channels, n_times) for one trial, or (n_trials, n_channels, n_times)
    data: list = Field(..., description="Nested list of EEG samples (channels x times).")
    sfreq: float = Field(..., gt=0, description="Sampling frequency in Hz.")

    model_config = {"json_schema_extra": {
        "example": {"data": [[0.1, 0.2], [0.0, -0.1]], "sfreq": 128.0}}}


class ClassProbability(BaseModel):
    label: str
    probability: float


class PredictResponse(BaseModel):
    prediction: int
    label: str
    probabilities: list[ClassProbability]
    model_name: str
    model_version: str
    backend: str
    inference_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_classes: int | None = None


class VersionResponse(BaseModel):
    model_name: str
    model_version: str
    backend: str
    n_classes: int | None = None
