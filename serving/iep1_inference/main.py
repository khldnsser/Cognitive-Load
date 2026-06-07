"""IEP-1 — Inference service.

Loads the champion FoldEnsembleModel at startup, runs feature extraction on
incoming signal windows, applies per-subject normalization via CalibrationParams,
and returns a binary cognitive-load prediction with probability.

Endpoints:
  POST /predict  — run inference on one 60s window
  GET  /health   — liveness check (includes model-loaded status)
  GET  /metrics  — Prometheus metrics
"""
from __future__ import annotations

import logging
import math
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import (
    Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST,
)

from cogload.config import NORMALIZE_FEATURES
from cogload.features.extract import extract_window_features, FEATURE_COLS
from cogload.models.ensemble import FoldEnsembleModel
from cogload.serving.model_loader import load_champion
from cogload.serving.schemas import (
    PredictRequest, PredictResponse, HealthResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT   = Counter("iep1_requests_total", "Total prediction requests", ["status"])
REQUEST_LATENCY = Histogram(
    "iep1_request_latency_seconds", "Prediction latency",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)
CONFIDENCE_HIST = Histogram(
    "iep1_prediction_confidence", "Ensemble probability (ML-specific signal)",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
DRIFT_GAUGE = Histogram(
    "iep1_input_drift_score",
    "Mean |z| of normalized features — proxy for calibration staleness / signal drift",
    buckets=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
)
UNCALIBRATED_COUNT = Counter("iep1_uncalibrated_requests_total", "Requests without calibration params")


# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

_model: Optional[FoldEnsembleModel] = None
_model_load_error: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_load_error
    logger.info("IEP-1 inference service starting — loading champion model")
    try:
        _model = load_champion()
        logger.info("Champion model loaded successfully")
    except Exception as exc:
        _model_load_error = str(exc)
        logger.error("Failed to load champion model: %s", exc)
    yield
    logger.info("IEP-1 inference service stopping")


app = FastAPI(title="IEP-1 Inference", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Error handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error in IEP-1: %s", exc)
    REQUEST_COUNT.labels(status="error").inc()
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _window_to_slices(window) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bvp_df = pd.DataFrame({"bvp": window.bvp_values})
    eda_df  = pd.DataFrame({"eda": window.eda_values, "time": window.eda_times})
    temp_df = pd.DataFrame({"temp": window.temp_values, "time": window.temp_times})
    return bvp_df, eda_df, temp_df


def _apply_calibration(features: dict, calib) -> tuple[dict, float]:
    """Normalize features in-place using CalibrationParams. Returns (features, drift_score).

    drift_score = mean |z| across normalized features.  Values near 1.0 are
    expected (standard normal).  Values >> 2 indicate stale calibration or
    significant signal drift.
    """
    z_vals = []
    for feat in NORMALIZE_FEATURES:
        if feat not in features or feat not in calib.mu:
            continue
        val = features[feat]
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        mu = calib.mu[feat]
        sd = calib.sigma.get(feat, 1.0)
        z = (val - mu) / (sd if sd > 1e-10 else 1.0)
        features[feat] = z
        z_vals.append(abs(z))

    drift_score = float(np.mean(z_vals)) if z_vals else float("nan")
    return features, drift_score


def _predict(model: FoldEnsembleModel, features: dict) -> tuple[float, int]:
    """Run ensemble predict_proba and apply threshold."""
    row = {f: features.get(f, float("nan")) for f in FEATURE_COLS}
    X = pd.DataFrame([row])
    prob = float(model.predict_proba(X)[0])
    pred = int(prob >= model.threshold)
    return prob, pred


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    start = time.perf_counter()

    if _model is None:
        REQUEST_COUNT.labels(status="error").inc()
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": f"Model not loaded: {_model_load_error}"},
        )

    try:
        bvp_df, eda_df, temp_df = _window_to_slices(request.window)
        t_start = request.window.bvp_times[0] if request.window.bvp_times else 0.0

        features = extract_window_features(
            t_start=t_start,
            bvp_slice=bvp_df,
            eda_slice=eda_df,
            temp_slice=temp_df,
            subject=0,
            label=-1,          # unknown at inference time
            cohort="live",
            session=None,
            condition="unknown",
        )

        calibrated = False
        drift_score = None

        if request.calib is not None:
            features, drift_score = _apply_calibration(features, request.calib)
            calibrated = True
            if not math.isnan(drift_score):
                DRIFT_GAUGE.observe(drift_score)
        else:
            UNCALIBRATED_COUNT.inc()

        prob, pred = _predict(_model, features)

        CONFIDENCE_HIST.observe(prob)
        REQUEST_COUNT.labels(status="success").inc()

        return PredictResponse(
            probability=prob,
            prediction=pred,
            threshold=_model.threshold,
            calibrated=calibrated,
            drift_score=drift_score if (drift_score is not None and not math.isnan(drift_score)) else None,
        )

    except Exception:
        REQUEST_COUNT.labels(status="error").inc()
        raise
    finally:
        REQUEST_LATENCY.observe(time.perf_counter() - start)


@app.get("/health", response_model=HealthResponse)
async def health():
    if _model is None:
        return HealthResponse(
            status="error",
            service="iep1-inference",
            details={"error": _model_load_error},
        )
    return HealthResponse(
        status="ok",
        service="iep1-inference",
        details={"fold_models": len(_model._models), "threshold": _model.threshold},
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
