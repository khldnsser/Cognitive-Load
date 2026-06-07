"""IEP-2 — Calibration service.

Accepts ≥30s of known-rest physiological signals, extracts features from each
window, and returns per-subject mean/std for the 13 NORMALIZE_FEATURES.

Endpoints:
  POST /calibrate  — compute CalibrationParams from rest windows
  GET  /health     — liveness check
  GET  /metrics    — Prometheus metrics
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import (
    Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST,
)

from cogload.config import NORMALIZE_FEATURES
from cogload.features.extract import extract_window_features
from cogload.serving.schemas import (
    CalibrateRequest, CalibrateResponse, CalibrationParams, HealthResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT   = Counter("iep2_requests_total", "Total calibration requests", ["status"])
REQUEST_LATENCY = Histogram("iep2_request_latency_seconds", "Calibration latency",
                            buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
WINDOWS_USED    = Histogram("iep2_windows_used", "Windows used per calibration",
                            buckets=[1, 2, 4, 8, 16, 32])


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("IEP-2 calibration service starting")
    yield
    logger.info("IEP-2 calibration service stopping")


app = FastAPI(title="IEP-2 Calibration", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Error handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error in IEP-2: %s", exc)
    REQUEST_COUNT.labels(status="error").inc()
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _window_to_slices(
    window,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convert a SignalWindow to the three DataFrames expected by extractors."""
    bvp_df = pd.DataFrame({"bvp": window.bvp_values})

    eda_df = pd.DataFrame({
        "eda":  window.eda_values,
        "time": window.eda_times,
    })

    temp_df = pd.DataFrame({
        "temp": window.temp_values,
        "time": window.temp_times,
    })

    return bvp_df, eda_df, temp_df


def _compute_calibration_params(windows) -> tuple[dict, dict, int]:
    """Extract features from all windows and return (mu, sigma, n_windows)."""
    rows = []
    for w in windows:
        bvp_df, eda_df, temp_df = _window_to_slices(w)
        t_start = w.bvp_times[0] if w.bvp_times else 0.0
        row = extract_window_features(
            t_start=t_start,
            bvp_slice=bvp_df,
            eda_slice=eda_df,
            temp_slice=temp_df,
            subject=0,       # unknown at calibration time — not used
            label=0,         # rest condition
            cohort="live",
            session=None,
            condition="baseline",
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    present = [f for f in NORMALIZE_FEATURES if f in df.columns]

    mu: dict[str, float] = {}
    sigma: dict[str, float] = {}

    for feat in present:
        vals = df[feat].dropna()
        if len(vals) == 0:
            continue
        mu[feat] = float(vals.mean())
        sd = float(vals.std())
        # Guard against zero variance — use 1.0 so normalization is a no-op
        sigma[feat] = sd if sd > 1e-10 else 1.0

    return mu, sigma, len(rows)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/calibrate", response_model=CalibrateResponse)
async def calibrate(request: CalibrateRequest):
    start = time.perf_counter()
    try:
        mu, sigma, n_windows = _compute_calibration_params(request.windows)

        if not mu:
            REQUEST_COUNT.labels(status="error").inc()
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={"detail": "Feature extraction returned no valid values. Check signal quality."},
            )

        REQUEST_COUNT.labels(status="success").inc()
        WINDOWS_USED.observe(n_windows)

        return CalibrateResponse(
            params=CalibrationParams(mu=mu, sigma=sigma),
            windows_used=n_windows,
            features_computed=sorted(mu.keys()),
        )
    except Exception:
        REQUEST_COUNT.labels(status="error").inc()
        raise
    finally:
        REQUEST_LATENCY.observe(time.perf_counter() - start)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", service="iep2-calibration")


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
