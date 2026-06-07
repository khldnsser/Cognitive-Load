"""EEP — External API Gateway.

The single public-facing service. Validates all inputs, rate-limits callers,
orchestrates IEP-2 (calibration) and IEP-1 (inference), handles timeouts and
retries, and returns a clean response to the client.

Endpoints:
  POST /calibrate  — proxy to IEP-2
  POST /predict    — proxy to IEP-1 (optionally pre-calling IEP-2)
  GET  /health     — aggregated health across EEP + IEP-1 + IEP-2
  GET  /metrics    — Prometheus metrics
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from cogload.serving.schemas import (
    CalibrateRequest, CalibrateResponse,
    PredictRequest, PredictResponse,
    HealthResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config from environment (overridable for Docker / K8s)
# ---------------------------------------------------------------------------
IEP1_URL     = os.environ.get("IEP1_URL", "http://localhost:8000")
IEP2_URL     = os.environ.get("IEP2_URL", "http://localhost:8001")
TIMEOUT_S    = float(os.environ.get("IEP_TIMEOUT_S", "10.0"))
MAX_RETRIES  = int(os.environ.get("IEP_MAX_RETRIES", "2"))

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT   = Counter("eep_requests_total", "Total EEP requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram(
    "eep_request_latency_seconds", "EEP end-to-end latency",
    ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)
DOWNSTREAM_ERRORS = Counter("eep_downstream_errors_total", "IEP call failures", ["iep"])

# ---------------------------------------------------------------------------
# Rate limiter — 60 requests/minute per IP
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


# ---------------------------------------------------------------------------
# HTTP client (shared, connection-pooled)
# ---------------------------------------------------------------------------
_http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client
    logger.info("EEP gateway starting — IEP1=%s  IEP2=%s", IEP1_URL, IEP2_URL)
    _http_client = httpx.AsyncClient(timeout=TIMEOUT_S)
    yield
    await _http_client.aclose()
    logger.info("EEP gateway stopping")


app = FastAPI(title="EEP Gateway", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ---------------------------------------------------------------------------
# Error handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error in EEP: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# IEP call helpers with retry logic
# ---------------------------------------------------------------------------

async def _call_iep(
    iep_name: str,
    url: str,
    payload: dict,
) -> tuple[int, dict]:
    """POST to an IEP with retries. Returns (status_code, body_dict)."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = await _http_client.post(url, json=payload)
            return resp.status_code, resp.json()
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            last_exc = exc
            logger.warning("%s attempt %d/%d failed: %s", iep_name, attempt + 1, MAX_RETRIES + 1, exc)

    DOWNSTREAM_ERRORS.labels(iep=iep_name).inc()
    raise RuntimeError(f"{iep_name} unreachable after {MAX_RETRIES + 1} attempts: {last_exc}")


async def _iep_health(url: str) -> dict:
    try:
        resp = await _http_client.get(f"{url}/health", timeout=3.0)
        return resp.json()
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/calibrate", response_model=CalibrateResponse)
@limiter.limit("60/minute")
async def calibrate(request: Request, body: CalibrateRequest):
    start = time.perf_counter()
    try:
        status_code, data = await _call_iep(
            "iep2", f"{IEP2_URL}/calibrate", body.model_dump()
        )
        REQUEST_COUNT.labels(endpoint="calibrate", status="success" if status_code == 200 else "error").inc()
        return JSONResponse(status_code=status_code, content=data)
    except RuntimeError as exc:
        REQUEST_COUNT.labels(endpoint="calibrate", status="error").inc()
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": str(exc)},
        )
    finally:
        REQUEST_LATENCY.labels(endpoint="calibrate").observe(time.perf_counter() - start)


@app.post("/predict", response_model=PredictResponse)
@limiter.limit("60/minute")
async def predict(request: Request, body: PredictRequest):
    start = time.perf_counter()
    try:
        status_code, data = await _call_iep(
            "iep1", f"{IEP1_URL}/predict", body.model_dump()
        )
        REQUEST_COUNT.labels(endpoint="predict", status="success" if status_code == 200 else "error").inc()
        return JSONResponse(status_code=status_code, content=data)
    except RuntimeError as exc:
        REQUEST_COUNT.labels(endpoint="predict", status="error").inc()
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": str(exc)},
        )
    finally:
        REQUEST_LATENCY.labels(endpoint="predict").observe(time.perf_counter() - start)


@app.get("/health", response_model=HealthResponse)
async def health():
    iep1_health = await _iep_health(IEP1_URL)
    iep2_health = await _iep_health(IEP2_URL)

    iep1_ok = iep1_health.get("status") == "ok"
    iep2_ok = iep2_health.get("status") == "ok"

    if iep1_ok and iep2_ok:
        overall = "ok"
    elif iep1_ok or iep2_ok:
        overall = "degraded"
    else:
        overall = "error"

    return HealthResponse(
        status=overall,
        service="eep-gateway",
        details={"iep1": iep1_health, "iep2": iep2_health},
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)
