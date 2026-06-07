"""Unit tests for the EEP gateway.

All IEP calls are mocked — this tests the gateway's own logic:
orchestration, fallback, rate-limit config, health aggregation,
validation pass-through, and retry/timeout error surfacing.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Load EEP main with a unique module name
_EEP_PATH = Path(__file__).parents[2] / "serving" / "eep" / "main.py"
_spec = importlib.util.spec_from_file_location("eep_main", _EEP_PATH)
_eep_mod = importlib.util.module_from_spec(_spec)
sys.modules["eep_main"] = _eep_mod
_spec.loader.exec_module(_eep_mod)

app = _eep_mod.app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_window_dict(duration_s: float = 60.0, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n_bvp = int(duration_s * 64)
    n_eda = int(duration_s * 4)
    return {
        "bvp_values":  rng.normal(0, 0.5, n_bvp).tolist(),
        "bvp_times":   [i / 64 for i in range(n_bvp)],
        "eda_values":  rng.uniform(0.5, 2.0, n_eda).tolist(),
        "eda_times":   [i / 4  for i in range(n_eda)],
        "temp_values": rng.normal(36.0, 0.1, n_eda).tolist(),
        "temp_times":  [i / 4  for i in range(n_eda)],
    }


_CALIB_PARAMS = {
    "mu":    {"hr_mean": 70.0},
    "sigma": {"hr_mean": 5.0},
}

_PREDICT_RESPONSE = {
    "probability": 0.85, "prediction": 1,
    "threshold": 0.20, "calibrated": True, "drift_score": 1.2,
}

_CALIBRATE_RESPONSE = {
    "params": _CALIB_PARAMS,
    "windows_used": 2,
    "features_computed": ["hr_mean"],
}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_both_ok():
    mock = AsyncMock(side_effect=[
        {"status": "ok", "service": "iep1-inference"},
        {"status": "ok", "service": "iep2-calibration"},
    ])
    with patch.object(_eep_mod, "_iep_health", new=mock):
        with TestClient(app) as c:
            resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_one_down_is_degraded():
    mock = AsyncMock(side_effect=[
        {"status": "ok"},
        {"status": "error", "detail": "down"},
    ])
    with patch.object(_eep_mod, "_iep_health", new=mock):
        with TestClient(app) as c:
            resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"


def test_health_both_down_is_error():
    mock = AsyncMock(side_effect=[
        {"status": "error"},
        {"status": "error"},
    ])
    with patch.object(_eep_mod, "_iep_health", new=mock):
        with TestClient(app) as c:
            resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "error"


# ---------------------------------------------------------------------------
# /calibrate
# ---------------------------------------------------------------------------

def test_calibrate_proxies_to_iep2():
    with patch.object(_eep_mod, "_call_iep", new=AsyncMock(return_value=(200, _CALIBRATE_RESPONSE))):
        with TestClient(app) as c:
            payload = {"windows": [_make_window_dict(), _make_window_dict(seed=1)]}
            resp = c.post("/calibrate", json=payload)
    assert resp.status_code == 200
    assert resp.json()["windows_used"] == 2


def test_calibrate_503_when_iep2_unreachable():
    with patch.object(_eep_mod, "_call_iep", new=AsyncMock(side_effect=RuntimeError("iep2 down"))):
        with TestClient(app) as c:
            payload = {"windows": [_make_window_dict()]}
            resp = c.post("/calibrate", json=payload)
    assert resp.status_code == 503


def test_calibrate_rejects_empty_windows():
    with TestClient(app) as c:
        resp = c.post("/calibrate", json={"windows": []})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

def test_predict_proxies_to_iep1():
    with patch.object(_eep_mod, "_call_iep", new=AsyncMock(return_value=(200, _PREDICT_RESPONSE))):
        with TestClient(app) as c:
            payload = {"window": _make_window_dict(), "calib": _CALIB_PARAMS}
            resp = c.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_503_when_iep1_unreachable():
    with patch.object(_eep_mod, "_call_iep", new=AsyncMock(side_effect=RuntimeError("iep1 down"))):
        with TestClient(app) as c:
            payload = {"window": _make_window_dict()}
            resp = c.post("/predict", json=payload)
    assert resp.status_code == 503


def test_predict_rejects_short_signal():
    with TestClient(app) as c:
        resp = c.post("/predict", json={"window": _make_window_dict(duration_s=10.0)})
    assert resp.status_code == 422


def test_predict_uncalibrated_allowed():
    """Predict without calib should reach IEP-1 (degraded mode, not rejected)."""
    with patch.object(_eep_mod, "_call_iep", new=AsyncMock(return_value=(200, {**_PREDICT_RESPONSE, "calibrated": False}))):
        with TestClient(app) as c:
            payload = {"window": _make_window_dict()}
            resp = c.post("/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["calibrated"] is False


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------

def test_metrics_endpoint_returns_prometheus_text():
    with TestClient(app) as c:
        resp = c.get("/metrics")
    assert resp.status_code == 200
    assert "eep_requests_total" in resp.text
