"""Integration tests — hit live services via HTTP.

Run with `docker compose up -d` first; tests skip automatically if services
are unreachable. Model-dependent tests additionally skip if IEP-1 hasn't
loaded the champion (or CI test fixture) model.
"""
from __future__ import annotations

import httpx
import pytest

from tests.integration.conftest import (
    IEP1_URL, IEP2_URL, EEP_URL,
    requires_iep1, requires_iep2, requires_eep, requires_model,
    make_window,
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@requires_iep2
def test_iep2_health():
    r = httpx.get(f"{IEP2_URL}/health", timeout=5)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@requires_iep1
def test_iep1_health_reachable():
    r = httpx.get(f"{IEP1_URL}/health", timeout=5)
    assert r.status_code == 200
    assert r.json()["status"] in ("ok", "error", "degraded")


@requires_model
def test_iep1_health_model_loaded():
    r = httpx.get(f"{IEP1_URL}/health", timeout=5)
    body = r.json()
    assert body["status"] == "ok"
    assert body["details"]["fold_models"] >= 1
    assert body["details"]["threshold"] == pytest.approx(0.20)


@requires_eep
def test_eep_health_aggregates():
    r = httpx.get(f"{EEP_URL}/health", timeout=5)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded", "error")
    assert "iep1" in body["details"]
    assert "iep2" in body["details"]


# ---------------------------------------------------------------------------
# IEP-2 calibration
# ---------------------------------------------------------------------------

@requires_iep2
def test_calibrate_returns_params():
    payload = {"windows": [make_window(seed=0), make_window(seed=1)]}
    r = httpx.post(f"{IEP2_URL}/calibrate", json=payload, timeout=30)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "params" in body
    assert "mu" in body["params"]
    assert "sigma" in body["params"]
    assert body["windows_used"] == 2
    # sigma values must all be positive
    for v in body["params"]["sigma"].values():
        assert v > 0


@requires_iep2
def test_calibrate_rejects_empty_windows():
    r = httpx.post(f"{IEP2_URL}/calibrate", json={"windows": []}, timeout=5)
    assert r.status_code == 422


@requires_iep2
def test_calibrate_rejects_short_signal():
    short = make_window(duration_s=10.0)
    r = httpx.post(f"{IEP2_URL}/calibrate", json={"windows": [short]}, timeout=5)
    assert r.status_code == 422


@requires_iep2
def test_calibrate_rejects_mismatched_lengths():
    w = make_window()
    w["eda_values"] = w["eda_values"][:10]  # length mismatch
    r = httpx.post(f"{IEP2_URL}/calibrate", json={"windows": [w]}, timeout=5)
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# IEP-1 prediction
# ---------------------------------------------------------------------------

@requires_model
def test_iep1_predict_returns_valid_response():
    r = httpx.post(f"{IEP1_URL}/predict", json={"window": make_window()}, timeout=30)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["calibrated"] is False
    assert body["threshold"] == pytest.approx(0.20)


@requires_iep1
def test_iep1_predict_rejects_short_signal():
    short = make_window(duration_s=10.0)
    r = httpx.post(f"{IEP1_URL}/predict", json={"window": short}, timeout=5)
    assert r.status_code == 422


@requires_model
def test_iep1_predict_with_calibration():
    # First calibrate
    calib_r = httpx.post(
        f"{IEP2_URL}/calibrate",
        json={"windows": [make_window(seed=5), make_window(seed=6)]},
        timeout=30,
    )
    calib_params = calib_r.json()["params"]

    r = httpx.post(
        f"{IEP1_URL}/predict",
        json={"window": make_window(), "calib": calib_params},
        timeout=30,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["calibrated"] is True
    assert body["drift_score"] is not None
    assert body["drift_score"] >= 0


# ---------------------------------------------------------------------------
# EEP gateway
# ---------------------------------------------------------------------------

@requires_eep
def test_eep_rejects_short_signal():
    short = make_window(duration_s=10.0)
    r = httpx.post(f"{EEP_URL}/predict", json={"window": short}, timeout=5)
    assert r.status_code == 422


@requires_eep
def test_eep_rejects_empty_calibrate():
    r = httpx.post(f"{EEP_URL}/calibrate", json={"windows": []}, timeout=5)
    assert r.status_code == 422


@requires_eep
@requires_model
def test_eep_predict_uncalibrated():
    r = httpx.post(f"{EEP_URL}/predict", json={"window": make_window()}, timeout=30)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["calibrated"] is False


@requires_eep
@requires_model
def test_eep_full_calibrate_then_predict():
    """Full cross-service flow: calibrate via IEP-2 then predict via IEP-1."""
    calib_r = httpx.post(
        f"{EEP_URL}/calibrate",
        json={"windows": [make_window(seed=10), make_window(seed=11)]},
        timeout=30,
    )
    assert calib_r.status_code == 200, calib_r.text
    calib_params = calib_r.json()["params"]

    pred_r = httpx.post(
        f"{EEP_URL}/predict",
        json={"window": make_window(seed=99), "calib": calib_params},
        timeout=30,
    )
    assert pred_r.status_code == 200, pred_r.text
    body = pred_r.json()
    assert body["calibrated"] is True
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] in (0, 1)


# ---------------------------------------------------------------------------
# Metrics endpoints
# ---------------------------------------------------------------------------

@requires_iep1
def test_iep1_metrics_endpoint():
    r = httpx.get(f"{IEP1_URL}/metrics", timeout=5)
    assert r.status_code == 200
    assert "iep1_requests_total" in r.text


@requires_iep2
def test_iep2_metrics_endpoint():
    r = httpx.get(f"{IEP2_URL}/metrics", timeout=5)
    assert r.status_code == 200
    assert "iep2_requests_total" in r.text


@requires_eep
def test_eep_metrics_endpoint():
    r = httpx.get(f"{EEP_URL}/metrics", timeout=5)
    assert r.status_code == 200
    assert "eep_requests_total" in r.text
