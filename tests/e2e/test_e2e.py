"""End-to-end test against a deployed EEP endpoint.

Pointed at EEP_URL env var (defaults to localhost:8080 for local docker stack;
set to the EKS LoadBalancer URL after Stage 8 cloud deploy).

Skipped automatically if the endpoint is not reachable.
"""
from __future__ import annotations

import os
import numpy as np
import pytest
import httpx

EEP_URL = os.environ.get("EEP_URL", "http://localhost:8080")


def _reachable() -> bool:
    try:
        httpx.get(f"{EEP_URL}/health", timeout=5.0)
        return True
    except Exception:
        return False


def _model_ok() -> bool:
    try:
        r = httpx.get(f"{EEP_URL}/health", timeout=5.0)
        details = r.json().get("details", {})
        return details.get("iep1", {}).get("status") == "ok"
    except Exception:
        return False


def make_window(duration_s: float = 60.0, seed: int = 0) -> dict:
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


@pytest.mark.skipif(not _reachable(), reason=f"EEP not reachable at {EEP_URL}")
def test_e2e_health():
    r = httpx.get(f"{EEP_URL}/health", timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert "iep1" in body.get("details", {})
    assert "iep2" in body.get("details", {})


@pytest.mark.skipif(not _reachable(), reason=f"EEP not reachable at {EEP_URL}")
@pytest.mark.skipif(not _model_ok(), reason="IEP-1 model not loaded")
def test_e2e_calibrate_then_predict():
    """Golden path: calibrate on rest data, predict on a 60s window."""
    # 1. Calibrate
    calib_r = httpx.post(
        f"{EEP_URL}/calibrate",
        json={"windows": [make_window(seed=20), make_window(seed=21), make_window(seed=22)]},
        timeout=60,
    )
    assert calib_r.status_code == 200, f"Calibrate failed: {calib_r.text}"
    calib = calib_r.json()
    assert calib["windows_used"] == 3
    calib_params = calib["params"]
    assert set(calib_params["mu"].keys()) == set(calib_params["sigma"].keys())
    print(f"\n  [calibrate] windows_used={calib['windows_used']}  features={len(calib_params['mu'])}")

    # 2. Predict with calibration
    pred_r = httpx.post(
        f"{EEP_URL}/predict",
        json={"window": make_window(seed=99), "calib": calib_params},
        timeout=60,
    )
    assert pred_r.status_code == 200, f"Predict failed: {pred_r.text}"
    result = pred_r.json()

    assert 0.0 <= result["probability"] <= 1.0
    assert result["prediction"] in (0, 1)
    assert result["calibrated"] is True
    assert result["threshold"] == pytest.approx(0.20)
    assert result["drift_score"] is not None
    label = "HIGH" if result["prediction"] == 1 else "LOW"
    print(f"  [predict]   probability={result['probability']:.3f}  prediction={label}"
          f"  drift_score={result['drift_score']:.3f}  calibrated={result['calibrated']}")

    # 3. Metrics populated
    metrics_r = httpx.get(f"{EEP_URL}/metrics", timeout=5)
    assert metrics_r.status_code == 200
    assert "eep_requests_total" in metrics_r.text
    print(f"  [metrics]   eep_requests_total present")


@pytest.mark.skipif(not _reachable(), reason=f"EEP not reachable at {EEP_URL}")
def test_e2e_validation_rejects_short_signal():
    """Data-validation gate: short signals must be rejected at the boundary."""
    r = httpx.post(
        f"{EEP_URL}/predict",
        json={"window": make_window(duration_s=10.0)},
        timeout=10,
    )
    assert r.status_code == 422


@pytest.mark.skipif(not _reachable(), reason=f"EEP not reachable at {EEP_URL}")
def test_e2e_validation_rejects_empty_calibrate():
    r = httpx.post(f"{EEP_URL}/calibrate", json={"windows": []}, timeout=10)
    assert r.status_code == 422
