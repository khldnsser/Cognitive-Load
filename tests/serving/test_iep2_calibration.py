"""Unit tests for IEP-2 calibration service."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import importlib.util
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Load iep2 main with a unique module name to avoid caching collision with iep1
_IEP2_PATH = Path(__file__).parents[2] / "serving" / "iep2_calibration" / "main.py"
_spec = importlib.util.spec_from_file_location("iep2_main", _IEP2_PATH)
_iep2_mod = importlib.util.module_from_spec(_spec)
sys.modules["iep2_main"] = _iep2_mod
_spec.loader.exec_module(_iep2_mod)

app = _iep2_mod.app
_compute_calibration_params = _iep2_mod._compute_calibration_params

from cogload.config import NORMALIZE_FEATURES
from cogload.serving.schemas import SignalWindow

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_window(duration_s: float = 60.0, bvp_hz: int = 64, eda_hz: int = 4) -> dict:
    """Build a minimal valid SignalWindow payload dict."""
    n_bvp  = int(duration_s * bvp_hz)
    n_eda  = int(duration_s * eda_hz)

    bvp_times  = [i / bvp_hz for i in range(n_bvp)]
    eda_times  = [i / eda_hz for i in range(n_eda)]
    temp_times = [i / eda_hz for i in range(n_eda)]

    rng = np.random.default_rng(42)
    return {
        "bvp_values":  (rng.normal(0, 0.5, n_bvp)).tolist(),
        "bvp_times":   bvp_times,
        "eda_values":  (rng.uniform(0.5, 2.0, n_eda)).tolist(),
        "eda_times":   eda_times,
        "temp_values": (rng.normal(36.0, 0.1, n_eda)).tolist(),
        "temp_times":  temp_times,
    }


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "iep2-calibration"


# ---------------------------------------------------------------------------
# /calibrate — happy path
# ---------------------------------------------------------------------------

def test_calibrate_returns_valid_params():
    payload = {"windows": [_make_window(), _make_window()]}
    resp = client.post("/calibrate", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "params" in body
    assert "mu" in body["params"]
    assert "sigma" in body["params"]
    assert body["windows_used"] == 2
    # Every key in mu must also be in NORMALIZE_FEATURES
    for key in body["params"]["mu"]:
        assert key in NORMALIZE_FEATURES


def test_calibrate_sigma_positive():
    payload = {"windows": [_make_window() for _ in range(4)]}
    resp = client.post("/calibrate", json=payload)
    assert resp.status_code == 200, resp.text
    sigma = resp.json()["params"]["sigma"]
    for feat, val in sigma.items():
        assert val > 0, f"{feat} sigma={val} must be positive"


def test_calibrate_mu_sigma_same_keys():
    payload = {"windows": [_make_window()]}
    resp = client.post("/calibrate", json=payload)
    assert resp.status_code == 200, resp.text
    params = resp.json()["params"]
    assert set(params["mu"].keys()) == set(params["sigma"].keys())


# ---------------------------------------------------------------------------
# /calibrate — validation rejections
# ---------------------------------------------------------------------------

def test_calibrate_rejects_empty_windows():
    resp = client.post("/calibrate", json={"windows": []})
    assert resp.status_code == 422


def test_calibrate_rejects_short_signal():
    """Window shorter than 30s must be rejected by SignalWindow validator."""
    short = _make_window(duration_s=10.0)
    resp = client.post("/calibrate", json={"windows": [short]})
    assert resp.status_code == 422


def test_calibrate_rejects_mismatched_lengths():
    w = _make_window()
    w["bvp_times"] = w["bvp_times"][:-1]  # one element shorter
    resp = client.post("/calibrate", json={"windows": [w]})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Calibration math
# ---------------------------------------------------------------------------

def test_compute_calibration_deterministic():
    """Same input twice produces identical mu/sigma."""
    windows_raw = [_make_window(), _make_window(duration_s=45.0)]
    windows = [SignalWindow(**w) for w in windows_raw]
    mu1, sig1, n1 = _compute_calibration_params(windows)
    mu2, sig2, n2 = _compute_calibration_params(windows)
    assert mu1 == mu2
    assert sig1 == sig2
    assert n1 == n2 == 2


def test_compute_calibration_no_nan_in_output():
    windows = [SignalWindow(**_make_window()) for _ in range(3)]
    mu, sigma, _ = _compute_calibration_params(windows)
    for feat, val in mu.items():
        assert not math.isnan(val), f"mu[{feat}] is NaN"
    for feat, val in sigma.items():
        assert not math.isnan(val), f"sigma[{feat}] is NaN"
