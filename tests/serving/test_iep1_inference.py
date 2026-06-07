"""Unit + golden-dataset regression tests for IEP-1 inference service.

Tests that do NOT require a live MLflow server:
  - validation rejection tests (no model needed)
  - normalization math
  - drift score computation

Tests that DO require the champion model (marked with @pytest.mark.requires_model):
  - golden-dataset regression test
  - end-to-end predict with calibration
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import importlib.util
import numpy as np
import pandas as pd
import pytest

# Load iep1 main with a unique module name to avoid caching collision with iep2
_IEP1_PATH = Path(__file__).parents[2] / "serving" / "iep1_inference" / "main.py"
_spec = importlib.util.spec_from_file_location("iep1_main", _IEP1_PATH)
_iep1_mod = importlib.util.module_from_spec(_spec)
sys.modules["iep1_main"] = _iep1_mod
_spec.loader.exec_module(_iep1_mod)

from cogload.config import NORMALIZE_FEATURES
from cogload.serving.schemas import CalibrationParams, SignalWindow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_window_dict(duration_s: float = 60.0, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n_bvp  = int(duration_s * 64)
    n_eda  = int(duration_s * 4)
    return {
        "bvp_values":  rng.normal(0, 0.5, n_bvp).tolist(),
        "bvp_times":   [i / 64 for i in range(n_bvp)],
        "eda_values":  rng.uniform(0.5, 2.0, n_eda).tolist(),
        "eda_times":   [i / 4  for i in range(n_eda)],
        "temp_values": rng.normal(36.0, 0.1, n_eda).tolist(),
        "temp_times":  [i / 4  for i in range(n_eda)],
    }


def _make_calib_params() -> dict:
    mu    = {f: 0.0 for f in NORMALIZE_FEATURES}
    sigma = {f: 1.0 for f in NORMALIZE_FEATURES}
    return {"mu": mu, "sigma": sigma}


# ---------------------------------------------------------------------------
# Normalization math (no model needed)
# ---------------------------------------------------------------------------

def test_apply_calibration_zero_mean_unit_std():
    """When mu=0 and sigma=1, features should be unchanged."""
    from iep1_main import _apply_calibration

    features = {f: 1.0 for f in NORMALIZE_FEATURES}
    calib = CalibrationParams(
        mu={f: 0.0 for f in NORMALIZE_FEATURES},
        sigma={f: 1.0 for f in NORMALIZE_FEATURES},
    )
    result, drift = _apply_calibration(dict(features), calib)
    for f in NORMALIZE_FEATURES:
        assert result[f] == pytest.approx(1.0), f"{f} changed unexpectedly"


def test_apply_calibration_shifts_correctly():
    """(x - mu) / sigma should produce expected z-score."""
    from iep1_main import _apply_calibration

    feat = NORMALIZE_FEATURES[0]
    features = {feat: 10.0}
    calib = CalibrationParams(mu={feat: 5.0}, sigma={feat: 2.0})
    result, _ = _apply_calibration(features, calib)
    assert result[feat] == pytest.approx(2.5)


def test_apply_calibration_drift_score_near_one_for_standard_normal():
    """z-scores of 1.0 → mean |z| = 1.0."""
    from iep1_main import _apply_calibration

    features = {f: 1.0 for f in NORMALIZE_FEATURES}
    calib = CalibrationParams(
        mu={f: 0.0 for f in NORMALIZE_FEATURES},
        sigma={f: 1.0 for f in NORMALIZE_FEATURES},
    )
    _, drift = _apply_calibration(dict(features), calib)
    assert drift == pytest.approx(1.0)


def test_apply_calibration_skips_missing_features():
    """Features not in calib.mu are left unchanged."""
    from iep1_main import _apply_calibration

    feat = NORMALIZE_FEATURES[0]
    features = {feat: 5.0, "some_other_feature": 99.0}
    calib = CalibrationParams(mu={feat: 2.0}, sigma={feat: 1.0})
    result, _ = _apply_calibration(dict(features), calib)
    assert result["some_other_feature"] == 99.0


# ---------------------------------------------------------------------------
# Validation (no model needed — uses mocked model)
# ---------------------------------------------------------------------------

@pytest.fixture
def client_with_mock_model():
    """TestClient with a mock FoldEnsembleModel injected."""
    from fastapi.testclient import TestClient
    import iep1_main

    mock_model = MagicMock()
    mock_model.threshold = 0.20
    mock_model._models = [MagicMock()] * 23
    mock_model.feature_cols = []

    # predict_proba returns shape (1,) array
    mock_model.predict_proba.return_value = np.array([0.75])

    original = iep1_main._model
    iep1_main._model = mock_model
    with TestClient(iep1_main.app) as c:
        yield c
    iep1_main._model = original


def test_health_no_model():
    from fastapi.testclient import TestClient
    import iep1_main
    with patch("iep1_main.load_champion", side_effect=RuntimeError("test error")):
        with TestClient(iep1_main.app) as c:
            resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "error"


def test_health_with_model(client_with_mock_model):
    resp = client_with_mock_model.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_rejects_short_signal(client_with_mock_model):
    short = _make_window_dict(duration_s=10.0)
    resp = client_with_mock_model.post("/predict", json={"window": short})
    assert resp.status_code == 422


def test_predict_returns_valid_response(client_with_mock_model):
    payload = {
        "window": _make_window_dict(),
        "calib":  _make_calib_params(),
    }
    resp = client_with_mock_model.post("/predict", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "probability" in body
    assert "prediction" in body
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["calibrated"] is True


def test_predict_uncalibrated_flagged(client_with_mock_model):
    payload = {"window": _make_window_dict()}
    resp = client_with_mock_model.post("/predict", json=payload)
    assert resp.status_code == 200, resp.text
    assert resp.json()["calibrated"] is False


def test_predict_503_when_model_not_loaded():
    from fastapi.testclient import TestClient
    import iep1_main
    with patch("iep1_main.load_champion", side_effect=RuntimeError("not loaded")):
        with TestClient(iep1_main.app) as c:
            resp = c.post("/predict", json={"window": _make_window_dict()})
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Golden-dataset regression test
# Skipped automatically if MLflow is not reachable.
# The frozen expected probability is captured from the champion model on
# Subject 24's first valid window. Tolerance ±0.05 to allow minor env diffs.
# ---------------------------------------------------------------------------

# Golden baseline: Subject 24, first cognitive_load window (parquet row index 697),
# champion model features_w60_h15_off30, already per-subject z-scored.
# Captured from live champion on 2026-06-09. Tolerance ±0.01.
GOLDEN_PARQUET     = Path(__file__).parents[2] / "data" / "processed" / "features_w60_h15_off30.parquet"
GOLDEN_ROW_INDEX   = 697
GOLDEN_EXPECTED_PROB = 0.994256
GOLDEN_TOLERANCE     = 0.01


@pytest.mark.skipif(
    not Path(GOLDEN_PARQUET).exists(),
    reason="Champion parquet not present (run build_features.py first)",
)
def test_golden_dataset_regression():
    """Champion model on frozen S24 window must stay within ±0.01 of baseline.

    Uses real Subject 24 data from the champion feature table (already
    per-subject z-scored). No calibration needed — features are in the
    same normalized space the model was trained on.
    """
    from iep1_main import _predict
    from cogload.serving.model_loader import load_champion
    from cogload.features.extract import FEATURE_COLS as FC

    model = load_champion()
    df = pd.read_parquet(GOLDEN_PARQUET)
    row = df.loc[GOLDEN_ROW_INDEX]

    assert row["subject"] == 24, "Unexpected subject in golden row"
    assert row["label"] == 1,   "Golden row should be cognitive_load (label=1)"

    features = {f: float(row[f]) if f in row.index else float("nan") for f in FC}
    prob, pred = _predict(model, features)

    assert abs(prob - GOLDEN_EXPECTED_PROB) <= GOLDEN_TOLERANCE, (
        f"Golden regression FAILED: got {prob:.6f}, "
        f"expected {GOLDEN_EXPECTED_PROB:.6f} ± {GOLDEN_TOLERANCE}"
    )
    assert pred == 1, f"Golden window should predict high load, got {pred}"
