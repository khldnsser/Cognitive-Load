"""Shared fixtures for integration tests.

These tests hit live services. They are skipped automatically if the services
are not reachable (e.g., running pytest without `docker compose up`).
"""
from __future__ import annotations

import os
import numpy as np
import pytest
import httpx

IEP1_URL = os.environ.get("IEP1_URL", "http://localhost:8000")
IEP2_URL = os.environ.get("IEP2_URL", "http://localhost:8001")
EEP_URL  = os.environ.get("EEP_URL",  "http://localhost:8080")


def _reachable(url: str) -> bool:
    try:
        httpx.get(f"{url}/health", timeout=3.0)
        return True
    except Exception:
        return False


def _model_loaded(url: str) -> bool:
    try:
        r = httpx.get(f"{url}/health", timeout=3.0)
        return r.json().get("status") == "ok"
    except Exception:
        return False


requires_iep1 = pytest.mark.skipif(
    not _reachable(IEP1_URL), reason="IEP-1 not reachable"
)
requires_iep2 = pytest.mark.skipif(
    not _reachable(IEP2_URL), reason="IEP-2 not reachable"
)
requires_eep = pytest.mark.skipif(
    not _reachable(EEP_URL), reason="EEP not reachable"
)
requires_model = pytest.mark.skipif(
    not _model_loaded(IEP1_URL), reason="IEP-1 champion model not loaded"
)


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
