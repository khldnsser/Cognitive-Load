"""Shared Pydantic contracts for all three serving services.

These types are the single source of truth for every request/response body.
Validation is enforced at the boundary (EEP input, IEP inputs) so internal
service calls can trust their payloads.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, field_validator, model_validator


# ---------------------------------------------------------------------------
# Raw signal container — used in both calibration and inference requests
# ---------------------------------------------------------------------------

class SignalWindow(BaseModel):
    """One multi-rate physiological window.

    Arrays are parallel time series; each value corresponds to one sample.
    Timestamps are Unix seconds (float). All three signals are required.

    BVP:  64 Hz  → 60s window = 3840 samples
    EDA:   4 Hz  → 60s window =  240 samples
    Temp:  4 Hz  → 60s window =  240 samples

    For calibration (IEP-2) windows can be shorter (≥30s minimum).
    """
    bvp_values:   list[float]
    bvp_times:    list[float]
    eda_values:   list[float]
    eda_times:    list[float]
    temp_values:  list[float]
    temp_times:   list[float]

    @field_validator("bvp_values", "eda_values", "temp_values", mode="before")
    @classmethod
    def no_empty(cls, v: list) -> list:
        if len(v) == 0:
            raise ValueError("signal array must not be empty")
        return v

    @model_validator(mode="after")
    def lengths_match(self) -> "SignalWindow":
        if len(self.bvp_values) != len(self.bvp_times):
            raise ValueError("bvp_values and bvp_times must have equal length")
        if len(self.eda_values) != len(self.eda_times):
            raise ValueError("eda_values and eda_times must have equal length")
        if len(self.temp_values) != len(self.temp_times):
            raise ValueError("temp_values and temp_times must have equal length")
        return self

    @model_validator(mode="after")
    def min_duration(self) -> "SignalWindow":
        """Require at least 30s of coverage for each signal."""
        for name, times in [
            ("bvp", self.bvp_times),
            ("eda", self.eda_times),
            ("temp", self.temp_times),
        ]:
            if len(times) >= 2:
                duration = max(times) - min(times)
                if duration < 30.0:
                    raise ValueError(
                        f"{name} signal covers only {duration:.1f}s — minimum is 30s"
                    )
        return self


# ---------------------------------------------------------------------------
# Calibration  (IEP-2)
# ---------------------------------------------------------------------------

class CalibrateRequest(BaseModel):
    """≥30s of known-rest signals from the current user.

    The service computes per-subject mean/std over NORMALIZE_FEATURES and
    returns a CalibrationParams token consumed by IEP-1 at predict time.
    """
    windows: list[SignalWindow]

    @field_validator("windows", mode="before")
    @classmethod
    def at_least_one(cls, v: list) -> list:
        if len(v) == 0:
            raise ValueError("at least one window is required for calibration")
        return v


class CalibrationParams(BaseModel):
    """Per-subject normalization statistics for NORMALIZE_FEATURES.

    mu and sigma are dicts keyed by feature name (same keys as NORMALIZE_FEATURES).
    Features absent from these dicts are passed through unchanged.
    """
    mu:    dict[str, float]
    sigma: dict[str, float]

    @model_validator(mode="after")
    def keys_match(self) -> "CalibrationParams":
        if set(self.mu.keys()) != set(self.sigma.keys()):
            raise ValueError("mu and sigma must have identical feature keys")
        return self


class CalibrateResponse(BaseModel):
    """Response from IEP-2 /calibrate."""
    params:         CalibrationParams
    windows_used:   int
    features_computed: list[str]


# ---------------------------------------------------------------------------
# Inference  (IEP-1)
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """One 60s physiological window plus calibration params.

    calib is optional: if None, features are passed to the model un-normalized
    (degraded mode — model still runs but accuracy may be reduced).
    """
    window: SignalWindow
    calib:  Optional[CalibrationParams] = None


class PredictResponse(BaseModel):
    """Response from IEP-1 /predict."""
    probability:  float
    prediction:   int        # 0 = low load, 1 = high load
    threshold:    float
    calibrated:   bool       # False when calib was None (degraded mode)
    drift_score:  Optional[float] = None   # mean |z| of normalized features


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status:  str   # "ok" | "degraded" | "error"
    service: str
    details: Optional[dict] = None
