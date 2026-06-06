"""Orchestrate per-window feature extraction across all modalities."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from cogload.features.bvp  import extract_bvp_features
from cogload.features.eda  import extract_eda_features
from cogload.features.temp import extract_temp_features

# bvp_quality is a diagnostic flag, not a predictor.
METADATA_COLS = ["subject", "label", "cohort", "session", "condition", "t_start"]

FEATURE_COLS = [
    "eda_tonic_mean", "eda_tonic_slope", "eda_tonic_std",
    "scr_count", "scr_rate", "scr_mean_amp", "scr_sum_amp",
    "hr_mean", "rmssd", "sdnn", "pnn50", "ibi_mean", "ibi_std",
    "temp_slope", "temp_mean", "temp_std", "temp_range", "temp_min",
]


def extract_window_features(
    t_start:    float,
    bvp_slice:  Optional[pd.DataFrame],
    eda_slice:  Optional[pd.DataFrame],
    temp_slice: Optional[pd.DataFrame],
    subject:    int,
    label:      int,
    cohort:     str,
    session:    Optional[str],
    condition:  str,
) -> dict:
    """Extract all features for one window and attach metadata."""
    row: dict = {
        "subject":   subject,
        "label":     label,
        "cohort":    cohort,
        "session":   session,
        "condition": condition,
        "t_start":   t_start,
    }
    row.update(extract_eda_features(eda_slice))
    row.update(extract_bvp_features(bvp_slice))
    row.update(extract_temp_features(temp_slice))
    return row
