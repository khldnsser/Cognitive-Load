"""Temperature feature extraction."""
from __future__ import annotations

from typing import Optional

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_NAN_FEATURES = {
    "temp_slope": np.nan,
    "temp_mean":  np.nan,
    "temp_std":   np.nan,
    "temp_range": np.nan,
    "temp_min":   np.nan,
}


def extract_temp_features(temp_slice: Optional[pd.DataFrame]) -> dict:
    """Compute slope (key vasoconstriction signal), mean, std, range, min."""
    if temp_slice is None or len(temp_slice) < 2:
        return dict(_NAN_FEATURES)

    try:
        vals = temp_slice["temp"].values.astype(float)
        t    = temp_slice["time"].values - temp_slice["time"].values[0]

        # Slope in °C/min — offset-invariant, not normalised later
        slope_per_s = float(np.polyfit(t, vals, 1)[0])
        slope_per_min = slope_per_s * 60.0

        return {
            "temp_slope": slope_per_min,
            "temp_mean":  float(np.mean(vals)),
            "temp_std":   float(np.std(vals)),
            "temp_range": float(np.ptp(vals)),
            "temp_min":   float(np.min(vals)),
        }

    except Exception as exc:
        log.debug("Temp feature extraction failed: %s", exc)
        return dict(_NAN_FEATURES)
