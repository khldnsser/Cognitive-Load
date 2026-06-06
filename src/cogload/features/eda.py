"""EDA feature extraction using neurokit2 phasic/tonic decomposition."""
from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from cogload.config import EDA_HZ

log = logging.getLogger(__name__)

_NAN_FEATURES = {
    "eda_tonic_mean":  np.nan,
    "eda_tonic_slope": np.nan,
    "eda_tonic_std":   np.nan,
    "scr_count":       np.nan,
    "scr_rate":        np.nan,
    "scr_mean_amp":    np.nan,
    "scr_sum_amp":     np.nan,
}


def extract_eda_features(eda_slice: Optional[pd.DataFrame]) -> dict:
    """Decompose EDA into tonic (SCL) + phasic (SCR) and return features.

    Returns NaN dict for slices that are too short or fail decomposition.
    """
    if eda_slice is None or len(eda_slice) < 10:
        return dict(_NAN_FEATURES)

    try:
        import neurokit2 as nk

        signal = eda_slice["eda"].values.astype(float)
        t = eda_slice["time"].values - eda_slice["time"].values[0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            signals, info = nk.eda_process(signal, sampling_rate=EDA_HZ)

        tonic = signals["EDA_Tonic"].values
        slope = float(np.polyfit(t, tonic, 1)[0]) if len(t) > 1 else np.nan

        # SCR peaks: neurokit2 marks them in SCR_Peaks column (value == 1)
        peak_mask = signals.get("SCR_Peaks", pd.Series(0, index=signals.index)) == 1
        scr_count = int(peak_mask.sum())

        duration_min = float(t[-1] - t[0]) / 60.0 if len(t) > 1 else 1.0
        scr_rate = scr_count / duration_min if duration_min > 0 else 0.0

        # Amplitude values are NaN except at peak locations
        amp_col = signals.get("SCR_Amplitude", pd.Series(np.nan, index=signals.index))
        amp_vals = amp_col[peak_mask].dropna().values

        return {
            "eda_tonic_mean":  float(np.mean(tonic)),
            "eda_tonic_slope": slope,
            "eda_tonic_std":   float(np.std(tonic)),
            "scr_count":       float(scr_count),
            "scr_rate":        float(scr_rate),
            "scr_mean_amp":    float(np.mean(amp_vals)) if len(amp_vals) > 0 else 0.0,
            "scr_sum_amp":     float(np.sum(amp_vals)) if len(amp_vals) > 0 else 0.0,
        }

    except Exception as exc:
        log.debug("EDA feature extraction failed: %s", exc)
        return dict(_NAN_FEATURES)
