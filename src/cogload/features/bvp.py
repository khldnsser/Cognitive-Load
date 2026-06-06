"""BVP feature extraction: peak detection → IBI → HRV via neurokit2."""
from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from cogload.config import BVP_HZ, MIN_BEATS

log = logging.getLogger(__name__)

_NAN_FEATURES = {
    "hr_mean":    np.nan,
    "rmssd":      np.nan,
    "sdnn":       np.nan,
    "pnn50":      np.nan,
    "ibi_mean":   np.nan,
    "ibi_std":    np.nan,
    "bvp_quality": 0.0,   # 0 = unusable window; not used as a predictor
}


def extract_bvp_features(bvp_slice: Optional[pd.DataFrame]) -> dict:
    """Detect BVP peaks, compute IBI series and time-domain HRV features.

    Returns NaN dict for slices with fewer than MIN_BEATS peaks or that
    fail neurokit2 processing.
    """
    if bvp_slice is None or len(bvp_slice) < BVP_HZ * 5:
        return dict(_NAN_FEATURES)

    try:
        import neurokit2 as nk

        signal = bvp_slice["bvp"].values.astype(float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            signals, info = nk.ppg_process(signal, sampling_rate=BVP_HZ)

        peaks = info.get("PPG_Peaks", np.array([], dtype=int))
        if len(peaks) < MIN_BEATS:
            return dict(_NAN_FEATURES)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hrv = nk.hrv_time(signals, sampling_rate=BVP_HZ, show=False)

        mean_nn = float(hrv["HRV_MeanNN"].iloc[0])   # milliseconds

        return {
            "hr_mean":     60_000.0 / mean_nn if mean_nn > 0 else np.nan,
            "rmssd":       float(hrv["HRV_RMSSD"].iloc[0]),
            "sdnn":        float(hrv["HRV_SDNN"].iloc[0]),
            "pnn50":       float(hrv["HRV_pNN50"].iloc[0]),
            "ibi_mean":    mean_nn / 1000.0,
            "ibi_std":     float(hrv["HRV_SDNN"].iloc[0]) / 1000.0,
            "bvp_quality": 1.0,
        }

    except Exception as exc:
        log.debug("BVP feature extraction failed: %s", exc)
        return dict(_NAN_FEATURES)
