"""Clip BVP/EDA/Temp signals to their shared time overlap window."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from cogload.config import BASELINE_OFFSET_S


def clip_to_overlap(
    bvp: Optional[pd.DataFrame],
    eda: Optional[pd.DataFrame],
    temp: Optional[pd.DataFrame],
    condition: str,
    baseline_offset_s: float = BASELINE_OFFSET_S,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Clip signals to [max(starts), min(ends)].

    For baseline recordings, advance the start by baseline_offset_s to
    discard the post-Stroop recovery period before windowing begins.

    Returns (None, None, None) if usable overlap is zero or negative.
    """
    present = {k: v for k, v in {"bvp": bvp, "eda": eda, "temp": temp}.items() if v is not None}
    if not present:
        return None, None, None

    t_start = max(df["time"].min() for df in present.values())
    t_end   = min(df["time"].max() for df in present.values())

    if condition == "baseline":
        t_start += baseline_offset_s

    if t_start >= t_end:
        return None, None, None

    def _clip(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None:
            return None
        mask = (df["time"] >= t_start) & (df["time"] <= t_end)
        return df[mask].reset_index(drop=True)

    return _clip(bvp), _clip(eda), _clip(temp)
