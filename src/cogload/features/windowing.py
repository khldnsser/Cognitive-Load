"""Timestamp-based sliding window over multi-rate signals."""
from __future__ import annotations

from typing import Generator, Optional

import pandas as pd

from cogload.config import HOP_S, WINDOW_S


def sliding_windows(
    bvp: Optional[pd.DataFrame],
    eda: Optional[pd.DataFrame],
    temp: Optional[pd.DataFrame],
    window_s: float = WINDOW_S,
    hop_s: float = HOP_S,
) -> Generator[tuple[float, dict[str, Optional[pd.DataFrame]]], None, None]:
    """Slide a fixed-length window across aligned signals.

    Slicing is timestamp-based so different sampling rates are handled
    automatically. Yields (t_start, {signal: slice_df}) for each window.
    """
    signals = {"bvp": bvp, "eda": eda, "temp": temp}
    present = [df for df in signals.values() if df is not None]
    if not present:
        return

    t_begin = max(df["time"].min() for df in present)
    t_final = min(df["time"].max() for df in present)

    if t_begin + window_s > t_final:
        return

    t_start = t_begin
    while t_start + window_s <= t_final:
        t_end = t_start + window_s
        slices: dict[str, Optional[pd.DataFrame]] = {}
        for name, df in signals.items():
            if df is None:
                slices[name] = None
            else:
                mask = (df["time"] >= t_start) & (df["time"] < t_end)
                slices[name] = df[mask].reset_index(drop=True)
        yield t_start, slices
        t_start += hop_s
