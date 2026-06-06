"""Per-subject z-score normalization (Operation 1).

Each subject is z-scored against their own mean/std computed across all their
windows (both conditions, no label information used). This removes the
individual identity offset — e.g. EDA 1 µS vs 10 µS baseline — so the model
learns cognitive-load vs rest rather than who the person is.

This is NOT leakage: it uses only each subject's own data and ignores labels,
which mirrors exactly what happens at deployment (calibration on a known-rest
window at session start).
"""
from __future__ import annotations

import pandas as pd

from cogload.config import NORMALIZE_FEATURES


def apply_per_subject_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each subject's level/amplitude features against their own stats.

    Features with zero variance for a subject are left unchanged (constant
    feature — the model will ignore it via zero variance).
    """
    df = df.copy()
    present = [f for f in NORMALIZE_FEATURES if f in df.columns]

    for subj in df["subject"].unique():
        mask = df["subject"] == subj
        for feat in present:
            vals = df.loc[mask, feat]
            mu = vals.mean()
            sd = vals.std()
            if pd.isna(sd) or sd < 1e-10:
                continue
            df.loc[mask, feat] = (vals - mu) / sd

    return df
