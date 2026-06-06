"""Build, cache, and load the windowed feature table."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from cogload.config import (
    BASELINE_OFFSET_S, HOP_S, PROCESSED_DIR, WINDOW_S,
)
from cogload.data.alignment import clip_to_overlap
from cogload.data.loader import get_all_recordings, load_bvp, load_eda, load_temp
from cogload.features.extract import FEATURE_COLS, METADATA_COLS, extract_window_features
from cogload.features.windowing import sliding_windows
from cogload.preprocessing.normalization import apply_per_subject_normalization

log = logging.getLogger(__name__)


def build_feature_table(
    window_s:          float = WINDOW_S,
    hop_s:             float = HOP_S,
    baseline_offset_s: float = BASELINE_OFFSET_S,
) -> pd.DataFrame:
    """Load all recordings, window them, extract features, normalise per subject.

    Returns the full feature table (all subjects, including the holdout).
    The caller is responsible for splitting off the holdout subject before
    entering the LOSO loop.
    """
    recordings = get_all_recordings()
    log.info("Found %d condition-recordings.", len(recordings))

    rows: list[dict] = []
    skipped = 0

    for rec in tqdm(recordings, desc="Extracting features"):
        bvp  = load_bvp(rec.path)
        eda  = load_eda(rec.path)
        temp = load_temp(rec.path)

        if bvp is None or eda is None or temp is None:
            skipped += 1
            log.warning("Skipping %s/%s — missing signal(s).", rec.subject, rec.condition)
            continue

        bvp, eda, temp = clip_to_overlap(
            bvp, eda, temp,
            condition=rec.condition,
            baseline_offset_s=baseline_offset_s,
        )

        if bvp is None:
            skipped += 1
            log.warning("Skipping %s/%s — no usable overlap.", rec.subject, rec.condition)
            continue

        for t_start, slices in sliding_windows(bvp, eda, temp, window_s, hop_s):
            rows.append(extract_window_features(
                t_start=t_start,
                bvp_slice=slices["bvp"],
                eda_slice=slices["eda"],
                temp_slice=slices["temp"],
                subject=rec.subject,
                label=rec.label,
                cohort=rec.cohort,
                session=rec.session,
                condition=rec.condition,
            ))

    if not rows:
        raise RuntimeError("No windows extracted — check data paths and config.")

    df = pd.DataFrame(rows)
    log.info(
        "Extracted %d windows from %d recordings (%d skipped). "
        "Class balance: %d cl=1 / %d cl=0.",
        len(df), len(recordings) - skipped, skipped,
        (df["label"] == 1).sum(), (df["label"] == 0).sum(),
    )

    return apply_per_subject_normalization(df)


def cache_feature_table(df: pd.DataFrame, window_s: float, hop_s: float, offset_s: float) -> Path:
    """Persist feature table to data/processed/ as parquet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / _fname(window_s, hop_s, offset_s)
    df.to_parquet(path, index=False)
    log.info("Feature table saved: %s", path)
    return path


def load_feature_table(window_s: float, hop_s: float, offset_s: float) -> pd.DataFrame:
    """Load a previously cached feature table."""
    path = PROCESSED_DIR / _fname(window_s, hop_s, offset_s)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached feature table at {path}. Run scripts/build_features.py first."
        )
    return pd.read_parquet(path)


def _fname(window_s: float, hop_s: float, offset_s: float) -> str:
    return f"features_w{int(window_s)}_h{int(hop_s)}_off{int(offset_s)}.parquet"
