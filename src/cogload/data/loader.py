"""Discover all condition-recordings and load raw Empatica CSVs."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from cogload.config import (
    BVP_FILE, EDA_FILE, EDA_TYPO, TEMP_FILE,
    CONDITIONS, PILOT_DIR, SURVEY_DIR,
)

log = logging.getLogger(__name__)


@dataclass
class RecordingMeta:
    subject:   int
    cohort:    str            # "pilot" | "survey"
    session:   Optional[str]  # None | "pre" | "post"
    condition: str            # "cognitive_load" | "baseline"
    label:     int            # 1 | 0
    path:      Path


def get_all_recordings() -> list[RecordingMeta]:
    """Return a RecordingMeta for every condition-recording present on disk."""
    records: list[RecordingMeta] = []

    if PILOT_DIR.exists():
        for subj_dir in sorted(PILOT_DIR.iterdir()):
            if not subj_dir.is_dir() or not subj_dir.name.isdigit():
                continue
            sid = int(subj_dir.name)
            for cond, label in CONDITIONS.items():
                p = subj_dir / cond
                if p.exists():
                    records.append(RecordingMeta(
                        subject=sid, cohort="pilot", session=None,
                        condition=cond, label=label, path=p,
                    ))

    if SURVEY_DIR.exists():
        for subj_dir in sorted(SURVEY_DIR.iterdir()):
            if not subj_dir.is_dir() or not subj_dir.name.isdigit():
                continue
            sid = int(subj_dir.name)
            for sess in ("pre", "post"):
                sess_dir = subj_dir / sess
                if not sess_dir.exists():
                    continue
                for cond, label in CONDITIONS.items():
                    p = sess_dir / cond
                    if p.exists():
                        records.append(RecordingMeta(
                            subject=sid, cohort="survey", session=sess,
                            condition=cond, label=label, path=p,
                        ))

    return records


def load_bvp(path: Path) -> Optional[pd.DataFrame]:
    return _safe_read(path, BVP_FILE)


def load_eda(path: Path) -> Optional[pd.DataFrame]:
    """Load EDA CSV, handling the subject-3 filename/column typo."""
    df = _safe_read(path, EDA_FILE)
    if df is None:
        df = _safe_read(path, EDA_TYPO)
    if df is not None and "emda" in df.columns:
        df = df.rename(columns={"emda": "eda"})
    return df


def load_temp(path: Path) -> Optional[pd.DataFrame]:
    return _safe_read(path, TEMP_FILE)


def _safe_read(path: Path, fname: str) -> Optional[pd.DataFrame]:
    fp = path / fname
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    if df.empty or "time" not in df.columns:
        log.warning("Empty or missing 'time' column: %s", fp)
        return None
    return df
