"""Core experiment runner: one MLflow run = one XGBoost config.

Importable by both train_loso.py (single run) and run_experiments.py (sweep).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from cogload.config import (
    BASELINE_OFFSET_S, HOLDOUT_SUBJECT, HOP_S,
    SELECTION_LAMBDA, THRESHOLDS, WINDOW_S,
)
from cogload.evaluation.loso import assert_no_leakage, run_loso
from cogload.pipeline import load_feature_table
from cogload.tracking.mlflow_utils import log_loso_run

log = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    window_s:          float       = WINDOW_S
    hop_s:             float       = HOP_S
    baseline_offset_s: float       = BASELINE_OFFSET_S
    n_estimators:      int         = 200
    max_depth:         int         = 4
    learning_rate:     float       = 0.05
    subsample:         float       = 0.8
    colsample:         float       = 0.8
    thresholds:        list[float] = field(default_factory=lambda: list(THRESHOLDS))
    lam:               float       = SELECTION_LAMBDA


def run_experiment(cfg: ExperimentConfig) -> str:
    """Execute one full LOSO experiment and return the MLflow run_id.

    Steps:
    1. Load the cached feature table for (window, hop, offset).
    2. Split off the holdout subject.
    3. Assert no LOSO leakage.
    4. Run LOSO — returns fold models + per-fold metrics.
    5. Log everything (params, datasets, metrics, plots, ensemble) to MLflow.
    """
    df_all = load_feature_table(cfg.window_s, cfg.hop_s, cfg.baseline_offset_s)

    df_holdout = df_all[df_all["subject"] == HOLDOUT_SUBJECT].copy()
    df_loso    = df_all[df_all["subject"] != HOLDOUT_SUBJECT].copy()

    if df_holdout.empty:
        raise ValueError(f"Holdout subject {HOLDOUT_SUBJECT} not found in feature table.")

    assert_no_leakage(df_loso)

    log.info(
        "LOSO pool: %d windows, %d subjects. Holdout (S%d): %d windows.",
        len(df_loso), df_loso["subject"].nunique(),
        HOLDOUT_SUBJECT, len(df_holdout),
    )

    xgb_kwargs = dict(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample,
    )

    fold_records, fold_models, oof_labels, oof_probs = run_loso(
        df_loso, cfg.thresholds, xgb_kwargs,
    )

    if not fold_records:
        raise RuntimeError("LOSO produced no fold results — check data.")

    params = {
        "model":               "XGBoost-Ensemble",
        "window_s":            cfg.window_s,
        "hop_s":               cfg.hop_s,
        "baseline_offset_s":   cfg.baseline_offset_s,
        "n_estimators":        cfg.n_estimators,
        "max_depth":           cfg.max_depth,
        "learning_rate":       cfg.learning_rate,
        "subsample":           cfg.subsample,
        "colsample_bytree":    cfg.colsample,
        "normalization":       "per_subject_zscore",
        "imbalance_strategy":  "scale_pos_weight",
        "n_loso_subjects":     df_loso["subject"].nunique(),
        "n_loso_windows":      len(df_loso),
        "n_loso_cl1":          int((df_loso["label"] == 1).sum()),
        "n_loso_cl0":          int((df_loso["label"] == 0).sum()),
        "holdout_subject":     HOLDOUT_SUBJECT,
        "n_holdout_windows":   len(df_holdout),
    }

    from cogload.pipeline import _fname
    from cogload.config import PROCESSED_DIR
    table_path = PROCESSED_DIR / _fname(cfg.window_s, cfg.hop_s, cfg.baseline_offset_s)

    run_id = log_loso_run(
        fold_records=fold_records,
        fold_models=fold_models,
        oof_labels=oof_labels,
        oof_probs=oof_probs,
        df_loso=df_loso,
        df_holdout=df_holdout,
        params=params,
        thresholds=cfg.thresholds,
        lam=cfg.lam,
        feature_table_path=table_path if table_path.exists() else None,
    )

    return run_id
