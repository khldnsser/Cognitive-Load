"""Leave-One-Subject-Out cross-validation loop.

Each fold trains XGBoost on N-1 subjects and tests on the held-out subject.
All fold models are returned so the caller can build an ensemble.
Out-of-fold (OOF) probabilities are collected across every fold for
pooled confusion-matrix analysis.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from cogload.evaluation.metrics import compute_fold_metrics
from cogload.features.extract import FEATURE_COLS
from cogload.models.xgboost_model import build_xgb, compute_scale_pos_weight

log = logging.getLogger(__name__)


def run_loso(
    df: pd.DataFrame,
    thresholds: list[float],
    xgb_kwargs: dict | None = None,
) -> tuple[list[dict], list[XGBClassifier], np.ndarray, np.ndarray]:
    """Run LOSO cross-validation.

    Args:
        df: feature table for LOSO subjects only (holdout already excluded).
            Must have columns 'subject', 'label', and all FEATURE_COLS.
        thresholds: decision thresholds to evaluate per fold.
        xgb_kwargs: optional XGBClassifier hyperparameter overrides.

    Returns:
        (fold_records, fold_models, oof_labels, oof_probs)
        fold_records: one dict per fold — subject + all metrics.
        fold_models:  one trained XGBClassifier per fold (same order).
        oof_labels/oof_probs: concatenated across all folds (for pooled metrics).
    """
    xgb_kwargs = xgb_kwargs or {}
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    subjects = sorted(df["subject"].unique())

    fold_records:  list[dict]          = []
    fold_models:   list[XGBClassifier] = []
    oof_labels:    list[np.ndarray]    = []
    oof_probs:     list[np.ndarray]    = []

    for subj in subjects:
        test_mask  = df["subject"] == subj
        train_mask = ~test_mask

        train_df = df[train_mask]
        test_df  = df[test_mask]

        if test_df.empty or len(test_df["label"].unique()) < 2:
            log.warning("Skipping subject %s — fewer than 2 classes in test set.", subj)
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_test  = test_df[feature_cols].values
        y_test  = test_df["label"].values

        spw   = compute_scale_pos_weight(y_train)
        model = build_xgb(scale_pos_weight=spw, **xgb_kwargs)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]

        record = compute_fold_metrics(y_test, y_prob, thresholds)
        record["subject"] = subj

        fold_records.append(record)
        fold_models.append(model)
        oof_labels.append(y_test)
        oof_probs.append(y_prob)

    return (
        fold_records,
        fold_models,
        np.concatenate(oof_labels) if oof_labels else np.array([]),
        np.concatenate(oof_probs)  if oof_probs  else np.array([]),
    )


def assert_no_leakage(df: pd.DataFrame) -> None:
    """Sanity check: subject IDs are unique (no duplicate subject rows across cohorts)."""
    dupes = df.groupby("subject")["cohort"].nunique()
    assert (dupes == 1).all(), f"Subjects with mixed cohorts: {dupes[dupes > 1].index.tolist()}"
