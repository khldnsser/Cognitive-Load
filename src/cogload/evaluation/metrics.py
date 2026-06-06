"""Per-fold and aggregate evaluation metrics with threshold sweep."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)


def thr_key(threshold: float) -> str:
    """Canonical string key for a threshold, e.g. 0.5 → 't050'."""
    return f"t{round(threshold * 100):03d}"


def compute_fold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float],
) -> dict:
    """Compute all metrics for one LOSO fold across every threshold.

    Threshold-independent metrics (ROC-AUC, PR-AUC) are logged once.
    Threshold-dependent metrics (F1, Precision, Recall) are logged per threshold.
    """
    metrics: dict = {}

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"]  = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"]  = np.nan

    for thr in thresholds:
        k = thr_key(thr)
        y_pred = (y_prob >= thr).astype(int)
        metrics[f"f1_{k}"]        = float(f1_score(y_true, y_pred, zero_division=0))
        metrics[f"precision_{k}"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics[f"recall_{k}"]    = float(recall_score(y_true, y_pred, zero_division=0))

    return metrics


def aggregate_fold_metrics(fold_records: list[dict]) -> dict:
    """Compute mean ± std across folds for every metric (NaN-safe)."""
    if not fold_records:
        return {}
    df = pd.DataFrame(fold_records)
    result: dict = {}
    for col in df.columns:
        if col == "subject":
            continue
        result[f"{col}_mean"] = float(df[col].mean())
        result[f"{col}_std"]  = float(df[col].std())
    return result


def selection_score(
    fold_records: list[dict],
    threshold: float,
    lam: float,
) -> float:
    """mean_f1 - lam * std_f1 for the given threshold across all folds."""
    k    = f"f1_{thr_key(threshold)}"
    vals = [r[k] for r in fold_records if k in r and not np.isnan(r[k])]
    if not vals:
        return -np.inf
    mean = float(np.mean(vals))
    std  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return mean - lam * std


def build_confusion_matrix_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        cm,
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    )
