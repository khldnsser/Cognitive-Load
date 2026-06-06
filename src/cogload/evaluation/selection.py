"""Model selection utilities: leaderboard and best (run, threshold) lookup."""
from __future__ import annotations

import pandas as pd

from cogload.config import SELECTION_LAMBDA, THRESHOLDS
from cogload.evaluation.metrics import thr_key


def build_leaderboard(
    runs: list,
    thresholds: list[float] = THRESHOLDS,
) -> pd.DataFrame:
    """Build a leaderboard DataFrame from MLflow run objects.

    Each row = one (run_id, threshold) pair with its selection score and
    LOSO mean/std F1.

    Args:
        runs: list of mlflow.entities.Run objects from client.search_runs().
    """
    rows = []
    for run in runs:
        metrics = run.data.metrics
        for thr in thresholds:
            k = thr_key(thr)
            score = metrics.get(f"f1_{k}_selscore", float("-inf"))
            rows.append({
                "run_id":         run.info.run_id,
                "threshold":      thr,
                "selscore":       score,
                "f1_mean":        metrics.get(f"f1_{k}_mean", float("nan")),
                "f1_std":         metrics.get(f"f1_{k}_std",  float("nan")),
                "pr_auc_mean":    metrics.get("pr_auc_mean",  float("nan")),
                "roc_auc_mean":   metrics.get("roc_auc_mean", float("nan")),
                "holdout_f1":     metrics.get(f"holdout_f1_{k}", float("nan")),
            })

    df = pd.DataFrame(rows).sort_values("selscore", ascending=False).reset_index(drop=True)
    return df


def best_run_and_threshold(
    runs: list,
    thresholds: list[float] = THRESHOLDS,
) -> tuple[str, float, float]:
    """Return (run_id, threshold, selection_score) for the top candidate."""
    lb = build_leaderboard(runs, thresholds)
    if lb.empty:
        raise ValueError("No runs found — run experiments first.")
    top = lb.iloc[0]
    return str(top["run_id"]), float(top["threshold"]), float(top["selscore"])
