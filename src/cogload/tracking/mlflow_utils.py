"""MLflow logging helpers.

The MLflow server runs with --serve-artifacts and a SQLite backend, so:
- No file:// artifact paths needed — the server handles storage.
- Model Registry is available (requires DB backend, not file store).
- Call setup_mlflow() once before logging.
"""
from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from cogload.config import (
    MLFLOW_EXPERIMENT, MLFLOW_URI,
    MODEL_CHAMPION_ALIAS, MODEL_REGISTRY_NAME,
    SELECTION_LAMBDA, THRESHOLDS,
)
from cogload.evaluation.metrics import (
    aggregate_fold_metrics, build_confusion_matrix_df,
    selection_score, thr_key,
)
from cogload.evaluation.selection import best_run_and_threshold
from cogload.features.extract import FEATURE_COLS
from cogload.models.ensemble import FoldEnsembleModel

log = logging.getLogger(__name__)


def setup_mlflow() -> None:
    """Set tracking URI and create the experiment if it does not exist."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def log_loso_run(
    fold_records:   list[dict],
    fold_models:    list,
    oof_labels:     np.ndarray,
    oof_probs:      np.ndarray,
    df_loso:        pd.DataFrame,
    df_holdout:     pd.DataFrame,
    params:         dict,
    thresholds:     list[float] = THRESHOLDS,
    lam:            float       = SELECTION_LAMBDA,
    feature_table_path: Optional[Path] = None,
) -> str:
    """Log one complete LOSO experiment run to MLflow.

    Logs params, dataset inputs, per-fold×threshold metrics, aggregates,
    selection scores, artifacts (plots, fold scores), holdout evaluation,
    and the ensemble model. Returns the MLflow run_id.
    """
    setup_mlflow()

    # Determine the best threshold from this run's fold records.
    scores_per_thr = {
        thr: selection_score(fold_records, thr, lam) for thr in thresholds
    }
    winning_thr = max(scores_per_thr, key=scores_per_thr.get)

    # Build ensemble and evaluate on holdout subject.
    ensemble = FoldEnsembleModel(fold_models, winning_thr, FEATURE_COLS)
    X_holdout = df_holdout[FEATURE_COLS].values
    y_holdout = df_holdout["label"].values
    holdout_probs = ensemble.predict_proba(df_holdout)
    holdout_preds = (holdout_probs >= winning_thr).astype(int)

    with mlflow.start_run() as run:
        # ── Params
        mlflow.log_params(params)
        mlflow.log_param("feature_cols",      ",".join(FEATURE_COLS))
        mlflow.log_param("n_features",        len(FEATURE_COLS))
        mlflow.log_param("thresholds",        str(thresholds))
        mlflow.log_param("selection_lambda",  lam)
        mlflow.log_param("winning_threshold", winning_thr)
        mlflow.log_param("git_commit",        _git_commit())
        mlflow.log_param("git_dirty",         _git_dirty())
        if feature_table_path:
            mlflow.log_param("feature_table_sha256", _sha256(feature_table_path))

        # ── Dataset inputs
        loso_dataset = mlflow.data.from_pandas(
            df_loso,
            name="loso_features",
            targets="label",
        )
        mlflow.log_input(loso_dataset, context="loso")

        holdout_dataset = mlflow.data.from_pandas(
            df_holdout,
            name="holdout_features",
            targets="label",
        )
        mlflow.log_input(holdout_dataset, context="holdout")

        # ── Per-fold metrics (step = fold index)
        for fold_idx, record in enumerate(fold_records):
            for key, val in record.items():
                if key == "subject":
                    continue
                mlflow.log_metric(f"fold_{key}", val, step=fold_idx)

        # ── Aggregate mean / std + selection scores
        agg = aggregate_fold_metrics(fold_records)
        for k, v in agg.items():
            mlflow.log_metric(k, v)

        for thr in thresholds:
            k = thr_key(thr)
            mlflow.log_metric(f"f1_{k}_selscore", scores_per_thr[thr])

        # ── Holdout metrics (ensemble on subject 24)
        from cogload.evaluation.metrics import compute_fold_metrics
        holdout_metrics = compute_fold_metrics(y_holdout, holdout_probs, thresholds)
        for metric_name, val in holdout_metrics.items():
            mlflow.log_metric(f"holdout_{metric_name}", val)

        # ── Tags
        mlflow.set_tag("stage",            "loso-xgboost-ensemble")
        mlflow.set_tag("holdout_subject",  str(params.get("holdout_subject", 24)))
        mlflow.set_tag("n_folds",          str(len(fold_records)))
        mlflow.set_tag("benchmark_f1",     "0.62")

        # ── Artifact: fold scores table
        fold_df = pd.DataFrame(fold_records)
        mlflow.log_text(fold_df.to_csv(index=False), "fold_scores.csv")
        mlflow.log_dict(
            {str(r["subject"]): {k: v for k, v in r.items() if k != "subject"}
             for r in fold_records},
            "fold_scores.json",
        )

        # ── Artifact: metric-vs-threshold table (aggregates)
        thr_rows = []
        for thr in thresholds:
            k = thr_key(thr)
            thr_rows.append({
                "threshold":  thr,
                "f1_mean":    agg.get(f"f1_{k}_mean"),
                "f1_std":     agg.get(f"f1_{k}_std"),
                "prec_mean":  agg.get(f"precision_{k}_mean"),
                "recall_mean":agg.get(f"recall_{k}_mean"),
                "selscore":   scores_per_thr[thr],
            })
        mlflow.log_text(pd.DataFrame(thr_rows).to_csv(index=False), "threshold_summary.csv")

        # ── Figures
        for thr in thresholds:
            k = thr_key(thr)
            if len(oof_labels) > 0:
                oof_preds = (oof_probs >= thr).astype(int)
                cm_df  = build_confusion_matrix_df(oof_labels, oof_preds)
                fig    = _confusion_matrix_fig(cm_df, title=f"OOF Confusion Matrix (thr={thr})")
                mlflow.log_figure(fig, f"confusion_matrix_oof_{k}.png")
                plt.close(fig)

        if len(fold_records) > 0:
            fig = _fold_f1_fig(fold_records, winning_thr)
            mlflow.log_figure(fig, "fold_f1.png")
            plt.close(fig)

            fig = _threshold_curve_fig(fold_records, thresholds, lam)
            mlflow.log_figure(fig, "threshold_curve.png")
            plt.close(fig)

        # Holdout confusion matrix
        if len(y_holdout) > 0:
            cm_df = build_confusion_matrix_df(y_holdout, holdout_preds)
            fig   = _confusion_matrix_fig(cm_df, title=f"Holdout Confusion Matrix (thr={winning_thr})")
            mlflow.log_figure(fig, "confusion_matrix_holdout.png")
            plt.close(fig)

        # Feature importance (mean across fold models)
        fig = _feature_importance_fig(fold_models)
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close(fig)

        # ── Ensemble model
        X_sample = df_loso[FEATURE_COLS].head(5)
        y_sample = ensemble.predict(None, X_sample)
        signature = infer_signature(X_sample, y_sample)

        mlflow.pyfunc.log_model(
            artifact_path="ensemble_model",
            python_model=ensemble,
            signature=signature,
            input_example=X_sample.head(1),
        )

        run_id = run.info.run_id

    log.info("MLflow run logged: %s", run_id)
    return run_id


def register_champion(run_id: str, winning_thr: float, loso_metrics: dict) -> None:
    """Register the best run's ensemble to the Model Registry as 'champion'.

    Tags the model version with the winning threshold and key LOSO metrics.
    """
    setup_mlflow()
    client = mlflow.tracking.MlflowClient()

    result = mlflow.register_model(
        f"runs:/{run_id}/ensemble_model",
        MODEL_REGISTRY_NAME,
    )
    version = result.version

    client.set_registered_model_alias(MODEL_REGISTRY_NAME, MODEL_CHAMPION_ALIAS, version)

    tag_pairs = {
        "winning_threshold":    str(winning_thr),
        "loso_f1_mean":         str(loso_metrics.get(f"f1_{thr_key(winning_thr)}_mean", "")),
        "loso_f1_std":          str(loso_metrics.get(f"f1_{thr_key(winning_thr)}_std",  "")),
        "loso_pr_auc_mean":     str(loso_metrics.get("pr_auc_mean", "")),
        "holdout_f1":           str(loso_metrics.get(f"holdout_f1_{thr_key(winning_thr)}", "")),
    }
    for tag, val in tag_pairs.items():
        client.set_model_version_tag(MODEL_REGISTRY_NAME, version, tag, val)

    log.info(
        "Registered %s v%s as @%s (threshold=%.2f)",
        MODEL_REGISTRY_NAME, version, MODEL_CHAMPION_ALIAS, winning_thr,
    )


# ── Private helpers

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL,
        ).decode().strip()
        return bool(out)
    except Exception:
        return False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _confusion_matrix_fig(cm_df: pd.DataFrame, title: str = "Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm_df.values, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred_0", "pred_1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["actual_0", "actual_1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_df.values[i, j]), ha="center", va="center", fontsize=12)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def _fold_f1_fig(fold_records: list[dict], winning_thr: float):
    k        = f"f1_{thr_key(winning_thr)}"
    subjects = [r["subject"] for r in fold_records]
    f1s      = [r[k] for r in fold_records]
    mean_f1  = float(np.nanmean(f1s))

    fig, ax = plt.subplots(figsize=(max(6, len(subjects) * 0.5), 4))
    ax.bar(range(len(subjects)), f1s, color="#4878D0", width=0.6)
    ax.axhline(mean_f1, color="red",    linestyle="--", linewidth=1.3, label=f"mean F1={mean_f1:.3f}")
    ax.axhline(0.62,    color="orange", linestyle=":",  linewidth=1.2, label="benchmark=0.62")
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels([f"S{s}" for s in subjects], rotation=45)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title(f"LOSO F1 per Fold (thr={winning_thr})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def _threshold_curve_fig(fold_records: list[dict], thresholds: list[float], lam: float):
    from cogload.evaluation.metrics import selection_score as sel_score
    means   = [float(np.nanmean([r[f"f1_{thr_key(t)}"] for r in fold_records])) for t in thresholds]
    stds    = [float(np.nanstd( [r[f"f1_{thr_key(t)}"] for r in fold_records])) for t in thresholds]
    selscores = [sel_score(fold_records, t, lam) for t in thresholds]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(thresholds, means, yerr=stds, marker="o", label="mean F1 ± std", color="#4878D0")
    ax.plot(thresholds, selscores, marker="s", linestyle="--", label="selection score", color="red")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("F1 vs Decision Threshold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def _feature_importance_fig(fold_models: list):
    importances = np.array([m.feature_importances_ for m in fold_models])
    mean_imp = importances.mean(axis=0)
    idx      = np.argsort(mean_imp)[::-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(mean_imp)), mean_imp[idx], color="#4878D0")
    ax.set_xticks(range(len(mean_imp)))
    ax.set_xticklabels([FEATURE_COLS[i] for i in idx], rotation=45, ha="right", fontsize=7)
    ax.set_title("Mean Feature Importance (across folds)")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    return fig
