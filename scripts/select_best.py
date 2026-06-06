#!/usr/bin/env python
"""Scan all MLflow runs, build a leaderboard, and register the champion ensemble.

Usage:
    python scripts/select_best.py
    python scripts/select_best.py --dry-run   # print leaderboard without registering
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mlflow
import pandas as pd

from cogload.config import MLFLOW_EXPERIMENT, MLFLOW_URI
from cogload.evaluation.selection import best_run_and_threshold, build_leaderboard
from cogload.tracking.mlflow_utils import register_champion, setup_mlflow

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print leaderboard without registering.")
    args = parser.parse_args()

    setup_mlflow()
    client = mlflow.tracking.MlflowClient(MLFLOW_URI)

    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if exp is None:
        print(f"Experiment '{MLFLOW_EXPERIMENT}' not found. Run experiments first.")
        sys.exit(1)

    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="tags.stage = 'loso-xgboost-ensemble'",
        max_results=500,
    )

    if not runs:
        print("No completed runs found.")
        sys.exit(1)

    leaderboard = build_leaderboard(runs)
    print("\n── Leaderboard (top 10) ────────────────────────────────────")
    print(leaderboard.head(10).to_string(index=False))
    print()

    # Save full leaderboard as artifact on the best run
    best_run_id, best_thr, best_score = best_run_and_threshold(runs)
    print(f"Champion: run_id={best_run_id}  threshold={best_thr}  selscore={best_score:.4f}")

    if args.dry_run:
        print("(dry-run — skipping registration)")
        return

    # Fetch the winning run's metrics to attach as registry tags
    best_run  = client.get_run(best_run_id)
    loso_metrics = best_run.data.metrics

    register_champion(best_run_id, best_thr, loso_metrics)

    # Log leaderboard CSV back onto the winning run
    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_text(leaderboard.to_csv(index=False), "leaderboard.csv")

    print(f"\nModel registered: cogload-ensemble@champion")
    print("View at http://localhost:5001/#/models")


if __name__ == "__main__":
    main()
