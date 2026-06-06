#!/usr/bin/env python
"""Run a single LOSO experiment and log results to MLflow.

Usage:
    python scripts/train_loso.py
    python scripts/train_loso.py --window 30 --hop 15 --offset 30
    python scripts/train_loso.py --n_estimators 300 --max_depth 6
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cogload.experiment import ExperimentConfig, run_experiment

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window",        type=float, default=30)
    parser.add_argument("--hop",           type=float, default=15)
    parser.add_argument("--offset",        type=float, default=30)
    parser.add_argument("--n_estimators",  type=int,   default=200)
    parser.add_argument("--max_depth",     type=int,   default=4)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--subsample",     type=float, default=0.8)
    parser.add_argument("--colsample",     type=float, default=0.8)
    args = parser.parse_args()

    cfg = ExperimentConfig(
        window_s=args.window,
        hop_s=args.hop,
        baseline_offset_s=args.offset,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample=args.colsample,
    )

    print(f"Running experiment: {cfg}")
    run_id = run_experiment(cfg)
    print(f"\nMLflow run: {run_id}")
    print("View at http://localhost:5001")


if __name__ == "__main__":
    main()
