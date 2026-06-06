#!/usr/bin/env python
"""Sweep over the experiment grid — builds all MLflow runs.

Each combination of (feature table config, XGBoost hyperparams) is one run.
Feature tables must already be cached via build_features.py.

Usage:
    python scripts/run_experiments.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cogload.experiment import ExperimentConfig, run_experiment

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ── Experiment grid ────────────────────────────────────────────────────────────
# Each entry specifies one (window, hop, offset) feature table and the
# XGBoost hyperparameters to evaluate against it.
# Add or remove entries freely; each produces one MLflow run.

GRID: list[ExperimentConfig] = [
    # 30s window, offset sweep
    ExperimentConfig(window_s=30, hop_s=15, baseline_offset_s=0,  max_depth=4),
    ExperimentConfig(window_s=30, hop_s=15, baseline_offset_s=30, max_depth=4),
    ExperimentConfig(window_s=30, hop_s=15, baseline_offset_s=60, max_depth=4),
    # 30s window, depth sweep (offset=30)
    ExperimentConfig(window_s=30, hop_s=15, baseline_offset_s=30, max_depth=3),
    ExperimentConfig(window_s=30, hop_s=15, baseline_offset_s=30, max_depth=6),
    # 60s window (helps frequency-domain HRV)
    ExperimentConfig(window_s=60, hop_s=15, baseline_offset_s=30, max_depth=4),
]
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    run_ids: list[str] = []
    for i, cfg in enumerate(GRID, 1):
        print(f"\n[{i}/{len(GRID)}] {cfg}")
        try:
            run_id = run_experiment(cfg)
            run_ids.append(run_id)
            print(f"  → run_id: {run_id}")
        except Exception as exc:
            print(f"  ✗ FAILED: {exc}")

    print(f"\nCompleted {len(run_ids)}/{len(GRID)} runs.")
    print("Run  python scripts/select_best.py  to register the champion model.")


if __name__ == "__main__":
    main()
