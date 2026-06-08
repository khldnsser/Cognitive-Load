#!/usr/bin/env python
"""Roll back the champion model to the previous version.

Swaps the 'champion' alias back to whichever version was saved as
'previous-champion' by the last select_best.py promotion. Both aliases
are swapped, so a second rollback undoes the first.

Usage:
    python scripts/rollback_champion.py [--dry-run]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mlflow

from cogload.config import MLFLOW_URI, MODEL_REGISTRY_NAME
from cogload.tracking.mlflow_utils import setup_mlflow

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    setup_mlflow()
    client = mlflow.tracking.MlflowClient(MLFLOW_URI)

    try:
        current_mv = client.get_model_version_by_alias(MODEL_REGISTRY_NAME, "champion")
    except Exception:
        print("No champion registered — nothing to roll back.")
        sys.exit(1)

    try:
        prev_mv = client.get_model_version_by_alias(MODEL_REGISTRY_NAME, "previous-champion")
    except Exception:
        print("No previous-champion alias found — cannot roll back further.")
        sys.exit(1)

    print(f"Current champion : v{current_mv.version}  (run {current_mv.run_id[:8]})")
    print(f"Previous champion: v{prev_mv.version}  (run {prev_mv.run_id[:8]})")

    if args.dry_run:
        print("(dry-run — no changes made)")
        return

    # Swap: previous → champion, current → previous-champion
    client.set_registered_model_alias(MODEL_REGISTRY_NAME, "champion", prev_mv.version)
    client.set_registered_model_alias(MODEL_REGISTRY_NAME, "previous-champion", current_mv.version)

    log.info(
        "Rolled back: champion is now v%s. Previous-champion is now v%s.",
        prev_mv.version, current_mv.version,
    )
    print(f"\nRollback complete. Restart IEP-1 to reload the model.")
    print("  docker compose restart iep1-inference")


if __name__ == "__main__":
    main()
