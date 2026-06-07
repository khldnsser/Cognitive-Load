"""Register a minimal FoldEnsembleModel in MLflow for CI integration tests.

Creates a 1-fold XGBoost model trained on random data with the correct feature
schema, then registers it as cogload-ensemble@champion so IEP-1 can load it.

Usage:
    python scripts/register_test_model.py [--mlflow-uri http://localhost:5001]
"""
from __future__ import annotations

import argparse
import logging
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.pyfunc

from cogload.config import MODEL_REGISTRY_NAME, MODEL_CHAMPION_ALIAS
from cogload.features.extract import FEATURE_COLS
from cogload.models.ensemble import FoldEnsembleModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(mlflow_uri: str) -> None:
    mlflow.set_tracking_uri(mlflow_uri)

    rng = np.random.default_rng(42)
    n = 50
    X = rng.standard_normal((n, len(FEATURE_COLS)))
    y = (rng.random(n) > 0.5).astype(int)

    clf = xgb.XGBClassifier(n_estimators=3, max_depth=2, n_jobs=1, eval_metric="logloss")
    clf.fit(X, y)

    model = FoldEnsembleModel(models=[clf], threshold=0.20, feature_cols=FEATURE_COLS)

    experiment = mlflow.set_experiment("ci-test-model")
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        mlflow.pyfunc.log_model(artifact_path="model", python_model=model)
        run_id = run.info.run_id

    logger.info("Logged model at run %s", run_id)

    client = mlflow.tracking.MlflowClient()
    try:
        client.create_registered_model(MODEL_REGISTRY_NAME)
    except Exception:
        pass  # already exists

    mv = client.create_model_version(
        name=MODEL_REGISTRY_NAME,
        source=f"runs:/{run_id}/model",
        run_id=run_id,
    )
    client.set_registered_model_alias(MODEL_REGISTRY_NAME, MODEL_CHAMPION_ALIAS, mv.version)
    logger.info("Registered %s@%s (version %s)", MODEL_REGISTRY_NAME, MODEL_CHAMPION_ALIAS, mv.version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow-uri", default="http://localhost:5001")
    args = parser.parse_args()
    main(args.mlflow_uri)
