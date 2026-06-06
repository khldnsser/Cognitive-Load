"""Champion model loader with local-MLflow / S3 fallback strategy.

Priority order at startup:
  1. MODEL_URI env var — caller can point at anything (s3://..., runs:/..., file://...)
  2. Local MLflow registry (MLFLOW_URI from config) — used during development
  3. Raises RuntimeError with a clear message if both fail

The loaded object is a FoldEnsembleModel instance, accessed via
mlflow.pyfunc.load_model(...).unwrap_python_model().  This gives access to
.predict_proba() which IEP-1 needs (the pyfunc wrapper only exposes .predict).
"""
from __future__ import annotations

import logging
import os

import mlflow.pyfunc

from cogload.config import MLFLOW_URI, MODEL_REGISTRY_NAME, MODEL_CHAMPION_ALIAS
from cogload.models.ensemble import FoldEnsembleModel

logger = logging.getLogger(__name__)

_REGISTRY_URI = f"models:/{MODEL_REGISTRY_NAME}@{MODEL_CHAMPION_ALIAS}"


def load_champion() -> FoldEnsembleModel:
    """Load and return the champion FoldEnsembleModel.

    Reads MODEL_URI env var first; falls back to the local MLflow registry.
    """
    uri = os.environ.get("MODEL_URI", "").strip()

    if uri:
        logger.info("Loading model from MODEL_URI=%s", uri)
        return _load(uri)

    # Fall back to local MLflow registry
    logger.info("MODEL_URI not set — loading from MLflow registry at %s", MLFLOW_URI)
    mlflow.set_tracking_uri(MLFLOW_URI)
    return _load(_REGISTRY_URI)


def _load(uri: str) -> FoldEnsembleModel:
    pyfunc_model = mlflow.pyfunc.load_model(uri)
    model = pyfunc_model.unwrap_python_model()
    if not isinstance(model, FoldEnsembleModel):
        raise TypeError(
            f"Expected FoldEnsembleModel, got {type(model).__name__}. "
            "Is the registered model the correct type?"
        )
    logger.info(
        "Champion loaded: %d fold models, threshold=%.2f, features=%d",
        len(model._models),
        model.threshold,
        len(model.feature_cols),
    )
    return model
