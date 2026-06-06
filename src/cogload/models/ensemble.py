"""Fold ensemble model for mlflow.pyfunc deployment.

Averages predict_proba across all fold models (bagging-style), then applies
a fixed decision threshold. The threshold and feature column list are baked
in at construction time and stored alongside the model in the registry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import mlflow.pyfunc


class FoldEnsembleModel(mlflow.pyfunc.PythonModel):
    """Average of N fold XGBoost models with a fixed decision threshold."""

    def __init__(
        self,
        models: list,
        threshold: float,
        feature_cols: list[str],
    ) -> None:
        self._models      = models
        self.threshold    = threshold
        self.feature_cols = feature_cols

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
    ) -> np.ndarray:
        """Return binary predictions (0/1) using the ensemble probability mean."""
        return (self.predict_proba(model_input) >= self.threshold).astype(int)

    def predict_proba(self, model_input: pd.DataFrame) -> np.ndarray:
        """Return averaged probability of class 1 across all fold models."""
        X = model_input[self.feature_cols].values
        probs = np.array([m.predict_proba(X)[:, 1] for m in self._models])
        return probs.mean(axis=0)
