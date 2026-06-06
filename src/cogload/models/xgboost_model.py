"""XGBoost estimator factory."""
from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


def build_xgb(scale_pos_weight: float = 2.0, **kwargs) -> XGBClassifier:
    """Return a configured XGBClassifier.

    Args:
        scale_pos_weight: count(negatives) / count(positives) from the training
                          fold — compensates for the ~2:1 time imbalance.
        **kwargs: override any XGBClassifier hyperparameter.
    """
    defaults = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    defaults.update(kwargs)
    return XGBClassifier(**defaults)


def compute_scale_pos_weight(y_train: pd.Series | np.ndarray) -> float:
    """count(negatives) / count(positives) from training labels."""
    y = np.asarray(y_train)
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    return float(n_neg) / float(n_pos) if n_pos > 0 else 1.0
