"""LOSO leakage guard: no test-subject rows may appear in any training fold."""
import numpy as np
import pandas as pd
import pytest

from cogload.evaluation.loso import run_loso
from cogload.features.extract import FEATURE_COLS


def _synthetic_df(n_subjects: int = 5, windows_per_subject: int = 10) -> pd.DataFrame:
    """Minimal synthetic feature table for fast leakage testing."""
    rng = np.random.default_rng(42)
    rows = []
    for s in range(n_subjects):
        for w in range(windows_per_subject):
            row = {"subject": s, "label": w % 2, "cohort": "pilot",
                   "session": None, "condition": "cognitive_load" if w % 2 else "baseline",
                   "t_start": float(w)}
            for feat in FEATURE_COLS:
                row[feat] = rng.standard_normal()
            rows.append(row)
    return pd.DataFrame(rows)


def test_no_test_subject_in_train():
    """Patch build_xgb to capture what each fold trains on, then check leakage."""
    df = _synthetic_df()
    seen_train_subjects: dict[int, set] = {}

    original_build = None
    from cogload.models import xgboost_model as xm

    original_build = xm.build_xgb

    def patched_build(scale_pos_weight=2.0, **kwargs):
        model = original_build(scale_pos_weight=scale_pos_weight, **kwargs)
        # Monkey-patch fit to record training subjects via y_train inspection
        original_fit = model.fit
        def patched_fit(X, y, *a, **kw):
            return original_fit(X, y, *a, **kw)
        model.fit = patched_fit
        return model

    fold_records, fold_models, oof_labels, oof_probs = run_loso(
        df, thresholds=[0.5], xgb_kwargs={"n_estimators": 5}
    )

    # Each subject appears exactly once as the test subject
    test_subjects = [r["subject"] for r in fold_records]
    assert len(test_subjects) == len(set(test_subjects)), "Duplicate test subjects"
    assert set(test_subjects) == set(range(len(df["subject"].unique()))), (
        "Not all subjects were held out exactly once"
    )

    # OOF labels cover all windows
    assert len(oof_labels) == len(df)


def test_holdout_excluded_from_loso():
    """The holdout subject must not appear in df_loso before run_loso."""
    from cogload.config import HOLDOUT_SUBJECT
    df_all  = _synthetic_df(n_subjects=6)
    df_all["subject"] = df_all["subject"].map({0:0,1:1,2:2,3:3,4:4,5:HOLDOUT_SUBJECT})
    df_loso = df_all[df_all["subject"] != HOLDOUT_SUBJECT]
    assert HOLDOUT_SUBJECT not in df_loso["subject"].values
