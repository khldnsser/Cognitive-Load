"""Tests for metric computation and model selection formula."""
import math
import numpy as np
import pytest

from cogload.evaluation.metrics import (
    aggregate_fold_metrics,
    compute_fold_metrics,
    selection_score,
    thr_key,
)


def test_thr_key():
    assert thr_key(0.2) == "t020"
    assert thr_key(0.5) == "t050"
    assert thr_key(0.8) == "t080"


def test_compute_fold_metrics_keys():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.3, 0.6, 0.9, 0.4, 0.7])
    m = compute_fold_metrics(y_true, y_prob, thresholds=[0.5])
    assert "roc_auc"     in m
    assert "pr_auc"      in m
    assert "f1_t050"     in m
    assert "precision_t050" in m
    assert "recall_t050" in m


def test_compute_fold_metrics_values():
    # Perfect classifier
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_fold_metrics(y_true, y_prob, thresholds=[0.5])
    assert m["f1_t050"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)


def test_single_class_fold_returns_nan():
    y_true = np.array([1, 1, 1])
    y_prob = np.array([0.8, 0.9, 0.7])
    m = compute_fold_metrics(y_true, y_prob, thresholds=[0.5])
    assert math.isnan(m["roc_auc"])
    assert math.isnan(m["pr_auc"])


def test_aggregate_mean_std():
    records = [
        {"subject": 0, "f1_t050": 0.6, "roc_auc": 0.7},
        {"subject": 1, "f1_t050": 0.8, "roc_auc": 0.9},
    ]
    agg = aggregate_fold_metrics(records)
    assert agg["f1_t050_mean"] == pytest.approx(0.7)
    assert agg["roc_auc_mean"] == pytest.approx(0.8)


def test_selection_score_penalises_high_std():
    records_stable   = [{"subject": i, "f1_t050": 0.7} for i in range(5)]
    records_unstable = [{"subject": i, "f1_t050": 0.7 + (0.3 if i % 2 == 0 else -0.3)} for i in range(5)]
    s_stable   = selection_score(records_stable,   0.5, lam=0.5)
    s_unstable = selection_score(records_unstable, 0.5, lam=0.5)
    assert s_stable > s_unstable
