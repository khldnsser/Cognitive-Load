# Cognitive Load Detection — XGBoost LOSO Pipeline Report

## Problem

Predict cognitive load (binary: high/low) from wrist-worn physiological signals (BVP, EDA, Temperature) using a Leave-One-Subject-Out cross-validation framework. Benchmark to beat: **F1 = 0.62** (Grzeszczyk et al., PPG-only).

---

## Dataset

- **24 subjects** (pilot 0–10, survey 11–24; subject 19 absent)
- **Signals:** BVP @ 64Hz, EDA @ 4Hz, Temp @ 4Hz (Empatica E4)
- **Subject 24 held out entirely** — never seen during any training or config selection. Only used for the final honest evaluation of the deployed ensemble.

---

## Pipeline

### 1. Feature Extraction

Signals are sliced into overlapping windows. Per window, ~18 features extracted:

| Signal | Features |
|--------|----------|
| EDA | tonic mean/slope/std, SCR count/rate/mean amp/sum amp |
| BVP | HR mean, RMSSD, SDNN, pNN50, IBI mean/std |
| Temp | slope, mean, std, range, min |

**Per-subject z-score normalization** applied on level/amplitude features only (not slopes). Done using each subject's own data only — no label or cross-subject information used, so this is not leakage.

**Baseline offset** trims the start of each baseline condition window. Subjects take time to settle after a stressor ends — including those contaminated early seconds inflates class separation artificially.

### 2. LOSO Cross-Validation (subjects 0–23)

23 folds. Each fold:
1. Train XGBoost on 22 subjects
2. Call `predict_proba` on the 1 held-out subject
3. Apply 3 thresholds (0.2, 0.5, 0.8) to the **same probabilities** — no retraining

**Why LOSO?** Physiological signals are highly person-specific. Random train/test split would let the model memorize a subject's baseline physiology, inflating metrics. LOSO grades how well the model generalizes to a *new, unseen person*.

**Why post-hoc threshold sweep?** The threshold is not a training parameter — it's applied after probabilities are computed. Sweeping it costs nothing extra (same 23 models, same probabilities) and lets us tune the precision/recall trade-off at deployment time.

### 3. Ensemble

After LOSO, all 23 fold models are kept. At inference:

```
ensemble_proba = mean(fold_model_i.predict_proba(X) for i in 0..22)
prediction = ensemble_proba >= threshold
```

Averaging probabilities reduces variance from any single fold's quirks.

### 4. Champion Selection

For each of the 18 combinations (6 configs × 3 thresholds):

```
selection_score = mean_F1 − 0.5 × std_F1
```

The `−0.5·std` term penalizes inconsistent configs — a model that's 0.95 F1 on some subjects and 0.60 on others is not deployable. We want a consistently good model, not a lucky one.

**Subject 24 is never used here.** Selection is purely on the LOSO folds over subjects 0–23.

### 5. Final Evaluation

The winning ensemble is run on **subject 24** at the winning threshold. This is the only unbiased number — subject 24 was never seen by any fold model and was not used in config selection.

---

## Experiment Grid

All runs: `n_estimators=200`, `lr=0.05`, `subsample=colsample=0.8`, `thresholds=[0.2, 0.5, 0.8]`

| Run | Window | Offset | Depth | F1 mean ± std | ROC AUC | Holdout F1 |
|-----|--------|--------|-------|----------------|---------|------------|
| **awesome-goat-735 ★** | **60s** | **30s** | **4** | **0.910 ± 0.083** | **0.919** | **0.785** |
| clumsy-vole-479 | 30s | 60s | 4 | 0.897 ± 0.078 | 0.893 | 0.791 |
| dashing-dog-750 | 30s | 30s | 4 | 0.871 ± 0.093 | 0.885 | 0.742 |
| selective-pig-494 | 30s | 30s | 3 | 0.868 ± 0.092 | 0.883 | 0.769 |
| secretive-hawk-552 | 30s | 30s | 6 | 0.862 ± 0.101 | 0.890 | 0.742 |
| dashing-crane-955 | 30s | 0s | 4 | 0.843 ± 0.126 | 0.868 | 0.667 |

---

## Champion

**Config:** w=60s, hop=15s, offset=30s, depth=4, **threshold=0.20**

| Metric | Value |
|--------|-------|
| LOSO F1 mean | 0.910 |
| LOSO F1 std | 0.083 |
| Selection score | 0.868 |
| ROC AUC mean | 0.919 |
| PR AUC mean | 0.956 |
| **Subject 24 holdout F1** | **0.785** |
| Benchmark (Grzeszczyk) | 0.620 |

Registered in MLflow Model Registry as `cogload-ensemble@champion`.

---

## Key Findings

**Longer windows win.** 60s captures more complete HRV cycles and EDA tonic drift than 30s. HRV metrics like RMSSD and pNN50 are computed from inter-beat intervals — you need enough beats for stable estimates.

**Offset matters more than depth.** offset=0 had the worst F1 mean (0.843) *and* highest std (0.126) — including the contaminated recovery period adds noise that hurts cross-subject generalization. Depth=6 also underperformed depth=4, suggesting mild overfitting on this small dataset.

**Threshold=0.20 won.** The ensemble's mean probabilities are well-calibrated above 0.5 for true positives. A low threshold captures high recall, which suits the use case: it's worse to miss a cognitive overload event than to trigger a false alert.

---

## Infrastructure

- **MLflow** (SQLite backend + `--serve-artifacts`): tracks params, metrics, datasets, artifacts, and model registry. All artifact I/O proxied through the server — no direct filesystem writes from the client.
- **Prometheus + Grafana**: monitoring stack, provisioned via Docker Compose.
- Package: installable via `pip install -e .` (`pyproject.toml`).
- Tests: 16 pytest tests covering windowing, alignment, metrics, and a LOSO leakage guard.
