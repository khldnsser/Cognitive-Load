# Cognitive Load Detection — CogWear

Binary classification (high / low cognitive load) from wrist physiological signals using the CogWear dataset.
Signals: BVP (64 Hz), EDA (4 Hz), Temperature (4 Hz) from Empatica E4. Muse EEG and Samsung BVP are out of scope.

**Champion result:** LOSO F1 = 0.910 ± 0.083 · Holdout (Subject 24) F1 = 0.785 · Benchmark F1 = 0.62 (Grzeszczyk et al.)

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Start infrastructure (MLflow · Prometheus · Grafana):

```bash
docker compose up -d
# MLflow  → http://localhost:5001
# Grafana → http://localhost:3000  (admin / admin)
```

---

## Running the Pipeline

```bash
# Build and cache feature tables
python scripts/build_features.py

# Run all 6 experiment configs
python scripts/run_experiments.py

# Register the champion model
python scripts/select_best.py

# Tests
pytest tests/
```

Cached parquet files live in `data/processed/`. Re-extract only if features, window size, or baseline offset change.

---

## Results

| Run | Window | Offset | Depth | F1 mean ± std | Holdout F1 |
|-----|--------|--------|-------|----------------|------------|
| **awesome-goat-735 ★** | **60s** | **30s** | **4** | **0.910 ± 0.083** | **0.785** |
| clumsy-vole-479 | 30s | 60s | 4 | 0.897 ± 0.078 | 0.791 |
| dashing-dog-750 | 30s | 30s | 4 | 0.871 ± 0.093 | 0.742 |
| selective-pig-494 | 30s | 30s | 3 | 0.868 ± 0.092 | 0.769 |
| secretive-hawk-552 | 30s | 30s | 6 | 0.862 ± 0.101 | 0.742 |
| dashing-crane-955 | 30s | 0s | 4 | 0.843 ± 0.126 | 0.667 |

Champion config: `window=60s  hop=15s  baseline_offset=30s  max_depth=4  threshold=0.20`
Registered as `cogload-ensemble@champion` in the MLflow Model Registry.

See `results_report.md` for full analysis.

---

## Architecture

### Training pipeline (complete)

```
raw signals
    └─ feature extraction (18 features per 60s window)
           └─ per-subject z-score normalization
                  └─ LOSO cross-validation (23 folds, XGBoost)
                         └─ FoldEnsembleModel (mean of 23 fold probabilities)
                                └─ champion selection + MLflow registry
```

### Serving layer (in progress)

```
client
  └─ EEP: FastAPI gateway  (/predict  /calibrate  /health  /metrics)
          ├─ IEP-1: Inference service   — loads champion, runs prediction
          └─ IEP-2: Calibration service — computes per-subject normalization params
                                               └─ Prometheus /metrics → Grafana dashboards
```

---

## Project Structure

```
src/cogload/
  config.py                    # all constants (sample rates, paths, thresholds, MLflow URI)
  data/loader.py               # discovers recordings; handles subject-3 filename typo
  data/alignment.py            # clips signals to overlap; applies baseline offset
  features/windowing.py        # timestamp-based sliding window generator
  features/eda.py              # neurokit2 EDA decomposition → 7 features
  features/bvp.py              # neurokit2 HRV extraction → 6 features
  features/temp.py             # temperature slope + statistics → 5 features
  features/extract.py          # FEATURE_COLS (18) + extract_window_features()
  preprocessing/normalization.py  # per-subject z-score (Operation 1)
  evaluation/loso.py           # run_loso() + assert_no_leakage()
  evaluation/metrics.py        # compute_fold_metrics(), selection_score = mean_F1 - 0.5*std_F1
  evaluation/selection.py      # build_leaderboard() from MLflow runs
  models/xgboost_model.py      # build_xgb() factory + scale_pos_weight
  models/ensemble.py           # FoldEnsembleModel: averages fold predict_proba, applies threshold
  tracking/mlflow_utils.py     # log_loso_run() + register_champion()
  pipeline.py                  # build / cache / load feature parquet tables
  experiment.py                # ExperimentConfig + run_experiment()

scripts/
  build_features.py            # extract and cache feature tables
  run_experiments.py           # 6-config experiment grid
  select_best.py               # register champion (--dry-run flag available)

tests/                         # 16 unit tests: alignment, windowing, metrics, leakage
data/processed/                # cached parquet files (features_w{W}_h{H}_off{O}.parquet)
docker/
  prometheus/prometheus.yml    # scrape config (serving targets not yet wired)
  grafana/provisioning/        # datasource provisioned; dashboards to be added
```

---

## Dataset

24 subjects. Pilot study: subjects 0–10. Survey gamification: subjects 11–24 (pre/post sessions).
Subject 24 is the final holdout — never used during training or champion selection.
Subject 19 is absent from the dataset.

See `raw_data/README.md` for signal and folder structure details.
