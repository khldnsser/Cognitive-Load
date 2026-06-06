from pathlib import Path

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent.parent

RAW_DATA_ROOT = REPO_ROOT / "raw_data"
PILOT_DIR     = RAW_DATA_ROOT / "pilot"
SURVEY_DIR    = RAW_DATA_ROOT / "survey_gamification"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

BVP_FILE  = "empatica_bvp.csv"
EDA_FILE  = "empatica_eda.csv"
TEMP_FILE = "empatica_temp.csv"
EDA_TYPO  = "empatica_emda.csv"   # subject 3 cognitive_load filename quirk

BVP_HZ  = 64
EDA_HZ  = 4
TEMP_HZ = 4

CONDITIONS = {"cognitive_load": 1, "baseline": 0}

# Subject held out entirely from LOSO — used only for final ensemble evaluation.
HOLDOUT_SUBJECT = 24

WINDOW_S          = 30
HOP_S             = 15
BASELINE_OFFSET_S = 30
MIN_BEATS         = 5    # minimum BVP peaks for valid HRV features

# Decision thresholds swept post-hoc on predict_proba output.
THRESHOLDS = [0.2, 0.5, 0.8]

# Penalises unstable configs in selection_score = mean_f1 - LAM * std_f1.
SELECTION_LAMBDA = 0.5

# Features z-scored per subject (level/amplitude features with large inter-subject offsets).
# Slope/derivative features are excluded — they are already offset-invariant.
NORMALIZE_FEATURES = [
    "eda_tonic_mean",
    "eda_tonic_std",
    "scr_mean_amp",
    "scr_sum_amp",
    "hr_mean",
    "rmssd",
    "sdnn",
    "pnn50",
    "ibi_mean",
    "ibi_std",
    "temp_mean",
    "temp_std",
    "temp_min",
]

MLFLOW_URI            = "http://localhost:5001"
MLFLOW_EXPERIMENT     = "cognitive-load-xgboost-loso"
MODEL_REGISTRY_NAME   = "cogload-ensemble"
MODEL_CHAMPION_ALIAS  = "champion"
