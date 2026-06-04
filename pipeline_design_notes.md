# Pipeline Design Notes — LOSO, Windowing & Inference

Concise summary of design decisions discussed. Scope: Empatica-only (BVP, EDA, Temp); Muse/EEG and Samsung dropped.

---

## LOSO (Leave-One-Subject-Out)

- **What it is:** an *evaluation* protocol, not a training mode. An outer loop that wraps train→test.
- **Mechanics:** for each subject S → train on all *other* subjects, test on S, record F1. ~24 subjects → 24 folds → 24 F1 scores → report **mean ± std**.
- **The 24 models are throwaway** — they only produce honest scores. The model you *ship* is retrained on all subjects once LOSO confirms the approach generalizes.
- **Why:** deployment target is new, unseen users. LOSO simulates that. Window-level splits leak (adjacent windows share data) and inflate metrics 15–20 pts — the leak behind Hassan et al.'s fake 93%. Honest benchmark to beat: **F1 ≈ 0.62**.
- **Normalization rule:** fit z-score mean/std on **train subjects only**, apply same numbers to held-out subject. Recomputing on test = leakage.
- **Hyperparameter tuning** happens *inside* the training folds, never on the held-out subject.
- **Model selection:** rank by mean LOSO F1; use fold **std** as risk measure and **PR-AUC** for threshold tuning. (Std/PR-AUC are recommendations, not in CLAUDE.md.)

---

## Windowing — Training vs Inference

- The model is **window-blind**: a pure function `feature_vector → probability`. No memory of neighbors. The sliding window lives in the *pipeline* (training) and *application* (inference), not the model.
- **Sliding window used in BOTH:**
  - *Training:* slide over each recording → many labeled rows (one row per window) → build dataset.
  - *Inference:* slide over live stream → one feature vector per hop → predict.
- **Window size MUST match** between train and inference (a 30s RMSSD ≠ a 60s RMSSD).
- Each row from one recording shares: **same label** (from folder), **same subject ID** (LOSO split key), same condition/session.

### Row generation (per condition-recording)
1. Start windowing at `max(all sensor starts)`, stop before `min(all sensor ends)` — the overlap region (near-trivial now that EEG is dropped; all Empatica signals start within ~1s).
2. For **baseline only**, start some seconds in to skip post-Stroop recovery contamination (separate offset from the alignment clip). **This start offset is a tunable hyperparameter — sweep 0–60s** (see Baseline Discard section).
3. Slice each signal by **timestamp, not index** → handles different sample rates for free (30s yields ~120 EDA, ~1920 BVP, ~120 Temp samples).
4. Extract features → one row. Slide by hop. Repeat to end.
5. Stack rows across all subjects × conditions × sessions → tabular dataset for the LOSO loop.

---

## Window Size vs Hop/Stride

| Parameter | Role | Tunable as | Coupling |
|---|---|---|---|
| **Window size** | How much recent data each decision sees | Hyperparameter — changes feature meaning → re-extract + retrain | **Coupled**: train & inference must match; frozen at deploy |
| **Training hop** | Dataset density | Weak knob (smaller hop = more *redundant*, correlated rows) | Affects row count / class balance only |
| **Inference hop** | How often a fresh decision fires | Pure application setting | **Decoupled** from model — change anytime, no retrain |

- Window=30s, hop=15s → a decision every **15s** over a **30s lookback** (the two are independent).
- **~30s latency floor:** newly-started load isn't fully detected until the window fills.

---

## Features (per window, Empatica-only)

Extract **raw** features in the pipeline; standardize later (see next section). ~18–20 features total — fine for XGBoost/logistic regression with ~24 subjects.

### EDA (4 Hz) — strongest sympathetic signal
First decompose into **tonic (SCL)** + **phasic (SCR)** (`neurokit2.eda_phasic` or cvxEDA).

| Feature | Standardize? |
|---|---|
| Tonic mean (SCL level) | **Yes** — the 10–20× inter-subject offset lives here |
| Tonic slope | Less critical (slope cancels offset) |
| Tonic std | Yes |
| SCR peak count | Mild |
| SCR rate (peaks/min) | Mild |
| SCR mean amplitude | Yes (amplitude scales with skin) |
| SCR sum of amplitudes | Yes |

### BVP (64 Hz) — cardiac autonomic; never use raw amplitude
Peak-detect → IBI → HRV (`neurokit2.ppg_process` + `hrv_time`).

| Feature | Standardize? |
|---|---|
| Mean HR (bpm) | **Yes** — resting HR varies 55 vs 75 bpm |
| RMSSD (workhorse; drops under load) | Yes |
| SDNN | Yes |
| pNN50 | Mild |
| Mean IBI / std IBI | Yes |
| LF/HF ratio *(OPTIONAL — shaky at 30s, needs ~60s)* | Yes |
| Signal-quality score | n/a (filter, not predict) |

### Temp (4 Hz) — slow vasoconstriction signal

| Feature | Standardize? |
|---|---|
| **Slope (°C/min)** — the key feature | **No** — slope already cancels the offset |
| Mean temp | **Yes** — individual offsets (31°C vs 34°C) |
| Std / range | Mild |
| Min temp in window | Yes |

### Optional features (only test if needed)
- **LF/HF (and frequency-domain HRV)** — unreliable below ~60s windows; keep optional, lean on time-domain HRV.
- **Signal-quality score** — used to flag/drop motion-corrupted windows, not as a predictor.
- **Lag/delta features** (e.g., `RMSSD_now − RMSSD_prev`) — add temporal context; only if smoothing proves insufficient. Must reset at recording boundaries.

### Empty/garbage windows
If BVP peak-detection finds <5 beats or EDA is flat → drop the window or emit NaN and let the model handle it.

---

## Standardization / Normalization (all notes)

**Two DISTINCT operations with DIFFERENT jobs.** They are easy to confuse but solve different problems — do both, in this order.

### Operation 1 — Per-subject normalization (this is what PERSONALIZES)
- **Job:** remove each person's individual baseline offset so a feature means the same thing across people.
- **Each subject is scaled against THEIR OWN mean/std:** `z = (x − subject_mean) / subject_std`, computed from *that subject's own data*.
- **Why a global mean/std can't do this:** one mean/std pooled over everyone does NOT personalize — it would encode *identity*, not load. Example (EDA): Subject A rest=1/load=2, Subject B rest=10/load=12. Global-z leaves A's *load* (2) below B's *rest* (10) → feature screams "who is this," not "are they loaded." Per-subject z (A vs own ~1.5, B vs own ~11) maps **both** to rest=−1, load=+1 → now aligned across people.
- **Not leakage** even though it touches the test subject's own data: it uses only that subject's *own* values and *no labels* — exactly what deployment does (calibrate on the user's own rest window).
- **In deployment:** estimate `subject_mean/std` from a ~30s known-rest calibration window at session start.

### Operation 2 — Train-fold standardization (the GLOBAL one; prevents leakage)
- **Job:** put the (already-personalized) feature columns on a common numeric scale (~mean 0, std 1) for the model's math. Does NOT personalize.
- **One mu/sd per feature column, pooled over training rows only**, applied to both train and test:
```python
mu = train_set[feature].mean()   # from the 23 training subjects ONLY
sd = train_set[feature].std()
train_set[feature] = (train_set[feature] - mu) / sd
test_set[feature]  = (test_set[feature]  - mu) / sd   # SAME mu, sd
```
- Computing `mu/sd` over the whole dataset (incl. held-out subject) = **leakage**, inflates LOSO score.
- Matters most for logistic regression / CNN; **tree models (XGBoost) are scale-invariant** so this step barely helps them.

### Where each lives in the pipeline
1. Build raw feature table (once).
2. **Operation 1 (per-subject):** normalize each subject against their own stats — can be applied up front (uses only own data, no fold dependency).
3. Enter LOSO loop → split by subject.
4. **Operation 2 (train-fold):** fit mu/sd on train rows, apply to train + test — **must be redone every fold**.
5. Train, evaluate.

### Which features need normalization
Proportional to how much the between-subject offset exceeds the load-induced change:
- **EDA absolute level (tonic mean)** → huge offset → **mandatory**.
- **BVP HR/HRV** (mean HR, RMSSD, SDNN, IBI) → moderate offset → **should** normalize (notebook: "HRV features will need individual-level baselines").
- **Temp slope** → offset already cancelled → **skip**. **Temp mean** → has offset → normalize.
- General rule: *level/amplitude* features normalize; *slope/derivative* features usually don't (they're offset-invariant).

> Note: CLAUDE.md currently frames per-subject normalization as EDA-only. Accurate framing: **mandatory for EDA, beneficial for HR/HRV, feature-dependent for Temp.**

---

## Inference / Application Layer (post-model, no retraining)

- Raw per-window output is i.i.d. and **jittery** — don't act on every output.
- **Smoothing / hysteresis:** require 2–3 consecutive "loaded" windows before suppressing notifications, and a few "rest" windows before re-enabling. Highest-value, lowest-effort, zero leakage risk.
- Optional model-side temporal context (only if smoothing insufficient): **lag/delta features** (must reset at recording boundaries) or a **sequence model** (1D CNN / LSTM).

---

## Class Imbalance & Baseline Discard (a real trade-off)

- The **baseline start offset is a tunable hyperparameter, swept anywhere in 0–60s** — how many seconds of post-Stroop recovery to discard before windowing baseline. It's a hypothesis (cleaner labels vs less data), so let LOSO decide the value.
- Counts balanced (35/35) but **time imbalanced**: cl≈280s vs baseline≈150s → ~2:1 rows before any discard.
- Discarding baseline seconds (label-noise fix) **worsens** this — it only trims the minority class:

| baseline start offset | usable | ~rows (10s hop) | ratio vs cl=1 |
|---|---|---|---|
| 0s | 150s | ~13 | 2 : 1 |
| 30s | 120s | ~10 | ~2.6 : 1 |
| 60s | 90s | ~7 | ~3.7 : 1 |

- **Resolution:**
  1. Treat **baseline start offset as a tunable knob** (sweep 0–60s via LOSO, e.g. 0/15/30/45/60) — let the data decide.
  2. Handle imbalance *separately*: class weights (`scale_pos_weight`, `class_weight='balanced'`) or balanced sampling.
  3. Recover baseline rows with a **smaller hop on cl=0 only** (hop is per-class-free, e.g. 5s baseline / 10s cl) — adds count for balance, but rows are correlated/redundant (fine under LOSO).
- **Suggested start:** baseline start offset ~30s + 5s baseline hop / 10s cl hop + class weights on; then sweep the offset (0–60s) to confirm cleaner labels actually raise LOSO F1.

---

## Parameters to Test (MLflow runs)

- **Window size:** 30s vs 45s vs 60s (60s helps frequency-domain HRV; 30s lowers latency)
- **Baseline start offset:** sweep 0–60s (e.g. 0 / 15 / 30 / 45 / 60s)
- **Hop:** training density; per-class hop for balance
- **Feature library:** `neurokit2` vs hand-rolled
- **Model:** logistic regression → XGBoost/LightGBM → 1D CNN
- **Imbalance strategy:** class weights vs balanced sampling
- **Normalization:** per-subject z-score and/or train-fold standardization
- Log feature list + all above as MLflow params; log per-fold F1 with `step=fold_index`.
