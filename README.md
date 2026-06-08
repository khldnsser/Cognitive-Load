# Cognitive Load Detection — CogWear

Binary classification (high / low cognitive load) from wrist physiological signals using the
CogWear dataset. Signals: BVP (64 Hz), EDA (4 Hz), Temperature (4 Hz) from Empatica E4.
Muse EEG and Samsung BVP are out of scope.

**Champion result:** LOSO F1 = **0.910 ± 0.083** · Holdout (Subject 24) **F1 0.785 vs 0.620
benchmark** (Grzeszczyk et al.). A production-served system on AWS EKS.

---

## Quickstart for reviewers / testers

There are **two ways to test this system**. You do **not** need AWS, Kubernetes, or any
training to test the live deployment — it is already running.

### Path A — Test the live cloud system (no infrastructure to run)

The public gateway (EEP) is deployed on AWS EKS behind a Network Load Balancer:

```
http://a55d869eb058c442580c72b3969c5d48-8766164aef600474.elb.eu-central-1.amazonaws.com
```

**A1. Instant liveness check — zero install** (just `curl`):

```bash
curl http://a55d869eb058c442580c72b3969c5d48-8766164aef600474.elb.eu-central-1.amazonaws.com/health
# → {"status":"ok", ...}
```

**A2. Headless end-to-end test — no dataset needed.** Runs the full calibrate→predict golden
path against the live cloud endpoint using synthetic windows (so it works from a bare clone):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                # dev extra provides pytest + httpx (needed by the suite)
bash scripts/run_e2e.sh                # defaults to the deployed cloud URL
```

**A3. Live demo — watch the model classify the held-out subject in real time.**
Replays Subject 24 (never seen in training) through the *cloud* endpoint, separating its rest
(LOW load) and Stroop (HIGH load) recordings and printing the predicted vs. true label per
60 s window. Needs only Python 3.11+ — **no extra pip packages** (uses the stdlib):

```bash
pip install -e .                       # installs the cogload package + its deps
python scripts/demo_s24_stream.py      # streams S24 → cloud EEP, labels live in the terminal
```

> ⚠️ **Dataset required for A3 only.** This demo reads Subject 24's raw signals from
> `raw_data/`. The CogWear dataset is **git-ignored** and therefore **not present in a bare
> `git clone`** — it ships with the submitted project folder. If you cloned from git, drop the
> CogWear `raw_data/` tree in the project root first (see [Dataset card](raw_data/README.md)),
> or just use **A1/A2**, which need no dataset.

Useful `demo_s24_stream.py` flags: `--delay 0.3` (faster), `--session post`,
`--url http://localhost:8080` (point at a local stack instead of the cloud).

> Hitting the cloud endpoint updates the **in-cluster** Prometheus/Grafana on EKS (private,
> `ClusterIP`). To view them: `kubectl port-forward -n cogload svc/grafana 3000:3000`.

### Path B — Run the whole stack on your machine

Prerequisites: **Python 3.11+** and **Docker** (with Compose). Then:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,serving]"
docker compose up -d        # serving stack + MLflow + Prometheus + Grafana, all local
make test                   # 69 tests: unit + integration + e2e against the local stack
```

Local service URLs once Compose is up:

| Service | URL | Notes |
|---------|-----|-------|
| EEP gateway | http://localhost:8080 | public-facing API (`/predict`, `/calibrate`, `/health`, `/metrics`) |
| MLflow | http://localhost:5001 | experiment tracking + model registry |
| Grafana | http://localhost:3000 | login `admin` / `admin` → dashboard **CogLoad Serving** |
| Prometheus | http://localhost:9090 | raw metrics (Grafana sits on top) |

Then `python scripts/demo_s24_stream.py --url http://localhost:8080` runs the same live demo
against your local stack (and updates the **local** Grafana).

---

## Retraining the model (optional — for ML reviewers)

The deployed model is already trained; this section is only for reproducing the modeling
pipeline. Assumes the **Path B** environment above (`pip install -e ".[dev,serving]"` and
`docker compose up -d` for MLflow).

```bash
make features      # build & cache windowed feature tables
make experiments   # run the experiment grid (one config = one MLflow run)
make select        # rank runs, apply promotion gate, register champion
make retrain       # all three of the above, end to end
make rollback      # swap champion ↔ previous-champion

make test          # unit + integration + e2e   (69 tests)
make test-unit     # unit only (no services needed)
```

Cached parquet files live in `data/processed/`. Re-extract only if features, window size, or
baseline offset change.

Champion config: `window=60s  hop=15s  baseline_offset=30s  max_depth=4  threshold=0.20`,
registered as `cogload-ensemble@champion` in the MLflow Model Registry. Full leaderboard and
analysis in [results_report.md](results_report.md).

---

## Live system

Three services behind a public AWS NLB (see [Documentation](docs/CogLoad_Documentation.docx)):

```
client → EEP (public gateway) ─┬─► IEP-2 calibration  (/calibrate)
                               └─► IEP-1 inference    (/predict)
                                        └─► Prometheus → Grafana
```

Routes: `POST /calibrate` · `POST /predict` · `GET /health` · `GET /metrics`.
Public URL: `http://a55d869eb058c442580c72b3969c5d48-8766164aef600474.elb.eu-central-1.amazonaws.com`

```bash
python scripts/demo_s24_stream.py  # live demo: stream held-out S24 → labels in the terminal
bash scripts/run_e2e.sh            # E2E suite against the deployed URL
bash scripts/run_e2e.sh http://localhost:8080   # against the local stack
```

See **[Quickstart for reviewers / testers](#quickstart-for-reviewers--testers)** above for the
full walkthrough.

---

## Repository layout

```
src/cogload/      ML pipeline + serving schemas (per-module map in CLAUDE.md)
serving/          EEP, IEP-1, IEP-2 FastAPI apps + Dockerfiles
scripts/          feature build, experiments, selection, deploy, e2e, live S24 demo
k8s/              EKS cluster config + Kubernetes manifests
docker/           Prometheus + Grafana provisioning
tests/            unit · integration · e2e
docs/             technical + business documentation
data/processed/   cached feature parquet tables
raw_data/         CogWear dataset (+ dataset card)
```

---

## Dataset

24 subjects — pilot (0–10) and survey gamification (11–24); subject 19 is absent. Subject 24
is the final holdout, never used in training or champion selection. Structure detailed in
[raw_data/README.md](raw_data/README.md).
</content>
