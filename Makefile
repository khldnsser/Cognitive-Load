PYTHON ?= .venv/bin/python

# Promotion gate: reject candidates below this selection_score.
#   selection_score = mean_F1 - 0.5 * std_F1
#   Current champion : ≈ 0.869  (F1=0.910, std=0.083)
#   Absolute minimum: > benchmark 0.620
# Override with: make retrain PROMOTE_MIN_SCORE=0.85
PROMOTE_MIN_SCORE ?= 0.80

.PHONY: help features experiments select select-dry retrain rollback \
        test test-unit test-integration test-e2e infra-up infra-down

# ── Default target ─────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "CogLoad ML pipeline"
	@echo ""
	@echo "Training pipeline:"
	@echo "  make features          Build & cache windowed feature tables"
	@echo "  make experiments       Run the full experiment grid (all configs)"
	@echo "  make select            Select best run & promote to @champion"
	@echo "  make select-dry        Leaderboard only — no registration"
	@echo "  make retrain           Full pipeline: features → experiments → select"
	@echo "  make rollback          Swap champion back to previous-champion"
	@echo ""
	@echo "Testing:"
	@echo "  make test              All tests (unit + integration + e2e)"
	@echo "  make test-unit         Unit tests only (no services needed)"
	@echo "  make test-integration  Integration tests (needs docker compose up)"
	@echo "  make test-e2e          E2E tests (needs docker compose up)"
	@echo ""
	@echo "Infrastructure:"
	@echo "  make infra-up          docker compose up -d"
	@echo "  make infra-down        docker compose down"
	@echo ""
	@echo "Overrides:"
	@echo "  PYTHON=...             Python interpreter (default: .venv/bin/python)"
	@echo "  PROMOTE_MIN_SCORE=...  Minimum selscore to promote (default: 0.80)"
	@echo ""

# ── Training pipeline ──────────────────────────────────────────────────────────
features:
	$(PYTHON) scripts/build_features.py

experiments:
	$(PYTHON) scripts/run_experiments.py

select:
	$(PYTHON) scripts/select_best.py --min-score $(PROMOTE_MIN_SCORE)

select-dry:
	$(PYTHON) scripts/select_best.py --dry-run

retrain: features experiments select

rollback:
	$(PYTHON) scripts/rollback_champion.py

# ── Tests ──────────────────────────────────────────────────────────────────────
test-unit:
	$(PYTHON) -m pytest tests/ \
		--ignore=tests/integration \
		--ignore=tests/e2e \
		-v --tb=short

test-integration:
	$(PYTHON) -m pytest tests/integration -v --tb=short

test-e2e:
	$(PYTHON) -m pytest tests/e2e -v --tb=short

test: test-unit test-integration test-e2e

# ── Infrastructure ─────────────────────────────────────────────────────────────
infra-up:
	docker compose up -d

infra-down:
	docker compose down
