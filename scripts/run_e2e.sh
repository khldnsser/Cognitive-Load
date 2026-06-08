#!/usr/bin/env bash
# Run E2E tests against the deployed EEP endpoint.
# Usage:
#   bash scripts/run_e2e.sh                          # uses EKS NLB URL
#   bash scripts/run_e2e.sh http://localhost:8080    # local docker stack
set -euo pipefail

EEP_URL=${1:-http://a55d869eb058c442580c72b3969c5d48-8766164aef600474.elb.eu-central-1.amazonaws.com}

echo "==> Running E2E tests against: ${EEP_URL}"
EEP_URL="${EEP_URL}" pytest tests/e2e/test_e2e.py -v -s
