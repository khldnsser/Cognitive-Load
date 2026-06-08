#!/usr/bin/env bash
# Build and push all 3 serving images to ECR.
# Usage: bash scripts/push_images.sh [tag]
#   tag defaults to "latest"
set -euo pipefail

ACCOUNT_ID=879381262015
REGION=eu-central-1
TAG=${1:-latest}
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "==> Logging in to ECR..."
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${REGISTRY}"

build_and_push() {
  local name=$1
  local dockerfile=$2
  local full="${REGISTRY}/${name}:${TAG}"

  echo ""
  echo "==> Building ${name} (linux/amd64)..."
  docker buildx build --platform linux/amd64 -f "${dockerfile}" -t "${full}" --load .

  echo "==> Pushing ${name}..."
  docker push "${full}"
  echo "    pushed: ${full}"
}

# All three images — build context is repo root so pyproject.toml + src/ are reachable
build_and_push cogload-iep2 serving/iep2_calibration/Dockerfile
build_and_push cogload-iep1 serving/iep1_inference/Dockerfile
build_and_push cogload-eep  serving/eep/Dockerfile

echo ""
echo "All images pushed to ${REGISTRY}."
