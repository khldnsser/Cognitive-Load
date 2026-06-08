#!/usr/bin/env bash
set -euo pipefail
ACCOUNT_ID=879381262015
REGION=eu-central-1
TAG=${1:-latest}
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
FULL="${REGISTRY}/cogload-iep1:${TAG}"
aws ecr get-login-password --region "${REGION}" | docker login --username AWS --password-stdin "${REGISTRY}"
docker buildx build --platform linux/amd64 -f serving/iep1_inference/Dockerfile -t "${FULL}" --load .
docker push "${FULL}"
echo "pushed: ${FULL}"
