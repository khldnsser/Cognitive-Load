#!/usr/bin/env bash
# Pull MODEL_URI from AWS SSM Parameter Store and create/update the K8s Secret
# that IEP-1 reads at startup.
#
# Call this once after export_model_to_s3.py and again after each model promotion.
#
# Usage:
#   bash scripts/sync_ssm_to_k8s_secret.sh
#   SSM_PARAM=/cogload/model_uri bash scripts/sync_ssm_to_k8s_secret.sh  # override param name
set -euo pipefail

REGION=${AWS_DEFAULT_REGION:-eu-central-1}
SSM_PARAM=${SSM_PARAM:-/cogload/model_uri}
NAMESPACE=cogload
SECRET_NAME=cogload-model-secret
SECRET_KEY=MODEL_URI

echo "==> Reading ${SSM_PARAM} from SSM..."
MODEL_URI=$(aws ssm get-parameter \
  --name "${SSM_PARAM}" \
  --region "${REGION}" \
  --query "Parameter.Value" \
  --output text)

if [ -z "${MODEL_URI}" ]; then
  echo "ERROR: parameter ${SSM_PARAM} is empty or not found" >&2
  exit 1
fi
echo "    MODEL_URI=${MODEL_URI}"

echo "==> Writing K8s Secret ${NAMESPACE}/${SECRET_NAME}..."
kubectl create secret generic "${SECRET_NAME}" \
  --namespace "${NAMESPACE}" \
  --from-literal="${SECRET_KEY}=${MODEL_URI}" \
  --dry-run=client -o yaml \
  | kubectl apply -f -

echo ""
echo "Secret ${SECRET_NAME} is up to date."
echo "IEP-1 will use the new URI on its next pod restart."
echo ""
echo "To force a rolling restart:"
echo "  kubectl rollout restart deployment/iep1-inference -n ${NAMESPACE}"
