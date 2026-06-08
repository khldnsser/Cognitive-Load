#!/usr/bin/env bash
# Create the IAM policy + IRSA role for IEP-1 to read the model bucket from EKS.
# Idempotent — safe to re-run.
#
# Prerequisites:
#   - eksctl cluster already running (OIDC issuer is needed)
#   - aws CLI configured with admin-level permissions
#
# Usage:
#   bash scripts/create_iam_role.sh
set -euo pipefail

ACCOUNT_ID=879381262015
REGION=eu-central-1
CLUSTER_NAME=cogload
BUCKET=cogload-models-879381262015
ROLE_NAME=cogload-iep1-s3-role
POLICY_NAME=cogload-iep1-s3-policy
NAMESPACE=cogload
SA_NAME=iep1-sa

echo "==> Fetching OIDC issuer for cluster ${CLUSTER_NAME}..."
OIDC_ISSUER=$(aws eks describe-cluster \
  --name "${CLUSTER_NAME}" \
  --region "${REGION}" \
  --query "cluster.identity.oidc.issuer" \
  --output text)

# Strip https://
OIDC_PROVIDER="${OIDC_ISSUER#https://}"
echo "    OIDC provider: ${OIDC_PROVIDER}"

echo "==> Creating IAM policy ${POLICY_NAME}..."
POLICY_DOCUMENT=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::${BUCKET}",
        "arn:aws:s3:::${BUCKET}/*"
      ]
    }
  ]
}
EOF
)

POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${POLICY_NAME}"
if aws iam get-policy --policy-arn "${POLICY_ARN}" > /dev/null 2>&1; then
  echo "    policy already exists: ${POLICY_ARN}"
else
  aws iam create-policy \
    --policy-name "${POLICY_NAME}" \
    --policy-document "${POLICY_DOCUMENT}" \
    --description "Allows IEP-1 in EKS to read the cogload model bucket"
  echo "    created policy: ${POLICY_ARN}"
fi

echo "==> Creating IAM role ${ROLE_NAME} with IRSA trust policy..."
TRUST_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::${ACCOUNT_ID}:oidc-provider/${OIDC_PROVIDER}"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "${OIDC_PROVIDER}:sub": "system:serviceaccount:${NAMESPACE}:${SA_NAME}",
          "${OIDC_PROVIDER}:aud": "sts.amazonaws.com"
        }
      }
    }
  ]
}
EOF
)

ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
if aws iam get-role --role-name "${ROLE_NAME}" > /dev/null 2>&1; then
  echo "    role already exists: ${ROLE_ARN}"
  echo "    updating trust policy..."
  aws iam update-assume-role-policy \
    --role-name "${ROLE_NAME}" \
    --policy-document "${TRUST_POLICY}"
else
  aws iam create-role \
    --role-name "${ROLE_NAME}" \
    --assume-role-policy-document "${TRUST_POLICY}" \
    --description "IRSA role for IEP-1 S3 read access to cogload model bucket"
  echo "    created role: ${ROLE_ARN}"
fi

echo "==> Attaching policy to role..."
aws iam attach-role-policy \
  --role-name "${ROLE_NAME}" \
  --policy-arn "${POLICY_ARN}"
echo "    attached: ${POLICY_ARN} → ${ROLE_NAME}"

echo ""
echo "Done. Role ARN:"
echo "  ${ROLE_ARN}"
echo ""
echo "This ARN is already baked into k8s/03-iep1.yml:"
echo "  eks.amazonaws.com/role-arn: ${ROLE_ARN}"
