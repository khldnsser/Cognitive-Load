"""
Download the champion model artifact from local MLflow and upload it to S3.
Run this once before the first EKS deploy (or after each promotion).

Usage:
    python scripts/export_model_to_s3.py [--bucket cogload-models-879381262015]
"""
import argparse
import os
import shutil
import tempfile

import boto3
import mlflow

from cogload.config import (
    MLFLOW_URI,
    MODEL_CHAMPION_ALIAS,
    MODEL_REGISTRY_NAME,
)

S3_KEY_PREFIX = "models/cogload-ensemble"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bucket",
        default=os.environ.get("MODEL_BUCKET", "cogload-models-879381262015"),
    )
    parser.add_argument("--alias", default=MODEL_CHAMPION_ALIAS)
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()

    mv = client.get_model_version_by_alias(MODEL_REGISTRY_NAME, args.alias)
    model_uri = f"models:/{MODEL_REGISTRY_NAME}/{mv.version}"
    print(f"Downloading {model_uri} (version {mv.version}, alias @{args.alias})…")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"{model_uri}", dst_path=tmpdir
        )
        print(f"  downloaded to {local_path}")

        s3 = boto3.client("s3")
        s3_prefix = f"{S3_KEY_PREFIX}/v{mv.version}"

        upload_count = 0
        for root, _, files in os.walk(local_path):
            for fname in files:
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, local_path)
                s3_key = f"{s3_prefix}/{rel}"
                s3.upload_file(fpath, args.bucket, s3_key)
                upload_count += 1

        # Write/overwrite a "latest" pointer file so IEP-1 can resolve the current version
        pointer_key = f"{S3_KEY_PREFIX}/champion_version"
        s3.put_object(
            Bucket=args.bucket,
            Key=pointer_key,
            Body=f"v{mv.version}".encode(),
        )

        model_s3_uri = f"s3://{args.bucket}/{s3_prefix}"
        print(f"  uploaded {upload_count} files → {model_s3_uri}")
        print(f"  pointer  → s3://{args.bucket}/{pointer_key}")

    print()
    print("==> Set this env var / SSM parameter for IEP-1:")
    print(f"    MODEL_URI={model_s3_uri}")
    print()
    print("To update SSM:")
    print(
        f'    aws ssm put-parameter --name /cogload/model_uri --value "{model_s3_uri}" '
        "--type String --overwrite"
    )


if __name__ == "__main__":
    main()
