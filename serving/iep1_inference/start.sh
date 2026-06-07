#!/bin/sh
# When running locally (no MODEL_URI), forward localhost:5001 → mlflow:5001 so
# that the HTTP Host header reads "localhost" and passes MLflow's rebinding check.
if [ -z "$MODEL_URI" ]; then
    socat TCP-LISTEN:5001,fork,reuseaddr TCP:${MLFLOW_SERVICE:-mlflow}:${MLFLOW_PORT:-5001} &
fi
exec uvicorn main:app --host 0.0.0.0 --port 8000
