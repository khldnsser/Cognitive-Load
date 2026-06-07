FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends socat && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir ".[serving]"

COPY serving/iep1_inference/main.py .
COPY serving/iep1_inference/start.sh .
RUN chmod +x start.sh

EXPOSE 8000
ENTRYPOINT ["./start.sh"]
