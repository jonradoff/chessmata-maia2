FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (CPU-only torch to keep image small)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch --no-deps && \
    pip install --no-cache-dir -r requirements.txt

# Clone the Maia2 engine
RUN git clone --depth 1 https://github.com/CSSLab/maia2.git maia2-engine

# Copy agent code and config
COPY agent/ agent/
COPY config.fly.yaml config.yaml

# Model weights will be cached in a volume at /data/models
ENV MAIA2_MODEL_DIR=/data/models

CMD ["python3", "-m", "agent.main"]
