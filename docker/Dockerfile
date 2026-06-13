# Multi-stage build: small runtime image, CPU torch by default.
# For GPU, swap the base to pytorch/pytorch:2.x-cuda12.x-cudnn8-runtime and drop the CPU index.
FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# MNE / matplotlib runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 git && rm -rf /var/lib/apt/lists/*

FROM base AS builder
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# Install CPU torch from the dedicated index, then the package + serving extras.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.1" && pip install ".[dl,features,serve]"

FROM base AS runtime
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src ./src
COPY configs ./configs
ENV EEGPIPE_CKPT=/models/best.ckpt
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "eegpipe.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
