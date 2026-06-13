# Deploying eegpipe

A decision guide and ready-to-edit manifests. The golden rule: **don't reach for Kubernetes
until a single container is genuinely the bottleneck.** `docker compose up` (see `../docker/`)
serves most use-cases.

## Which path?

| You have… | Use | Files |
|-----------|-----|-------|
| One service, modest traffic | The Docker image directly | `../docker/` |
| A k8s cluster, want autoscaling | Plain Deployment + HPA | `k8s/` |
| A k8s cluster + want canary/scale-to-zero/traffic-split | **KServe** | `kserve/` |
| Many models, max GPU throughput, server-side batching | **Triton** | `triton/` |

All paths serve the **exported ONNX** artefact (`eegpipe-export`), versioned in object storage
(DVC remote or the MLflow registry) — never baked into the image.

## 1. Plain Kubernetes (`k8s/`)
```bash
kubectl create secret generic eegpipe-secrets --from-literal=api-key=$(openssl rand -hex 16)
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/hpa.yaml
```
- Readiness probe hits `/ready` (only routes traffic once the model is loaded); liveness hits
  `/health`. Populate `/models/model.onnx` with an initContainer (`dvc pull` or `aws s3 cp`).
- HPA scales 2→10 pods at 70% CPU. For latency SLOs, scale on a custom metric via the
  Prometheus Adapter rather than CPU.

## 2. KServe (`kserve/`)
```bash
kubectl apply -f kserve/inferenceservice.yaml
```
- Knative autoscaling (incl. scale-to-zero) on concurrency, plus built-in canary traffic
  splitting (`canaryTrafficPercent`). Option A runs our FastAPI container; Option B uses
  KServe's managed **Triton** runtime via `storageUri`.

## 3. Triton (`triton/`)
```bash
eegpipe-export model.ckpt triton/model_repository/eegpipe/1/model.onnx --kind trial --channels 22 --times 256
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 \
  -v $PWD/triton/model_repository:/models nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models
```
- Set `dims` in `config.pbtxt` to your `(channels, times)`. `dynamic_batching` gives server-side
  micro-batching; flip `instance_group` to `KIND_GPU` for CUDA.

## Observability (all paths)
Scrape the API's `X-Process-Time-ms` / a `/metrics` endpoint, log input feature distributions,
and alert on **drift** — EEG is non-stationary across sessions/headsets, so drift is expected.
Gate promotions with the CI **regression gate** (`scripts/check_regression.py`).
