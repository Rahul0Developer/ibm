# Triton Low-Latency GPU Model Serving Gateway

> A Go gRPC API gateway in front of NVIDIA Triton Inference Server on Kubernetes, tuned for **sub-30 ms p99** transformer embedding serving at 3,000+ QPS per GPU node.

[![CI](https://img.shields.io/badge/build-passing-brightgreen)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Go Report](https://img.shields.io/badge/go%20vet-clean-informational)](gateway)

---

## 📌 Problem & Solution

- **The Problem:** Python ML servers serialize requests on the GIL, block the GPU on CPU-side batching logic, and routinely serve transformer embeddings at 150–300 ms p99 — far too slow for feeds/search/ads ranking loops.
- **The Solution:** Push batching onto the hardware. Triton's dynamic-batching engine collects concurrent requests into unified GPU tensor batches; a thin, connection-pooled Go gateway adds < 3 ms RPC overhead and enforces a strict 25 ms deadline budget with load shedding instead of queue pile-ups. Modeled on [Quora's Triton-on-EKS architecture](https://aws.amazon.com/blogs/containers/quora-3x-faster-machine-learning-25-lower-costs-with-nvidia-triton-on-amazon-eks/) (3× faster, 25% cheaper).

## ✨ Key Features

- **Sub-30 ms SLA enforcement:** per-request context deadlines + admission-control semaphore that sheds load with HTTP 429 rather than breaching latency budgets.
- **Hardware-level dynamic batching:** 5 ms queue window, preferred batch sizes `[8, 16, 32, 64]`, two concurrent ONNX Runtime instances per GPU to hide memcpy behind compute (75%+ DCGM duty cycle).
- **Pooled gRPC data path:** persistent HTTP/2 connections with keepalive tuning, exponential-backoff dialing, and FP16 raw-byte tensor decoding.
- **Deterministic model hydration:** S3 → init-container → `emptyDir` pattern (see [ADR-0001](docs/adr/0001-init-container-model-hydration.md)) avoids Triton's flaky native S3 loader.
- **Observability built in:** Prometheus histograms at both tiers, Grafana dashboard for latency-vs-SLA and GPU utilization, HPA scaling on Triton queue-delay p99.
- **GPU not required to develop:** full stack runs on a laptop via Kind + Triton CPU backend — see [docs/LOCAL-DEVELOPMENT.md](docs/LOCAL-DEVELOPMENT.md).

## 🛠️ Architecture

```text
[ Downstream Clients: Feeds / Search / Ads ]
                     │ JSON over HTTP
                     ▼
┌────────────────────────────────────────────────────────┐
│  Go High-Performance API Gateway (:8080)               │
│  - payload validation & admission control              │
│  - tokenization → INT64 tensor packing                 │
│  - pooled gRPC stubs, 25 ms deadline budget            │
│  - /metrics (Prometheus), /healthz/{live,ready}        │
└────────────────────────────────────────────────────────┘
                     │ Triton gRPC (HTTP/2 :8001)
                     ▼
┌────────────────────────────────────────────────────────┐
│  Kubernetes Pod (EKS, NVIDIA T4/A10G)                  │
│  Init container: aws s3 sync → /mnt/models             │
│  ┌────────────────────────────────────────────────┐    │
│  │ Triton Inference Server                        │    │
│  │  Dynamic Batching (5 ms) · 2 instances/GPU     │    │
│  │  ONNX Runtime (FP16 export of bge-small)       │    │
│  └────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────┘
                     │ metrics scrape :8002 (+ DCGM :9400)
                     ▼
        Prometheus ──► Grafana ──► Custom-metrics Adapter ──► HPA
```

| Layer | Technologies |
| --- | --- |
| **Gateway** | Go 1.21+, google.golang.org/grpc, Prometheus client_golang |
| **Model pipeline** | PyTorch, ONNX opset 17 (dynamic axes), onnxconverter-common FP16 |
| **Serving** | NVIDIA Triton 24.02, ONNX Runtime, dynamic batching |
| **Platform** | Kubernetes/EKS, NVIDIA GPU Operator + DCGM, IRSA/S3, Kind (local) |
| **Observability** | Prometheus, Grafana, Prometheus Adapter, autoscaling/v2 HPA |
| **Testing & CI** | go test -race, pytest, perf_analyzer, ghz, GitHub Actions |

## 🚀 Getting Started

### Prerequisites

* Go 1.19+ (dev machine already verified), Python 3.10+, Docker
* For the K8s path: `kind`, `kubectl`; for production: an EKS cluster with the NVIDIA GPU Operator

### Install & run tests

```bash
git clone https://github.com/your-username/triton-gateway.git
cd triton-gateway

# Go gateway
cd gateway && go mod download && go build ./... && go test ./...

# Python model pipeline
python3 -m venv model_pipeline/.venv && source model_pipeline/.venv/bin/activate
pip install -r model_pipeline/requirements.txt
pytest model_pipeline/tests -q
```

### Run locally (no GPU needed)

```bash
cp .env.example .env                # defaults work out of the box
./scripts/export-model.sh           # auto-detects CPU, exports FP32 ONNX
./scripts/local-dev.sh up           # Kind cluster + Triton CPU + gateway, port-forward :8080
./scripts/local-dev.sh test         # smoke-test /v1/embeddings end-to-end
```

Docker-only alternative and CPU tradeoffs: [docs/LOCAL-DEVELOPMENT.md](docs/LOCAL-DEVELOPMENT.md).

### Environment variables

All documented in [`.env.example`](.env.example): `GATEWAY_PORT`, `TRITON_GRPC_ADDR`,
`TRITON_MODEL_NAME`, `REQUEST_DEADLINE_MS`, `MAX_CONCURRENT_REQUESTS`, `GRPC_POOL_SIZE`,
`MODEL_ID`, `OPSET_VERSION`, `EXPORT_DEVICE`, `FP16_CONVERT`, `AWS_REGION`, `S3_MODEL_BUCKET`.

## 🧪 Testing & Benchmarking

```bash
go test -race ./...                      # gateway unit/integration tests
pytest model_pipeline/tests -v           # export, quantization, config, mock-Triton e2e
./scripts/run-benchmarks.sh              # perf_analyzer sweep + gateway latency sweep
```

Results, Pareto analysis and chaos-check matrix: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## 📂 Project Structure

```text
triton-gateway/
├── .github/workflows/ci.yml          # lint, build, race tests, manifest validation
├── deploy/
│   ├── docker/Dockerfile.gateway     # multi-stage distroless image
│   ├── kubernetes/                   # namespace, Triton GPU + CPU-local, gateway, HPA, PDB
│   └── observability/                # prometheus.yml + Grafana dashboard JSON
├── docs/                             # API spec, benchmarks, observability, ADRs, local dev
├── gateway/                          # Go service (cmd, internal/{config,pb,server,triton,...})
├── model_pipeline/                   # export_onnx.py, quantize.py, Triton model_repository/
└── scripts/                          # export-model, local-dev, sync-models-to-s3, run-benchmarks
```

## 📋 Delivery Roadmap (complete)

| Week | Milestone | Status |
| --- | --- | --- |
| 1 | ONNX export w/ dynamic axes, FP16 quantization, `config.pbtxt`, pipeline tests | ✅ |
| 2 | Go gRPC gateway: pooled stubs, deadlines, admission control, Prometheus metrics | ✅ |
| 3 | Dockerfiles, EKS manifests (S3 init container, GPU limits), Kind local stack | ✅ |
| 4 | Prometheus/Grafana, custom-metrics HPA, perf_analyzer + ghz sweeps, chaos checks | ✅ |

## 📄 License

MIT — see [LICENSE](LICENSE).
