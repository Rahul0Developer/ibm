# Local Development — No GPU Required

This whole stack runs on a laptop without an NVIDIA GPU. The only thing you lose
is the latency/throughput SLAs (which are GPU-specific); every code path —
validation → admission control → tokenization → gRPC → ONNX inference → FP16
decode — is exercised identically on CPU.

## What works on CPU-only hardware

| Component | CPU status | Notes |
| --- | --- | --- |
| Go gateway (`gateway/`) | ✅ full | `go build ./... && go test ./...` — no GPU deps at all |
| Python pipeline tests | ✅ full | pytest mocks Triton; export tests use tiny dummy models |
| ONNX export | ✅ | run `scripts/export-model.sh` — it auto-detects no CUDA and exports with `--device cpu` (FP32) |
| Triton serving locally | ✅ | same image, ONNX Runtime **CPU** backend; remove `nvidia.com/gpu` requests (see below) |
| Kind cluster + manifests | ✅ | `scripts/local-dev.sh up` applies `local-triton-cpu.yaml` (no GPU resources) |
| Grafana/Prometheus | ✅ | metrics and dashboards are hardware-independent |
| FP16 quantization | ⚠️ converts fine | conversion is graph-level; execution on CPU falls back to FP32 math and is slow |
| perf_analyzer GPU sweeps / DCGM panels | ❌ | need real NVIDIA hardware — run those on a cloud GPU instance |

## Fastest path (Docker only, no Kubernetes)

```bash
# 1. Export the model for CPU
./scripts/export-model.sh

# 2. Serve it with Triton's CPU execution (no --gpus flag!)
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$PWD/model_pipeline/model_repository:/mnt/models:ro" \
  nvcr.io/nvidia/tritonserver:24.02-py3 \
  tritonserver --model-repository=/mnt/models --strict-model-config=false

# NOTE: with strict-model-config=false Triton overrides the FP16/GPU bits of
# config.pbtxt to match what the CPU runtime can actually serve. For an explicit
# setup, edit config.pbtxt locally: kind: KIND_CPU and output TYPE_FP32.

# 3. Run the gateway against it
cd gateway && go run ./cmd/gateway
export TRITON_GRPC_ADDR=localhost:8001 REQUEST_DEADLINE_MS=500  # relax SLA for CPU

# 4. Smoke test
curl -s localhost:8080/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"texts":["the quick brown fox"]}' | head -c 300
```

The `REQUEST_DEADLINE_MS` bump is expected on CPU: bge-small on a laptop takes
~20–80 ms per batched forward pass, so the production 25 ms budget would shed
every request. The deadline is configuration, not code — nothing else changes.

## Full stack in Kind

```bash
./scripts/local-dev.sh up     # creates cluster, builds image, applies CPU manifests
./scripts/local-dev.sh test   # smoke-test through port-forward
./scripts/local-dev.sh down
```

## When you get GPU access

Run the exact same manifests minus the CPU override:
`deploy/kubernetes/10-triton-deployment.yaml` (GPU resources + S3 init container),
then `scripts/run-benchmarks.sh` to regenerate `docs/BENCHMARKS.md` artifacts.
