#!/usr/bin/env bash
# Local end-to-end stack WITHOUT a GPU: Kind cluster + Triton CPU image + Go gateway.
# Usage: ./scripts/local-dev.sh up | down | test
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KIND_CLUSTER="triton-local"
NAMESPACE="ml-serving"

require() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not found. Install it first (see README)."; exit 1; }
}

up() {
  require kind
  require kubectl
  require docker

  if ! kind get clusters 2>/dev/null | grep -q "^${KIND_CLUSTER}$"; then
    echo ">> Creating Kind cluster '${KIND_CLUSTER}'"
    kind create cluster --name "${KIND_CLUSTER}"
  fi

  echo ">> Building gateway image locally"
  docker build -t inference-gateway:local -f "${ROOT}/deploy/docker/Dockerfile.gateway" "${ROOT}/gateway"
  kind load docker-image inference-gateway:local --name "${KIND_CLUSTER}"

  echo ">> Applying namespace + CPU Triton override"
  kubectl apply -f "${ROOT}/deploy/kubernetes/00-namespace.yaml"
  # CPU-only Triton: no GPU node available on Kind/laptops. Model must be exported
  # with EXPORT_DEVICE=cpu and config.pbtxt switched to KIND_CPU / TYPE_FP32
  # (scripts/export-model.sh handles this automatically).
  kubectl apply -f "${ROOT}/deploy/kubernetes/local-triton-cpu.yaml"
  kubectl apply -f "${ROOT}/deploy/kubernetes/20-gateway-deployment.yaml"

  echo ">> Waiting for rollouts (may take a few minutes on cold pull)"
  kubectl -n "${NAMESPACE}" rollout status deploy/triton-serving --timeout=600s || true
  kubectl -n "${NAMESPACE}" rollout status deploy/inference-gateway --timeout=300s || true

  echo ">> Port-forward gateway :8080 (Ctrl-C to stop)"
  kubectl -n "${NAMESPACE}" port-forward svc/inference-gateway 8080:80
}

test_smoke() {
  echo ">> POST /v1/embeddings"
  curl -sf localhost:8080/v1/embeddings \
    -H 'Content-Type: application/json' \
    -d '{"texts":["the quick brown fox"]}' | head -c 400; echo
  echo ">> GET /metrics (gateway counters)"
  curl -sf localhost:8080/metrics | grep '^gateway_' | head -5
}

down() {
  require kind
  kind delete cluster --name "${KIND_CLUSTER}"
  echo ">> Cluster deleted."
}

case "${1:-up}" in
  up)    up ;;
  down)  down ;;
  test)  test_smoke ;;
  *)     echo "usage: $0 {up|down|test}"; exit 1 ;;
esac
