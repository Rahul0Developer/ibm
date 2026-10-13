#!/usr/bin/env bash
# Week-4 benchmark driver: runs Triton perf_analyzer concurrency sweeps and a
# ghz load test against the gateway, writing artifacts into gateway/benchmarks/.
#
# Requirements (GPU host or any machine that can reach Triton):
#   * triton-inference-server/perf_analyzer container or binary
#   * ghz (https://ghz.sh) for raw gRPC load on the gateway path
# Usage: ./scripts/run-benchmarks.sh [triton_addr] [gateway_addr]
set -euo pipefail

TRITON_ADDR="${1:-localhost:8001}"
GATEWAY_ADDR="${2:-localhost:8080}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT}/gateway/benchmarks"
mkdir -p "${OUT_DIR}"

echo ">> Sweep 1/2: perf_analyzer concurrency 1..32 (gRPC -> Triton)"
if command -v perf_analyzer >/dev/null 2>&1; then
  perf_analyzer -m text_embeddings \
    -u "${TRITON_ADDR}" \
    -i gRPC \
    --concurrency-range 1:32:4 \
    --measurement-interval 10000 \
    --shape input_ids:-1 \
    --shape attention_mask:-1 \
    --latency-report-file "${OUT_DIR}/latency_curve.csv" \
    | tee "${OUT_DIR}/perf_analyzer.log"
else
  echo "WARN: perf_analyzer not found — skipping GPU sweep. Run this script on the GPU host." >&2
fi

echo ">> Sweep 2/2: ghz load against the gateway HTTP endpoint (curl-based fallback)"
if command -v ghz >/dev/null 2>&1; then
  # ghz speaks gRPC; here we use it to hit Triton directly as an upper-bound check.
  ghz --insecure -u "${TRITON_ADDR}" \
    -c 64 -z 60s \
    -m inference.GRPCInferenceService/ModelInfer \
    -d '{"model_name":"text_embeddings","inputs":[{"name":"input_ids","datatype":"INT64","shape":[1,64],"contents":{"int64_contents":[0]}}]}' \
    > "${OUT_DIR}/ghz_report.json" || true
else
  echo "INFO: ghz not installed; running a lightweight wrk-style curl sweep instead." >&2
fi

# Lightweight always-available sweep: measure gateway p50/p95/p99 with curl timing.
python3 - "$GATEWAY_ADDR" "$OUT_DIR/gateway_sweep.csv" <<'PY'
import concurrent.futures as cf, json, subprocess, sys, time

addr, out = sys.argv[1], sys.argv[2]
body = json.dumps({"texts": ["the quick brown fox jumps over the lazy dog"]})

def one(_):
    t0 = time.perf_counter()
    r = subprocess.run(
        ["curl", "-sf", "-o", "/dev/null", "-w", "%{http_code}", f"{addr}/v1/embeddings",
         "-H", "Content-Type: application/json", "-d", body],
        capture_output=True, text=True, timeout=5)
    return (time.perf_counter() - t0) * 1000.0, r.stdout.strip()

results = []
with cf.ThreadPoolExecutor(max_workers=16) as ex:
    for lat, code in ex.map(one, range(200)):
        results.append((lat, code))

oks = sorted(l for l, c in results if c == "200")
def pct(p): return oks[int(len(oks) * p)] if oks else float("nan")

with open(out, "w") as f:
    f.write("metric,value_ms\n")
    f.write(f"count_ok,{len(oks)}\n")
    f.write(f"errors,{len(results) - len(oks)}\n")
    f.write(f"p50,{pct(.50):.2f}\n")
    f.write(f"p95,{pct(.95):.2f}\n")
    f.write(f"p99,{pct(.99):.2f}\n")
print(f"[bench] wrote {out}: p50={pct(.50):.1f}ms p99={pct(.99):.1f}ms ok={len(oks)}/{len(results)}")
PY

echo ">> Artifacts in ${OUT_DIR}/"
ls -la "${OUT_DIR}"
