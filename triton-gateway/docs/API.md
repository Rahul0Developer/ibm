# Triton Gateway — API Specification

Base URL: `http://<gateway-host>:8080`

All inference calls are POSTs with a JSON body; all responses are JSON. Errors use
the same envelope: `{"error": "<code>", "message": "<human readable>"}`.

---

## `POST /v1/embeddings`

Embed one or more texts (max 64 per call, each 1–2048 bytes).

**Request**

```json
{ "texts": ["the quick brown fox", "jumps over the lazy dog"] }
```

**200 OK**

```json
{
  "model": "text_embeddings",
  "dimension": 384,
  "embeddings": [[0.0153, ...], [0.0179, ...]],
  "latency_ms": 8.42
}
```

**Errors**

| Status | `error` code | Meaning |
| --- | --- | --- |
| 400 | `invalid_json` | Body not `{ "texts": [...] }` |
| 400 | `invalid_batch` | `texts` length outside `[1, 64]` |
| 400 | `invalid_input` | A text is empty or > 2048 bytes |
| 405 | `bad_method` | Non-POST to this route |
| 429 | `overloaded` | Admission-control semaphore full — retry shortly |
| 502 | `upstream` | Triton returned an error / embedding count mismatch |
| 504 | `deadline` | 25 ms SLA budget exceeded end-to-end |

**Example**

```bash
curl -s localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"texts":["hello world"]}'
```

---

## `GET /healthz/live`

Liveness probe. Always `200 {"status":"alive"}` while the process runs.

## `GET /healthz/ready`

Readiness probe. `200 {"status":"ready"}` only when the gateway has verified
Triton `ServerLive`; otherwise `503 {"error":"backend_down"|...}`. Kubernetes
routes traffic based on this endpoint.

## `GET /metrics`

Prometheus text exposition. Key series:

* `gateway_http_request_duration_seconds_{bucket,sum,count}` — end-to-end latency histogram
* `gateway_http_requests_total{outcome="ok|invalid|overloaded|deadline|upstream"}`
* `gateway_http_inflight_requests` — current admitted concurrency
* `gateway_triton_infer_latency_seconds_*` — RPC-only latency (subtract from HTTP
  histogram to isolate gateway overhead)
* `gateway_triton_pool_connections_in_use`

---

## gRPC contract (gateway → Triton)

The gateway speaks Triton's native `inference.GRPCInferenceService/ModelInfer`:

| Tensor | Datatype | Shape |
| --- | --- | --- |
| `input_ids` | INT64 | `[batch, 64]` (right-padded token IDs) |
| `attention_mask` | INT64 | `[batch, 64]` |
| `sentence_embedding` (out) | FP16→FP32 decoded | `[batch, 384]` |

Generated stubs live in `gateway/internal/pb/` (`grpc_service.proto`). Keepalive:
10 s ping interval, 3 s timeout, `PermitWithoutStream=true`; max recv message 10 MiB.
