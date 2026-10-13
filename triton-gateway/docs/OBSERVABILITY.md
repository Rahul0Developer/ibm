# Observability & Auto-Scaling

Week 4 deliverable: what we scrape, why, and how scaling decisions are made.

## Metrics sources

| Source | Endpoint | Key metrics |
| --- | --- | --- |
| Go gateway | `:8080/metrics` | `gateway_http_request_duration_seconds` (histogram), `gateway_http_requests_total{outcome=...}`, `gateway_http_inflight_requests`, `gateway_triton_infer_latency_seconds`, `gateway_triton_pool_connections_in_use` |
| Triton server | `:8002/metrics` | `nvrx_infer_stats_latency_queue_*` (dynamic-batch queue time), `nvrx_infer_stats_latency_compute_*` (GPU exec), `nvrx_infer_stats_success/failed_total` |
| DCGM Exporter (GPU Operator) | `:9400/metrics` | `DCGM_FI_DEV_GPU_UTIL` (duty cycle %), `DCGM_FI_DEV_FB_USED` (framebuffer), `DCGM_FI_DEV_POWER_USAGE` |

The Grafana dashboard (`deploy/observability/grafana/dashboards/gateway-overview.json`)
plots p50/p95/p99 against the 30 ms SLA threshold, Triton queue delay against the
5 ms batching-window budget, GPU duty cycle against the 75% target, and request
outcomes (ok / deadline / shed / upstream).

## Local stack (no Kubernetes required)

```bash
docker run -d --name prometheus -p 9090:9090 \
  -v "$PWD/deploy/observability/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
  prom/prometheus:v2.51.0

docker run -d --name grafana -p 3000:3000 grafana/grafana:10.4.2
# then import deploy/observability/grafana/dashboards/gateway-overview.json
```

Prometheus must be able to reach the host ports; on Linux add `--network host` or
export `TRITON_GRPC_ADDR`/gateway via `host.docker.internal`.

## Kubernetes (production)

* Scrape config: pod annotations `prometheus.io/scrape|port|path` are already set
  on both deployments — kube-prometheus-stack's default job picks them up. For a
  stricter setup, add ServiceMonitors targeting the same ports.
* GPU metrics come from the **NVIDIA GPU Operator** with DCGM enabled
  (`nvidia-operator-profile.yaml`: `dcgmExporter.enabled=true`).

## Custom-metrics adapter for HPA

The Triton HPA scales on `triton_queue_delay_p99_ms`, which Prometheus Adapter
must expose. Core of the ConfigMap consumed by the adapter:

```yaml
rules:
  - seriesQuery: 'nvrx_infer_stats_latency_queue_count{namespace="ml-serving",pod=~"triton-serving.*"}'
    resources: { overrides: { namespace: {resource: "namespace"}, pod: {resource: "pod"} } }
    name: { matches: "^(.*)$", as: "triton_queue_delay_p99_ms" }
    metricsQuery: 'histogram_quantile(0.99, sum by (le) (rate(nvrx_infer_stats_latency_queue_bucket[2m]))) / 1000'
```

## Scaling policy

| Tier | Signal | Target | Bounds |
| --- | --- | --- | --- |
| Gateway (CPU pods) | CPU utilization | 70% | 3 → 20 replicas |
| Triton (GPU pods) | queue-delay p99 | ≤ 5 ms avg | 2 → 8 replicas |

Rationale: the gateway is stateless and cheap to scale; GPU pods are expensive and
slow to start (S3 model sync + ONNX load ≈ 40 s), so they scale only on real GPU
queue pressure, with a 300 s stabilization window to avoid flapping. PodDisruptionBudgets
(`40-pdb.yaml`) guarantee at least one GPU replica survives node drains.

## Alerting ideas (PrometheusRule)

```yaml
- alert: GatewayP99SLABreach
  expr: histogram_quantile(0.99, sum(rate(gateway_http_request_duration_seconds_bucket[5m])) by (le)) > 0.03
  for: 5m
- alert: TritonQueueSaturating
  expr: histogram_quantile(0.99, sum by (le) (rate(nvrx_infer_stats_latency_queue_bucket[5m]))) > 10000
  for: 5m
- alert: LoadShedActive
  expr: sum(rate(gateway_http_requests_total{outcome="overloaded"}[2m])) > 0
  for: 2m
```
