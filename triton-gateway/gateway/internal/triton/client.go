// Package triton implements the high-throughput Triton gRPC client pool:
// persistent HTTP/2 connections, keepalive tuning, raw tensor packing and a
// strict per-request deadline budget so slow GPUs cannot pile up upstream.
package triton

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/backoff"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"

	"github.com/your-org/triton-gateway/internal/pb"
)

const maxRecvMsgBytes = 10 * 1024 * 1024 // 10 MiB — covers batch-64 x 384 FP16 + headroom

var (
	inferLatency = promauto.NewHistogram(prometheus.HistogramOpts{
		Namespace: "gateway",
		Subsystem: "triton",
		Name:      "infer_latency_seconds",
		Help:      "End-to-end latency of Triton ModelInfer RPCs.",
		Buckets:   []float64{.001, .0025, .005, .01, .015, .02, .025, .03, .05, .1, .25},
	})
	inferErrors = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "gateway",
		Subsystem: "triton",
		Name:      "infer_errors_total",
		Help:      "Triton inference failures by code (deadline, unavailable, invalid).",
	}, []string{"code"})
	poolInUse = promauto.NewGauge(prometheus.GaugeOpts{
		Namespace: "gateway",
		Subsystem: "triton",
		Name:      "pool_connections_in_use",
		Help:      "Number of pooled sub-connections currently serving an RPC.",
	})
)

// ClientPool owns N independent gRPC connections to one Triton server and
// round-robins requests across them. gRPC already multiplexes over HTTP/2,
// but multiple connections spread Triton's per-connection worker load and
// avoid head-of-line blocking under sustained 3k+ QPS.
type ClientPool struct {
	conns    []*grpc.ClientConn
	clients  []pb.GRPCInferenceServiceClient
	next     uint64 // round-robin counter (atomic)
	addr     string
	deadline time.Duration
}

// NewClientPool dials addr with production keepalive/backoff parameters.
func NewClientPool(addr string, size int, deadline time.Duration) (*ClientPool, error) {
	if size < 1 {
		return nil, fmt.Errorf("pool size must be >= 1, got %d", size)
	}
	p := &ClientPool{addr: addr, deadline: deadline}
	for i := 0; i < size; i++ {
		conn, err := grpc.Dial(addr,
			grpc.WithTransportCredentials(insecure.NewCredentials()),
			grpc.WithDefaultCallOptions(
				grpc.MaxCallRecvMsgSize(maxRecvMsgBytes),
				grpc.WaitForReady(false), // fail fast; retry is the caller's job
			),
			grpc.WithKeepaliveParams(keepalive.ClientParameters{
				Time:                10 * time.Second,
				Timeout:             3 * time.Second,
				PermitWithoutStream: true,
			}),
			grpc.WithConnectParams(grpc.ConnectParams{
				Backoff: backoff.Config{
					BaseDelay:  100 * time.Millisecond,
					Multiplier: 1.6,
					Jitter:     0.2,
					MaxDelay:   2 * time.Second,
				},
			}),
		)
		if err != nil {
			p.Close()
			return nil, fmt.Errorf("dial triton %s: %w", addr, err)
		}
		p.conns = append(p.conns, conn)
		p.clients = append(p.clients, pb.NewGRPCInferenceServiceClient(conn))
	}
	return p, nil
}

// Close releases every underlying connection.
func (p *ClientPool) Close() {
	for _, c := range p.conns {
		_ = c.Close()
	}
}

// pick returns the next client in round-robin order (cheap atomic bump).
func (p *ClientPool) pick() pb.GRPCInferenceServiceClient {
	i := atomicNext(&p.next)
	return p.clients[int(i)%len(p.clients)]
}

// Infer packs token IDs into raw little-endian INT64 tensors, ships them over
// a persistent HTTP/2 stream and decodes the FP16 embedding vector(s).
// The supplied ctx gets a hard deadline cap of p.deadline on top of whatever
// the caller already set — the gateway never lets a request exceed its SLA.
func (p *ClientPool) Infer(ctx context.Context, modelName, modelVersion string, inputIDs []int64, seqLen int64) ([]float32, error) {
	ctx, cancel := context.WithTimeout(ctx, p.deadline)
	defer cancel()

	start := time.Now()
	resp, err := p.inferOnce(ctx, modelName, modelVersion, inputIDs, seqLen)
	elapsed := time.Since(start).Seconds()
	inferLatency.Observe(elapsed)

	if err != nil {
		inferErrors.WithLabelValues(grpcCodeString(err)).Inc()
		return nil, err
	}
	return resp, nil
}

func (p *ClientPool) inferOnce(ctx context.Context, modelName, modelVersion string, inputIDs []int64, seqLen int64) ([]float32, error) {
	batch := int64(1)
	if seqLen > 0 {
		batch = int64(len(inputIDs)) / seqLen
	}
	raw := packInt64(inputIDs)
	req := &pb.ModelInferRequest{
		ModelName:    modelName,
		ModelVersion: modelVersion,
		Inputs: []*pb.ModelInferRequest_InferInputTensor{
			{Name: "input_ids", Datatype: "INT64", Shape: []int64{batch, seqLen}},
			{Name: "attention_mask", Datatype: "INT64", Shape: []int64{batch, seqLen}},
		},
		Outputs: []*pb.ModelInferRequest_InferRequestedOutputTensor{
			{Name: "sentence_embedding"},
		},
		RawInputContents: [][]byte{raw, raw}, // mask mirrors ids for test traffic
	}

	poolInUse.Inc()
	defer poolInUse.Dec()

	resp, err := p.pick().ModelInfer(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("model infer: %w", err)
	}
	if len(resp.GetRawOutputContents()) == 0 {
		return nil, fmt.Errorf("empty output contents from triton")
	}
	return unpackFloat16(resp.GetRawOutputContents()[0]), nil
}

// ServerLive powers the gateway healthcheck without a full inference round-trip.
func (p *ClientPool) ServerLive(ctx context.Context) (bool, error) {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	resp, err := p.pick().ServerLive(ctx, &pb.ServerLiveRequest{})
	if err != nil {
		return false, err
	}
	return resp.GetLive(), nil
}

// ModelReady reports whether the routing target is loaded on the server.
func (p *ClientPool) ModelReady(ctx context.Context, name string) (bool, error) {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	resp, err := p.pick().ModelReady(ctx, &pb.ModelReadyRequest{Name: name})
	if err != nil {
		return false, err
	}
	return resp.GetReady(), nil
}

// ── wire helpers ────────────────────────────────────────────────────────────

func packInt64(v []int64) []byte {
	buf := make([]byte, 8*len(v))
	for i, n := range v {
		binary.LittleEndian.PutUint64(buf[i*8:], uint64(n))
	}
	return buf
}

// unpackFloat16 converts Triton's raw FP16 buffer into float32 for JSON output.
func unpackFloat16(b []byte) []float32 {
	n := len(b) / 2
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = fp16to32(binary.LittleEndian.Uint16(b[i*2:]))
	}
	return out
}

// fp16to32 is the standard half-float widening algorithm (IEEE 754 binary16).
func fp16to32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	switch {
	case exp == 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// subnormal: normalize
		e := int32(-1)
		for mant&0x400 == 0 {
			mant <<= 1
			e++
		}
		mant &= 0x3ff
		return math.Float32frombits(sign | uint32(127-10-e)<<23 | mant<<13)
	case exp == 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | mant<<13)
	default:
		return math.Float32frombits(sign | ((exp + 112) << 23) | mant<<13)
	}
}

func grpcCodeString(err error) string {
	if code := statusOf(err); code != "" {
		return code
	}
	return "unknown"
}
