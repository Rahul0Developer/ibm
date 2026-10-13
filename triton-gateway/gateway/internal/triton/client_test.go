package triton

import (
	"context"
	"log"
	"net"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/your-org/triton-gateway/internal/pb"
)

// testBackend is an in-process Triton stand-in implementing the same wire
// contract as a real tritonserver, so unit tests never need a GPU.
type testBackend struct {
	pb.UnimplementedGRPCInferenceServiceServer
	delay    time.Duration
	failWith error
	gotReqs  atomic.Int64
}

func (b *testBackend) ModelInfer(ctx context.Context, req *pb.ModelInferRequest) (*pb.ModelInferResponse, error) {
	b.gotReqs.Add(1)
	if b.delay > 0 {
		select {
		case <-time.After(b.delay):
		case <-ctx.Done():
			return nil, status.Error(codes.DeadlineExceeded, "backend deadline")
		}
	}
	if b.failWith != nil {
		return nil, b.failWith
	}
	// Echo back a deterministic FP16 vector: [1.0, 2.0, 3.0]
	vec := []float32{1.0, 2.0, 3.0}
	raw := make([]byte, 2*len(vec))
	for i, f := range vec {
		binary := float32ToFP16Bytes(f)
		copy(raw[i*2:], binary)
	}
	return &pb.ModelInferResponse{
		ModelName:         req.GetModelName(),
		RawOutputContents: [][]byte{raw},
		Outputs: []*pb.ModelInferResponse_InferOutputTensor{
			{Name: "sentence_embedding", Datatype: "FP16", Shape: []int64{1, 3}},
		},
	}, nil
}

func (b *testBackend) ServerLive(context.Context, *pb.ServerLiveRequest) (*pb.ServerLiveResponse, error) {
	return &pb.ServerLiveResponse{Live: true}, nil
}

func (b *testBackend) ModelReady(context.Context, *pb.ModelReadyRequest) (*pb.ModelReadyResponse, error) {
	return &pb.ModelReadyResponse{Ready: true}, nil
}

func startBackend(t *testing.T, b *testBackend) string {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	srv := grpc.NewServer()
	pb.RegisterGRPCInferenceServiceServer(srv, b)
	go func() {
		if err := srv.Serve(lis); err != nil {
			log.Default().Println("backend serve:", err)
		}
	}()
	t.Cleanup(srv.Stop)
	return lis.Addr().String()
}

func newPool(t *testing.T, addr string) *ClientPool {
	t.Helper()
	p, err := NewClientPool(addr, 2, 25*time.Millisecond)
	if err != nil {
		t.Fatalf("NewClientPool: %v", err)
	}
	t.Cleanup(p.Close)
	return p
}

func TestPackInt64RoundTrip(t *testing.T) {
	in := []int64{-5, 0, 7, 1 << 40}
	raw := packInt64(in)
	if len(raw) != 8*len(in) {
		t.Fatalf("packed size = %d, want %d", len(raw), 8*len(in))
	}
	for i, want := range in {
		got := int64(uint64Raw(raw[i*8:]))
		if got != want {
			t.Errorf("elem %d = %d, want %d", i, got, want)
		}
	}
}

func TestUnpackFloat16KnownVectors(t *testing.T) {
	// 0x3C00 == 1.0h, 0xC000 == -2.0h, 0x0000 == +0.0h
	out := unpackFloat16([]byte{0x00, 0x3C, 0x00, 0xC0, 0x00, 0x00})
	want := []float32{1.0, -2.0, 0.0}
	for i := range want {
		if out[i] != want[i] {
			t.Errorf("fp16[%d] = %v, want %v", i, out[i], want[i])
		}
	}
}

func TestFp16SpecialCases(t *testing.T) {
	if v := fp16to32(0x7C00); !isInf(v, 1) {
		t.Errorf("+Inf decode = %v", v)
	}
	if v := fp16to32(0xFC00); !isInf(v, -1) {
		t.Errorf("-Inf decode = %v", v)
	}
	if v := fp16to32(0x0001); v <= 0 || v > 6.2e-5 { // smallest subnormal ≈ 5.96e-8
		t.Errorf("subnormal decode out of range: %v", v)
	}
}

func TestInferHappyPath(t *testing.T) {
	b := &testBackend{}
	pool := newPool(t, startBackend(t, b))

	emb, err := pool.Infer(context.Background(), "text_embeddings", "1", []int64{1, 2, 3, 4}, 4)
	if err != nil {
		t.Fatalf("Infer: %v", err)
	}
	if len(emb) != 3 || emb[0] != 1.0 {
		t.Fatalf("unexpected embedding %v", emb)
	}
	if b.gotReqs.Load() != 1 {
		t.Fatalf("backend saw %d requests, want 1", b.gotReqs.Load())
	}
}

func TestInferRespectsDeadlineBudget(t *testing.T) {
	b := &testBackend{delay: 200 * time.Millisecond} // exceeds 25ms pool deadline
	pool := newPool(t, startBackend(t, b))

	start := time.Now()
	_, err := pool.Infer(context.Background(), "text_embeddings", "1", []int64{1}, 1)
	elapsed := time.Since(start)

	if err == nil || !strings.Contains(err.Error(), "DeadlineExceeded") {
		t.Fatalf("want DeadlineExceeded, got %v", err)
	}
	if elapsed > 100*time.Millisecond {
		t.Fatalf("deadline not enforced: took %s", elapsed)
	}
}

func TestInferPropagatesBackendError(t *testing.T) {
	b := &testBackend{failWith: status.Error(codes.Unavailable, "gpu oom")}
	pool := newPool(t, startBackend(t, b))

	_, err := pool.Infer(context.Background(), "text_embeddings", "1", []int64{1}, 1)
	if err == nil || !strings.Contains(err.Error(), "Unavailable") {
		t.Fatalf("want Unavailable, got %v", err)
	}
}

func TestServerLiveAndModelReady(t *testing.T) {
	b := &testBackend{}
	pool := newPool(t, startBackend(t, b))

	live, err := pool.ServerLive(context.Background())
	if err != nil || !live {
		t.Fatalf("ServerLive = %v, %v", live, err)
	}
	ready, err := pool.ModelReady(context.Background(), "text_embeddings")
	if err != nil || !ready {
		t.Fatalf("ModelReady = %v, %v", ready, err)
	}
}

func TestPoolRejectsBadSize(t *testing.T) {
	if _, err := NewClientPool("localhost:1", 0, time.Second); err == nil {
		t.Fatal("expected error for pool size 0")
	}
}

func uint64Raw(b []byte) uint64 {
	var v uint64
	for i := 7; i >= 0; i-- {
		v = v<<8 | uint64(b[i])
	}
	return v
}

func isInf(f float32, sign int) bool {
	bits := fp32bits(f)
	exp := (bits >> 23) & 0xff
	mant := bits & 0x7fffff
	if exp != 0xff || mant != 0 {
		return false
	}
	s := bits>>31 == 1
	return (sign > 0 && !s) || (sign < 0 && s)
}
