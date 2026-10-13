package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/your-org/triton-gateway/internal/config"
)

// fakePool implements InferencePool for handler tests.
type fakePool struct {
	vectors []float32
	err     error
	live    bool
	gotIDs  []int64
}

func (f *fakePool) Infer(ctx context.Context, name, version string, ids []int64, seqLen int64) ([]float32, error) {
	f.gotIDs = ids
	if f.err != nil {
		return nil, f.err
	}
	return f.vectors, nil
}

func (f *fakePool) ServerLive(ctx context.Context) (bool, error) { return f.live, nil }

func newTestServer(pool InferencePool) *Server {
	cfg := &config.Config{
		GatewayPort:       8080,
		TritonGRPCAddr:    "localhost:8001",
		ModelName:         "text_embeddings",
		RequestDeadline:   25_000_000, // 25ms as time.Duration ns
		MaxConcurrentReqs: 2,
		GRPCPoolSize:      4,
	}
	return New(cfg, pool)
}

func postJSON(s *Server, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewBufferString(body))
	rec := httptest.NewRecorder()
	s.httpSrv.Handler.ServeHTTP(rec, req)
	return rec
}

func TestEmbeddingsHappyPath(t *testing.T) {
	vecs := make([]float32, 2*384)
	for i := range vecs {
		vecs[i] = float32(i%7) / 7.0
	}
	pool := &fakePool{vectors: vecs, live: true}
	s := newTestServer(pool)

	rec := postJSON(s, `{"texts":["hello world","second text"]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("want 200 got %d body=%s", rec.Code, rec.Body.String())
	}
	var out embeddingsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if len(out.Embeddings) != 2 || out.Dimension != 384 {
		t.Fatalf("bad response shape: %d x %d", len(out.Embeddings), out.Dimension)
	}
	// 2 texts x 64 tokens must have been packed into the flat ID tensor.
	if len(pool.gotIDs) != 128 {
		t.Fatalf("want 128 packed ids, got %d", len(pool.gotIDs))
	}
}

func TestEmbeddingsRejectsEmptyBatch(t *testing.T) {
	s := newTestServer(&fakePool{})
	rec := postJSON(s, `{"texts":[]}`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("want 400 got %d", rec.Code)
	}
}

func TestEmbeddingsRejectsOversizedBatch(t *testing.T) {
	s := newTestServer(&fakePool{})
	texts := make([]string, 65)
	for i := range texts {
		texts[i] = "x"
	}
	body, _ := json.Marshal(map[string]interface{}{"texts": texts})
	rec := postJSON(s, string(body))
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "invalid_batch") {
		t.Fatalf("want 400 invalid_batch got %d %s", rec.Code, rec.Body.String())
	}
}

func TestEmbeddingsRejectsBadJSON(t *testing.T) {
	s := newTestServer(&fakePool{})
	rec := postJSON(s, `{not json`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("want 400 got %d", rec.Code)
	}
}

func TestUpstreamDeadlineMapsTo504(t *testing.T) {
	pool := &fakePool{err: context.DeadlineExceeded}
	s := newTestServer(pool)
	rec := postJSON(s, `{"texts":["hi"]}`)
	if rec.Code != http.StatusGatewayTimeout {
		t.Fatalf("want 504 got %d body=%s", rec.Code, rec.Body.String())
	}
}

func TestUpstreamErrorMapsTo502(t *testing.T) {
	pool := &fakePool{err: errors.New("triton unavailable")}
	s := newTestServer(pool)
	rec := postJSON(s, `{"texts":["hi"]}`)
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("want 502 got %d", rec.Code)
	}
}

func TestAdmissionControlShedsLoad(t *testing.T) {
	// Semaphore capacity is 2; occupy all slots then fire a third request.
	s := newTestServer(&fakePool{vectors: make([]float32, 384)})
	s.sem <- struct{}{}
	s.sem <- struct{}{}
	rec := postJSON(s, `{"texts":["hi"]}`)
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("want 429 shed, got %d body=%s", rec.Code, rec.Body.String())
	}
}

func TestReadyProbeReflectsBackend(t *testing.T) {
	pool := &fakePool{live: false}
	s := newTestServer(pool)
	s.MarkReady(true)

	rec := httptest.NewRecorder()
	s.httpSrv.Handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/healthz/ready", nil))
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("backend down: want 503 got %d", rec.Code)
	}

	pool.live = true
	rec = httptest.NewRecorder()
	s.httpSrv.Handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/healthz/ready", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("backend live: want 200 got %d", rec.Code)
	}
}

func TestMethodNotAllowed(t *testing.T) {
	s := newTestServer(&fakePool{})
	rec := httptest.NewRecorder()
	s.httpSrv.Handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/embeddings", nil))
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("want 405 got %d", rec.Code)
	}
}
