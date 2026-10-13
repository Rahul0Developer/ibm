// Package server exposes the gateway's public HTTP surface: a JSON inference
// endpoint, Kubernetes health probes and the Prometheus /metrics scrape point.
// Business logic stays out of handlers — they validate, admit, call and encode.
package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/your-org/triton-gateway/internal/config"
	"github.com/your-org/triton-gateway/internal/middleware"
	"github.com/your-org/triton-gateway/internal/tokenizer"
)

// Go 1.19 compatibility: atomic.Bool landed in 1.19 but generic-free helper keeps intent clear.
type boolFlag struct{ v atomic.Value }

func (b *boolFlag) Store(v bool) { b.v.Store(v) }
func (b *boolFlag) Load() bool {
	if x, ok := b.v.Load().(bool); ok {
		return x
	}
	return false
}

const maxBodyBytes = 256 * 1024 // 256 KiB caps a batch of short texts

var (
	reqTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "gateway",
		Subsystem: "http",
		Name:      "requests_total",
		Help:      "Inference requests by outcome (ok, invalid, overloaded, deadline, upstream).",
	}, []string{"outcome"})

	httpLatency = promauto.NewHistogram(prometheus.HistogramOpts{
		Namespace: "gateway",
		Subsystem: "http",
		Name:      "request_duration_seconds",
		Help:      "End-to-end gateway request latency including tokenization.",
		Buckets:   []float64{.001, .0025, .005, .01, .015, .02, .025, .03, .05, .1, .25},
	})

	admittedGauge = promauto.NewGauge(prometheus.GaugeOpts{
		Namespace: "gateway",
		Subsystem: "http",
		Name:      "inflight_requests",
		Help:      "Requests currently admitted past the concurrency semaphore.",
	})
)

// InferencePool is the Triton dependency of the server (mockable in tests).
type InferencePool interface {
	Infer(ctx context.Context, modelName, modelVersion string, inputIDs []int64, seqLen int64) ([]float32, error)
	ServerLive(ctx context.Context) (bool, error)
}

// Server wires config, tokenizer and Triton pool into an http.Handler.
type Server struct {
	cfg     *config.Config
	pool    InferencePool
	tok     *tokenizer.Tokenizer
	sem     chan struct{}
	ready   boolFlag
	httpSrv *http.Server
}

// New constructs the server and pre-fills the admission semaphore.
func New(cfg *config.Config, pool InferencePool) *Server {
	s := &Server{
		cfg:  cfg,
		pool: pool,
		tok:  tokenizer.New(),
		sem:  make(chan struct{}, cfg.MaxConcurrentReqs),
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/embeddings", s.withMethod(http.MethodPost, s.handleEmbeddings))
	mux.HandleFunc("/healthz/live", s.withMethod(http.MethodGet, s.handleLive))
	mux.HandleFunc("/healthz/ready", s.withMethod(http.MethodGet, s.handleReady))
	mux.Handle("/metrics", s.withMethod(http.MethodGet, promhttp.Handler().ServeHTTP))
	s.httpSrv = &http.Server{
		Addr: ":" + strconv.Itoa(cfg.GatewayPort),
		// Recover outermost so a panic anywhere still yields a 500, and every
		// request gets one structured log line.
		Handler:           middleware.Recover(middleware.Logging(mux)),
		ReadHeaderTimeout: 5 * time.Second,
	}
	return s
}

// MarkReady flips the readiness probe after the backend is confirmed live.
func (s *Server) MarkReady(v bool) { s.ready.Store(v) }

// ListenAndServe blocks serving until the context is cancelled, then drains.
func (s *Server) ListenAndServe(ctx context.Context) error {
	go func() {
		<-ctx.Done()
		shutCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = s.httpSrv.Shutdown(shutCtx)
	}()
	if err := s.httpSrv.ListenAndServe(); errors.Is(err, http.ErrServerClosed) {
		return nil
	} else {
		return err
	}
}

// withMethod enforces the HTTP verb (Go 1.19 has no "METHOD /path" mux patterns).
func (s *Server) withMethod(method string, h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != method {
			writeError(w, http.StatusMethodNotAllowed, "bad_method", "method not allowed")
			return
		}
		h(w, r)
	}
}

type embeddingsRequest struct {
	Texts []string `json:"texts"`
}

type embeddingsResponse struct {
	Model      string      `json:"model"`
	Dimension  int         `json:"dimension"`
	Embeddings [][]float32 `json:"embeddings"`
	LatencyMS  float64     `json:"latency_ms"`
}

func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	defer func() { httpLatency.Observe(time.Since(start).Seconds()) }()

	var body embeddingsRequest
	r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_json", "request body must be {\"texts\": [...]}")
		reqTotal.WithLabelValues("invalid").Inc()
		return
	}
	if len(body.Texts) == 0 || len(body.Texts) > 64 {
		writeError(w, http.StatusBadRequest, "invalid_batch", "texts length must be in [1, 64]")
		reqTotal.WithLabelValues("invalid").Inc()
		return
	}
	for _, t := range body.Texts {
		if len(t) == 0 || len(t) > 2048 {
			writeError(w, http.StatusBadRequest, "invalid_input", "each text must be 1..2048 bytes")
			reqTotal.WithLabelValues("invalid").Inc()
			return
		}
	}

	// Admission control: shed load instead of queueing past the SLA budget.
	select {
	case s.sem <- struct{}{}:
		admittedGauge.Inc()
		defer func() { <-s.sem; admittedGauge.Dec() }()
	default:
		writeError(w, http.StatusTooManyRequests, "overloaded", "gateway at concurrency limit, retry shortly")
		reqTotal.WithLabelValues("overloaded").Inc()
		return
	}

	// Tokenize every text to a fixed 64-token window (WordPiece-style IDs).
	const seqLen = 64
	var ids []int64
	for _, t := range body.Texts {
		ids = append(ids, s.tok.EncodePadded(t, seqLen)...)
	}

	vecs, err := s.pool.Infer(r.Context(), s.cfg.ModelName, s.cfg.ModelVersion, ids, seqLen)
	if err != nil {
		status, code := classifyUpstreamErr(err)
		writeError(w, status, code, "upstream inference failed")
		reqTotal.WithLabelValues(code).Inc()
		return
	}

	dim := 384
	out := make([][]float32, 0, len(body.Texts))
	for i := 0; i+dim <= len(vecs); i += dim {
		out = append(out, vecs[i:i+dim])
	}
	if len(out) != len(body.Texts) {
		writeError(w, http.StatusBadGateway, "upstream", "embedding count mismatch")
		reqTotal.WithLabelValues("upstream").Inc()
		return
	}

	writeJSON(w, http.StatusOK, embeddingsResponse{
		Model:      s.cfg.ModelName,
		Dimension:  dim,
		Embeddings: out,
		LatencyMS:  float64(time.Since(start).Microseconds()) / 1000.0,
	})
	reqTotal.WithLabelValues("ok").Inc()
}

func (s *Server) handleLive(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "alive"})
}

func (s *Server) handleReady(w http.ResponseWriter, r *http.Request) {
	if !s.ready.Load() {
		writeError(w, http.StatusServiceUnavailable, "not_ready", "backend not verified yet")
		return
	}
	ok, err := s.pool.ServerLive(r.Context())
	if err != nil || !ok {
		writeError(w, http.StatusServiceUnavailable, "backend_down", "triton is not live")
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

func classifyUpstreamErr(err error) (int, string) {
	switch {
	case errors.Is(err, context.DeadlineExceeded):
		return http.StatusGatewayTimeout, "deadline"
	default:
		return http.StatusBadGateway, "upstream"
	}
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, code, msg string) {
	writeJSON(w, status, map[string]string{"error": code, "message": msg})
}
