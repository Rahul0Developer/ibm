// Command gateway runs the low-latency Triton inference API gateway:
// HTTP/JSON in -> tokenization + admission control -> pooled gRPC to Triton.
package main

import (
	"context"
	"encoding/json"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/your-org/triton-gateway/internal/config"
	"github.com/your-org/triton-gateway/internal/server"
	"github.com/your-org/triton-gateway/internal/triton"
)

func main() {
	if err := run(); err != nil {
		log.Printf("level=ERROR msg=fatal error=%q", err)
		os.Exit(1)
	}
}

func logJSON(level, msg string, kv map[string]interface{}) {
	b, _ := json.Marshal(kv)
	log.Printf("level=%s msg=%s %s", level, msg, b)
}

func run() error {
	cfg, err := config.FromEnv()
	if err != nil {
		return err
	}

	pool, err := triton.NewClientPool(cfg.TritonGRPCAddr, cfg.GRPCPoolSize, cfg.RequestDeadline)
	if err != nil {
		return err
	}
	defer pool.Close()

	srv := server.New(cfg, pool)

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	// Verify backend liveness asynchronously so K8s readiness gates traffic
	// until Triton answers ServerLive; startup never blocks on a cold GPU pod.
	go func() {
		for {
			live, err := pool.ServerLive(context.Background())
			if err == nil && live {
				srv.MarkReady(true)
				logJSON("INFO", "backend verified live", map[string]interface{}{
					"triton": cfg.TritonGRPCAddr, "model": cfg.ModelName,
				})
				return
			}
			select {
			case <-ctx.Done():
				return
			case <-time.After(time.Second):
			}
		}
	}()

	logJSON("INFO", "gateway listening", map[string]interface{}{
		"port": cfg.GatewayPort, "deadline_ms": cfg.RequestDeadline.Milliseconds(),
	})
	return srv.ListenAndServe(ctx)
}
