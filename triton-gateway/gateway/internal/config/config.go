// Package config loads gateway configuration from environment variables with
// validated defaults. Fail fast at boot on invalid values — never at request time.
package config

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

// Config is the immutable runtime configuration of the inference gateway.
type Config struct {
	GatewayPort       int           // HTTP listen port (inference + health + metrics)
	TritonGRPCAddr    string        // Triton gRPC endpoint host:port
	ModelName         string        // Triton model name to route to
	ModelVersion      string        // pinned version; "" => server default
	RequestDeadline   time.Duration // strict end-to-end SLA deadline per request
	MaxConcurrentReqs int           // admission-control semaphore capacity
	GRPCPoolSize      int           // pooled sub-connections toward Triton
	LogLevel          string
}

const (
	envPrefix          = "GATEWAY_"
	defaultPort        = 8080
	defaultDeadlineMS  = 25
	defaultConcurrency = 512
	defaultPoolSize    = 4
)

// FromEnv builds a Config, applying defaults and rejecting nonsensical values.
func FromEnv() (*Config, error) {
	c := &Config{
		GatewayPort:       envInt("PORT", defaultPort),
		TritonGRPCAddr:    envStr("TRITON_GRPC_ADDR", "localhost:8001"),
		ModelName:         envStr("MODEL_NAME", "text_embeddings"),
		ModelVersion:      os.Getenv("MODEL_VERSION"),
		RequestDeadline:   time.Duration(envInt("REQUEST_DEADLINE_MS", defaultDeadlineMS)) * time.Millisecond,
		MaxConcurrentReqs: envInt("MAX_CONCURRENT_REQUESTS", defaultConcurrency),
		GRPCPoolSize:      envInt("GRPC_POOL_SIZE", defaultPoolSize),
		LogLevel:          envStr("LOG_LEVEL", "info"),
	}
	if err := c.validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}
	return c, nil
}

func (c *Config) validate() error {
	if c.GatewayPort <= 0 || c.GatewayPort > 65535 {
		return fmt.Errorf("gateway port %d out of range", c.GatewayPort)
	}
	if c.TritonGRPCAddr == "" {
		return fmt.Errorf("TRITON_GRPC_ADDR must not be empty")
	}
	if c.RequestDeadline < 5*time.Millisecond || c.RequestDeadline > 5*time.Second {
		return fmt.Errorf("request deadline %s outside sane bounds [5ms, 5s]", c.RequestDeadline)
	}
	if c.MaxConcurrentReqs <= 0 {
		return fmt.Errorf("max concurrent requests must be positive")
	}
	if c.GRPCPoolSize <= 0 || c.GRPCPoolSize > 64 {
		return fmt.Errorf("grpc pool size %d out of range (1..64)", c.GRPCPoolSize)
	}
	return nil
}

func envStr(key, def string) string {
	if v := os.Getenv(envPrefix + key); v != "" {
		return v
	}
	// Also accept the unprefixed .env.example names for local runs.
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envInt(key string, def int) int {
	raw := envStr(key, "")
	if raw == "" {
		return def
	}
	n, err := strconv.Atoi(raw)
	if err != nil {
		return def
	}
	return n
}
