package triton

import (
	"sync/atomic"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// atomicNext atomically bumps the round-robin counter. Kept in a tiny helper so
// client.go stays focused on transport concerns.
func atomicNext(p *uint64) uint64 { return atomic.AddUint64(p, 1) }

// statusOf maps an error to its gRPC status code string for metric labels.
func statusOf(err error) string {
	if s, ok := status.FromError(err); ok {
		switch s.Code() {
		case codes.DeadlineExceeded:
			return "deadline_exceeded"
		case codes.Unavailable:
			return "unavailable"
		case codes.InvalidArgument:
			return "invalid_argument"
		case codes.ResourceExhausted:
			return "resource_exhausted"
		default:
			return s.Code().String()
		}
	}
	return ""
}
