// Package middleware provides composable net/http middleware for the gateway:
// structured request logging and panic recovery. Handlers stay focused on
// validation, admission control and encoding.
package middleware

import (
	"log"
	"net/http"
	"time"
)

// statusRecorder captures the written status code for logging.
type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

// Logging wraps a handler with one structured key=value line per request.
func Logging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rec, req)
		log.Printf("level=INFO msg=request method=%s path=%s status=%d dur_ms=%.2f remote=%s",
			req.Method, req.URL.Path, rec.status,
			float64(time.Since(start).Microseconds())/1000.0, req.RemoteAddr)
	})
}

// Recover converts panics in downstream handlers into 500s instead of killing
// the process — a single bad payload must never take the whole pod offline.
func Recover(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		defer func() {
			if err := recover(); err != nil {
				log.Printf("level=ERROR msg=panic_recovered path=%s err=%v", req.URL.Path, err)
				w.WriteHeader(http.StatusInternalServerError)
				_, _ = w.Write([]byte(`{"error":"internal","message":"recovered from panic"}`))
			}
		}()
		next.ServeHTTP(w, req)
	})
}
