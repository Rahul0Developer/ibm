package middleware

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestLoggingPassesThroughStatus(t *testing.T) {
	h := Logging(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTeapot)
	}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/embeddings", nil))
	if rec.Code != http.StatusTeapot {
		t.Fatalf("want 418, got %d", rec.Code)
	}
}

func TestRecoverTurnsPanicInto500(t *testing.T) {
	h := Recover(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		panic("boom")
	}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/embeddings", nil))
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("want 500, got %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "recovered") {
		t.Fatalf("unexpected body: %s", rec.Body.String())
	}
}
