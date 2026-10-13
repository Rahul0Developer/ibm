package github

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// newTestServer returns a fake GitHub API server and a client pointed at it.
func newTestServer(t *testing.T, handler http.HandlerFunc) (*Client, func()) {
	t.Helper()
	srv := httptest.NewServer(handler)
	c := New("")
	c.BaseURL = srv.URL
	return c, srv.Close
}

func TestListUserRepos_OK(t *testing.T) {
	c, cleanup := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/users/octocat/repos" {
			t.Errorf("unexpected path %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[
		  {"name":"hello-world","full_name":"octocat/hello-world","description":"My first repo","stargazers_count":42,"forks_count":3,"language":"Go","html_url":"https://github.com/octocat/hello-world","updated_at":"2024-05-01T10:00:00Z"},
		  {"name":"spoon-knife","full_name":"octocat/spoon-knife","description":null,"stargazers_count":7,"forks_count":1,"language":null,"html_url":"https://github.com/octocat/spoon-knife","updated_at":"2023-01-02T03:04:05Z"}
		]`))
	})
	defer cleanup()

	repos, err := c.ListUserRepos("octocat", 10)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(repos) != 2 {
		t.Fatalf("got %d repos, want 2", len(repos))
	}
	if repos[0].FullName != "octocat/hello-world" || repos[0].Stars != 42 {
		t.Errorf("bad decode: %+v", repos[0])
	}
	if repos[1].Description != nil {
		t.Errorf("expected null description to decode to nil pointer")
	}
}

func TestListUserRepos_EmptyUser(t *testing.T) {
	c, cleanup := newTestServer(t, func(w http.ResponseWriter, _ *http.Request) {})
	defer cleanup()

	if _, err := c.ListUserRepos("   ", 5); err == nil {
		t.Fatal("expected error for empty user")
	}
}

func TestGetUser_NotFound(t *testing.T) {
	c, cleanup := newTestServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	})
	defer cleanup()

	_, err := c.GetUser("does-not-exist")
	if err == nil || !strings.Contains(err.Error(), "not found") {
		t.Fatalf("expected 'not found' error, got: %v", err)
	}
}

func TestAuthHeaderSent(t *testing.T) {
	var got string
	c, cleanup := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Get("Authorization")
		_, _ = w.Write([]byte(`{"login":"octocat"}`))
	})
	defer cleanup()

	c.APIToken = "secret-token"
	if _, err := c.GetUser("octocat"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "Bearer secret-token" {
		t.Errorf("Authorization = %q, want %q", got, "Bearer secret-token")
	}
}

func TestNoAuthHeaderWhenTokenEmpty(t *testing.T) {
	var got string
	c, cleanup := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Get("Authorization")
		_, _ = w.Write([]byte(`{"login":"octocat"}`))
	})
	defer cleanup()

	if _, err := c.GetUser("octocat"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "" {
		t.Errorf("expected no Authorization header, got %q", got)
	}
}

func TestRateLimitError(t *testing.T) {
	c, cleanup := newTestServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusForbidden)
	})
	defer cleanup()

	_, err := c.GetUser("octocat")
	if err == nil || !strings.Contains(err.Error(), "GITHUB_TOKEN") {
		t.Fatalf("expected rate-limit/auth error, got: %v", err)
	}
}

func TestBadJSON(t *testing.T) {
	c, cleanup := newTestServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`not json`))
	})
	defer cleanup()

	if _, err := c.GetUser("octocat"); err == nil {
		t.Fatal("expected decode error")
	}
}
