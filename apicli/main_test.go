package main

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

// withTestAPI points GOAPI_BASE_URL at a fake server for the duration of a test.
func withTestAPI(t *testing.T) {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/posts", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`[{"userId":1,"id":1,"title":"one","body":"b1"},{"userId":1,"id":2,"title":"two","body":"b2"}]`))
	})
	mux.HandleFunc("/posts/", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"userId":1,"id":1,"title":"one","body":"b1"}`))
	})
	mux.HandleFunc("/users", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`[{"id":1,"name":"Leanne","username":"Bret","email":"s@april.biz"}]`))
	})
	mux.HandleFunc("/users/", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"id":3,"name":"Clementine","username":"Kamren","email":"k@april.biz"}`))
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	os.Setenv("GOAPI_BASE_URL", srv.URL)
	t.Cleanup(func() { os.Unsetenv("GOAPI_BASE_URL") })
}

func runCLI(t *testing.T, args ...string) (string, error) {
	t.Helper()
	var out bytes.Buffer
	err := run(args, &out, &out)
	return out.String(), err
}

func TestVersionAndHelp(t *testing.T) {
	out, err := runCLI(t, "version")
	if err != nil || !strings.Contains(out, version) {
		t.Fatalf("version output wrong: %q err=%v", out, err)
	}
	out, err = runCLI(t, "--help")
	if err != nil || !strings.Contains(out, "Usage:") {
		t.Fatalf("help output wrong: %q err=%v", out, err)
	}
}

func TestPostsListTable(t *testing.T) {
	withTestAPI(t)
	out, err := runCLI(t, "posts", "list")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "TITLE") || !strings.Contains(out, "one") {
		t.Errorf("unexpected table output:\n%s", out)
	}
}

func TestPostsListLimitAndJSON(t *testing.T) {
	withTestAPI(t)
	out, err := runCLI(t, "posts", "list", "-n", "1", "--format", "json")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(out, `"two"`) || !strings.Contains(out, `"one"`) {
		t.Errorf("-n 1 --format json did not limit output:\n%s", out)
	}
}

func TestFlagsAfterPositionalArgs(t *testing.T) {
	withTestAPI(t)
	out, err := runCLI(t, "posts", "get", "1", "--format", "json")
	if err != nil {
		t.Fatal(err)
	}
	if !jsonLooksValid(out) {
		t.Errorf("expected JSON output, got:\n%s", out)
	}
}

func jsonLooksValid(s string) bool {
	return strings.HasPrefix(strings.TrimSpace(s), "{") && strings.HasSuffix(strings.TrimSpace(s), "}")
}

func TestUsersGet(t *testing.T) {
	withTestAPI(t)
	out, err := runCLI(t, "users", "get", "3")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "Kamren") {
		t.Errorf("unexpected user output:\n%s", out)
	}
}

func TestUnknownCommand(t *testing.T) {
	if _, err := runCLI(t, "frobnicate"); err == nil {
		t.Fatal("expected error for unknown command")
	}
}

func TestMissingID(t *testing.T) {
	withTestAPI(t)
	if _, err := runCLI(t, "posts", "get"); err == nil {
		t.Fatal("expected usage error when id missing")
	}
}

func TestInvalidFormat(t *testing.T) {
	withTestAPI(t)
	if _, err := runCLI(t, "posts", "list", "--format", "yaml"); err == nil {
		t.Fatal("expected error for invalid format")
	}
}

func TestCreateDryRun(t *testing.T) {
	withTestAPI(t)
	out, err := runCLI(t, "posts", "create", "--title", "Hello", "--body", "World", "--dry-run")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "[dry-run]") || !strings.Contains(out, "Hello") {
		t.Errorf("unexpected dry-run output:\n%s", out)
	}
}

func TestParseArgsMix(t *testing.T) {
	set := newFlagSetForTest()
	positional, err := parseArgs(set, []string{"get", "-n", "5", "--format", "json", "42"})
	if err != nil {
		t.Fatal(err)
	}
	if len(positional) != 2 || positional[0] != "get" || positional[1] != "42" {
		t.Errorf("wrong positional args: %v", positional)
	}
}
