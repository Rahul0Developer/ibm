package api

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/yourusername/apicli/models"
)

// newTestServer starts an httptest server that routes a few fake API paths.
func newTestServer(t *testing.T) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()

	mux.HandleFunc("/posts", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			var np models.NewPost
			body, _ := io.ReadAll(r.Body)
			if err := json.Unmarshal(body, &np); err != nil {
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusCreated)
			json.NewEncoder(w).Encode(models.Post{ID: 101, UserID: np.UserID, Title: np.Title, Body: np.Body})
			return
		}
		w.Write([]byte(`[{"userId":1,"id":1,"title":"sunt aut facere","body":"quia et suscipit"}]`))
	})

	mux.HandleFunc("/posts/", func(w http.ResponseWriter, r *http.Request) {
		id := strings.TrimPrefix(r.URL.Path, "/posts/")
		if id == "999" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if r.Method == http.MethodDelete {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`{}`))
			return
		}
		w.Write([]byte(`{"userId":1,"id":` + id + `,"title":"hello","body":"world"}`))
	})

	mux.HandleFunc("/users", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`[{"id":1,"name":"Leanne Graham","username":"Bret","email":"Sincere@april.biz"}]`))
	})

	mux.HandleFunc("/users/", func(w http.ResponseWriter, r *http.Request) {
		id := strings.TrimPrefix(r.URL.Path, "/users/")
		if id == "404" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Write([]byte(`{"id":` + id + `,"name":"Leanne Graham","username":"Bret","email":"Sincere@april.biz"}`))
	})

	mux.HandleFunc("/auth", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer secret-token" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.Write([]byte(`ok`))
	})

	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func TestGetPosts(t *testing.T) {
	c := NewClient(newTestServer(t).URL, "")
	posts, err := c.GetPosts()
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(posts) != 1 || posts[0].ID != 1 {
		t.Fatalf("unexpected posts: %+v", posts)
	}
}

func TestGetPost(t *testing.T) {
	c := NewClient(newTestServer(t).URL, "")
	post, err := c.GetPost("1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if post.ID != 1 || post.Title != "hello" {
		t.Errorf("unexpected post: %+v", post)
	}
}

func TestGetPostNotFound(t *testing.T) {
	c := NewClient(newTestServer(t).URL, "")
	_, err := c.GetPost("999")
	if err == nil || !strings.Contains(err.Error(), "not found") {
		t.Fatalf("expected not-found error, got %v", err)
	}
}

func TestGetPostInvalidID(t *testing.T) {
	c := NewClient(newTestServer(t).URL, "")
	for _, bad := range []string{"abc", "-3", "0", ""} {
		if _, err := c.GetPost(bad); err == nil {
			t.Errorf("expected error for invalid id %q", bad)
		}
	}
}

func TestCreatePost(t *testing.T) {
	c := NewClient(newTestServer(t).URL, "")
	created, err := c.CreatePost(models.NewPost{Title: "t", Body: "b", UserID: 7})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if created.ID != 101 || created.UserID != 7 || created.Title != "t" {
		t.Errorf("unexpected created post: %+v", created)
	}
}

func TestDeletePost(t *testing.T) {
	c := NewClient(newTestServer(t).URL, "")
	if err := c.DeletePost("1"); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if err := c.DeletePost("nope"); err == nil {
		t.Fatal("expected error for non-numeric id")
	}
}

func TestGetUsersAndUser(t *testing.T) {
	c := NewClient(newTestServer(t).URL, "")
	users, err := c.GetUsers()
	if err != nil || len(users) != 1 || users[0].Username != "Bret" {
		t.Fatalf("unexpected users/error: %+v %v", users, err)
	}
	user, err := c.GetUser("3")
	if err != nil || user.ID != 3 {
		t.Fatalf("unexpected user/error: %+v %v", user, err)
	}
	if _, err := c.GetUser("404"); err == nil {
		t.Fatal("expected not-found error")
	}
}

func TestTokenIsSent(t *testing.T) {
	srv := newTestServer(t)
	c := NewClient(srv.URL, "secret-token")
	if err := c.do(http.MethodGet, "/auth", nil, nil); err != nil {
		t.Fatalf("expected auth to succeed with token, got %v", err)
	}
	cNoAuth := NewClient(srv.URL, "")
	if err := cNoAuth.do(http.MethodGet, "/auth", nil, nil); err == nil {
		t.Fatal("expected auth failure without token")
	}
}

func TestBadJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`this is not json`))
	}))
	defer srv.Close()
	c := NewClient(srv.URL, "")
	if _, err := c.GetPosts(); err == nil || !strings.Contains(err.Error(), "decode") {
		t.Fatalf("expected decode error, got %v", err)
	}
}
