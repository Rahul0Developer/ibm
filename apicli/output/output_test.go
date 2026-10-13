package output

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/yourusername/apicli/models"
)

var testPosts = []models.Post{
	{UserID: 1, ID: 1, Title: "first post", Body: "body one"},
	{UserID: 2, ID: 2, Title: "second post", Body: "body two"},
}

var testUsers = []models.User{
	{ID: 1, Name: "Leanne Graham", Username: "Bret", Email: "s@april.biz"},
}

func TestParseFormat(t *testing.T) {
	for _, ok := range []string{"table", "json"} {
		if _, err := ParseFormat(ok); err != nil {
			t.Errorf("format %q should be valid: %v", ok, err)
		}
	}
	if _, err := ParseFormat("yaml"); err == nil {
		t.Error("expected error for unknown format")
	}
}

func TestPostsTable(t *testing.T) {
	var buf bytes.Buffer
	if err := Posts(&buf, Table, testPosts); err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	if !strings.Contains(out, "ID") || !strings.Contains(out, "first post") {
		t.Errorf("table output missing expected content:\n%s", out)
	}
}

func TestPostsJSON(t *testing.T) {
	var buf bytes.Buffer
	if err := Posts(&buf, JSON, testPosts); err != nil {
		t.Fatal(err)
	}
	var got []models.Post
	if err := json.Unmarshal(buf.Bytes(), &got); err != nil {
		t.Fatalf("output is not valid JSON: %v\n%s", err, buf.String())
	}
	if len(got) != 2 || got[1].Title != "second post" {
		t.Errorf("unexpected JSON round-trip: %+v", got)
	}
}

func TestUsersTableAndJSON(t *testing.T) {
	var buf bytes.Buffer
	if err := Users(&buf, Table, testUsers); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "Bret") {
		t.Errorf("user table missing username:\n%s", buf.String())
	}

	buf.Reset()
	if err := Users(&buf, JSON, testUsers); err != nil {
		t.Fatal(err)
	}
	if !json.Valid(buf.Bytes()) {
		t.Errorf("user JSON invalid:\n%s", buf.String())
	}
}

func TestSinglePostAndUser(t *testing.T) {
	var buf bytes.Buffer
	if err := Post(&buf, Table, &testPosts[0]); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "first post") {
		t.Errorf("single post output wrong:\n%s", buf.String())
	}
	buf.Reset()
	if err := User(&buf, JSON, &testUsers[0]); err != nil {
		t.Fatal(err)
	}
	if !json.Valid(buf.Bytes()) {
		t.Errorf("single user JSON invalid:\n%s", buf.String())
	}
}
