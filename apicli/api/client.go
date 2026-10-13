// Package api contains a small HTTP client for the JSONPlaceholder REST API.
package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/yourusername/apicli/models"
)

// DefaultBaseURL is the public API endpoint used when nothing else is configured.
const DefaultBaseURL = "https://jsonplaceholder.typicode.com"

// Client is a simple HTTP client for the API.
type Client struct {
	baseURL    string
	httpClient *http.Client
	token      string // optional bearer token, sent when present
}

// NewClient creates a new API client.
// baseURL may be empty to use DefaultBaseURL; token may be empty for no auth.
func NewClient(baseURL, token string) *Client {
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	return &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{
			Timeout: 10 * time.Second,
		},
		token: token,
	}
}

// do performs a request against the given path and decodes the JSON response
// into out (when out is non-nil). Errors are wrapped with context.
func (c *Client) do(method, path string, body io.Reader, out interface{}) error {
	req, err := http.NewRequest(method, c.baseURL+path, body)
	if err != nil {
		return fmt.Errorf("failed to build request: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read response body: %w", err)
	}

	switch {
	case resp.StatusCode == http.StatusNotFound:
		return fmt.Errorf("resource not found (404)")
	case resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden:
		return fmt.Errorf("authentication failed (%s): check your GOAPI_TOKEN", resp.Status)
	case resp.StatusCode < 200 || resp.StatusCode >= 300:
		return fmt.Errorf("unexpected status: %s", resp.Status)
	}

	if out != nil {
		if err := json.Unmarshal(data, out); err != nil {
			return fmt.Errorf("failed to decode JSON response: %w", err)
		}
	}
	return nil
}

// validateID makes sure an ID argument is a positive integer.
func validateID(id string) error {
	n, err := strconv.Atoi(id)
	if err != nil || n <= 0 {
		return fmt.Errorf("invalid id %q: must be a positive number", id)
	}
	return nil
}

// GetPosts fetches all posts.
func (c *Client) GetPosts() ([]models.Post, error) {
	var posts []models.Post
	if err := c.do(http.MethodGet, "/posts", nil, &posts); err != nil {
		return nil, fmt.Errorf("failed to fetch posts: %w", err)
	}
	return posts, nil
}

// GetPost fetches a single post by ID.
func (c *Client) GetPost(id string) (*models.Post, error) {
	if err := validateID(id); err != nil {
		return nil, err
	}
	var post models.Post
	if err := c.do(http.MethodGet, "/posts/"+id, nil, &post); err != nil {
		return nil, fmt.Errorf("failed to fetch post %s: %w", id, err)
	}
	return &post, nil
}

// CreatePost creates a new post and returns the server's response.
func (c *Client) CreatePost(np models.NewPost) (*models.Post, error) {
	payload, err := json.Marshal(np)
	if err != nil {
		return nil, fmt.Errorf("failed to encode post: %w", err)
	}
	var created models.Post
	if err := c.do(http.MethodPost, "/posts", bytes.NewReader(payload), &created); err != nil {
		return nil, fmt.Errorf("failed to create post: %w", err)
	}
	return &created, nil
}

// DeletePost deletes a post by ID.
func (c *Client) DeletePost(id string) error {
	if err := validateID(id); err != nil {
		return err
	}
	if err := c.do(http.MethodDelete, "/posts/"+id, nil, nil); err != nil {
		return fmt.Errorf("failed to delete post %s: %w", id, err)
	}
	return nil
}

// GetUsers fetches all users.
func (c *Client) GetUsers() ([]models.User, error) {
	var users []models.User
	if err := c.do(http.MethodGet, "/users", nil, &users); err != nil {
		return nil, fmt.Errorf("failed to fetch users: %w", err)
	}
	return users, nil
}

// GetUser fetches a single user by ID.
func (c *Client) GetUser(id string) (*models.User, error) {
	if err := validateID(id); err != nil {
		return nil, err
	}
	var user models.User
	if err := c.do(http.MethodGet, "/users/"+id, nil, &user); err != nil {
		return nil, fmt.Errorf("failed to fetch user %s: %w", id, err)
	}
	return &user, nil
}
