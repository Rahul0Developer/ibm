// Package github provides a minimal client for the public GitHub REST API.
package github

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Repo represents the subset of fields we care about from a GitHub repository.
type Repo struct {
	Name        string  `json:"name"`
	FullName    string  `json:"full_name"`
	Description *string `json:"description"`
	Stars       int     `json:"stargazers_count"`
	Forks       int     `json:"forks_count"`
	Language    *string `json:"language"`
	URL         string  `json:"html_url"`
	UpdatedAt   string  `json:"updated_at"`
	Archived    bool    `json:"archived"`
}

// User represents the subset of fields we care about from a GitHub user.
type User struct {
	Login       string  `json:"login"`
	Name        *string `json:"name"`
	Bio         *string `json:"bio"`
	Followers   int     `json:"followers"`
	Following   int     `json:"following"`
	PublicRepos int     `json:"public_repos"`
	URL         string  `json:"html_url"`
}

// Client calls the GitHub REST API. It optionally authenticates with a token.
type Client struct {
	BaseURL    string // overridable for tests
	APIToken   string // read from GITHUB_TOKEN env var by the caller
	HTTPClient *http.Client
}

// New returns a Client with sane defaults.
func New(apiToken string) *Client {
	return &Client{
		BaseURL:    "https://api.github.com",
		APIToken:   apiToken,
		HTTPClient: &http.Client{Timeout: 10 * time.Second},
	}
}

// do performs the HTTP request, applies auth headers and maps non-2xx
// responses to descriptive errors.
func (c *Client) do(path string) ([]byte, error) {
	url := strings.TrimRight(c.BaseURL, "/") + path

	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("building request: %w", err)
	}
	req.Header.Set("Accept", "application/vnd.github+json")
	req.Header.Set("User-Agent", "go-api-cli")
	// Simple authentication: bearer token via env var (optional).
	if c.APIToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.APIToken)
	}

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("calling GitHub API: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response body: %w", err)
	}

	switch {
	case resp.StatusCode == http.StatusNotFound:
		return nil, fmt.Errorf("resource not found: %s", path)
	case resp.StatusCode == http.StatusForbidden || resp.StatusCode == http.StatusUnauthorized:
		return nil, fmt.Errorf("auth/rate-limit error (HTTP %d): check your GITHUB_TOKEN", resp.StatusCode)
	case resp.StatusCode < 200 || resp.StatusCode >= 300:
		return nil, fmt.Errorf("unexpected status %d from %s", resp.StatusCode, path)
	}
	return body, nil
}

// ListUserRepos returns up to perPage repositories for the given user.
func (c *Client) ListUserRepos(user string, perPage int) ([]Repo, error) {
	if strings.TrimSpace(user) == "" {
		return nil, fmt.Errorf("user must not be empty")
	}
	if perPage <= 0 || perPage > 100 {
		perPage = 30
	}
	path := fmt.Sprintf("/users/%s/repos?per_page=%d&sort=updated", user, perPage)
	body, err := c.do(path)
	if err != nil {
		return nil, err
	}
	var repos []Repo
	if err := json.Unmarshal(body, &repos); err != nil {
		return nil, fmt.Errorf("decoding repo list: %w", err)
	}
	return repos, nil
}

// GetUser fetches a single GitHub user profile.
func (c *Client) GetUser(login string) (*User, error) {
	if strings.TrimSpace(login) == "" {
		return nil, fmt.Errorf("login must not be empty")
	}
	body, err := c.do("/users/" + login)
	if err != nil {
		return nil, err
	}
	var u User
	if err := json.Unmarshal(body, &u); err != nil {
		return nil, fmt.Errorf("decoding user: %w", err)
	}
	return &u, nil
}
