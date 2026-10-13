// Package models defines the data structures returned by the JSONPlaceholder API.
package models

// Post represents a post from the API.
type Post struct {
	UserID int    `json:"userId"`
	ID     int    `json:"id"`
	Title  string `json:"title"`
	Body   string `json:"body"`
}

// User represents a user from the API.
type User struct {
	ID       int    `json:"id"`
	Name     string `json:"name"`
	Username string `json:"username"`
	Email    string `json:"email"`
}

// NewPost is the payload used when creating a post.
type NewPost struct {
	Title  string `json:"title"`
	Body   string `json:"body"`
	UserID int    `json:"userId"`
}
