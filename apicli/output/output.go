// Package output renders API results as human-friendly tables or raw JSON.
package output

import (
	"encoding/json"
	"fmt"
	"io"
	"text/tabwriter"

	"github.com/yourusername/apicli/models"
)

// Format identifies a supported output format.
type Format string

const (
	Table Format = "table"
	JSON  Format = "json"
)

// ParseFormat validates a user-supplied --format value.
func ParseFormat(s string) (Format, error) {
	switch Format(s) {
	case Table, JSON:
		return Format(s), nil
	default:
		return "", fmt.Errorf("unknown format %q: use \"table\" or \"json\"", s)
	}
}

// writeJSON pretty-prints any value as indented JSON.
func writeJSON(w io.Writer, v interface{}) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to encode JSON: %w", err)
	}
	_, err = fmt.Fprintf(w, "%s\n", data)
	return err
}

// Posts prints a list of posts in the requested format.
func Posts(w io.Writer, format Format, posts []models.Post) error {
	if format == JSON {
		return writeJSON(w, posts)
	}
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "ID\tUSER\tTITLE")
	for _, p := range posts {
		fmt.Fprintf(tw, "%d\t%d\t%s\n", p.ID, p.UserID, p.Title)
	}
	return tw.Flush()
}

// Post prints a single post in the requested format.
func Post(w io.Writer, format Format, p *models.Post) error {
	if format == JSON {
		return writeJSON(w, p)
	}
	_, err := fmt.Fprintf(w, "ID:     %d\nUser:   %d\nTitle:  %s\nBody:   %s\n",
		p.ID, p.UserID, p.Title, p.Body)
	return err
}

// Users prints a list of users in the requested format.
func Users(w io.Writer, format Format, users []models.User) error {
	if format == JSON {
		return writeJSON(w, users)
	}
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "ID\tNAME\tUSERNAME\tEMAIL")
	for _, u := range users {
		fmt.Fprintf(tw, "%d\t%s\t%s\t%s\n", u.ID, u.Name, u.Username, u.Email)
	}
	return tw.Flush()
}

// User prints a single user in the requested format.
func User(w io.Writer, format Format, u *models.User) error {
	if format == JSON {
		return writeJSON(w, u)
	}
	_, err := fmt.Fprintf(w, "ID:       %d\nName:     %s\nUsername: %s\nEmail:    %s\n",
		u.ID, u.Name, u.Username, u.Email)
	return err
}
