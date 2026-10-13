package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"text/tabwriter"

	"github.com/example/go-api-cli/internal/github"
)

const usage = `api-cli - a small Go CLI that calls the GitHub REST API.

Usage:
  api-cli repos <user> [-n count] [-format json|table]
  api-cli user  <login> [-format json|table]

Flags:
  -format string   output format: "table" (default) or "json"
  -n int           number of repositories to fetch (1-100, default 10)

Authentication:
  Set GITHUB_TOKEN in your environment to raise the API rate limit.
`

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		fmt.Fprint(os.Stderr, usage)
		return fmt.Errorf("no subcommand given")
	}

	sub := args[0]

	fs := flag.NewFlagSet(sub, flag.ExitOnError)
	format := fs.String("format", "table", `output format: "table" or "json"`)
	n := fs.Int("n", 10, "number of repositories (1-100)")
	// Allow flags to appear after the positional argument, e.g.
	//   api-cli repos golang -n 5
	flagArgs, positional := splitFlagsAndArgs(args[1:])
	if err := fs.Parse(flagArgs); err != nil {
		return err
	}

	switch sub {
	case "repos":
		user := ""
		if len(positional) > 0 {
			user = positional[0]
		}
		if user == "" {
			return fmt.Errorf("usage: api-cli repos <user> [-n count] [-format ...]")
		}
		client := newClient()
		repos, err := client.ListUserRepos(user, *n)
		if err != nil {
			return err
		}
		return printRepos(repos, *format)

	case "user":
		login := ""
		if len(positional) > 0 {
			login = positional[0]
		}
		if login == "" {
			return fmt.Errorf("usage: api-cli user <login> [-format ...]")
		}
		client := newClient()
		u, err := client.GetUser(login)
		if err != nil {
			return err
		}
		return printUser(u, *format)

	case "-h", "--help", "help":
		fmt.Print(usage)
		return nil

	default:
		fmt.Fprint(os.Stderr, usage)
		return fmt.Errorf("unknown subcommand %q", sub)
	}
}

// splitFlagsAndArgs separates flag tokens (and their values) from positional
// arguments so that users may write either `repos -n 5 golang` or
// `repos golang -n 5`.
func splitFlagsAndArgs(args []string) (flags, positional []string) {
	for i := 0; i < len(args); i++ {
		a := args[i]
		if !strings.HasPrefix(a, "-") || a == "-" {
			positional = append(positional, a)
			continue
		}
		flags = append(flags, a)
		// `-name value` style flags consume the next token (but not `-name=value`).
		if !strings.Contains(a, "=") && i+1 < len(args) && !strings.HasPrefix(args[i+1], "-") {
			i++
			flags = append(flags, args[i])
		}
	}
	return flags, positional
}

// newClient reads the optional API key from the environment.
func newClient() *github.Client {
	return github.New(strings.TrimSpace(os.Getenv("GITHUB_TOKEN")))
}

func printRepos(repos []github.Repo, format string) error {
	if format == "json" {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(repos)
	}
	if len(repos) == 0 {
		fmt.Println("no repositories found")
		return nil
	}
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "REPO\tSTARS\tFORKS\tLANGUAGE\tUPDATED")
	for _, r := range repos {
		lang := valueOr(r.Language, "-")
		updated := shortDate(r.UpdatedAt)
		fmt.Fprintf(w, "%s\t%d\t%d\t%s\t%s\n", r.FullName, r.Stars, r.Forks, lang, updated)
	}
	return w.Flush()
}

func printUser(u *github.User, format string) error {
	if format == "json" {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(u)
	}
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintf(w, "LOGIN\t%s\n", u.Login)
	fmt.Fprintf(w, "NAME\t%s\n", valueOr(u.Name, "-"))
	fmt.Fprintf(w, "BIO\t%s\n", valueOr(u.Bio, "-"))
	fmt.Fprintf(w, "FOLLOWERS\t%d\n", u.Followers)
	fmt.Fprintf(w, "FOLLOWING\t%d\n", u.Following)
	fmt.Fprintf(w, "PUBLIC_REPOS\t%d\n", u.PublicRepos)
	fmt.Fprintf(w, "URL\t%s\n", u.URL)
	return w.Flush()
}

func valueOr(s *string, def string) string {
	if s == nil || *s == "" {
		return def
	}
	return *s
}

// shortDate trims an ISO timestamp like 2024-05-01T12:34:56Z to 2024-05-01.
func shortDate(iso string) string {
	if i := strings.IndexByte(iso, 'T'); i > 0 {
		return iso[:i]
	}
	return iso
}
