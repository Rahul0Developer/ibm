// apicli is a small command-line tool that turns the JSONPlaceholder REST
// API into simple, memorable commands.
//
// Usage:
//
//	apicli posts list [--format table|json] [-n N]
//	apicli posts get <id> [--format table|json]
//	apicli posts create --title "..." --body "..." [--user N] [--dry-run]
//	apicli posts delete <id> [--dry-run]
//	apicli users list [--format table|json]
//	apicli users get <id> [--format table|json]
//	apicli version | help
package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/yourusername/apicli/api"
	"github.com/yourusername/apicli/models"
	"github.com/yourusername/apicli/output"
)

// version is the CLI version string.
const version = "v0.1.0"

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

// run parses arguments and dispatches the command. It takes writers so it
// can be tested without touching the real stdout/stderr.
func run(args []string, stdout io.Writer, stderr io.Writer) error {
	if len(args) == 0 {
		printHelp(stderr)
		return fmt.Errorf("no command given")
	}

	client := api.NewClient(os.Getenv("GOAPI_BASE_URL"), os.Getenv("GOAPI_TOKEN"))

	switch args[0] {
	case "posts":
		return handlePosts(client, args[1:], stdout)
	case "users":
		return handleUsers(client, args[1:], stdout)
	case "version", "--version":
		fmt.Fprintf(stdout, "apicli %s\n", version)
		return nil
	case "help", "--help", "-h":
		printHelp(stdout)
		return nil
	default:
		printHelp(stderr)
		return fmt.Errorf("unknown command: %s", args[0])
	}
}

// devNull discards writes (used to silence the flag package's own usage text;
// we print help ourselves when needed).
type devNull struct{}

func (devNull) Write(p []byte) (int, error) { return len(p), nil }

// newFlagSetForTest builds a flag set with the common flags, for unit tests
// of parseArgs.
func newFlagSetForTest() *flag.FlagSet {
	set := flag.NewFlagSet("test", flag.ContinueOnError)
	set.SetOutput(devNull{})
	set.String("format", "table", `output format: "table" or "json"`)
	set.Int("n", 0, "limit number of results")
	return set
}

func handlePosts(client *api.Client, args []string, stdout io.Writer) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: apicli posts [list|get <id>|create|delete <id>]")
	}
	sub := args[0]
	rest := args[1:]

	set := flag.NewFlagSet("posts "+sub, flag.ContinueOnError)
	set.SetOutput(devNull{})
	formatFlag := set.String("format", "table", `output format: "table" or "json"`)
	count := set.Int("n", 0, "limit number of results (0 = all)")
	title := set.String("title", "", "post title (create)")
	body := set.String("body", "", "post body (create)")
	userID := set.Int("user", 1, "user id (create)")
	dryRun := set.Bool("dry-run", false, "print what would be sent without calling the API")

	positional, err := parseArgs(set, rest)
	if err != nil {
		return err
	}
	format, err := output.ParseFormat(*formatFlag)
	if err != nil {
		return err
	}

	switch sub {
	case "list":
		posts, err := client.GetPosts()
		if err != nil {
			return err
		}
		if *count > 0 && *count < len(posts) {
			posts = posts[:*count]
		}
		return output.Posts(stdout, format, posts)

	case "get":
		if len(positional) < 1 {
			return fmt.Errorf("usage: apicli posts get <id>")
		}
		post, err := client.GetPost(positional[0])
		if err != nil {
			return err
		}
		return output.Post(stdout, format, post)

	case "create":
		if *title == "" || *body == "" {
			return fmt.Errorf("create requires --title and --body")
		}
		np := models.NewPost{Title: *title, Body: *body, UserID: *userID}
		if *dryRun {
			fmt.Fprintf(stdout, "[dry-run] would POST /posts: %+v\n", np)
			return nil
		}
		created, err := client.CreatePost(np)
		if err != nil {
			return err
		}
		return output.Post(stdout, format, created)

	case "delete":
		if len(positional) < 1 {
			return fmt.Errorf("usage: apicli posts delete <id>")
		}
		if *dryRun {
			fmt.Fprintf(stdout, "[dry-run] would DELETE /posts/%s\n", positional[0])
			return nil
		}
		if err := client.DeletePost(positional[0]); err != nil {
			return err
		}
		fmt.Fprintf(stdout, "Deleted post %s\n", positional[0])
		return nil

	default:
		return fmt.Errorf("unknown posts subcommand: %q (want list|get|create|delete)", sub)
	}
}

func handleUsers(client *api.Client, args []string, stdout io.Writer) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: apicli users [list|get <id>]")
	}
	sub := args[0]

	set := flag.NewFlagSet("users "+sub, flag.ContinueOnError)
	set.SetOutput(devNull{})
	formatFlag := set.String("format", "table", `output format: "table" or "json"`)
	count := set.Int("n", 0, "limit number of results (0 = all)")

	positional, err := parseArgs(set, args[1:])
	if err != nil {
		return err
	}
	format, err := output.ParseFormat(*formatFlag)
	if err != nil {
		return err
	}

	switch sub {
	case "list":
		users, err := client.GetUsers()
		if err != nil {
			return err
		}
		if *count > 0 && *count < len(users) {
			users = users[:*count]
		}
		return output.Users(stdout, format, users)

	case "get":
		if len(positional) < 1 {
			return fmt.Errorf("usage: apicli users get <id>")
		}
		user, err := client.GetUser(positional[0])
		if err != nil {
			return err
		}
		return output.User(stdout, format, user)

	default:
		return fmt.Errorf("unknown users subcommand: %q (want list|get)", sub)
	}
}

// parseArgs interleaves flags and positional arguments: it pulls positional
// args out first, then parses the remaining flags. This lets users write
// either `posts get 1 --format json` or `posts get --format json 1`.
func parseArgs(set *flag.FlagSet, args []string) ([]string, error) {
	var positional []string
	var flagArgs []string
	for i := 0; i < len(args); i++ {
		a := args[i]
		if len(a) > 1 && a[0] == '-' {
			flagArgs = append(flagArgs, a)
			// consume a value for non-boolean flags like -format json / -n 5
			if !isBoolFlag(set, a) && i+1 < len(args) && len(args[i+1]) > 0 && args[i+1][0] != '-' {
				i++
				flagArgs = append(flagArgs, args[i])
			}
			continue
		}
		positional = append(positional, a)
	}
	if err := set.Parse(flagArgs); err != nil {
		return nil, err
	}
	return positional, nil
}

func isBoolFlag(set *flag.FlagSet, arg string) bool {
	name := arg
	for _, c := range []string{"--", "-"} {
		if len(name) >= len(c) && name[:len(c)] == c {
			name = name[len(c):]
			break
		}
	}
	if f := set.Lookup(name); f != nil {
		_, ok := f.Value.(interface{ IsBoolFlag() bool })
		return ok
	}
	return false
}

func printHelp(w io.Writer) {
	fmt.Fprint(w, `apicli `+version+` - a simple CLI for the JSONPlaceholder REST API

Usage:
  apicli posts list                 [--format table|json] [-n N]
  apicli posts get <id>             [--format table|json]
  apicli posts create --title T --body B [--user N] [--dry-run]
  apicli posts delete <id>          [--dry-run]
  apicli users list                 [--format table|json] [-n N]
  apicli users get <id>             [--format table|json]
  apicli version
  apicli help

Environment:
  GOAPI_BASE_URL   override the API base URL (default: `+api.DefaultBaseURL+`)
  GOAPI_TOKEN      optional bearer token sent as Authorization header
`)
}
