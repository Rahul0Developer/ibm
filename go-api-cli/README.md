# go-api-cli

A small Go command-line tool that calls the public **GitHub REST API** and prints
the results as a clean table or pretty JSON. It supports optional token
authentication via the `GITHUB_TOKEN` environment variable and has solid error
handling with unit tests (`httptest`).

## Features

- Two subcommands: `repos <user>` (list a user's repositories) and `user <login>` (fetch a profile)
- Flags parsed with the standard `flag` package: `-n` (repo count), `-format json|table`
- Output as an aligned table (`text/tabwriter`) or indented JSON (`encoding/json`)
- Optional authentication: reads `GITHUB_TOKEN` from the environment and sends it as a `Bearer` header
- Error handling for network failures, 404s, rate limits (401/403), and malformed JSON
- Unit tests using `net/http/httptest` — no live network required

## Requirements

- Go 1.19+

## Install / Build

```bash
cd go-api-cli
go build -o bin/api-cli ./cmd/api-cli
# or run directly without building:
go run ./cmd/api-cli help
```

## Usage

```
api-cli repos <user> [-n count] [-format json|table]
api-cli user  <login> [-format json|table]
```

Set an API token to raise GitHub's rate limit (optional):

```bash
export GITHUB_TOKEN=ghp_your_token_here
```

## Example Commands & Output

### Table output (default)

```bash
$ ./bin/api-cli repos golang -n 5
REPO             STARS   FORKS  LANGUAGE  UPDATED
golang/go        139287  22147  Go        2026-10-13
golang/net       3045    1493   Go        2026-10-13
golang/protobuf  10076   1571   Go        2026-10-13
golang/vuln      526     72     Go        2026-10-13
golang/mod       211     102    Go        2026-10-13
```

### JSON output

```bash
$ ./bin/api-cli user octocat -format json
{
  "login": "octocat",
  "name": "The Octocat",
  "bio": null,
  "followers": 24572,
  "following": 9,
  "public_repos": 8,
  "html_url": "https://github.com/octocat"
}
```

### Error handling

```bash
$ ./bin/api-cli repos this-user-does-not-exist-xyz
error: resource not found: /users/this-user-does-not-exist-xyz/repos?per_page=10&sort=updated
$ echo $?
1
```

## Tests

Run the test suite:

```bash
go test ./...
# with coverage:
go test -cover ./...
```

Test file: [`internal/github/client_test.go`](internal/github/client_test.go)

Covered cases:

- Successful repo listing + JSON decoding (including nullable fields)
- Empty-input validation
- 404 → "resource not found" error
- 403 → rate-limit/auth error message mentioning `GITHUB_TOKEN`
- `Authorization: Bearer <token>` header sent only when a token is set
- Malformed JSON → decode error

All tests use `httptest.NewServer`, so they are fast and deterministic.

## Project Layout

```
go-api-cli/
├── cmd/api-cli/          # main package: flag parsing + output formatting
│   └── main.go
├── internal/github/      # reusable API client package
│   ├── client.go
│   └── client_test.go
├── bin/                  # built binary (gitignored)
├── go.mod
└── README.md
```

## What I Learned / Design Decisions

- **Standard library only** — chose `flag` over Cobra to keep the module
  dependency-free; a tiny `splitFlagsAndArgs` helper lets flags appear after
  positional args (`repos golang -n 5`), which the stock `flag` package doesn't do.
- **Separation of concerns** — HTTP/JSON logic lives in `internal/github`
  (structs `Repo`/`User`, methods on `Client`); `main.go` only parses args and
  renders output. This made the client easy to unit-test.
- **Testability** — `BaseURL` and `HTTPClient` are exported fields so tests can
  point the client at an `httptest` server instead of hitting the real API.
- **Nullable JSON fields** — GitHub returns `null` for things like
  `description`/`language`; using `*string` avoids scanning `null` into a plain
  `string` (a classic `json.Unmarshal` pitfall).
- **Error wrapping** — used `fmt.Errorf("...: %w", err)` throughout so callers
  get context while preserving the underlying error for `errors.Is/As`.
- **Auth via env var** — the API key is never a CLI flag (flags leak into shell
  history / `ps`); `GITHUB_TOKEN` is read once in `newClient()`.
- **Status-code mapping** — 404, 401/403 (rate limit) and other non-2xx
  responses produce distinct, actionable messages instead of raw bodies.
