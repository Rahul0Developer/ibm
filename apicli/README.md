# apicli

A simple Go CLI tool that turns the [JSONPlaceholder](https://jsonplaceholder.typicode.com) REST API into intuitive, memorable commands — no API key required. Built with the Go standard library only (`net/http`, `encoding/json`, `flag`, `text/tabwriter`, `testing`).

## Features

- `posts list` / `posts get <id>` / `posts create` / `posts delete` commands
- `users list` / `users get <id>` commands
- Two output formats: aligned **table** (default) or pretty **JSON** (`--format json`)
- `-n N` result limiting for list commands
- `--dry-run` safety mode for create/delete
- Input validation (IDs must be positive numbers), friendly error messages, 10s HTTP timeout
- Optional auth via `GOAPI_TOKEN` env var (sent as `Authorization: Bearer …`) and configurable base URL via `GOAPI_BASE_URL`
- Unit tests with `httptest` (no live network needed in CI)

## Requirements

- Go 1.19+ (developed/tested on Go 1.19; nothing newer is required)

## Install & Run

```bash
git clone <this-repo-url>
cd apicli
go build -o apicli .

./apicli posts list
./apicli posts get 1
./apicli users get 3
./apicli --help
```

Or without building: `go run . posts list`

## Example commands & output

```console
$ ./apicli posts list -n 5
ID  USER  TITLE
1   1     sunt aut facere repellat provident occaecati excepturi optio reprehenderit
2   1     qui est esse
3   1     ea molestias quasi exercitationem repellat qui ipsa sit aut
4   1     eum et est occaecati
5   1     nesciunt quas odio

$ ./apicli posts get 1
ID:     1
User:   1
Title:  sunt aut facere repellat provident occaecati excepturi optio reprehenderit
Body:   quia et suscipit
suscipit recusandae consequuntur expedita et cum
...

$ ./apicli users get 3 --format json
{
  "id": 3,
  "name": "Clementine Bauch",
  "username": "Samantha",
  "email": "Nathan@yesenia.net"
}

$ ./apicli posts create --title "Hello" --body "World" --dry-run
[dry-run] would POST /posts: {Title:Hello Body:World UserID:1}

$ ./apicli frobnicate
Error: unknown command: frobnicate   (exit code 1, help printed to stderr)
```

## Configuration (environment variables)

| Variable         | Purpose                                             |
|------------------|-----------------------------------------------------|
| `GOAPI_BASE_URL` | Override the API base URL (useful for tests/mocks)  |
| `GOAPI_TOKEN`    | Optional bearer token sent in the `Authorization` header |

## Tests

Run everything:

```bash
go test ./... -cover
```

Current coverage: `api` ≈ 84%, `output` ≈ 87%, `main` (CLI wiring) ≈ 69%.

Test files:
- [`api/client_test.go`](api/client_test.go) — HTTP client: fetch/list/create/delete, 404 handling, invalid IDs, bad JSON, token auth (all against an in-process `httptest` server)
- [`output/output_test.go`](output/output_test.go) — table/JSON rendering, format validation, JSON round-trip
- [`main_test.go`](main_test.go) — CLI argument parsing, flags after positional args, dry-run, error paths

## Project structure

```
apicli/
├── go.mod              # module definition (stdlib only, zero dependencies)
├── main.go             # CLI entry point: flag parsing + command dispatch
├── main_test.go        # CLI-level tests
├── api/
│   ├── client.go       # net/http client for the REST API
│   └── client_test.go  # unit tests with httptest
├── models/
│   └── models.go       # Post / User / NewPost structs + JSON tags
├── output/
│   ├── output.go       # table (tabwriter) and JSON renderers
│   └── output_test.go
└── README.md
```

## Design decisions & what I learned

- **Standard library only.** No cobra/viper — the `flag` package plus a small hand-written `parseArgs` helper was enough and keeps the binary dependency-free. The project maps cleanly onto the same patterns real CLIs use.
- **Flags after positional args.** Go's `flag` package stops parsing at the first non-flag argument, so `posts get 1 --format json` wouldn't work out of the box. I wrote a ~30-line interleaving parser (`parseArgs`) that separates positional args from flags before parsing — both orders now work. This is one reason real CLIs adopt cobra; it's a good example of understanding *why* those libraries exist.
- **Separate packages by responsibility** (`api` = transport, `models` = data shapes, `output` = presentation, `main` = command wiring). Each is independently testable, and swapping JSONPlaceholder for another REST API would only touch `api/`.
- **One shared `do()` method** in the client handles headers, auth, timeouts, status-code checks, and JSON decoding, so every endpoint method is three lines of business logic. Errors are wrapped with `%w` and mapped to human-friendly messages (404 → "resource not found", 401/403 → "check your GOAPI_TOKEN").
- **Testability drove the design**: `run(args, stdout, stderr)` takes `io.Writer`s instead of printing directly, and the base URL is injectable via `GOAPI_BASE_URL` — so all tests run against `httptest.NewServer` with zero live network calls.
- **Input validation at the boundary** (IDs must be positive integers) prevents malformed URLs like `/posts/abc`.
- **Dry-run mode** for mutating commands (create/delete) is a small taste of the safety UX that production developer CLIs (e.g. Confluent CLI, Terraform plan) invest heavily in.

## Possible next steps

- Migrate command wiring to `cobra` (the industry standard for Go CLIs)
- Add `comments` resource and pagination for list commands
- Simple retry-with-backoff for transient network errors
- YAML output format and colorized tables
