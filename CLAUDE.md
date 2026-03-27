# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`aivo` is a Rust CLI tool that provides unified access to multiple AI coding assistants (Claude, Codex, Gemini, OpenCode, Pi) with local API key management and secure storage. Supports OpenAI-compatible providers (Cloudflare Workers AI, Moonshot, DeepSeek), GitHub Copilot, OpenRouter, Ollama (local models), and native APIs.

> [!IMPORTANT]
> **Rebuild before testing**: After code changes, always run `cargo build --release && cargo install --path .` before testing the binary. Never test a stale build.

## Build & Test

```bash
cargo build --release   # Compile optimized binary to target/release/aivo
cargo test --features test-fast-crypto  # Run all tests (~1900 tests, fast crypto for CI/dev)
cargo test -- test_name                 # Run a single test by name
cargo clippy            # Lint (fix all warnings before committing)
cargo fmt               # Format code (run before committing)
```

The `test-fast-crypto` feature uses reduced PBKDF2 iterations for faster test runs. Tests also work without it (`cargo test`), just slower.

A `Makefile` wraps common workflows: `make test`, `make build`, `make clippy`, `make install`, `make release`.

## Git Conventions

* Always squash merge to main. Never fast-forward. Command: `git merge --squash <branch> && git commit`
* Do not commit automatically to the fix.

## Release Process

1. Optionally add release notes to `CHANGELOG.md` under a `## vX.Y.Z` heading (auto-generated from commits if omitted).
2. Bump version in both `Cargo.toml` and `npm/package.json` **first** — never tag without updating the version.
3. Run `cargo fmt`, `cargo clippy -- -D warnings`, and `cargo test`.
4. `cargo build --release && cargo install --path .` to verify the binary.
5. Commit: `git add -A && git commit -m "chore: release vX.Y.Z"`
6. Tag and push: `git tag vX.Y.Z && git push origin main --tags`

## CLI / UX Conventions

> [!NOTE]
> When formatting CLI help text, pay close attention to alignment, spacing, bracket style, and description consistency. Match existing patterns exactly rather than inventing new formatting.

When implementing interactive UI (pickers, prompts, formatted output), verify before presenting as done:
* **Keyboard handling**: arrow keys, Ctrl+P/N navigation, ESC to cancel (restore terminal state), Ctrl+C cleanup
* **Selection state**: pre-select the currently active item when editing existing values
* **Alignment**: consistent padding and column alignment with existing UI in the codebase
* **Edge cases**: empty input, single item, long strings that could break alignment

## Code Review

Be concise and deliver findings quickly. Focus on the diff and immediate context only — do not explore the entire codebase first.

## Architecture

`src/main.rs` initializes all services via dependency injection:

```
SessionStore → EnvironmentInjector → AILauncher
                     ↓
             Command Handlers
```

### Source Layout

#### `src/`

| File               | Purpose                                               |
| ------------------ | ----------------------------------------------------- |
| `main.rs`          | Entry point, dependency injection, command dispatch   |
| `lib.rs`           | Library root re-exporting public modules              |
| `cli.rs`           | Argument parsing with clap                            |
| `constants.rs`     | Application-wide constants (placeholder URL, content types) |
| `errors.rs`        | Error classification, exit codes (0/1/2/3), suggestions |
| `key_resolution.rs`| API key resolution logic (selection, cancellation, missing auth) |
| `style.rs`         | Terminal styling with console crate                   |
| `tui.rs`           | Custom TUI components (FuzzySelect)                   |
| `version.rs`       | Version constant from `CARGO_PKG_VERSION`             |

#### `src/commands/`

| File                     | Purpose                                                    |
| ------------------------ | ---------------------------------------------------------- |
| `run.rs`                 | Launch AI tools (claude, codex, gemini, opencode, pi); falls back to `start` flow when no tool given |
| `start.rs`               | Interactive remembered-start flow (key + tool + model picker) |
| `chat.rs`                | Interactive chat command; routes to TUI or one-shot mode   |
| `chat_tui.rs`            | Full-screen interactive chat TUI entry point (ratatui + crossterm) |
| `chat_tui/`              | Modular chat TUI components (app state, event loop, input, rendering, sessions, storage) |
| `chat_tui_format.rs`     | Display formatting for chat TUI (elapsed time, token counts) |
| `chat_request_builder.rs`| HTTP request body construction for OpenAI/Anthropic chat APIs |
| `chat_response_parser.rs`| SSE chunk parsing, usage extraction, response format handling |
| `alias.rs`               | Model alias management (short names → full model names)    |
| `info.rs`                | System info and health check (keys, tools, directory state; `--ping` for key pinging) |
| `keys.rs`                | API key management (add, rm, use, edit, cat, list)         |
| `logs.rs`                | Query local SQLite logs for chat, run, and serve activity  |
| `models.rs`              | List available models from active provider (1h cache)      |
| `serve.rs`               | Local OpenAI-compatible API server                         |
| `stats.rs`               | Usage statistics display (aivo chat + global tool stats)   |
| `update.rs`              | Self-update via GitHub Releases                            |

#### `src/services/`

| File                            | Purpose                                                                 |
| ------------------------------- | ----------------------------------------------------------------------- |
| `session_store.rs`              | Top-level session store facade coordinating extracted sub-stores        |
| `session_crypto.rs`             | AES-256-GCM encryption/decryption with PBKDF2 key derivation           |
| `api_key_store.rs`              | API key CRUD operations with encryption/decryption                      |
| `chat_session_store.rs`         | Chat session persistence and titling                                    |
| `directory_starts.rs`           | Per-directory remembered key + tool selections with stale detection     |
| `usage_stats_store.rs`          | Usage statistics persistence with file locking                          |
| `log_store.rs`                  | SQLite-backed event log (WAL mode) for chat, run, and serve activity    |
| `ai_launcher.rs`                | Process spawning, signal forwarding (SIGINT/SIGTERM), stdio passthrough |
| `environment_injector.rs`       | Tool-specific env var configuration, placeholder URL + router flag injection |
| `provider_protocol.rs`          | Protocol detection from base URL                                        |
| `protocol_fallback.rs`          | Multi-protocol fallback strategy with attempt tracking                  |
| `provider_profile.rs`           | Provider kind classification (Copilot, Ollama, OpenRouter, etc.) and model listing flags |
| `known_providers.rs`            | Registry of known AI provider names and base URLs                       |
| `model_names.rs`                | Model name transformations (e.g. `claude-sonnet-4-6` → `anthropic/claude-sonnet-4.6`) |
| `codex_model_map.rs`            | Model name mapping for Codex CLI compatibility                          |
| `openai_models.rs`              | OpenAI chat request/response data structures                            |
| `anthropic_router.rs`           | Proxy for Claude + OpenRouter                                           |
| `anthropic_to_openai_router.rs` | Proxy for Anthropic-format clients + OpenAI-compatible providers        |
| `copilot_router.rs`             | Proxy for Claude/Codex/Gemini/Pi + GitHub Copilot                       |
| `copilot_auth.rs`               | GitHub Copilot OAuth device flow and token refresh                      |
| `responses_to_chat_router.rs`   | Proxy for Responses API clients + non-OpenAI providers (Responses API → Chat Completions) |
| `gemini_router.rs`              | Proxy for Gemini + non-Google providers (Gemini format → Chat Completions) |
| `serve_router.rs`               | Shared router server scaffolding                                        |
| `serve_upstream.rs`             | Upstream request forwarding with protocol routing                       |
| `serve_responses.rs`            | OpenAI → Responses API format conversion                                |
| `serve_stream_converters.rs`    | Stream format translation between providers during proxying             |
| `http_utils.rs`                 | Shared HTTP utilities (request parsing, header extraction, SSE)         |
| `request_log.rs`                | Async JSONL request logger for serve mode (timestamp, path, model, status, latency) |
| `openai_anthropic_bridge.rs`    | Anthropic Messages ↔ OpenAI Chat Completions conversion                 |
| `openai_gemini_bridge.rs`       | Gemini native ↔ OpenAI Chat Completions conversion                      |
| `anthropic_route_pipeline.rs`   | Shared pipeline for Anthropic-format router requests                    |
| `anthropic_chat_request.rs`     | Anthropic chat request types                                            |
| `anthropic_chat_response.rs`    | Anthropic chat response types                                           |
| `models_cache.rs`               | 1-hour file-backed cache for model lists                                |
| `ollama.rs`                     | Ollama lifecycle management (detect, auto-start, model pull)            |
| `path_search.rs`                | PATH scanning to find executables with platform-specific extensions     |
| `system_env.rs`                 | System environment helpers (CWD, home dir, etc.)                        |
| `launch_runtime.rs`             | Router startup, temp dir writing (Pi agent dir), runtime env patching   |
| `global_stats.rs`               | Cross-tool stats aggregation (Claude/Codex/Gemini/OpenCode/Pi) with per-file caching |
| `launch_args.rs`                | CLI arg injection (model flags, teammate mode, codex/pi model prefixing)|

### Cross-Platform

Platform-specific code is gated behind `cfg(unix)` / `cfg(windows)`. Unix uses `libc` for signal handling; Windows uses `windows-sys` for file locking. Ensure new platform-specific code is similarly gated.

### Data Model

`ApiKey`: `id`, `name`, `base_url`, `key`, `created_at`. Stored AES-256-GCM encrypted in `~/.config/aivo/config.json`. The sentinel `base_url` values `"copilot"` and `"ollama"` identify GitHub Copilot and local Ollama keys respectively.

### Exit Codes

| Code | Meaning    |
| ---- | ---------- |
| `0`  | Success    |
| `1`  | User error |
| `2`  | Network    |
| `3`  | Auth       |
