# CLAUDE.md

## Project Overview

`aivo` is a Rust CLI tool that provides unified access to multiple AI coding assistants (Claude, Codex, Gemini) with local API key management and secure storage. Supports OpenAI-compatible providers (Cloudflare Workers AI, Moonshot, DeepSeek), GitHub Copilot, OpenRouter, and native APIs.

> [!IMPORTANT]
> **Rebuild before testing**: After code changes, always run `cargo build --release && cargo install --path .` before testing the binary. Never test a stale build.

## Build & Test

```bash
cargo build --release   # Compile optimized binary to target/release/aivo
cargo test              # Run all tests (~400 tests)
cargo clippy            # Lint (fix all warnings before committing)
cargo fmt               # Format code (run before committing)
```

## Git Conventions

* Always squash merge to main. Never fast-forward. Command: `git merge --squash <branch> && git commit`
* Do not commit automatically to the fix.

## CLI / UX Conventions

> [!NOTE]
> When formatting CLI help text, pay close attention to alignment, spacing, bracket style, and description consistency. Match existing patterns exactly rather than inventing new formatting.

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

| File            | Purpose                                               |
| --------------- | ----------------------------------------------------- |
| `main.rs`       | Entry point, dependency injection, command dispatch   |
| `cli.rs`        | Argument parsing with clap                            |
| `errors.rs`     | Error classification, exit codes (0/1/2/3), suggestions |
| `style.rs`      | Terminal styling with console crate                   |
| `tui.rs`        | Custom TUI components (FuzzySelect)                   |
| `version.rs`    | Version constant from `CARGO_PKG_VERSION`             |

#### `src/commands/`

| File        | Purpose                                                    |
| ----------- | ---------------------------------------------------------- |
| `run.rs`    | Launch AI tools; falls back to `start` flow when no tool given |
| `start.rs`  | Interactive remembered-start flow (key + tool + model picker) |
| `chat.rs`   | Interactive chat command; routes to TUI or one-shot mode   |
| `chat_tui.rs` | Full-screen interactive chat TUI (ratatui + crossterm)   |
| `keys.rs`   | API key management (add, rm, use, edit, cat, list)         |
| `models.rs` | List available models from active provider (1h cache)      |
| `serve.rs`  | Local OpenAI-compatible API server                         |
| `update.rs` | Self-update via GitHub Releases                            |

#### `src/services/`

| File                          | Purpose                                                                 |
| ----------------------------- | ----------------------------------------------------------------------- |
| `session_store.rs`            | Key persistence, AES-256-GCM encryption, chat sessions, directory starts, usage stats |
| `ai_launcher.rs`              | Process spawning, signal forwarding (SIGINT/SIGTERM), stdio passthrough |
| `environment_injector.rs`     | Tool-specific env var configuration, placeholder URL + router flag injection |
| `provider_protocol.rs`        | Protocol detection from base URL                                        |
| `model_names.rs`              | Model name transformations (e.g. `claude-sonnet-4-6` → `anthropic/claude-sonnet-4.6`) |
| `anthropic_router.rs`         | Proxy for Claude + OpenRouter                                           |
| `anthropic_to_openai_router.rs` | Proxy for Anthropic-format clients + OpenAI-compatible providers     |
| `copilot_router.rs`           | Proxy for Claude/Codex/Gemini + GitHub Copilot                          |
| `copilot_auth.rs`             | GitHub Copilot OAuth device flow and token refresh                      |
| `responses_to_chat_router.rs` | Proxy for Responses API clients + non-OpenAI providers (Responses API → Chat Completions) |
| `gemini_router.rs`            | Proxy for Gemini + non-Google providers (Gemini format → Chat Completions) |
| `serve_router.rs`             | Shared router server scaffolding                                        |
| `http_utils.rs`               | Shared HTTP utilities (request parsing, header extraction, SSE)        |
| `openai_anthropic_bridge.rs`  | Anthropic Messages ↔ OpenAI Chat Completions conversion                 |
| `openai_gemini_bridge.rs`     | Gemini native ↔ OpenAI Chat Completions conversion                      |
| `anthropic_route_pipeline.rs` | Shared pipeline for Anthropic-format router requests                    |
| `anthropic_chat_request.rs`   | Anthropic chat request types                                            |
| `anthropic_chat_response.rs`  | Anthropic chat response types                                           |
| `models_cache.rs`             | 1-hour file-backed cache for model lists                                |
| `system_env.rs`               | System environment helpers (CWD, home dir, etc.)                        |

### Data Model

`ApiKey`: `id`, `name`, `base_url`, `key`, `created_at`. Stored AES-256-GCM encrypted in `~/.config/aivo/config.json`. The sentinel `base_url` value `"copilot"` identifies GitHub Copilot keys.

### Exit Codes

| Code | Meaning    |
| ---- | ---------- |
| `0`  | Success    |
| `1`  | User error |
| `2`  | Network    |
| `3`  | Auth       |
