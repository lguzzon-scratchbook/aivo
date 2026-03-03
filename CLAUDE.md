# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

aivo is a **Rust** CLI tool that provides unified access to multiple AI coding assistants (Claude, Codex, Gemini) with local API key management and secure storage.

## Commands

```bash
# Build
cargo build --release  # Compile optimized binary to target/release/aivo
cargo build --release --target <target>  # Cross-compile for specific platform

# Test
cargo test             # Run all tests (~200 tests)
cargo test --release   # Run tests on release build

# Format
cargo fmt              # Format code (always run before committing)

# Check
cargo clippy           # Lint with clippy
cargo check            # Quick type check
```

## Development Workflow

After making code changes to CLI tools or binaries, always rebuild and reinstall before testing. Run `cargo build --release && cargo install --path .` (or equivalent) to avoid testing stale binaries.

## Testing & Quality

This project uses Rust as the primary language. Run `cargo clippy` before committing and fix all warnings. Run `cargo test` after any code changes and ensure all tests pass before committing.

## Git Conventions

Always use squash merge when merging branches to main. Never use fast-forward merge. Command: `git merge --squash <branch> && git commit`

## Code Review

For code reviews, be concise and deliver findings quickly. Do not extensively explore the entire codebase before providing review feedback. Focus on the diff and immediate context only.

## CLI / UX Conventions

When formatting CLI help text, pay close attention to alignment, spacing, bracket style, and description consistency. Match existing patterns exactly rather than inventing new formatting.

## Architecture

### Entry Point & Dependency Injection

`src/main.rs` initializes all services and injects them into command handlers:

```
SessionStore → EnvironmentInjector → AILauncher
                     ↓
             Command Handlers
```

### CLI Structure

`src/cli.rs` - Command parsing with clap, handles:
- Help/version display
- Unknown command detection
- Argument validation and routing

`src/style.rs` - Terminal styling with console crate
`src/version.rs` - Version management from CARGO_PKG_VERSION
`src/errors.rs` - Centralized error classification with context-specific suggestions

### Service Layer (`src/services/`)

- **SessionStore** (`session_store.rs`) - Persists API keys to `~/.config/aivo/config.json` with AES-256-GCM encryption. Machine-specific key derivation using username + home directory.

- **AILauncher** (`ai_launcher.rs`) - Spawns AI tool processes (claude, codex, gemini) with environment injection using tokio. Forwards signals (SIGINT, SIGTERM) and inherits stdio for interactive passthrough. Injects `--teammate-mode in-process` for Claude to ensure single-window mode. Starts the appropriate built-in router when needed, then overwrites the placeholder base URL with the actual bound port.

- **EnvironmentInjector** (`environment_injector.rs`) - Configures tool-specific environment variables. Includes model name transformation for OpenRouter compatibility (converts hyphenated versions like `claude-sonnet-4-6` to dotted format `claude-sonnet-4.6`):
  - Claude (direct): `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY` (empty), `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`, `ANTHROPIC_MODEL` and related model env vars (optional)
  - Claude (Copilot): uses placeholder `ANTHROPIC_BASE_URL` + sets `AIVO_USE_COPILOT_ROUTER=1` to trigger `CopilotRouter`
  - Claude (OpenRouter): uses placeholder `ANTHROPIC_BASE_URL` + sets `AIVO_USE_ROUTER=1` to trigger `ClaudeCodeRouter`
  - Codex (OpenAI): `OPENAI_API_KEY`, `OPENAI_BASE_URL` (direct)
  - Codex (non-OpenAI): uses placeholder `OPENAI_BASE_URL` + sets `AIVO_USE_CODEX_ROUTER=1` to trigger `CodexRouter`
  - Gemini (Google): `GEMINI_API_KEY`, `GOOGLE_GEMINI_BASE_URL` (direct)
  - Gemini (non-Google): uses placeholder `GOOGLE_GEMINI_BASE_URL` + sets `AIVO_USE_GEMINI_ROUTER=1` to trigger `GeminiRouter`

- **ClaudeCodeRouter** (`claude_code_router.rs`) - Built-in HTTP proxy for OpenRouter. Intercepts Claude Code's `/v1/messages` and `/v1/chat/completions` requests, transforms model names (`claude-sonnet-4-6` → `anthropic/claude-sonnet-4.6`), and forwards to OpenRouter. Binds to a random port.

- **CopilotRouter** (`copilot_router.rs`) - Built-in HTTP proxy for GitHub Copilot. Intercepts Claude Code's `/v1/messages` requests (Anthropic Messages format), converts to OpenAI Chat Completions format, and forwards to the Copilot API. Converts responses back to Anthropic format including SSE streaming. Uses `CopilotTokenManager` from `copilot_auth.rs` for token exchange and auto-refresh. Binds to a random port.

- **CopilotAuth** (`copilot_auth.rs`) - GitHub Copilot authentication. Implements OAuth device flow using the VS Code Copilot client ID (`Iv1.b507a08c87ecfe98`). Manages Copilot token lifecycle: exchanges GitHub OAuth token → short-lived Copilot token via `api.github.com/copilot_internal/v2/token`, caches with expiry-based auto-refresh.

- **CodexRouter** (`codex_router.rs`) - Built-in HTTP proxy for non-OpenAI providers. Strips unsupported built-in tool types (`computer_use`, `file_search`, `web_search`, `code_interpreter`) that most third-party providers reject. Converts between Codex CLI's Responses API (`/v1/responses`) and the Chat Completions API (`/v1/chat/completions`) for providers that only support the latter. Binds to a random port.

- **GeminiRouter** (`gemini_router.rs`) - Built-in HTTP proxy for non-Google providers. Converts Gemini CLI's native API format (`/v1beta/models/{model}:generateContent`) to OpenAI Chat Completions format, then converts the response back. Handles streaming, tool calls, function responses, and generation config. Binds to a random port.

### Command Handlers (`src/commands/`)

Each command receives injected services. Commands return exit codes for testing.

**Available Commands:**
- **keys** - API key management:
  - `list` - List all keys
  - `use <id|name>` - Activate a specific key
  - `add` - Add an API key interactively (`add copilot` triggers GitHub Copilot device flow)
  - `rm <id|name>` - Remove an API key
  - `cat <id|name>` - Display full key details
- **run** - Launch AI tools with unified interface
- **chat** - Interactive REPL with streaming responses. Supports OpenAI-compatible, Anthropic, and GitHub Copilot providers. Automatically transforms model names for OpenRouter compatibility (e.g., `claude-sonnet-4-6` → `claude-sonnet-4.6`).
- **models** - List available models from the active provider (cached, with `--refresh` to bypass)
- **update** - Self-update with download progress display, cross-platform binary download from GitHub Releases

### Error Handling (`src/errors.rs`)

Exit codes: 0=success, 1=user error, 2=network, 3=auth. Errors are classified by pattern matching and formatted with suggestions.

### Data Model

Single `ApiKey` struct with fields: `id`, `name`, `base_url`, `key`, `created_at`. Keys are stored encrypted in a `StoredConfig` containing `api_keys: Vec<ApiKey>` and `active_key_id: Option<String>`. The sentinel `base_url` value `"copilot"` identifies GitHub Copilot keys.

## Testing Patterns

- Unit tests in `#[cfg(test)]` modules within source files
- Integration tests in `tests/` directory
- Command handlers return exit codes for verification
- **Test Coverage:** ~200 tests covering encryption, services, router logic, and command handlers

## Build & Deployment

- **Runtime:** Rust (native binary)
- **Build:** `cargo build --release` creates optimized binary at `target/release/aivo`
- **Cross-platform:** Supports linux/darwin x64/arm64, windows x64

## Project Structure

```
aivo/
├── src/
│   ├── cli.rs                       # CLI argument parsing (clap)
│   ├── main.rs                      # Main entry point with dependency injection
│   ├── lib.rs                       # Library exports for testing
│   ├── version.rs                   # Version constant
│   ├── style.rs                     # Terminal styling with console crate
│   ├── tui.rs                       # Custom TUI components (FuzzySelect)
│   ├── errors.rs                    # Centralized error handling & exit codes
│   ├── commands/
│   │   ├── mod.rs
│   │   ├── chat.rs                  # Interactive chat REPL
│   │   ├── keys.rs                  # API key management
│   │   ├── models.rs                # List available models from provider
│   │   ├── run.rs                   # Unified AI tool launcher
│   │   └── update.rs               # Self-update via GitHub Releases
│   └── services/
│       ├── mod.rs
│       ├── session_store.rs         # Key persistence & AES-256-GCM encryption
│       ├── environment_injector.rs  # Tool-specific env configuration
│       ├── ai_launcher.rs          # Process spawning & signal forwarding
│       ├── claude_code_router.rs   # Built-in proxy for Claude + OpenRouter
│       ├── copilot_auth.rs         # GitHub Copilot OAuth device flow & token management
│       ├── copilot_router.rs       # Built-in proxy for Claude + GitHub Copilot
│       ├── codex_router.rs         # Built-in proxy for Codex + non-OpenAI providers
│       ├── gemini_router.rs        # Built-in proxy for Gemini + non-Google providers
│       └── models_cache.rs         # 24h file-backed cache for model lists
├── tests/
│   ├── encryption_test.rs
│   ├── encryption_property.rs
│   ├── environment_injector_test.rs
│   ├── errors_test.rs
│   └── integration/
│       └── cli_workflow_test.rs
├── Cargo.toml
├── Cargo.lock
├── CLAUDE.md                        # This file
├── README.md
└── LICENSE
```

## Encryption & Security

**AES-256-GCM Encryption:**
- API keys encrypted with AES-256-GCM using machine-specific derivation
- Key derivation: PBKDF2 with SHA-256, 100k iterations
- Salt derived from HMAC of machine data (username + home directory)
- 16-byte IV and 16-byte auth tag

## Router Architecture

### Built-in Proxy Pattern

All routers follow a consistent pattern:

1. **Placeholder Base URL**: EnvironmentInjector sets a placeholder URL (e.g., `http://localhost:0`) and a flag environment variable (e.g., `AIVO_USE_ROUTER=1`)

2. **Router Discovery**: The appropriate router (ClaudeCodeRouter, CopilotRouter, CodexRouter, GeminiRouter) detects the flag and starts an HTTP server on a random available port

3. **URL Replacement**: The router binds to the random port, then overwrites the placeholder base URL with the actual bound URL

4. **Request Interception**: The router intercepts requests from the AI tool and forwards them to the target provider with necessary transformations

### Router-Specific Transformations

- **ClaudeCodeRouter**: Converts model names (`claude-sonnet-4-6` → `anthropic/claude-sonnet-4.6`) for OpenRouter compatibility
- **CopilotRouter**: Converts between Anthropic Messages format and OpenAI Chat Completions format, handles SSE streaming
- **CodexRouter**: Strips unsupported tool types and converts between Responses API and Chat Completions API
- **GeminiRouter**: Converts between Gemini native API format and OpenAI Chat Completions format

### Signal Handling

AILauncher forwards SIGINT and SIGTERM signals to spawned processes and inherits stdio for interactive passthrough, ensuring proper cleanup and user experience.
