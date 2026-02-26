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
cargo test             # Run all tests (~53 tests)
cargo test --release   # Run tests on release build

# Format
cargo fmt              # Format code (always run before committing)

# Check
cargo clippy           # Lint with clippy
cargo check            # Quick type check
```

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

- **AILauncher** (`ai_launcher.rs`) - Spawns AI tool processes (claude, codex, gemini) with environment injection using tokio. Forwards signals (SIGINT, SIGTERM) and inherits stdio for interactive passthrough.

- **EnvironmentInjector** (`environment_injector.rs`) - Configures tool-specific environment variables:
  - Claude: `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`, `ANTHROPIC_MODEL` (optional)
  - Codex: `OPENAI_API_KEY`, `OPENAI_BASE_URL`
  - Gemini: `GEMINI_API_KEY`, `GOOGLE_GEMINI_BASE_URL`

### Command Handlers (`src/commands/`)

Each command receives injected services. Commands return exit codes for testing.

**Available Commands:**
- **keys** - API key management:
  - `list` - List all keys
  - `use <id|name>` - Activate a specific key
  - `add` - Add an API key interactively
  - `rm <id|name>` - Remove an API key
  - `cat <id|name>` - Display full key details
- **run** - Launch AI tools with unified interface
- **chat** - Interactive REPL with streaming responses via OpenAI-compatible `/v1/chat/completions` endpoint
- **update** - Self-update with download progress display, cross-platform binary download from GitHub Releases

### Error Handling (`src/errors.rs`)

Exit codes: 0=success, 1=user error, 2=network, 3=auth. Errors are classified by pattern matching and formatted with suggestions.

### Data Model

Single `ApiKey` struct with fields: `id`, `name`, `base_url`, `key`, `created_at`. Keys are stored encrypted in a `StoredConfig` containing `api_keys: Vec<ApiKey>` and `active_key_id: Option<String>`.

## Testing Patterns

- Unit tests in `#[cfg(test)]` modules within source files
- Integration tests in `tests/` directory
- Command handlers return exit codes for verification
- **Test Coverage:** ~116 tests covering encryption, services, and command logic

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
│   ├── errors.rs                    # Centralized error handling & exit codes
│   ├── commands/
│   │   ├── mod.rs
│   │   ├── chat.rs                  # Interactive chat REPL
│   │   ├── keys.rs                  # API key management
│   │   ├── run.rs                   # Unified AI tool launcher
│   │   └── update.rs               # Self-update via GitHub Releases
│   └── services/
│       ├── mod.rs
│       ├── session_store.rs         # Key persistence & AES-256-GCM encryption
│       ├── environment_injector.rs  # Tool-specific env configuration
│       └── ai_launcher.rs          # Process spawning & signal forwarding
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
