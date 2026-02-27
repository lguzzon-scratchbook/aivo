# aivo

CLI tool for unified access to AI coding assistants (Claude, Codex, Gemini) with local API key management.

## Features

- **Unified interface** for multiple AI coding assistants (Claude, Codex, Gemini)
- **Multi-provider support** - works with OpenRouter, Vercel AI Gateway, and other compatible providers
- **Interactive chat** - built-in REPL with streaming responses via OpenAI-compatible API
- **API key management** - add, activate, and remove keys
- **Secure storage** - API keys encrypted with AES-256-GCM
- **Direct passthrough** of all tool arguments and flags
- **Cross-platform support** - macOS, Linux, and Windows

## Installation

```bash
curl -fsSL https://raw.githubusercontent.com/yuanchuan/aivo/main/scripts/install.sh | sh
```

Or download a binary manually from [GitHub Releases](https://github.com/yuanchuan/aivo/releases).

## Quick Start

```bash
# Add an API key
aivo keys add

# Run an AI tool (these are equivalent)
aivo claude
aivo run claude
```

## Usage

```
aivo v1.0.0 — CLI for AI coding assistants

Usage: aivo <command> [options]

Commands:
  run <claude|codex|gemini>  Launch AI tool with local API keys
  chat [--model]             Start an interactive chat REPL
  keys [action]              Manage API keys (list, use, rm, add, cat)
  update                     Update to the latest version

Shortcuts: aivo claude, aivo codex, aivo gemini

Options:
  -h, --help      Display help information
  -v, --version   Display the current version
```

### Run AI Tools

Run any supported AI tool with automatic API key injection:

```bash
# Quick shortcuts
aivo claude
aivo codex
aivo gemini

# Or use the run command
aivo run claude
aivo run codex
aivo run gemini
```

All arguments are passed through directly to the underlying tool:

```bash
# Specify a model
aivo claude --model claude-sonnet-4-5-20251001

# Select a specific API key by ID or name
aivo claude --key my-proxy
aivo claude -k a1b2

# Inject environment variables
aivo claude --env DEBUG=true --env CUSTOM_VAR=value

# Pass tool-specific options
aivo codex --model o4-mini file.ts

# Enable debug output (shows injected env vars)
aivo claude --debug
```

### Chat REPL

Start an interactive chat session with streaming responses:

```bash
aivo chat                        # Start with default model (gpt-4o)
aivo chat --model claude-sonnet-4-5  # Use a specific model (saved for next session)
aivo chat -m gpt-4o              # Short flag
aivo chat --key my-proxy         # Use a specific API key
aivo chat -k a1b2 -m gpt-4o     # Combine key and model
```

Uses the active API key's base URL with the OpenAI-compatible `/v1/chat/completions` endpoint. Model choice is remembered across sessions.

### Manage API Keys

```bash
aivo keys                    # List all keys
aivo keys add                # Add a new API key (interactive)
aivo keys use <id|name>      # Activate a specific key
aivo keys cat <id|name>      # Display full key details
aivo keys rm <id|name>       # Remove an API key
```

### Other Commands

```bash
# Update CLI to latest version
aivo update

# Show help
aivo --help
```

## How It Works

1. **Key Management**: API keys are stored in `~/.config/aivo/config.json` with AES-256-GCM encryption. Machine-specific key derivation ensures keys cannot be used on another machine.
2. **Environment Injection**: When you run a tool, the CLI injects the appropriate environment variables:
   - **Claude**: `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`, `ANTHROPIC_MODEL` (when `--model` is used)
   - **Codex**: `OPENAI_BASE_URL`, `OPENAI_API_KEY`
   - **Gemini**: `GOOGLE_GEMINI_BASE_URL`, `GEMINI_API_KEY`
3. **Process Spawning**: The AI tool is spawned with the injected environment and all arguments passed through. Signals (SIGINT, SIGTERM) are forwarded to child processes.

## Configuration

Keys are stored in `~/.config/aivo/config.json` with restricted file permissions (0600).

**Encryption**: API key values are encrypted using AES-256-GCM with machine-specific key derivation (PBKDF2 with SHA-256, 100k iterations) based on system username and home directory path.

## Provider Compatibility

aivo works with the official Anthropic API and third-party providers. All URL normalization and API compatibility is handled automatically.

### OpenRouter

Add your OpenRouter key using `https://openrouter.ai/api/v1` as the base URL.

```bash
# aivo chat - uses OpenAI-compatible endpoint
aivo chat --model claude-sonnet-4-6

# aivo run claude - uses a built-in proxy that handles OpenRouter's API format
aivo claude --model claude-sonnet-4-6
aivo claude --model claude-opus-4-6
aivo claude --model claude-haiku-4-5
```

Both `aivo chat` and `aivo run claude` work out of the box. aivo automatically:
- Converts model names to OpenRouter's format (`claude-sonnet-4-6` → `anthropic/claude-sonnet-4.6`)
- Starts a lightweight background proxy to bridge Claude Code's API with OpenRouter

### Vercel AI Gateway

Add your Vercel key using `https://ai-gateway.vercel.sh` as the base URL.

```bash
aivo claude    # works directly, no special setup needed
aivo chat --model claude-sonnet-4-6
```

### Other providers

Any Anthropic-compatible provider works with `aivo run claude`. Any OpenAI-compatible provider works with `aivo chat`. Use the provider's base URL when adding the key — aivo handles trailing `/v1` automatically.

## Development

```bash
cargo build --release    # Build release binary
cargo test               # Run all tests
cargo clippy             # Lint
cargo check              # Quick type check
```

## Prerequisites

The AI tools you want to use must be installed separately:

### macOS (Homebrew)

```bash
brew install claude              # Claude Code
brew install openai/codex        # Codex
brew tap google-gemini/gemini-cli && brew install gemini-cli  # Gemini CLI
```

### All Platforms (npm)

```bash
npm install -g @anthropic-ai/claude-code  # Claude Code
npm install -g @openai/codex              # Codex
npm install -g @google/gemini-cli         # Gemini CLI
```

## License

MIT
