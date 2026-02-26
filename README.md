# aivo

CLI tool for unified access to AI coding assistants (Claude, Codex, Gemini) with local API key management.

## Features

- **Unified interface** for multiple AI coding assistants (Claude, Codex, Gemini)
- **API key management** - add, activate, and remove keys
- **Secure storage** - API keys encrypted with AES-256-GCM
- **Direct passthrough** of all tool arguments and flags
- **Cross-platform support** - macOS, Linux, and Windows

## Installation

```bash
curl -fsSL https://raw.githubusercontent.com/yuanchuan/aivo/main/scripts/install.sh | sh
```

Or download a binary manually from [GitHub Releases](https://github.com/yuanchuan/aivo/releases).

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

## Quick Start

```bash
# Add an API key
aivo keys add

# Run an AI tool (these are equivalent)
aivo claude
aivo run claude
```

## Usage

### Run AI Tools

Run any supported AI tool with automatic API key injection:

```bash
# Quick aliases
aivo claude
aivo codex
aivo gemini

# Or use the full form
aivo run claude
aivo run codex
aivo run gemini
```

All arguments are passed through directly to the underlying tool:

```bash
# Specify a model
aivo claude --model claude-sonnet-4-5-20251001

# Inject environment variables
aivo claude --env DEBUG=true --env CUSTOM_VAR=value

# Pass tool-specific options
aivo codex --model o4-mini file.ts

# Enable debug output (shows injected env vars)
aivo claude --debug
```

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

## Development

```bash
cargo build --release    # Build release binary
cargo test               # Run all tests (~53 tests)
cargo clippy             # Lint
cargo check              # Quick type check
```

## License

MIT
