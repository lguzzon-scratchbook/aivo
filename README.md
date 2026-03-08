# aivo

Unified key management tool and protocol bridge for Claude Code, Codex, and Gemini.

## Install

```bash
brew install yuanchuan/tap/aivo
```

Or via install script:

```bash
curl -fsSL https://raw.githubusercontent.com/yuanchuan/aivo/main/scripts/install.sh | sh
```

Or download a binary from [GitHub Releases](https://github.com/yuanchuan/aivo/releases).

## Quick Start

```bash
# 1) Add a provider key (OpenRouter, Vercel AI Gateway, etc.)
aivo keys add --name openrouter --base-url https://openrouter.ai/api/v1 --key sk-or-v1-...

# 2) Launch your tool
aivo claude

# 3) Optionally pin a model
aivo claude --model moonshotai/kimi-k2.5
```

Use GitHub Copilot in claude code

```bash
aivo keys add copilot
aivo claude
```

## Common Commands

| Command | Description |
|---|---|
| `aivo claude` | Run Claude Code |
| `aivo codex` | Run Codex |
| `aivo gemini` | Run Gemini |
| `aivo opencode` | Run OpenCode |
| `aivo chat` | Interactive chat REPL (or one-shot with `-x`) |
| `aivo models` | List available models from active provider |
| `aivo use [name]` | Switch active key |
| `aivo keys add` | Add an API key |
| `aivo keys` | List all keys |
| `aivo update` | Update `aivo` |

All extra flags pass through to the underlying tool:

```bash
aivo claude --dangerously-skip-permissions
aivo claude --resume 16354407-050e-4447-a068-4db7922ff841
aivo claude --model moonshotai/kimi-k2.5

aivo claude --key my-proxy       # use a specific saved key
aivo claude --env DEBUG=true     # inject extra env vars

aivo chat --model openai/gpt-4o
aivo chat -x "hello"
git diff --cached | aivo chat -x "Summarize these changes in one sentence"

aivo models                      # cached for 1h
aivo models --refresh            # force-refresh
```

## Provider Compatibility

### OpenRouter

Add your key with base URL `https://openrouter.ai/api/v1`.

```bash
aivo claude --model claude-sonnet-4-6   # model name auto-conversion
aivo chat --model openai/gpt-4o-mini
```

### Vercel AI Gateway

Add your key with base URL `https://ai-gateway.vercel.sh/v1`.

```bash
aivo claude
aivo chat --model claude-sonnet-4-6
```

### GitHub Copilot

Use your existing Copilot subscription to run Claude Code (no separate Anthropic API key required).

```bash
aivo keys add copilot         # GitHub device-flow login
aivo claude
aivo models
aivo chat --model claude-sonnet-4.6
```

### Other Providers

Any Anthropic-compatible provider works with `aivo claude`.
Any OpenAI-compatible provider works with `aivo chat` and `aivo codex`.

Use the provider base URL when adding a key; trailing `/v1` is handled automatically.

## Key Management

```bash
aivo keys            # list all keys
aivo keys add        # add a new key (interactive)
aivo keys add --name openrouter --base-url https://openrouter.ai/api/v1 --key sk-or-v1-...
aivo keys use [id]   # switch active key
aivo keys cat <id>   # show key details
aivo keys rm <id>    # remove a key
aivo keys edit <id>  # edit a key
```

Keys are encrypted in `~/.config/aivo/config.json` (AES-256-GCM, machine-specific key derivation).

## How It Works

1. **Encrypted storage**: API keys are encrypted locally in `~/.config/aivo/config.json`.
2. **Env injection**: `aivo` injects provider-specific env vars (`ANTHROPIC_BASE_URL`, `OPENAI_API_KEY`, etc.) only for the launched process.
3. **Protocol translation**: built-in local routing smooths over API incompatibilities across providers.
4. **Native terminal behavior**: tools run as child processes with proper signal forwarding (`SIGINT`, `SIGTERM`).

## Prerequisites

Install Claude Code:

**macOS (Homebrew)**
```bash
brew install claude
```

**All platforms (npm)**
```bash
npm install -g @anthropic-ai/claude-code
```

For Codex, Gemini, and OpenCode:

**macOS (Homebrew)**
```bash
brew install openai/codex
brew tap google-gemini/gemini-cli && brew install gemini-cli
```

**All platforms (npm)**
```bash
npm install -g @openai/codex
npm install -g @google/gemini-cli
```

## Development

```bash
cargo build
cargo test
cargo clippy
cargo check

# only when packaging or benchmarking the final binary
cargo build --release
```

## License

MIT
