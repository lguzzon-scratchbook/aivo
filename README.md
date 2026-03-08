# aivo

Unified key management and protocol bridge tool for Claude Code, Codex, and Gemini.

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
aivo keys add

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


## Key Management

```bash
aivo keys       # list all keys
aivo keys add   # add a new key (interactive)
aivo keys use   # switch active key
aivo keys cat   # show key details
aivo keys rm    # remove a key
aivo keys edit  # edit a key
```

### Adding popular providers

```bash
# OpenRouter
aivo keys add --base-url=https://openrouter.ai/api/v1 --key=xxx

# Vercel AI Gateway
aivo keys add --base-url=https://ai-gateway.vercel.sh/v1 --key=xxx

# DeepSeek
aivo keys add --base-url=https://api.deepseek.com --key=xxx

# Fireworks
aivo keys add --base-url=https://api.fireworks.ai/inference --key=xxx

# MiniMax
aivo keys add --base-url=https://api.minimax.io/anthropic --key=xxx

# Moonshot
aivo keys add --base-url=https://api.moonshot.ai/v1 --key=xxx

# Cloudflare Workers AI
aivo keys add --base-url=https://api.cloudflare.com/client/v4/accounts/<id>/ai/v1 --key=xxx
```

Use `aivo keys` to view your saved providers and `aivo use` to switch between them.

## How It Works

1. **Encrypted storage**: API keys are encrypted locally in `~/.config/aivo/config.json`.
2. **Env injection**: `aivo` injects provider-specific env vars (`ANTHROPIC_BASE_URL`, `OPENAI_API_KEY`, etc.) only for the launched process.
3. **Protocol translation**: built-in local routing smooths over API incompatibilities across providers.

## Development

```bash
make build
make test
make clippy
make check

# only when packaging or benchmarking the final binary
make build --release
```

## License

MIT
