# aivo

Run Claude Code, Codex, and Gemini CLI across multiple providers.

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

Use your GitHub Copilot subscription

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
| `aivo chat` | Full-screen interactive chat TUI (or one-shot with `-x`) |
| `aivo models` | List available models from active provider |
| `aivo serve` | Start a local OpenAI-compatible API server |
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

aivo serve                       # start on default port 24860
aivo serve --port 8080           # start on custom port
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

**OpenRouter**
```bash
aivo keys add --base-url=https://openrouter.ai/api/v1 --key=xxx
```

**Vercel AI Gateway**
```bash
aivo keys add --base-url=https://ai-gateway.vercel.sh/v1 --key=xxx
```

**DeepSeek**
```bash
aivo keys add --base-url=https://api.deepseek.com/v1 --key=xxx
```

**Fireworks**
```bash
aivo keys add --base-url=https://api.fireworks.ai/inference/v1 --key=xxx
```

**MiniMax**
```bash
aivo keys add --base-url=https://api.minimax.io/anthropic --key=xxx
```

**Moonshot**
```bash
aivo keys add --base-url=https://api.moonshot.cn/v1 --key=xxx
```

**Groq**
```bash
aivo keys add --base-url=https://api.groq.com/openai/v1 --key=xxx
```

**xAI (Grok)**
```bash
aivo keys add --base-url=https://api.x.ai/v1 --key=xxx
```

**Mistral**
```bash
aivo keys add --base-url=https://api.mistral.ai/v1 --key=xxx
```

**Cloudflare Workers AI**
```bash
aivo keys add --base-url=https://api.cloudflare.com/client/v4/accounts/<id>/ai/v1 --key=xxx
```

Use `aivo keys` to view your saved providers and `aivo use` to switch between them.

## Multiple Providers

Save multiple keys and switch between them on the fly:

```bash
# save a few providers
aivo keys add --name=openrouter --base-url=https://openrouter.ai/api/v1 --key=xxx
aivo keys add --name=groq --base-url=https://api.groq.com/openai/v1 --key=xxx
aivo keys add copilot

# switch active provider
aivo use openrouter

# or use a specific key for a single run
aivo claude --key groq
aivo claude --key copilot
```

## Local API Server

`aivo serve` exposes your active provider as a local OpenAI-compatible endpoint — useful for MCP servers, scripts, or any tool that speaks the OpenAI API:

```bash
aivo serve                  # listens on http://127.0.0.1:24860
aivo serve --port 8080      # custom port
aivo serve --key openrouter # serve a specific saved key
```

Then point any OpenAI client at `http://127.0.0.1:24860`.

## Development

```bash
make build
make build-debug
make test
make clippy
make check

# build the final optimized binary explicitly
make build-release
```

## License

MIT
