# aivo

Run Claude Code via OpenRouter, Vercel AI Gateway, and more — with unified API key management.

## Install

```bash
brew install yuanchuan/tap/aivo
```
Or via shell script

```bash
curl -fsSL https://raw.githubusercontent.com/yuanchuan/aivo/main/scripts/install.sh | sh
```

Or download a binary from [GitHub Releases](https://github.com/yuanchuan/aivo/releases).

## Quick Start

```bash
# Add API key (OpenRouter, Vercel AI Gateway, etc.)
aivo keys add
aivo claude
aivo claude --model moonshotai/kimi-k2.5

# Or use your GitHub Copilot subscription
aivo keys add copilot
aivo claude
```

## Commands

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
| `aivo keys list` | List all keys |
| `aivo update` | Update aivo |

Flags pass through directly:

```bash
aivo claude --dangerously-skip-permissions
aivo claude --resume 16354407-050e-4447-a068-4db7922ff841
aivo claude --model moonshotai/kimi-k2.5

aivo claude --key my-proxy       # use a specific saved key
aivo claude --env DEBUG=true     # inject extra env vars

aivo chat --model openai/gpt-4o  # chat with any model
aivo chat -x "Summarize this repo in 3 bullets"  # one-shot chat, prints and exits
git diff --cached | aivo chat -x "Summarize these changes in one sentence"  # stdin context + one-shot

aivo models                      # list models (cached for 24h)
aivo models --refresh            # force-refresh the model list
```

## Provider Compatibility

### OpenRouter

Add your key with `https://openrouter.ai/api/v1` as the base URL.

```bash
aivo claude --model claude-sonnet-4-6   # auto-converts model name
aivo chat --model openai/gpt-4o-mini
```

### Vercel AI Gateway

Add your key with `https://ai-gateway.vercel.sh/v1` as the base URL.

```bash
aivo claude
aivo chat --model claude-sonnet-4-6
```

### Other providers

Any Anthropic-compatible provider works with `aivo claude`.
Any OpenAI-compatible provider works with `aivo chat` and `aivo codex`.

Use the provider's base URL when adding a key — trailing `/v1` is handled automatically.

### GitHub Copilot

Use your existing GitHub Copilot subscription to run Claude Code — no separate Anthropic API key needed.

```bash
aivo keys add copilot         # authenticate via GitHub device flow
aivo claude                   # run Claude Code with Copilot
aivo models                   # list available Copilot models
aivo chat --model claude-sonnet-4.6
```

## Managing Keys

```bash
aivo keys            # list all keys
aivo keys add        # add a new key (interactive)
aivo keys use [id]   # switch active key
aivo keys cat <id>   # show key details
aivo keys rm <id>    # remove a key
```

Keys are stored encrypted in `~/.config/aivo/config.json` (AES-256-GCM, machine-specific).

## How It Works

1. **Key storage** — Keys are encrypted with AES-256-GCM in `~/.config/aivo/config.json`. Machine-specific key derivation (PBKDF2-SHA256, 100k iterations) means they can't be copied to another machine.
2. **Environment injection** — When you run a tool, aivo injects the right env vars for that provider (`ANTHROPIC_BASE_URL`, `OPENAI_API_KEY`, etc.) without touching your shell environment.
3. **Built-in routers** — For third-party providers, aivo starts a lightweight local HTTP proxy that handles API format differences automatically:
   - Claude + GitHub Copilot: OAuth device flow auth, Copilot token exchange, converts between Anthropic Messages and OpenAI Chat Completions
   - Claude + OpenRouter: translates model names and proxies Anthropic API requests
   - Codex + non-OpenAI: strips unsupported tool types, converts between Responses and Chat Completions API
   - Gemini + non-Google: converts Gemini's native format to/from OpenAI Chat Completions
4. **Process passthrough** — The AI tool runs as a child process with your terminal attached. Signals (SIGINT, SIGTERM) are forwarded correctly.

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
cargo build --release
cargo test
cargo clippy
cargo check
```

## License

MIT
