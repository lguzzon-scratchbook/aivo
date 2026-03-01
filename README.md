# aivo

Run Claude Code (and Codex, Gemini, OpenCode) with any API provider — OpenRouter, Vercel AI Gateway, or your own.

No env var juggling. No config files. Just add a key and go.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/yuanchuan/aivo/main/scripts/install.sh | sh
```

Or download a binary from [GitHub Releases](https://github.com/yuanchuan/aivo/releases).

```bash
# Add a key (OpenRouter, Vercel, or any compatible provider)
aivo keys add

# Run Claude Code with it
aivo claude
```

## Commands

| Command | Description |
|---|---|
| `aivo claude` | Run Claude Code |
| `aivo codex` | Run Codex |
| `aivo gemini` | Run Gemini |
| `aivo opencode` | Run OpenCode |
| `aivo chat` | Interactive chat REPL |
| `aivo keys add` | Add an API key |
| `aivo keys use <name>` | Switch active key |
| `aivo keys list` | List all keys |
| `aivo update` | Update aivo |

Flags pass through directly:

```bash
aivo claude --model claude-sonnet-4-6
aivo opencode --model gpt-5
aivo claude --key my-proxy          # use a specific saved key
aivo claude --env DEBUG=true        # inject extra env vars
aivo chat --model gpt-4o            # chat with any model
aivo chat --key my-proxy -m gpt-4o
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

## Managing Keys

```bash
aivo keys            # list all keys
aivo keys add        # add a new key (interactive)
aivo keys use <id>   # switch active key
aivo keys cat <id>   # show key details
aivo keys rm <id>    # remove a key
```

Keys are stored encrypted in `~/.config/aivo/config.json` (AES-256-GCM, machine-specific).

## How It Works

1. **Key storage** — Keys are encrypted with AES-256-GCM in `~/.config/aivo/config.json`. Machine-specific key derivation (PBKDF2-SHA256, 100k iterations) means they can't be copied to another machine.
2. **Environment injection** — When you run a tool, aivo injects the right env vars for that provider (`ANTHROPIC_BASE_URL`, `OPENAI_API_KEY`, etc.) without touching your shell environment.
3. **Built-in routers** — For third-party providers, aivo starts a lightweight local HTTP proxy that handles API format differences automatically:
   - Claude + OpenRouter: translates model names and proxies Anthropic API requests
   - Codex + non-OpenAI: strips unsupported tool types, converts between Responses and Chat Completions API
   - Gemini + non-Google: converts Gemini's native format to/from OpenAI Chat Completions
4. **Process passthrough** — The AI tool runs as a child process with your terminal attached. Signals (SIGINT, SIGTERM) are forwarded correctly.

## Prerequisites

Install the AI tools you want to use:

**macOS (Homebrew)**
```bash
brew install claude
brew install openai/codex
brew tap google-gemini/gemini-cli && brew install gemini-cli
```

**All platforms (npm)**
```bash
npm install -g @anthropic-ai/claude-code
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
