# aivo

A lightweight CLI for managing API keys and running Claude Code, Codex, Gemini, OpenCode, and Pi CLI across providers.


## What it does

- Securely manages multiple API keys for different providers.
- Runs `claude`, `codex`, `gemini`, `opencode`, and `pi` CLI tools seamlessly.
- Provides a simple chat TUI and a one-shot `-x` mode.
- Can expose the active provider as a local OpenAI-compatible server.

## Install

Homebrew:

```bash
brew install yuanchuan/tap/aivo
```

npm:

```bash
npm install -g @yuanchuan/aivo
```

Install script:

```bash
curl -fsSL https://yuanchuan.dev/aivo/install.sh | sh
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

Use your GitHub Copilot subscription.

```bash
aivo keys add copilot
aivo claude
```

Use local models via Ollama.

```bash
aivo keys add ollama

# auto pull the model if not present
aivo claude --model llama3.2
```

## run

Launch an AI tool, or use the saved start flow.

### Supported tools:

- `claude` [Claude Code](https://github.com/anthropics/claude-code)
- `codex` [Codex](https://github.com/openai/codex)
- `gemini` [Gemini CLI](https://github.com/google-gemini/gemini-cli)
- `opencode` [OpenCode](https://github.com/anomalyco/opencode)
- `pi` [Pi Coding Agent](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent)

```bash
aivo claude --dangerously-skip-permissions
aivo claude --resume 16354407-050e-4447-a068-4db222ff841
```

Pick a model for one run:

```bash
aivo claude --model moonshotai/kimi-k2.5
aivo chat --model openai/gpt-4o
```

Or let `--model` open the model picker if the provider supports the model list API:

```bash
aivo claude --model
aivo chat -m  # or just use -m
```

Use a different saved key without changing the active one:

```bash
aivo claude --key openrouter
aivo codex --key copilot
aivo claude --key  # open key picker for this run only
```

Preview what `aivo` would launch without starting the tool:

```bash
aivo claude --dry-run
aivo run codex --model gpt-5 --dry-run
```

Inject extra env vars into the child process:

```bash
aivo claude --env=BASH_DEFAULT_TIMEOUT_MS=60000
```

Use the interactive start flow for the current directory:

```bash
aivo run
```

`aivo run` without a tool will reuse the saved selection for that directory when it has one.

## chat

`aivo chat` starts the full-screen chat UI. `-x` sends a single prompt and exits.

```bash
aivo chat
aivo chat -x "Summarize this repository"
aivo chat --attach README.md --attach screenshot.png
git diff --cached | aivo chat -x "Write a one-line commit message"
```

Omit the message to read from stdin instead (Ctrl-D to send):

```bash
aivo chat -x
aivo -x "hello"
```

The selected chat model is remembered per saved key.

## serve

`aivo serve` exposes the active provider as a local OpenAI-compatible endpoint:

```bash
aivo serve
aivo serve --port 8080
aivo serve --key openrouter
aivo serve --log
```

Default address:

```text
http://127.0.0.1:24860
```

This is handy for scripts and tools that already speak the OpenAI API.

Options:

```bash
aivo serve --host 0.0.0.0 -p 8080       # bind to all interfaces
aivo serve --cors                        # enable CORS for browser clients
aivo serve --timeout 60                  # upstream timeout in seconds (default: 300)
aivo serve --auth-token                  # require bearer token (auto-generated)
aivo serve --auth-token my-secret        # require a specific bearer token
aivo serve --failover                    # multi-key failover on 429/5xx errors
aivo serve --log /tmp/requests.jsonl     # log requests to a file
aivo serve --log | jq .                  # log requests to stdout as JSONL
```

## keys

List, inspect, switch, edit, or remove saved keys:

```bash
aivo keys
aivo keys add
aivo keys use
aivo keys cat
aivo keys edit
aivo keys rm
aivo keys ping
aivo keys ping --all # ping all keys and show status
```

`aivo use <name>` is a shortcut for `aivo keys use <name>`.
`aivo ping` is a shortcut for `aivo keys ping`.

Examples:

```bash
aivo keys add openrouter --base-url https://openrouter.ai/api/v1 --key xxx
aivo keys add groq --base-url https://api.groq.com/openai/v1 --key xxx
aivo keys add deepseek --base-url https://api.deepseek.com/v1 --key xxx
aivo keys add copilot
aivo keys add ollama
```

You are not limited to the providers above.
Any endpoint that matches the supported protocols can be saved with `aivo keys add`.
Keys are stored locally and encrypted in the user config directory.

## models

List models for the active provider:

```bash
aivo models
aivo models --refresh
aivo models --key openrouter
aivo models -s sonnet
```

Model lists are cached for one hour. `--refresh` bypasses the cache.

## alias

Create short names for models:

```bash
aivo alias                          # list all aliases
aivo alias fast=claude-haiku-4-5    # create an alias
aivo alias best claude-sonnet-4-6   # alternative syntax
aivo alias rm fast                  # remove an alias

# then you can use the alias in place of the model name:
aivo claude -m fast
```

## ls

`aivo ls` shows a compact overview of:

- saved keys and the active key
- installed tool binaries on `PATH`
- the remembered tool/model for the current directory
- the saved chat model and cached model count for the active key

Use `--ping` to also health-check all keys:

```bash
aivo ls --ping
```

## stats

Show usage statistics: token counts, request counts, and cost breakdowns.

```bash
aivo stats                  # human-readable summary
aivo stats -n               # exact numbers
aivo stats -s openrouter    # filter by key, model, or tool
```

## update

Update to the latest version:

```bash
aivo update
```

## Development

```bash
make build
make build-debug
make check
make test
make clippy
make build-release
```

## License

MIT
