# aivo

A lightweight CLI for managing API keys and running Claude Code, Codex, and Gemini CLI across providers.


## What it does

- Securely manages multiple API keys for different providers.
- Runs `claude`, `codex`, and `gemini` CLI tools seamlessly.
- Provides a simple chat TUI and a one-shot `-x` mode.
- Can expose the active provider as a local OpenAI-compatible server.

## Install

Homebrew:

```bash
brew install yuanchuan/tap/aivo
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

Use your GitHub Copilot subscription

```bash
aivo keys add copilot
aivo claude
```

## Everyday usage

Run a tool with its normal flags:

```bash
aivo claude --dangerously-skip-permissions
aivo claude --resume 16354407-050e-4447-a068-4db222ff841
aivo claude --model moonshotai/kimi-k2.5
```

Pick a model for one run:

```bash
aivo claude --model moonshotai/kimi-k2.5
aivo chat --model openai/gpt-4o
```

Or let `--model` open the model picker if the provider supports the model list API:

```bash
aivo claude --model
aivo chat --model
```

Use a different saved key without changing the active one:

```bash
aivo claude --key openrouter
aivo codex --key copilot
aivo claude --key              # open key picker for this run only
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

## Keys and providers

List, inspect, switch, edit, or remove saved keys:

```bash
aivo keys
aivo keys add
aivo keys use
aivo keys cat
aivo keys edit
aivo keys rm
```

`aivo use <name>` is a shortcut for `aivo keys use <name>`.

Examples:

```bash
aivo keys add openrouter --base-url https://openrouter.ai/api/v1 --key xxx
aivo keys add groq --base-url https://api.groq.com/openai/v1 --key xxx
aivo keys add deepseek --base-url https://api.deepseek.com/v1 --key xxx
aivo keys add copilot
```

You are not limited to the providers above.
Any endpoint that matches the supported protocols can be saved with `aivo keys add`.
Keys are stored locally and encrypted in the user config directory.

## Models

List models for the active provider:

```bash
aivo models
aivo models --refresh
aivo models --key openrouter
```

Model lists are cached for one hour. `--refresh` bypasses the cache.

## Chat

`aivo chat` starts the full-screen chat UI. `-x` sends a single prompt and exits.

```bash
aivo chat
aivo chat -x "Summarize this repository"
git diff --cached | aivo chat -x "Write a one-line commit message"
```

Omit the message to read from stdin instead (Ctrl-D to send):

```bash
aivo chat -x
aivo -x "hello"
```

The selected chat model is remembered per saved key.

## Local API server

`aivo serve` exposes the active provider as a local OpenAI-compatible endpoint:

```bash
aivo serve
aivo serve --port 8080
aivo serve --key openrouter
```

Default address:

```text
http://127.0.0.1:24860
```

This is handy for scripts, MCP servers, and tools that already speak the OpenAI API.

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
