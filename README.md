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

Install script:

```bash
curl -fsSL https://yuanchuan.dev/aivo/install.sh | sh
```

Via npm (only recommended for windows users):

```bash
npm install -g @yuanchuan/aivo
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

## Commands

| Command | Description |
| ------- | ----------- |
| [run](#run) | Launch an AI tool (claude, codex, gemini, opencode, pi) |
| [chat](#chat) | Interactive chat TUI or one-shot `-x` mode |
| [serve](#serve) | Local OpenAI-compatible API server |
| [keys](#keys) | Manage API keys (add, use, rm, cat, edit, ping) |
| [models](#models) | List available models from the active provider |
| [alias](#alias) | Create short names for models |
| [ls](#ls) | Show system info, keys, tools, and directory state |
| [stats](#stats) | Show usage statistics |
| [update](#update) | Update to the latest version |

## run

Launch an AI tool with the active provider key. All extra arguments are passed through to the underlying tool.

Supported tools:

- `claude` [Claude Code](https://github.com/anthropics/claude-code)
- `codex` [Codex](https://github.com/openai/codex)
- `gemini` [Gemini CLI](https://github.com/google-gemini/gemini-cli)
- `opencode` [OpenCode](https://github.com/anomalyco/opencode)
- `pi` [Pi Coding Agent](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent)

The `run` keyword is optional — tool names work directly as shortcuts, so `aivo claude` is equivalent to `aivo run claude`.

```bash
aivo run claude
aivo claude --dangerously-skip-permissions
aivo claude --resume 16354407-050e-4447-a068-4db222ff841
```

#### `--model, -m`

Pick a model for one run, or omit the value to open the model picker:

```bash
aivo claude --model moonshotai/kimi-k2.5
aivo claude --model                      # opens model picker
aivo claude -m                           # short form
```

#### `--key, -k`

Use a different saved key without changing the active one:

```bash
aivo claude --key openrouter
aivo claude --key copilot
aivo claude --key                        # opens key picker
```

#### `--dry-run`

Preview the resolved command and environment without launching:

```bash
aivo claude --dry-run
```

#### `--env, -e`

Inject extra environment variables into the child process:

```bash
aivo claude --env BASH_DEFAULT_TIMEOUT_MS=60000
```

#### `aivo run`

Without a tool name, `aivo run` uses the interactive start flow, which remembers your key + tool selection per directory,
so next time you run `aivo run` in the same directory, it will skip the selection step and go straight to launching the tool.

```bash
aivo run
```

## chat

`aivo chat` starts the full-screen chat UI.

```bash
aivo chat
```

#### `--model, -m`

Specify or change the chat model. Omit the value to open the model picker. The selected model is remembered per saved key.

```bash
aivo chat --model gpt-4o
aivo chat -m claude-sonnet-4-5
aivo chat --model                        # opens model picker
```

#### `--key, -k`

Use a different saved key for this chat session:

```bash
aivo chat --key openrouter
aivo chat -k                             # opens key picker
```

#### `--execute, -x`

Send a single prompt and exit. When `-x` has a message, piped stdin is appended as context. When `-x` has no message, the entire stdin becomes the prompt.

```bash
aivo chat -x "Summarize this repository"
git diff | aivo -x "Write a one-line commit message"
cat error.log | aivo -x
aivo -x                                 # type interactively, Ctrl-D to send
```

`aivo -x` is a shortcut for `aivo chat -x`.

#### `--attach`

Attach text files or images to the next message (repeatable):

```bash
aivo chat --attach README.md --attach screenshot.png
```

## serve

`aivo serve` exposes the active provider as a local OpenAI-compatible endpoint. Handy for scripts and tools that already speak the OpenAI API.

```bash
aivo serve                               # http://127.0.0.1:24860
```

#### `--port, -p`

Listen on a custom port (default: 24860):

```bash
aivo serve --port 8080
aivo serve -p 8080
```

#### `--host`

Bind to a specific address (default: 127.0.0.1):

```bash
aivo serve --host 0.0.0.0               # expose on all interfaces
```

#### `--key, -k`

Use a different saved key:

```bash
aivo serve --key openrouter
aivo serve -k                            # opens key picker
```

#### `--log`

Enable request logging. Logs to stdout by default, or to a file if a path is given:

```bash
aivo serve --log | jq .                  # JSONL to stdout
aivo serve --log /tmp/requests.jsonl     # JSONL to file
```

#### `--failover`

Enable multi-key failover on 429/5xx errors. Automatically retries with other saved keys:

```bash
aivo serve --failover
```

#### `--cors`

Enable CORS headers for browser-based clients:

```bash
aivo serve --cors
```

#### `--timeout`

Upstream request timeout in seconds (default: 300, 0 = no timeout):

```bash
aivo serve --timeout 60
```

#### `--auth-token`

Require a bearer token. Auto-generated if no value given:

```bash
aivo serve --auth-token                  # auto-generated token
aivo serve --auth-token my-secret        # specific token
```

## keys

Manage saved API keys. Keys are stored locally and encrypted in the user config directory.

```bash
aivo keys                                # list all keys
```

#### `keys add`

Add a new provider key. Interactive by default, or pass `--name`, `--base-url`, and `--key` for scripted setup:

```bash
aivo keys add
aivo keys add --name openrouter --base-url https://openrouter.ai/api/v1 --key sk-xxx
aivo keys add --name groq --base-url https://api.groq.com/openai/v1 --key sk-xxx
aivo keys add --name deepseek --base-url https://api.deepseek.com/v1 --key sk-xxx
```

Any endpoint that speaks a supported protocol can be saved — you are not limited to the providers above.

Two special names skip the base-url/key prompts:

- **`copilot`** — uses your GitHub Copilot subscription via OAuth device flow
- **`ollama`** — connects to a local Ollama instance (auto-starts if needed)

```bash
aivo keys add copilot
aivo keys add ollama
```

#### `keys use`

Switch the active key by name or ID:

```bash
aivo keys use openrouter
aivo keys use                            # opens key picker
aivo use openrouter                      # shortcut
```

#### `keys cat`

Print the decrypted key details:

```bash
aivo keys cat
aivo keys cat openrouter
```

#### `keys edit`

Edit a saved key interactively:

```bash
aivo keys edit
aivo keys edit openrouter
```

#### `keys rm`

Remove a saved key:

```bash
aivo keys rm openrouter
```

#### `keys ping`

Health-check the active key, or all keys:

```bash
aivo keys ping
aivo keys ping --all
aivo ping                                # shortcut
```

## models

List models available from the active provider. Model lists are cached for one hour.

```bash
aivo models
```

#### `--refresh, -r`

Bypass the cache and fetch a fresh model list:

```bash
aivo models --refresh
```

#### `--key, -k`

List models for a different saved key:

```bash
aivo models --key openrouter
```

#### `--search, -s`

Filter models by substring:

```bash
aivo models -s sonnet
```

## alias

Create short names for models. Aliases work anywhere a model name is accepted.

```bash
aivo alias                               # list all aliases
```

#### Create an alias

```bash
aivo alias fast=claude-haiku-4-5
aivo alias best claude-sonnet-4-6        # alternative syntax
```

Then use it in place of the full model name:

```bash
aivo claude -m fast
aivo chat -m best
```

#### Remove an alias

```bash
aivo alias rm fast
```

## ls

Show a compact overview of saved keys, installed tools, the remembered tool/model for the current directory, and the cached model count for the active key.

```bash
aivo ls
```

#### `--ping`

Also health-check all keys:

```bash
aivo ls --ping
```

## stats

Show usage statistics: token counts, request counts, and cost breakdowns.

```bash
aivo stats
```

#### `--numbers, -n`

Show exact numbers instead of human-readable approximations:

```bash
aivo stats -n
```

#### `--search, -s`

Filter by key, model, or tool name:

```bash
aivo stats -s openrouter
```

## update

Update to the latest version. Delegates to Homebrew or npm when installed by those package managers.

```bash
aivo update
```

#### `--force`

Force update even if installed via a package manager:

```bash
aivo update --force
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
