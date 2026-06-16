# CLAUDE.md

## Project Overview

`aivo` is a Rust CLI tool providing unified access to multiple AI coding assistants (Claude, Codex, Gemini, OpenCode, Pi) with local API key management and secure storage. Supports OpenAI-compatible providers, GitHub Copilot, OpenRouter, Ollama, and native APIs.

> [!IMPORTANT]
> **Rebuild before testing**: After code changes, always run `cargo build && cargo install --path . --debug` before testing the binary. Never test a stale build. Only use `--release` for final testing before a release.

## Build & Test

```bash
cargo build                             # Debug build (~1s incremental)
cargo test --features test-fast-crypto  # All tests (~1900, fast crypto)
cargo test -- test_name                 # Single test
cargo clippy                            # Lint (fix all warnings before committing)
cargo fmt                               # Format (run before committing)
```

`test-fast-crypto` uses reduced PBKDF2 iterations. A `Makefile` wraps common workflows: `make test`, `make build`, `make clippy`, `make install`, `make release`.

## Git Conventions

- Always squash merge to main: `git merge --squash <branch> && git commit`
- Do not commit automatically to the fix.

## Release Process

1. Bump version in `Cargo.toml` and `npm/package.json` first — never tag without updating.
2. Run `cargo fmt`, `cargo clippy -- -D warnings`, `cargo test`.
3. `cargo build --release && cargo install --path .` to verify.
4. Commit: `git add -A && git commit -m "chore: release vX.Y.Z"`
5. Tag and push: `git tag vX.Y.Z && git push origin main --tags`

## CLI / UX Conventions

Match existing CLI help text formatting exactly (alignment, spacing, bracket style). When implementing interactive UI, verify: keyboard handling (arrows, Ctrl+P/N, ESC, Ctrl+C), selection state pre-selection, column alignment, and edge cases (empty input, single item, long strings).

## Architecture

```
src/main.rs → SessionStore → EnvironmentInjector → AILauncher → Command Handlers
```

- **`src/`**: Entry point, CLI parsing, error handling, TUI components, styling
- **`src/commands/`**: `run` (launch tools), `start` (interactive picker), `chat` (chat TUI + one-shot), `keys`, `serve`, `info`, `models`, `alias`, `logs`, `stats`, `update`
- **`src/services/`**: Session/key/stats storage, AI process launching, provider routing/bridging (Anthropic, OpenAI, Gemini, Copilot, Ollama), model name transforms, HTTP utilities

**Data model**: `ApiKey` (`id`, `name`, `base_url`, `key`, `created_at`) stored AES-256-GCM encrypted in `~/.config/aivo/config.json`. Sentinel `base_url` values `"copilot"` and `"ollama"` identify special provider types.

**Cross-platform**: Platform-specific code gated behind `cfg(unix)` / `cfg(windows)`.

**Exit codes**: 0 = success, 1 = user error, 2 = network, 3 = auth.

## Instructions

Restate the question in fully concrete terms, making every implicit detail explicit. Then answer.



<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:6cd5cc61 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->
