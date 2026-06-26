# CLAUDE.md

## Project Overview

`aivo` is a Rust CLI tool providing unified access to multiple AI coding assistants (Claude, Codex, Gemini, OpenCode, Pi) with local API key management and secure storage. Supports OpenAI-compatible providers, GitHub Copilot, OpenRouter, Ollama, and native APIs.

> [!IMPORTANT]
> **Rebuild before testing**: After code changes, always run `cargo build && cargo install --path . --debug` before testing the binary. Never test a stale build. Only use `--release` for final testing before a release.

## Build & Test

```bash
cargo build                             # Debug build (~1s incremental)
cargo test --features __internal_test_fast_crypto  # All tests (~1900, fast crypto)
cargo test -- test_name                 # Single test
cargo clippy                            # Lint (fix all warnings before committing)
cargo fmt                               # Format (run before committing)
```

`__internal_test_fast_crypto` uses reduced PBKDF2 iterations. A `Makefile` wraps common workflows: `make test`, `make build`, `make clippy`, `make install`, `make release`.

## Git Conventions

- Always squash merge to main: `git merge --squash <branch> && git commit`
- Do not commit automatically to the fix.

## Release Process

> [!IMPORTANT]
> **Tag only after CI is green on main.** `ci.yml` runs the test matrix on every `main` push. Tagging before tests pass burns the version number — a failed release can't be re-cut on the same tag, and any `chore: release vX.Y.Z` commit becomes a zombie. Push main, wait for the green check, then tag.

1. Bump version in `Cargo.toml` and `npm/package.json` first — never tag without updating.
2. Run `cargo fmt`, `cargo clippy -- -D warnings`, `cargo test`.
3. `cargo build --release && cargo install --path .` to verify.
4. Commit: `git add -A && git commit -m "chore: release vX.Y.Z"` and `git push origin main`.
5. Wait for the CI workflow on the release commit to pass on **all three runners** (Linux, macOS, Windows). `#[cfg(windows)]` code is invisible to Linux/macOS clippy; Windows-only lint failures only surface on the Windows runner. Use `gh run watch $(gh run list --workflow=ci.yml --branch=main --limit=1 --json databaseId --jq '.[0].databaseId') --exit-status`.
6. Tag and push: `git tag vX.Y.Z && git push origin vX.Y.Z` (this triggers `release.yml`).

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


## AgentOps Knowledge Flywheel

Knowledge compounds automatically across sessions:

- **MEMORY.md** is auto-loaded by your AI coding tool every session
- **Session hooks** extract learnings, update MEMORY.md, and prune stale knowledge
- **Skills** invoke flywheel commands at the right moments (no manual ao commands needed)

Verify the flywheel any time:

```bash
ao flywheel status    # escape velocity check
ao status             # current knowledge inventory
```


<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:7510c1e2 -->
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

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
