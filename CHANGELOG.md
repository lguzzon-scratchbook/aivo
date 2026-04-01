# Changelog

## v0.16.1

### Bug Fixes

- Fix `ETXTBSY` on Linux during self-update: close write file descriptor before smoke test

## v0.16.0

### Features

- **Google Gemini native API**: Direct support for `generativelanguage.googleapis.com` as a provider across all tools
- **Open aivo-starter model list**: aivo-starter users can now access the full model catalog
- **Enriched `aivo models`**: Show context window, max output tokens, and pricing from the provider API
- **Prompt to install missing tools**: Interactively offer to install tool binaries when not found on PATH, with cross-platform support
- **R2 download mirror**: GitHub-first binary downloads with Cloudflare R2 fallback for faster installs and updates

### Bug Fixes

- Clear `last_selection` on key add so the newly added key is actually used
- Resolve aivo-starter sentinel URL in serve router and model picker
- Normalize Pi stats token counting to include cached input tokens
- Reduce GitHub download timeout and deduplicate mirror fallback messages
- Fix mirror fallback for self-update flow

## v0.15.0

### Features

- **aivo-starter**: Zero-config provider — start using aivo without any API key setup
- **Update rollback**: Automatically roll back failed updates; added config migration tests and CI clippy gate
- **Local session logging**: SQLite-backed `aivo logs` command for browsing session history
- **Native top session view**: Opt-in `aivo stats --top` for a live session overview
- **Combined short flags**: Support Unix-style combined flags like `-xr`, `-nar`
- **Ollama lifecycle management**: Auto-stop Ollama on exit using PID-file refcount for safe concurrent instances
- **DeepSeek reasoning streaming**: Stream `reasoning_content` through routers for DeepSeek-reasoner models
- **Conditional default model option**: Only show "default model" in the picker when the selected tool supports it

### Bug Fixes

- Cap `max_tokens` for aivo-starter and DeepSeek in chat requests
- Remove last two production `unwrap()` calls for safer error handling
- Fix device auth for starter provider across all tools
- Hide default model option in chat mode since it requires a concrete model
- Support Responses API-only Copilot models (e.g. gpt-5.4) for Claude and Gemini
- Resolve tilde paths and add PDF/binary support for chat attachments
- Remove tool name from active key display, show only key and model

### Performance

- Avoid PBKDF2 decryption when displaying active selection label
- Warm model cache in background after adding API key

### Refactors

- Redesign key/model selection: per-directory → global last-selection
- Replace `sqlite3` CLI with `rusqlite` for OpenCode stats reading
- Route OpenCode through router for providers with quirks

## v0.14.5

Major update with stats aggregation, better tool support

### Improvements

- Global stats aggregation across all AI tools (Claude, Codex,
  Gemini, OpenCode, Pi).
- Mask API key input with asterisks during entry
- Show install guide when a tool is not found on PATH
- Support Pi tool with Copilot subscription
- Rename `ls` command to `info` (keep `ls` as alias)
- Embed provider registry as JSON with table-driven tests
- Remove redundant token stats recording from run tool
- Bump GitHub Actions to v5 for Node.js 24 compatibility

### Fixes

- Remove custom User-Agent headers from API requests
- Use Codex `model_provider` config to bypass `auth.json` and
  `OPENAI_BASE_URL` deprecation
- Wire `--refresh` flag through run command for model picker
  cache bypass
- Auto-strip `anthropic-beta` headers for Bedrock/Vertex providers


## v0.14.4

Stability hardening. Fixed panics from char-boundary slicing
and API response handling. Switched Linux builds to musl
targets for better portability.

## v0.14.3

Added Responses API fallback for models that require the
/v1/responses endpoint. Fixed /attach command autocomplete.

## v0.14.2

Bug fixes and CI improvements.
