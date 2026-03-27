# Changelog

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
