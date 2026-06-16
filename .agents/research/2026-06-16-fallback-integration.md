---
id: research-2026-06-16-fallback-integration
type: research
date: 2026-06-16
---

# Research: Fallback Integration Points

**Scope:** App init flow, alias resolution, provider launch chain

## Integration Points

| Point | File:Line | Description |
|---|---|---|
| `resolve_model_alias` | `src/main.rs:695-708` | Entry point for all model resolution — calls `resolve_alias_or_fallback` |
| StoredConfig.fallbacks | `src/services/session_store.rs:550` | Persistent fallback definitions in config JSON |
| `from_ctx` | `src/services/session_store.rs:881` | SessionStore construction — initializes empty OnceLock |
| `fallback_manager()` | `src/services/session_store.rs:1305` | Lazy-loaded: gets_fallbacks → FallbackManager::load → cache in OnceLock |
| `resolve_alias_or_fallback()` | `src/services/session_store.rs:1340` | Plain alias → fallback check → pass-through |
| `get_fallback_targets()` | `src/services/session_store.rs:1353` | Returns flat target list for a fallback alias |
| `RunCommand::execute` | `src/commands/run.rs` | CLI launch path — receives model string |
| `resolve_model` | `src/commands/run.rs:59` | Model resolution for run — uses flag_model as-is |

## Data Flow

1. CLI args parsed → `--model <value>` passed through `extract_aivo_flags`
2. `resolve_model_alias(session_store, model)` calls `resolve_alias_or_fallback`
3. `resolve_alias_or_fallback` tries: plain alias chain → fallback check → pass-through
4. If it's a fallback alias, the model string passes through unchanged as the fallback ID
5. Downstream (AILauncher, serve) gets the fallback ID as the model name

## What Fallback Manager Provides

- `is_fallback(name)` — check if name is a registered fallback
- `resolve_targets(fallback_id)` — get flat list of `(provider, model)` to try
- `resolve_with(fallback_id, invoker, timeout)` — full resolution via InvokeProvider
- `metrics()` — access to atomic counters

## Next Integration Waves

- **InvokeProvider for AILauncher**: wrap the existing launch logic as an InvokeProvider
- **Serve mode**: hook into `serve_router.rs` model selection
- **Chat TUI**: show fallback aliases in model picker
- **Fallback command**: `aivo fallback set/list/rm` (mirroring `aivo alias`)
