---
name: fallback-implementation-patterns
description: Architectural decisions for fallback mechanism in aivo
metadata:
  type: project
  date: 2026-06-16
---

# Fallback Mechanism Implementation

## Key Decisions

1. **Eager flattening at load time** rather than lazy resolution. Flat `Arc<FlattenedIndex>` is immutable for the lifetime of a config instance. Avoids nested resolution at request time.

2. **InvokeProvider trait** decouples fallback resolution from actual provider invocation. CLI mode uses `AILauncherInvoker` (spawns tokio runtime per call); serve mode uses protocol fallback loop directly.

3. **Metrics wrapper via MetricsInvoker** rather than callback-based. The invoker wraps an inner InvokeProvider and records per-attempt attempt/success/error/duration metrics transparently.

4. **OnceLock for lazy init** of FallbackManager in SessionStore. Thread-safe one-time initialization avoids loading fallback config on every request.

5. **Untagged serde enum** for Entry discrimination (ProviderModelPair vs FallbackReference). JSON shape is self-describing (provider+model vs fallbackId).

## What Worked

- Spec-first implementation: writing types from spec §2, then building up through validation → flattening → resolution → integration
- 84 unit tests in the fallback module caught bugs early (Internal advancement rule, metrics pre-recording)
- `BTreeSet` for error category deduplication in exhaustion summary
- Validate step found 7 issues that were fixed before integration

## What to Watch

- `InvokeProvider::invoke()` is sync — creating a tokio runtime per call in `AILauncherInvoker` works for CLI but wouldn't scale for high-throughput serve mode
- Serve mode fallback reimplements the protocol fallback loop inside `handle_fallback_chat` — duplicated logic with the existing loop in `handle_chat_body`
- `OnceLock` has no `reset()` — config changes during long-running serve process require restart to reload fallback definitions
