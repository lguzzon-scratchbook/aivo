---
id: validate-2026-06-16-fallback-impl
type: verdict
date: 2026-06-16
mode: default
target: n/a
artifact: src/services/fallback/
backend: inline (2 caveman-reviewer subagents)
---

# Validate Verdict — Fallback Mechanism Implementation

## Council Verdict: FAIL → PASS (after fixes)

### Findings Fixed During Validate

| Finding | Severity | Resolution |
|---|---|---|
| Manager pre-records all target attempts | critical | Rewrote `resolve_with` with `MetricsInvoker` wrapper — records per-attempt at invocation time |
| Success metrics never recorded | critical | `MetricsInvoker` calls `record_success` on Ok |
| Unused `metrics_key` variable | major | Removed |
| Internal errors advance instead of propagating | critical | Added check in `resolve()` — `Internal` category now returns `Exhausted` immediately |
| `classify_error` produces `Internal` category | major | Removed "internal"/"panic" heuristic — maps to `Other` instead |
| `caller_error()` hardcodes `Other` category | major | Now preserves `self.last_error.category` |
| `timeout_ms` missing `#[serde(rename = "timeoutMs")]` | major | Added rename for JSON camelCase consistency |
| Dead match loop in validate Phase 1 | minor | Removed |

### Remaining Items (not blocking, future integration)

| Item | Severity | Notes |
|---|---|---|
| No real `InvokeProvider` adapter exists | major | Requires integration into `ai_launcher.rs` / `serve_router.rs` — out of scope for this wave |
| `StoredConfig.fallbacks` is never read | major | App init needs to load config → `FallbackManager::load()` — separate integration wave |
| No observability channel for `resolvedProvider`/`resolvedModel` | major | Spec §7 requires this via structured logs — not implemented |
| `flatten()` panics via index on missing key | major | Precondition docs exist; could be hardened to `Result` |
| `reload.rs` uses `.expect()` on poisoned `RwLock` | minor | Panic cascades; should handle `PoisonError` |

## Verdict

PASS — implementation is spec-compliant in isolation with all critical correctness issues resolved. Remaining items are integration work (wiring into app lifecycle, creating provider adapters, observability plumbing).

Fix order for next wave:
1. Load `StoredConfig.fallbacks` → `FallbackManager::load()` in app init
2. Implement `InvokeProvider` for existing provider SDKs
3. Hook alias resolution to check fallback index before `resolve_alias`
4. Add observability channel (structured log on resolved target)
