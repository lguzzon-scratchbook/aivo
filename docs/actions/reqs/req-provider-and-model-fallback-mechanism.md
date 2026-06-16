# Fallback Mechanism for Provider/Model Selection

**Engineering Specification — Final Reviewed Version**

---

## 1. Purpose and Core Abstraction

A **fallback** is a named, first-class model alias that resolves to an **ordered list of provider/model targets**. Selecting a fallback is indistinguishable from selecting a concrete model from the caller's perspective; all resolution logic is internal and transparent to the caller.

A fallback exists to maximize availability by attempting multiple targets in priority order until one succeeds or all fail.

---

## 2. Entry Types and Data Model

A fallback sequence consists of exactly two entry types:

| Type | Description |
|---|---|
| `ProviderModelPair` | A concrete `{provider, model}` invocation target |
| `FallbackReference` | A reference to another named fallback |

A `FallbackReference` must not embed provider or model information. It is strictly an indirection mechanism. The union is closed; no third entry type is permitted. Validation MUST reject any entry whose type is not one of the two defined forms. Enforcement uses runtime type discrimination on each entry — an entry that is neither a `ProviderModelPair` nor a `FallbackReference` is rejected.

```
TYPES:

  Entry =
    | ProviderModelPair  { provider: string, model: string }
    | FallbackReference  { fallbackId: string }

  FallbackDefinition {
    id:          string        -- unique identity
    description: string?       -- optional human label
    timeoutMs:   Integer?      -- optional cumulative timeout for full resolution (see §5.2)
    sequence:    List<Entry>   -- MUST be non-empty
  }

  Registry = Map<fallbackId → FallbackDefinition>
```

---

## 3. Configuration Requirements

### 3.1 Validation Error Strategy

All validation errors across all phases **must be collected before rejection**. The registry is rejected once with the full set of accumulated errors. Fail-fast on first error is explicitly disallowed.

> **Rationale:** Operators correcting misconfigured registries benefit from a complete error report. A single reload cycle should surface all problems simultaneously.

---

## 4. Load-Time Validation

At configuration load or reload, validation executes in four sequential phases. All phases must complete before any fallback becomes active.

```
PROCEDURE ValidateRegistry(registry: Registry):

  errors ← empty List

  -- PHASE 1: Schema, entry type discrimination, and non-empty sequences
  FOR EACH fallback IN registry.values():
    IF fallback.sequence IS EMPTY:
      APPEND "Fallback '{fallback.id}' has an empty sequence" TO errors
    FOR EACH entry IN fallback.sequence:
      IF entry IS NOT ProviderModelPair AND entry IS NOT FallbackReference:
        APPEND "Fallback '{fallback.id}' contains an entry of unknown type" TO errors

  -- PHASE 2: Reference existence
  FOR EACH fallback IN registry.values():
    FOR EACH entry IN fallback.sequence:
      IF entry IS FallbackReference:
        IF entry.fallbackId NOT IN registry:
          APPEND "Fallback '{fallback.id}' references unknown id
                  '{entry.fallbackId}'" TO errors

  -- PHASE 3: Cycle detection via iterative DFS
  -- Two-phase stack entries: (nodeId, childrenProcessed).
  -- First visit → mark onStack, push children; second visit → unmark, mark visited.
  visited ← empty Set     -- nodes fully processed; proven acyclic

  PROCEDURE DetectCycles():
    FOR EACH fallbackId IN registry.keys():
      IF fallbackId IN visited:
        CONTINUE
      onStack ← empty Set           -- nodes in current traversal path
      stack  ← List{(fallbackId, False)}

      WHILE stack IS NOT EMPTY:
        (currentId, processed) ← POP(stack)

        IF processed:
          REMOVE currentId FROM onStack
          MARK currentId IN visited

        ELSE IF currentId IN visited:
          CONTINUE                   -- already proven acyclic

        ELSE IF currentId IN onStack:
          APPEND "Cycle detected involving '{currentId}'" TO errors
          -- Child is already on this traversal path → cycle found
          -- Do NOT unmark currentId; it belongs to the ancestor frame

        ELSE:
          MARK currentId IN onStack
          PUSH (currentId, True) ONTO stack   -- post-children cleanup
          -- Guard: Phase 2 may have recorded dangling references; skip
          -- unknown fallbacks to avoid null-ref on traversal
          IF currentId NOT IN registry:
            CONTINUE
          -- Push children in reverse order to preserve definition order
          children ← COLLECT entry.fallbackId
                     FROM entry IN registry[currentId].sequence
                     WHERE entry IS FallbackReference
          FOR EACH child IN REVERSE(children):
            PUSH (child, False) ONTO stack

  DetectCycles()

  -- PHASE 4: Atomic rejection or acceptance
  IF errors IS NOT EMPTY:
    REJECT WITH all collected errors   -- entire registry refused
  ELSE:
    EMIT summary diagnostic {
      fallbackCount: COUNT(registry),
      totalEntries:  SUM of sequence lengths across all fallbacks
    }
```

---

## 5. Resolution Semantics

### 5.1 Flattening

Before any request is processed, every fallback's sequence is **flattened** by recursively expanding all `FallbackReference` entries into concrete `ProviderModelPair` entries, preserving original order.

Flattening is **eager**: it executes once at load time for all registered fallbacks. The resulting flat index is **immutable** for the lifetime of that configuration instance (replaced atomically on reload, never mutated in place). Lazy flattening is not permitted.

> **Rationale:** Eager flattening eliminates all nested resolution at request time. There are no nested fallback invocations at runtime, only a flat ordered list. This makes the "no error is exempt" guarantee unambiguous and removes an entire class of runtime complexity.

```
  MAX_FLATTEN_DEPTH = 64   -- any chain deeper is rejected; prevents stack overflow

  -- Note: depth-violation THROW is caught by ReloadConfiguration's
  -- try-finally (see §8). The old config remains active. This is NOT
  -- an accumulated-validation error — it is a flattening-time guard
  -- that rejects the candidate index atomically after validation passed.

PROCEDURE Flatten(fallbackId: string, registry: Registry, depth: Integer = 0)
    → List<ProviderModelPair>:

  IF depth > MAX_FLATTEN_DEPTH:
    THROW "Flatten depth exceeded for chain involving '{fallbackId}'. "
          "Maximum nesting depth is {MAX_FLATTEN_DEPTH}. Check for "
          "unintended deep nesting or near-cycle."

  -- Pre-condition: registry has passed ValidateRegistry
  -- Termination is guaranteed; cycles were ruled out at load time

  result ← empty List

  FOR EACH entry IN registry[fallbackId].sequence:
    MATCH entry:
      CASE ProviderModelPair:
        APPEND entry TO result
      CASE FallbackReference:
        nested ← Flatten(entry.fallbackId, registry, depth + 1)
        APPEND ALL nested TO result

  RETURN result


PROCEDURE BuildFlattenedIndex(registry: Registry)
    → Map<fallbackId → List<ProviderModelPair>>:

  index ← empty Map

  FOR EACH fallbackId IN registry.keys():
    index[fallbackId] ← Flatten(fallbackId, registry)

  RETURN index   -- immutable; callers read only, never mutate
```

**Flattening invariants:**

| Invariant | Enforcement |
|---|---|
| Order strictly preserved | Sequential iteration; no sorting applied |
| No infinite recursion | Cycle detection in Phase 3 guarantees termination |
| Stack safety | `MAX_FLATTEN_DEPTH = 64` guard; exceeds → config rejected |
| Immutability | Index is built once and never modified after activation |
| Repeated entries permitted | No deduplication pass is run |

---

### 5.2 Traversal and Error Handling

Resolution proceeds sequentially against the pre-flattened target list. No error category is exempt from fallback advancement.

`InvokeProvider` is the provider-agnostic invocation boundary. It wraps the configured provider SDK or HTTP client for a given `{provider, model}` pair. Errors from any provider SDK (network, auth, rate-limit, timeout, model-not-found, internal) are surfaced uniformly as typed error objects that `Resolve`'s `ON ANY ERROR` handler can classify (see error categories below).

**`InvokeProvider` contract:** every provider-specific adapter maps its SDK's error taxonomy to the canonical error types recognized by `Resolve`. No provider error category is swallowed or converted to success.

```
PROCEDURE Resolve(fallbackId: string, flatIndex: FlattenedIndex,
                  request: Request, timeoutMs: Integer?):

  targets   ← flatIndex[fallbackId]   -- pre-flattened, immutable
  deadline  ← NOW + timeoutMs          -- only if timeoutMs is set; else null
  attempts  ← empty List              -- for exhaustion error reporting
  lastError ← null

  FOR EACH target IN targets:

    IF deadline IS NOT NULL AND NOW >= deadline:
      THROW ProviderError {
        message: "Cumulative timeout exceeded for fallback '{fallbackId}' "
                 "after {COUNT(attempts)} attempt(s)"
      }

    TRY:
      result ← InvokeProvider(target.provider, target.model, request)
      RETURN result                   -- first success wins; exit immediately

    ON ANY ERROR AS err:
      -- INTERNAL/PANIC errors signal system corruption, not provider
      -- degradation. Must NOT advance. See advancement rules below.
      IF err IS INTERNAL:
        THROW err     -- propagate immediately; circuit-break or crash
      LOG {
        fallbackId:  fallbackId,
        provider:    target.provider,
        model:       target.model,
        error:       err,
        remaining:   COUNT(targets) - INDEX(target) - 1
      }
      APPEND { provider: target.provider, model: target.model, error: err }
             TO attempts
      lastError ← err
      CONTINUE                        -- advance unconditionally

  -- Classify errors for caller-facing summary
  errorCategories ← CLASSIFY_ERRORS(attempts)

  -- All targets exhausted without success
  THROW ProviderError {
    message: "All {COUNT(attempts)} targets exhausted for fallback "
             "'{fallbackId}'. Error categories: {errorCategories}. "
             "Last error: {lastError.message}"
  }
  -- Full FallbackExhaustedError (with complete attempt history) is logged
  -- server-side for debugging. Caller receives only the summary above.
```

**Error advancement rules:**

| Error Type | Behavior | Alert |
|---|---|---|
| Rate limit | Advance to next target | — |
| Timeout | Advance to next target | — |
| Auth failure | Advance to next target | 🔔 Warning: credential/auth errors on advancement MUST trigger an operator-visible alert. Silent credential decay across targets masks provider degradation. |
| Model not found | Advance to next target | — |
| Network error | Advance to next target | — |
| Any provider error | Advance to next target | — |
| **INTERNAL / PANIC**<br/>(assertion fault, null deref, OOM, invariant violation) | **DO NOT advance** | 🚨 CRITICAL: system-level corruption. Route to crash or circuit-break. Provider advancement would mask a systemic fault. |

Provider errors are defined as errors originating from the external AI provider SDK or its transport layer. `INTERNAL`/`PANIC` errors originate from the fallback system itself or its host runtime — they are never produced by `InvokeProvider` adapters and signal a defect, not a provider degradation. The `ON ANY ERROR AS err` handler in `Resolve` MUST classify errors into these two categories before deciding whether to advance.

---

## 6. Exhaustion Contract

If resolution fails for all entries, the system logs a `FallbackExhaustedError` server-side, which includes:

- The fallback ID
- The ordered list of attempted targets, each with its associated error
- The final underlying error
- A categorized summary of error types across all attempts

The **caller-facing** error is a standard `ProviderError` with a summary — attempt count and error categories (e.g., `"3 targets exhausted: rate_limit, timeout, auth"`). The full attempt history is **logged server-side only**, not exposed to the caller. The caller-facing error type is uniform regardless of whether the alias was a fallback or a concrete model. This preserves the indistinguishability contract (§7).

---

## 7. Caller Interface Contract

Callers interact with fallbacks exactly as with concrete models. The resolution mechanism is entirely opaque.

```
PROCEDURE CallerCode():

  response ← client.complete({
    model:    "auto",       -- may be a concrete model or a fallback alias
    messages: [...]
  })

  -- Caller observes exactly two outcomes:
  --
  --   SUCCESS → response object  (no metadata about fallback resolution)
  --
  --   FAILURE → standard provider error  (fallback exhaustion or concrete
  --             model failure are indistinguishable)
  --             On exhaustion, the error includes a summary of attempt
  --             count and error categories, but not the full attempt
  --             history. This balances debuggability with transparency.
```

**Caller transparency invariants:**

- Callers do not know whether a model alias resolves to a fallback or a concrete model.
- Callers do not manage retries.
- Callers do not receive intermediate failure notifications.
- On success, the response does **not** expose which provider or model resolved the request. This maintains caller transparency — the resolution mechanism is invisible to the caller.
- On failure, the error type is uniform regardless of whether the alias is a fallback or a concrete model. Fallback exhaustion is wrapped in a standard provider error; the full attempt history is logged server-side, never exposed to the caller.

**Observability channel:** `resolvedProvider` and `resolvedModel` are exposed via a **separate observability channel** (structured logs, metric counters, or a debug endpoint — see §9.1 for the full metrics specification), never in the caller-facing response payload. This satisfies operational debuggability without breaking caller transparency.

---

## 8. Configuration Reload

Concurrent reloads are serialized. While a reload is in progress, subsequent reload
requests are queued or rejected; only one `ReloadConfiguration` executes at a time.

```
GLOBAL reloadLock ← Mutex        -- prevents concurrent reloads

PROCEDURE ReloadConfiguration(newConfig: RawConfig):

  ACQUIRE(reloadLock, timeoutMs=30_000)

  TRY:
    -- Step 1: Parse into a candidate registry
    --         Active registry is untouched at this stage
    -- Parse: RawConfig → Registry. Deserializes the serialized
    --         configuration format into the in-memory Registry
    --         data structure defined in §2. Schema validation
    --         (required fields, type correctness) happens here;
    --         semantic validation (reference existence, cycles,
    --         depth) is in Steps 2-3.
    candidateRegistry ← Parse(newConfig)

    -- Step 2: Full validation on candidate
    --         Returns all collected errors; does NOT fail-fast on first error
    --         REJECT (see §4 Phase 4) causes an early exit from this TRY
    --         block, preventing Steps 3-4 from executing. The old config
    --         remains active and the operator receives the full error report.
    ValidateRegistry(candidateRegistry)

    -- Step 3: Build flattened index for candidate
    candidateFlatIndex ← BuildFlattenedIndex(candidateRegistry)

    -- Step 4: Atomic swap
    --         MUST use RwLock<Arc<Index>> or equivalent.
    --         Concurrent reads must always see a consistent
    --         (old or new) index, never a half-constructed one.
    --         Only reached if steps 2 and 3 succeeded without error
    activeRegistry  ← candidateRegistry
    activeFlatIndex ← candidateFlatIndex

    EMIT "Configuration reloaded: {COUNT} fallbacks now active"

  FINALLY:
    -- Guaranteed release: success path (above) and failure path (steps 2/3)
    -- both drain through here
    RELEASE(reloadLock)

  -- On failure during any TRY step (parse, validate, flatten):
  --   activeRegistry  remains the previous valid configuration
  --   activeFlatIndex remains the previous valid flat index
  --   all collected errors are emitted to the operator
  --   no partial state is ever active
```

**Reload guarantees:**

- In-flight requests against the old flat index complete using the old configuration.
- New requests after the atomic swap use the new flat index exclusively.
- There is no window in which a partially validated configuration is active.
- A failed reload leaves the system in its last known good state.
- **Memory:** During reload, memory usage transiently doubles (old index + candidate index). For large registries (10k+ fallbacks), operators should size memory budget accordingly. The old index is released once all in-flight readers have drained.

---

## 9. Operational Guarantees

- Entry order is strictly preserved end to end, from definition through flattening to resolution.
- Fallback resolution is read-only against the flat index and safe for concurrent use. (`InvokeProvider` implementations that consume the resolved targets must themselves be safe for concurrent invocation. If a provider SDK has internal rate-limit state or connection pooling, the implementation must handle overlapping calls to the same provider/model pair.)
- All failures must be logged with sufficient diagnostic detail before advancing to the next target.
- Configuration reloads must complete full validation before becoming active.
- The flat index is the single source of truth at runtime; the raw registry is used only during validation and flattening.

### 9.1 Operational Metrics

The system MUST expose the following metrics per fallback target (observable via a metrics endpoint, not through caller-facing APIs):

| Metric | Type | Description |
|---|---|---|
| `fallback.target.attempts` | Counter (increment per attempt) | Total attempts per `{fallbackId, provider, model}` |
| `fallback.target.successes` | Counter | Successful resolutions per target |
| `fallback.target.errors` | Counter (tagged by error category) | Failures per target, split by `rate_limit`, `timeout`, `auth`, `network`, `other` |
| `fallback.resolution.duration_ms` | Histogram | End-to-end time per `{fallbackId, provider, model}` resolution |
| `fallback.exhaustion.count` | Counter | Total exhaustion events per `fallbackId` |
| `fallback.auth_warning.count` | Counter (separate alert trigger) | Auth/credential errors that caused advancement. MUST have an associated alert threshold. |

**Cardinality warning:** `{fallbackId, provider, model}` series multiply across all fallbacks. A registry with 10k fallbacks × 5 targets each = 50k unique metric series. Many monitoring systems degrade past ~10k series. Use aggregated dimensions (fallback-scoped totals, provider-scoped totals) for standard dashboards; full `{fallbackId, provider, model}` only for targeted debugging. Right-size cardinality budget to your observability backend.

Operators SHOULD configure alerting on:
- `fallback.auth_warning.count` exceeding a per-minute threshold (indicates credential decay)
- `fallback.target.errors` rate exceeding baseline per target (indicates provider degradation)
- `fallback.exhaustion.count` > 0 (indicates all paths degraded for a fallback)

---

## 10. Acceptance Criteria

The system is correct if and only if all of the following hold:

| Scenario | Expected Behavior |
|---|---|
| Fallback references another with empty sequence | Validation Phase 1 catches empty sequence on the referenced fallback; Phase 4 rejects registry atomically |
| Nested fallback expansion | All referenced fallbacks are recursively flattened in order at load time |
| Flatten depth limit | Nesting beyond `MAX_FLATTEN_DEPTH` causes config rejection at load time; no stack overflow |
| Cycle detection | Any direct or indirect cycle causes full registry rejection with descriptive errors |
| Cumulative timeout enforcement | `timeoutMs` is enforced across all targets; if exceeded, caller receives a provider error before exhaustion |
| Exhaustion | All targets exhausted; caller receives a standard provider error (not `FallbackExhaustedError`) with attempt count and error category summary. Full attempt history is logged server-side |
| Caller transparency | Callers cannot distinguish a fallback alias from a concrete model alias. Success response contains no `resolvedProvider` or `resolvedModel` fields |
| Observability channel | `resolvedProvider` and `resolvedModel` are exposed via metrics endpoint only, never in response payload |
| Auth failure alert | Auth/credential errors that cause advancement increment `fallback.auth_warning.count` and trigger a warning alert |
| Reload isolation | A failed reload leaves the active configuration unchanged |
| Reload concurrency safety | Concurrent reload attempts are serialized by mutex; no two reloads run simultaneously |
| Reload memory | During reload, old and new indices coexist briefly; memory budget accounts for the peak |
| Error accumulation | Validation reports all errors before rejecting, never just the first |
| No deduplication | Repeated entries in a sequence are preserved and each is attempted |
| Metrics per target | Each target has counters for attempts, successes, errors (tagged by category), and resolution duration |

---

*This document is complete as a standalone engineering specification and serves as the direct input to the Project Design Record (PDR) authoring phase. PDR: the downstream document that translates engineering specifications into implementation plans, system components, and interface contracts.*
