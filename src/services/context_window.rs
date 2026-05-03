//! Static fallback for context-window detection.
//!
//! Used by `aivo run claude` to auto-inject `--1m` / `--2m` when the
//! cached metadata from `aivo models` is missing (e.g. the user has
//! never warmed the cache, or the upstream `/v1/models` endpoint
//! doesn't expose a `context_length` field — Anthropic's doesn't).
//!
//! Source: <https://models.dev/api.json>. Patterns are matched as
//! case-insensitive substrings of the resolved model id, longest-first,
//! so `grok-4-fast-non-reasoning` wins over `grok-4-fast`, and
//! `anthropic/claude-sonnet-4-6` matches `claude-sonnet-4-6`.

/// Returns the published context window (in tokens) for a model id, or
/// `None` if the id doesn't match any known long-context model.
pub fn static_context_window(model: &str) -> Option<u64> {
    let needle = model.to_ascii_lowercase();
    KNOWN_LONG_CONTEXT
        .iter()
        .find(|(pat, _)| needle.contains(pat))
        .map(|(_, ctx)| *ctx)
}

/// `(substring_pattern, context_tokens)`. **Sorted longest-first** so
/// substring matching picks the most specific entry. Keep this invariant
/// when editing — `debug_assert_sorted_longest_first` enforces it under
/// `cargo test`.
const KNOWN_LONG_CONTEXT: &[(&str, u64)] = &[
    // ── 10M context ────────────────────────────────────────────────────
    ("llama-4-scout", 10_000_000),
    // ── 2M context ─────────────────────────────────────────────────────
    ("grok-4-20-beta-0309-non-reasoning", 2_000_000),
    ("grok-4.20-beta-0309-non-reasoning", 2_000_000),
    ("grok-4.20-multi-agent-beta-0309", 2_000_000),
    ("grok-4-20-beta-0309-reasoning", 2_000_000),
    ("grok-4.20-beta-0309-reasoning", 2_000_000),
    ("grok-4.20-0309-non-reasoning", 2_000_000),
    ("grok-4.20-non-reasoning-beta", 2_000_000),
    ("grok-4-1-fast-non-reasoning", 2_000_000),
    ("grok-4.1-fast-non-reasoning", 2_000_000),
    ("grok-4.2-fast-non-reasoning", 2_000_000),
    ("grok-4.20-multi-agent-beta", 2_000_000),
    ("grok-4-fast-non-reasoning", 2_000_000),
    ("grok-4.20-0309-reasoning", 2_000_000),
    ("grok-4.20-reasoning-beta", 2_000_000),
    ("grok-4-1-fast-reasoning", 2_000_000),
    ("grok-4.1-fast-reasoning", 2_000_000),
    ("grok-4.2-fast-reasoning", 2_000_000),
    ("grok-4.20-non-reasoning", 2_000_000),
    ("gemini-2.0-flash-lite", 2_000_000),
    ("grok-4-20-multi-agent", 2_000_000),
    ("grok-4-fast-reasoning", 2_000_000),
    ("grok-4.20-multi-agent", 2_000_000),
    ("grok-4.20-reasoning", 2_000_000),
    ("gemini-2.0-pro-exp", 2_000_000),
    ("gemini-flash-1.5", 2_000_000),
    ("qwen3-coder-next", 2_000_000),
    ("gemini-exp-1206", 2_000_000),
    ("grok-4.20-beta", 2_000_000),
    ("grok-4-1-fast", 2_000_000),
    ("grok-4.1-fast", 2_000_000),
    ("grok-4.2-fast", 2_000_000),
    ("grok-4-fast", 2_000_000),
    ("grok-4-20", 2_000_000),
    ("grok-4.20", 2_000_000),
    // ── 1M context ─────────────────────────────────────────────────────
    ("claude-sonnet-4-5-20250929", 1_000_000),
    ("claude-opus-4-6-thinking", 1_000_000),
    ("claude-sonnet-4-thinking", 1_000_000),
    ("gemini-flash-lite-latest", 1_000_000),
    ("gemini-2.5-flash-lite", 1_000_000),
    ("gemini-3.1-flash-lite", 1_000_000),
    ("nemotron-3-super-120b", 1_000_000),
    ("claude-sonnet-latest", 1_000_000),
    ("gemini-2.0-flash-001", 1_000_000),
    ("gemini-1.5-flash-8b", 1_000_000),
    ("gemini-flash-latest", 1_000_000),
    ("claude-opus-latest", 1_000_000),
    ("qwen-deep-research", 1_000_000),
    ("claude-sonnet-4-5", 1_000_000),
    ("claude-sonnet-4-6", 1_000_000),
    ("claude-sonnet-4.5", 1_000_000),
    ("claude-sonnet-4.6", 1_000_000),
    ("deepseek-reasoner", 1_000_000),
    ("deepseek-v4-flash", 1_000_000),
    ("gemini-pro-latest", 1_000_000),
    ("qwen3-coder-flash", 1_000_000),
    ("gemini-1.5-flash", 1_000_000),
    ("gemini-2.0-flash", 1_000_000),
    ("gemini-2.5-flash", 1_000_000),
    ("llama-4-maverick", 1_000_000),
    ("qwen3-coder-plus", 1_000_000),
    ("claude-opus-4-6", 1_000_000),
    ("claude-opus-4-7", 1_000_000),
    ("claude-opus-4.6", 1_000_000),
    ("claude-opus-4.7", 1_000_000),
    ("claude-sonnet-4", 1_000_000),
    ("deepseek-v4-pro", 1_000_000),
    ("minimax-text-01", 1_000_000),
    ("nemotron-3-nano", 1_000_000),
    ("gemini-1.5-pro", 1_000_000),
    ("gemini-2.5-pro", 1_000_000),
    ("gemini-3-flash", 1_000_000),
    ("gemini-3.1-pro", 1_000_000),
    ("qwen3-vl-flash", 1_000_000),
    ("deepseek-chat", 1_000_000),
    ("mimo-v2.5-pro", 1_000_000),
    ("qwen3.5-flash", 1_000_000),
    ("qwen3.6-flash", 1_000_000),
    ("gemini-3-pro", 1_000_000),
    ("gpt-4.1-mini", 1_000_000),
    ("gpt-4.1-nano", 1_000_000),
    ("minimax-m2.1", 1_000_000),
    ("minimax-m2.5", 1_000_000),
    ("nova-premier", 1_000_000),
    ("qwen3.5-plus", 1_000_000),
    ("qwen3.6-plus", 1_000_000),
    ("gpt-5.4-pro", 1_000_000),
    ("gpt-5.5-pro", 1_000_000),
    ("mimo-v2-pro", 1_000_000),
    ("glm-4-long", 1_000_000),
    ("minimax-01", 1_000_000),
    ("minimax-m1", 1_000_000),
    ("minimax-m2", 1_000_000),
    ("palmyra-x5", 1_000_000),
    ("qwen-flash", 1_000_000),
    ("qwen-turbo", 1_000_000),
    ("mimo-v2.5", 1_000_000),
    ("qwen-long", 1_000_000),
    ("qwen-plus", 1_000_000),
    ("grok-4-3", 1_000_000),
    ("grok-4.3", 1_000_000),
    ("gpt-4.1", 1_000_000),
    ("gpt-5.1", 1_000_000),
    ("gpt-5.4", 1_000_000),
    ("gpt-5.5", 1_000_000),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_assert_sorted_longest_first() {
        // Within a single context tier, patterns must be sorted longest-first
        // so the `find` in `static_context_window` picks the most specific
        // match. Across tiers it's fine if a shorter 1M pattern appears after
        // a longer 2M pattern — they're disjoint families.
        let mut prev_ctx: u64 = u64::MAX;
        let mut prev_len_in_tier: usize = usize::MAX;
        for (pat, ctx) in KNOWN_LONG_CONTEXT {
            if *ctx != prev_ctx {
                prev_ctx = *ctx;
                prev_len_in_tier = usize::MAX;
            }
            assert!(
                pat.len() <= prev_len_in_tier,
                "KNOWN_LONG_CONTEXT not sorted longest-first within tier {ctx}: \
                 {pat:?} (len {}) follows length {prev_len_in_tier}",
                pat.len(),
            );
            prev_len_in_tier = pat.len();
        }
    }

    #[test]
    fn claude_sonnet_4_6_resolves_to_1m() {
        assert_eq!(static_context_window("claude-sonnet-4-6"), Some(1_000_000));
    }

    #[test]
    fn claude_opus_4_7_resolves_to_1m() {
        assert_eq!(static_context_window("claude-opus-4-7"), Some(1_000_000));
    }

    #[test]
    fn provider_prefixed_model_still_matches() {
        assert_eq!(
            static_context_window("anthropic/claude-opus-4-7"),
            Some(1_000_000)
        );
        assert_eq!(
            static_context_window("us.anthropic.claude-sonnet-4-6"),
            Some(1_000_000)
        );
        assert_eq!(
            static_context_window("openrouter/anthropic/claude-sonnet-4.6"),
            Some(1_000_000)
        );
    }

    #[test]
    fn case_insensitive_match() {
        assert_eq!(static_context_window("Claude-Sonnet-4-6"), Some(1_000_000));
    }

    #[test]
    fn grok_4_3_resolves_to_1m() {
        // The example the user called out.
        assert_eq!(static_context_window("grok-4.3"), Some(1_000_000));
        assert_eq!(static_context_window("xai/grok-4.3"), Some(1_000_000));
    }

    #[test]
    fn grok_4_fast_resolves_to_2m() {
        assert_eq!(static_context_window("grok-4-fast"), Some(2_000_000));
        assert_eq!(static_context_window("x-ai/grok-4-fast"), Some(2_000_000));
        assert_eq!(
            static_context_window("grok-4-fast-reasoning"),
            Some(2_000_000)
        );
    }

    #[test]
    fn gpt_4_1_resolves_to_1m() {
        assert_eq!(static_context_window("gpt-4.1"), Some(1_000_000));
        assert_eq!(
            static_context_window("openai/gpt-4.1-mini"),
            Some(1_000_000)
        );
    }

    #[test]
    fn gemini_long_context_families_match() {
        assert_eq!(static_context_window("gemini-2.5-pro"), Some(1_000_000));
        assert_eq!(
            static_context_window("gemini-2.0-flash-lite"),
            Some(2_000_000)
        );
    }

    #[test]
    fn unknown_models_return_none() {
        assert_eq!(static_context_window("gpt-3.5"), None);
        assert_eq!(static_context_window("claude-3-haiku"), None);
        assert_eq!(static_context_window(""), None);
    }
}
