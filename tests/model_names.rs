use aivo::services::model_names::{
    anthropic_native_model_name, copilot_model_name, google_native_model_name,
    infer_provider_name_from_model, normalize_claude_version, transform_model_for_openrouter,
    transform_model_for_provider,
};

// ── normalize_claude_version ──────────────────────────────────────────

#[test]
fn normalize_version_standard_models() {
    assert_eq!(
        normalize_claude_version("claude-sonnet-4-6"),
        "claude-sonnet-4.6"
    );
    assert_eq!(
        normalize_claude_version("claude-haiku-4-5"),
        "claude-haiku-4.5"
    );
    assert_eq!(
        normalize_claude_version("claude-opus-4-6"),
        "claude-opus-4.6"
    );
}

#[test]
fn normalize_version_preserves_date_suffix() {
    assert_eq!(
        normalize_claude_version("claude-haiku-4-5-20251001"),
        "claude-haiku-4-5-20251001"
    );
}

#[test]
fn normalize_version_noop_for_non_claude() {
    assert_eq!(normalize_claude_version("gpt-4o"), "gpt-4o");
    assert_eq!(normalize_claude_version("o4-mini"), "o4-mini");
}

#[test]
fn normalize_version_single_version() {
    assert_eq!(
        normalize_claude_version("claude-sonnet-4"),
        "claude-sonnet-4"
    );
}

#[test]
fn normalize_version_empty_string() {
    assert_eq!(normalize_claude_version(""), "");
}

// ── transform_model_for_openrouter ────────────────────────────────────

#[test]
fn openrouter_adds_prefix_and_dots() {
    assert_eq!(
        transform_model_for_openrouter("claude-sonnet-4-6"),
        "anthropic/claude-sonnet-4.6"
    );
}

#[test]
fn openrouter_already_prefixed_passthrough() {
    assert_eq!(
        transform_model_for_openrouter("anthropic/claude-sonnet-4.6"),
        "anthropic/claude-sonnet-4.6"
    );
}

#[test]
fn openrouter_non_claude_passthrough() {
    assert_eq!(transform_model_for_openrouter("gpt-4o"), "gpt-4o");
    assert_eq!(
        transform_model_for_openrouter("gemini-2.5-pro"),
        "gemini-2.5-pro"
    );
}

// ── transform_model_for_provider ──────────────────────────────────────

#[test]
fn provider_transform_openrouter_url() {
    assert_eq!(
        transform_model_for_provider("https://openrouter.ai/api/v1", "claude-sonnet-4-6"),
        "anthropic/claude-sonnet-4.6"
    );
}

#[test]
fn provider_transform_non_openrouter_noop() {
    assert_eq!(
        transform_model_for_provider("https://api.anthropic.com/v1", "claude-sonnet-4-6"),
        "claude-sonnet-4-6"
    );
}

// ── copilot_model_name ────────────────────────────────────────────────

#[test]
fn copilot_strips_date_and_converts_version() {
    assert_eq!(
        copilot_model_name("claude-sonnet-4-6-20250603"),
        "claude-sonnet-4.6"
    );
    assert_eq!(
        copilot_model_name("claude-haiku-4-5-20250501"),
        "claude-haiku-4.5"
    );
}

#[test]
fn copilot_converts_without_date() {
    assert_eq!(copilot_model_name("claude-sonnet-4-6"), "claude-sonnet-4.6");
}

#[test]
fn copilot_single_version_no_change() {
    assert_eq!(copilot_model_name("claude-sonnet-4"), "claude-sonnet-4");
}

#[test]
fn copilot_non_claude_passthrough() {
    assert_eq!(copilot_model_name("gpt-4o"), "gpt-4o");
}

// ── google_native_model_name ──────────────────────────────────────────

#[test]
fn google_strips_prefix() {
    assert_eq!(
        google_native_model_name("google/gemini-2.5-pro"),
        "gemini-2.5-pro"
    );
}

#[test]
fn google_no_prefix_passthrough() {
    assert_eq!(google_native_model_name("gemini-2.5-pro"), "gemini-2.5-pro");
}

// ── anthropic_native_model_name ───────────────────────────────────────

#[test]
fn anthropic_native_strips_prefix_and_converts_dots() {
    assert_eq!(
        anthropic_native_model_name("anthropic/claude-sonnet-4.6"),
        "claude-sonnet-4-6"
    );
}

#[test]
fn anthropic_native_dot_with_date() {
    assert_eq!(
        anthropic_native_model_name("claude-haiku-4.5-20251001"),
        "claude-haiku-4-5-20251001"
    );
}

#[test]
fn anthropic_native_non_claude_passthrough() {
    assert_eq!(anthropic_native_model_name("MiniMax-M1"), "MiniMax-M1");
}

#[test]
fn anthropic_native_already_hyphenated() {
    assert_eq!(
        anthropic_native_model_name("claude-sonnet-4-6"),
        "claude-sonnet-4-6"
    );
}

// ── infer_provider_name_from_model ────────────────────────────────────

#[test]
fn infer_provider_claude() {
    assert_eq!(
        infer_provider_name_from_model("claude-sonnet-4-6").as_deref(),
        Some("anthropic")
    );
}

#[test]
fn infer_provider_explicit_prefix() {
    assert_eq!(
        infer_provider_name_from_model("moonshot/kimi-k2.5").as_deref(),
        Some("moonshot")
    );
}

#[test]
fn infer_provider_gemini() {
    assert_eq!(
        infer_provider_name_from_model("gemini-2.5-pro").as_deref(),
        Some("google")
    );
}

#[test]
fn infer_provider_openai() {
    assert_eq!(
        infer_provider_name_from_model("gpt-4o").as_deref(),
        Some("openai")
    );
}

#[test]
fn infer_provider_empty() {
    assert_eq!(infer_provider_name_from_model(""), None);
}

#[test]
fn infer_provider_unknown() {
    assert_eq!(infer_provider_name_from_model("llama-3.1-70b"), None);
}
