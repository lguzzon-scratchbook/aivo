//! Shared model name normalization utilities.
//!
//! Different providers expect different model name formats:
//! - OpenRouter: `anthropic/claude-sonnet-4.6` (prefix + dots)
//! - Copilot: `claude-sonnet-4.6` (dots, no prefix)
//! - Anthropic: `claude-sonnet-4-6` (hyphens)
//!
//! This module consolidates the version conversion logic that was previously
//! duplicated across Anthropic router code, copilot_router, and chat.rs.

use crate::services::provider_protocol::ProviderProtocol;

/// Converts Claude model version separators from hyphens to dots.
///
/// Examples:
/// - `claude-sonnet-4-6` → `claude-sonnet-4.6`
/// - `claude-haiku-4-5` → `claude-haiku-4.5`
/// - `claude-haiku-4-5-20251001` → `claude-haiku-4-5-20251001` (date suffix preserved)
/// - `gpt-4o` → `gpt-4o` (non-Claude models pass through)
pub fn normalize_claude_version(model: &str) -> String {
    if let Some(last_hyphen_pos) = model.rfind('-') {
        let after_last_hyphen = &model[last_hyphen_pos + 1..];

        // Date suffix (8 digits): keep as-is
        if after_last_hyphen.len() == 8 && after_last_hyphen.chars().all(|c| c.is_ascii_digit()) {
            return model.to_string();
        }

        // Version number: convert the separating hyphen to a dot
        if after_last_hyphen
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_digit())
            && let Some(second_last_hyphen) = model[..last_hyphen_pos].rfind('-')
            && model[second_last_hyphen + 1..last_hyphen_pos]
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_digit())
        {
            let mut result = model.to_string();
            result.replace_range(last_hyphen_pos..=last_hyphen_pos, ".");
            return result;
        }
    }
    model.to_string()
}

/// Transforms a model name for OpenRouter compatibility.
/// Adds `anthropic/` prefix and normalizes version separators.
///
/// Examples:
/// - `claude-sonnet-4-6` → `anthropic/claude-sonnet-4.6`
/// - `anthropic/claude-sonnet-4.6` → `anthropic/claude-sonnet-4.6` (already prefixed)
/// - `gpt-4o` → `gpt-4o` (non-Claude models pass through)
pub fn transform_model_for_openrouter(model: &str) -> String {
    if !model.starts_with("claude-") || model.starts_with("anthropic/") {
        return model.to_string();
    }
    format!("anthropic/{}", normalize_claude_version(model))
}

/// Transforms a model name based on the provider's base URL.
/// Currently, only OpenRouter requires transformation.
pub fn transform_model_for_provider(base_url: &str, model: &str) -> String {
    if base_url.contains("openrouter") {
        transform_model_for_openrouter(model)
    } else {
        model.to_string()
    }
}

pub fn google_native_model_name(model: &str) -> &str {
    model.strip_prefix("google/").unwrap_or(model)
}

pub fn anthropic_native_model_name(model: &str) -> String {
    let stripped = model.strip_prefix("anthropic/").unwrap_or(model);
    if !stripped.starts_with("claude-") {
        return stripped.to_string();
    }

    if let Some(dot_pos) = stripped.find('.')
        && stripped[..dot_pos]
            .chars()
            .next_back()
            .is_some_and(|c| c.is_ascii_digit())
        && stripped[dot_pos + 1..]
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_digit())
    {
        let mut normalized = stripped.to_string();
        normalized.replace_range(dot_pos..=dot_pos, "-");
        return normalized;
    }

    stripped.to_string()
}

pub fn is_gateway_style_endpoint(base_url: &str) -> bool {
    let lower = base_url.trim().to_ascii_lowercase();
    lower.contains("/endpoint") || lower.contains("gateway")
}

pub fn infer_provider_name_from_model(model: &str) -> Option<String> {
    let trimmed = model.trim();
    if trimmed.is_empty() {
        return None;
    }

    if let Some((provider, _)) = trimmed.split_once('/')
        && !provider.trim().is_empty()
    {
        return Some(provider.trim().to_ascii_lowercase());
    }

    match infer_model_protocol(trimmed) {
        Some(ProviderProtocol::Anthropic) => Some("anthropic".to_string()),
        Some(ProviderProtocol::Google) => Some("google".to_string()),
        Some(ProviderProtocol::Openai) | Some(ProviderProtocol::ResponsesApi) => {
            Some("openai".to_string())
        }
        None => None,
    }
}

pub fn should_preserve_cross_protocol_model(
    base_url: &str,
    model: &str,
    target_protocol: ProviderProtocol,
) -> bool {
    match infer_model_protocol(model) {
        Some(protocol) if model_family(protocol) != model_family(target_protocol) => {
            model_family(target_protocol) == ProviderProtocol::Openai
                && is_gateway_style_endpoint(base_url)
        }
        _ => false,
    }
}

/// Converts Claude model names from Anthropic/Claude Code format to Copilot format.
///
/// Claude Code sends names like `claude-sonnet-4-6-20250603` or `claude-sonnet-4-6`.
/// Copilot API expects names like `claude-sonnet-4.6` (dots for minor versions).
///
/// Steps:
///   1. Strip trailing date suffix `-YYYYMMDD`
///   2. Convert `claude-{family}-{major}-{minor}` → `claude-{family}-{major}.{minor}`
pub fn copilot_model_name(model: &str) -> String {
    // Strip trailing -YYYYMMDD date suffix
    let base = if model.len() > 9 {
        let (prefix, suffix) = model.split_at(model.len() - 9);
        if suffix.starts_with('-') && suffix[1..].chars().all(|c| c.is_ascii_digit()) {
            prefix
        } else {
            model
        }
    } else {
        model
    };

    // Convert hyphenated version to dotted: claude-sonnet-4-6 → claude-sonnet-4.6
    // Pattern: claude-{family}-{major}-{minor} where major/minor are digits
    if let Some(stripped) = base.strip_prefix("claude-") {
        let parts: Vec<&str> = stripped.split('-').collect();
        // e.g. ["sonnet", "4", "6"] or ["sonnet", "4"] or ["haiku", "4", "5"]
        if parts.len() >= 3 {
            let family = parts[0]; // sonnet, haiku, opus
            let major = parts[1]; // "4"
            let minor = parts[2]; // "6", "5"
            if major.chars().all(|c| c.is_ascii_digit())
                && minor.chars().all(|c| c.is_ascii_digit())
            {
                // Rejoin any remaining parts (e.g. "-thinking") after the version
                let rest = if parts.len() > 3 {
                    format!("-{}", parts[3..].join("-"))
                } else {
                    String::new()
                };
                return format!("claude-{}-{}.{}{}", family, major, minor, rest);
            }
        }
    }

    base.to_string()
}

pub fn default_model_for_protocol(protocol: ProviderProtocol) -> &'static str {
    match protocol {
        ProviderProtocol::Openai | ProviderProtocol::ResponsesApi => "gpt-4o",
        ProviderProtocol::Anthropic => "claude-sonnet-4-5",
        ProviderProtocol::Google => "gemini-2.5-pro",
    }
}

pub fn select_model_for_protocol(
    requested_model: Option<&str>,
    explicit_model: Option<&str>,
    target_protocol: ProviderProtocol,
) -> String {
    if let Some(model) = explicit_model.filter(|model| !model.trim().is_empty()) {
        return model.to_string();
    }

    match requested_model.filter(|model| !model.trim().is_empty()) {
        Some(model) => match infer_model_protocol(model) {
            Some(protocol) if model_family(protocol) != model_family(target_protocol) => {
                default_model_for_protocol(target_protocol).to_string()
            }
            _ => model.to_string(),
        },
        None => default_model_for_protocol(target_protocol).to_string(),
    }
}

pub fn select_model_for_provider_attempt(
    base_url: &str,
    requested_model: Option<&str>,
    explicit_model: Option<&str>,
    target_protocol: ProviderProtocol,
) -> String {
    if let Some(model) = explicit_model.filter(|model| !model.trim().is_empty()) {
        return model.to_string();
    }

    if let Some(model) = requested_model.filter(|model| !model.trim().is_empty())
        && should_preserve_cross_protocol_model(base_url, model, target_protocol)
    {
        return model.to_string();
    }

    select_model_for_protocol(requested_model, explicit_model, target_protocol)
}

/// Normalize protocol for model comparison — ResponsesApi uses the same models as Openai.
fn model_family(p: ProviderProtocol) -> ProviderProtocol {
    match p {
        ProviderProtocol::ResponsesApi => ProviderProtocol::Openai,
        other => other,
    }
}

fn infer_model_protocol(model: &str) -> Option<ProviderProtocol> {
    let lower = model.to_ascii_lowercase();
    let name_only = lower.split('/').next_back().unwrap_or(&lower);

    if name_only.contains("claude") {
        Some(ProviderProtocol::Anthropic)
    } else if name_only.contains("gemini") {
        Some(ProviderProtocol::Google)
    } else if name_only.starts_with("gpt-")
        || name_only.starts_with("o1")
        || name_only.starts_with("o3")
        || name_only.starts_with("o4")
        || name_only.starts_with("chatgpt")
        || name_only.contains("codex")
    {
        Some(ProviderProtocol::Openai)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_claude_version_basic() {
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
    fn test_normalize_claude_version_date_suffix_preserved() {
        assert_eq!(
            normalize_claude_version("claude-haiku-4-5-20251001"),
            "claude-haiku-4-5-20251001"
        );
    }

    #[test]
    fn test_normalize_claude_version_no_change() {
        assert_eq!(normalize_claude_version("gpt-4o"), "gpt-4o");
        assert_eq!(
            normalize_claude_version("claude-sonnet-4"),
            "claude-sonnet-4"
        );
    }

    #[test]
    fn test_transform_model_for_openrouter() {
        assert_eq!(
            transform_model_for_openrouter("claude-sonnet-4-6"),
            "anthropic/claude-sonnet-4.6"
        );
        assert_eq!(
            transform_model_for_openrouter("anthropic/claude-sonnet-4.6"),
            "anthropic/claude-sonnet-4.6"
        );
        assert_eq!(transform_model_for_openrouter("gpt-4o"), "gpt-4o");
    }

    #[test]
    fn test_transform_model_for_provider() {
        assert_eq!(
            transform_model_for_provider("https://openrouter.ai/api/v1", "claude-sonnet-4-6"),
            "anthropic/claude-sonnet-4.6"
        );
        assert_eq!(
            transform_model_for_provider("https://api.example.com/v1", "claude-sonnet-4-6"),
            "claude-sonnet-4-6"
        );
    }

    #[test]
    fn test_google_native_model_name_strips_provider_prefix() {
        assert_eq!(
            google_native_model_name("google/gemini-2.5-pro"),
            "gemini-2.5-pro"
        );
        assert_eq!(google_native_model_name("gemini-2.5-pro"), "gemini-2.5-pro");
    }

    #[test]
    fn test_anthropic_native_model_name_normalizes_claude_versions() {
        assert_eq!(
            anthropic_native_model_name("anthropic/claude-sonnet-4.6"),
            "claude-sonnet-4-6"
        );
        assert_eq!(
            anthropic_native_model_name("claude-haiku-4.5-20251001"),
            "claude-haiku-4-5-20251001"
        );
        assert_eq!(anthropic_native_model_name("MiniMax-M1"), "MiniMax-M1");
    }

    #[test]
    fn test_copilot_model_name_strips_date() {
        assert_eq!(
            copilot_model_name("claude-sonnet-4-20250514"),
            "claude-sonnet-4"
        );
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
    fn test_copilot_model_name_converts_dots() {
        assert_eq!(copilot_model_name("claude-sonnet-4"), "claude-sonnet-4");
        assert_eq!(copilot_model_name("claude-sonnet-4-6"), "claude-sonnet-4.6");
        assert_eq!(copilot_model_name("gpt-4o"), "gpt-4o");
    }

    #[test]
    fn test_select_model_for_protocol_keeps_provider_native_models() {
        assert_eq!(
            select_model_for_protocol(Some("MiniMax-M1"), None, ProviderProtocol::Anthropic),
            "MiniMax-M1"
        );
        assert_eq!(
            select_model_for_protocol(
                Some("google/gemini-2.5-pro"),
                None,
                ProviderProtocol::Google
            ),
            "google/gemini-2.5-pro"
        );
    }

    #[test]
    fn test_select_model_for_protocol_remaps_cross_protocol_defaults() {
        assert_eq!(
            select_model_for_protocol(Some("gpt-5-codex"), None, ProviderProtocol::Anthropic),
            "claude-sonnet-4-5"
        );
        assert_eq!(
            select_model_for_protocol(Some("claude-sonnet-4-5"), None, ProviderProtocol::Google),
            "gemini-2.5-pro"
        );
        assert_eq!(
            select_model_for_protocol(Some("gemini-2.0-flash"), None, ProviderProtocol::Openai),
            "gpt-4o"
        );
    }

    #[test]
    fn test_should_preserve_cross_protocol_model_for_gateway_endpoint() {
        assert!(should_preserve_cross_protocol_model(
            "https://api.ai.example-gateway.net/endpoint",
            "claude-sonnet-4-6",
            ProviderProtocol::Openai
        ));
        assert!(should_preserve_cross_protocol_model(
            "http://localhost:3005/endpoint",
            "claude-sonnet-4-6",
            ProviderProtocol::Openai
        ));
        assert!(is_gateway_style_endpoint("https://ai-gateway.vercel.sh/v1"));
    }

    #[test]
    fn test_should_not_preserve_cross_protocol_model_for_plain_openai_endpoint() {
        assert!(!should_preserve_cross_protocol_model(
            "https://api.openai.com/v1",
            "claude-sonnet-4-6",
            ProviderProtocol::Openai
        ));
    }

    #[test]
    fn test_infer_provider_name_from_model() {
        assert_eq!(
            infer_provider_name_from_model("claude-sonnet-4-6").as_deref(),
            Some("anthropic")
        );
        assert_eq!(
            infer_provider_name_from_model("moonshot/kimi-k2.5").as_deref(),
            Some("moonshot")
        );
        assert_eq!(infer_provider_name_from_model("").as_deref(), None);
    }

    #[test]
    fn test_select_model_for_protocol_prefers_explicit_model() {
        assert_eq!(
            select_model_for_protocol(
                Some("gpt-5-codex"),
                Some("claude-3-opus"),
                ProviderProtocol::Anthropic
            ),
            "claude-3-opus"
        );
    }

    #[test]
    fn test_select_model_for_provider_attempt_preserves_cross_protocol_gateway_models() {
        assert_eq!(
            select_model_for_provider_attempt(
                "https://api.ai.example-gateway.net/endpoint",
                Some("claude-sonnet-4.6"),
                None,
                ProviderProtocol::Openai
            ),
            "claude-sonnet-4.6"
        );
    }

    #[test]
    fn test_select_model_for_provider_attempt_still_remaps_plain_openai_endpoints() {
        assert_eq!(
            select_model_for_provider_attempt(
                "https://api.openai.com/v1",
                Some("claude-sonnet-4.6"),
                None,
                ProviderProtocol::Openai
            ),
            "gpt-4o"
        );
    }
}
