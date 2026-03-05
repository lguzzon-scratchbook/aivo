//! Shared model name normalization utilities.
//!
//! Different providers expect different model name formats:
//! - OpenRouter: `anthropic/claude-sonnet-4.6` (prefix + dots)
//! - Copilot: `claude-sonnet-4.6` (dots, no prefix)
//! - Anthropic: `claude-sonnet-4-6` (hyphens)
//!
//! This module consolidates the version conversion logic that was previously
//! duplicated across claude_code_router, copilot_router, and chat.rs.

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
}
