//! Central registry of known AI providers with auto-fill base URLs.
//!
//! Provider data is embedded from `src/data/providers.json` at compile time
//! and parsed once via `LazyLock`. Used by `keys add` for name-based URL
//! auto-detection.

use std::sync::LazyLock;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct KnownProvider {
    pub name: String,
    pub base_url: String,
}

/// All known providers, ordered so that more specific names come first
/// (e.g. "openrouter" before "openai") to avoid substring false-positives.
static KNOWN_PROVIDERS: LazyLock<Vec<KnownProvider>> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../data/providers.json"))
        .expect("embedded providers.json must be valid")
});

/// Find a provider whose name appears as a substring in the input
/// (case-insensitive). Used by `keys add` for auto-detecting base URLs from
/// key names like "my-openrouter-key".
pub fn find_by_name_substring(input: &str) -> Option<&KnownProvider> {
    KNOWN_PROVIDERS.iter().find(|p| {
        input.len() >= p.name.len()
            && input
                .as_bytes()
                .windows(p.name.len())
                .any(|w| w.eq_ignore_ascii_case(p.name.as_bytes()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_by_name_substring_matches() {
        let p = find_by_name_substring("my-openrouter-key").unwrap();
        assert_eq!(p.name, "openrouter");

        let p = find_by_name_substring("work_groq").unwrap();
        assert_eq!(p.name, "groq");
    }

    #[test]
    fn find_by_name_substring_no_match() {
        assert!(find_by_name_substring("random").is_none());
        assert!(find_by_name_substring("").is_none());
    }

    #[test]
    fn preserves_original_detect_base_url_behavior() {
        let cases = [
            ("openrouter", "https://openrouter.ai/api/v1"),
            ("vercel", "https://ai-gateway.vercel.sh/v1"),
            ("fireworks", "https://api.fireworks.ai/inference/v1"),
            ("minimax", "https://api.minimax.io/anthropic"),
            ("deepseek", "https://api.deepseek.com/v1"),
            ("moonshot", "https://api.moonshot.ai/v1"),
            ("anthropic", "https://api.anthropic.com"),
            ("openai", "https://api.openai.com"),
            ("qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            ("zai", "https://api.z.ai/v1"),
            ("groq", "https://api.groq.com/openai/v1"),
            ("xai", "https://api.x.ai/v1"),
            ("mistral", "https://api.mistral.ai/v1"),
        ];
        for (name, expected_url) in cases {
            let p = find_by_name_substring(name)
                .unwrap_or_else(|| panic!("should find provider for '{}'", name));
            assert_eq!(p.base_url, expected_url, "mismatch for '{}'", name);
        }
    }

    #[test]
    fn substring_match_case_insensitive() {
        let p = find_by_name_substring("My-OpenRouter-Key").unwrap();
        assert_eq!(p.name, "openrouter");

        let p = find_by_name_substring("GROQ_KEY").unwrap();
        assert_eq!(p.name, "groq");
    }
}
