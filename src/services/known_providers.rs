//! Central registry of known AI providers with auto-fill base URLs.
//!
//! Provider data is embedded from `src/data/providers.json` at compile time
//! and parsed once via `LazyLock`. Used by `keys add` for name-based URL
//! auto-detection.

use std::sync::LazyLock;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct KnownProvider {
    pub id: String,
    pub name: String,
    pub base_url: String,
}

/// All known providers, ordered so that more specific ids come first
/// (e.g. "openrouter" before "openai") to avoid substring false-positives.
static KNOWN_PROVIDERS: LazyLock<Vec<KnownProvider>> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../data/providers.json"))
        .expect("embedded providers.json must be valid")
});

/// Find a provider whose id appears as a substring in the input
/// (case-insensitive). Used by `keys add` for auto-detecting base URLs from
/// key names like "my-openrouter-key".
pub fn find_by_name_substring(input: &str) -> Option<&KnownProvider> {
    KNOWN_PROVIDERS.iter().find(|p| {
        input.len() >= p.id.len()
            && input
                .as_bytes()
                .windows(p.id.len())
                .any(|w| w.eq_ignore_ascii_case(p.id.as_bytes()))
    })
}

/// Returns all known providers (for the provider picker UI).
pub fn all() -> &'static [KnownProvider] {
    &KNOWN_PROVIDERS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_by_name_substring_matches() {
        let p = find_by_name_substring("my-openrouter-key").unwrap();
        assert_eq!(p.id, "openrouter");

        let p = find_by_name_substring("work_groq").unwrap();
        assert_eq!(p.id, "groq");
    }

    #[test]
    fn find_by_name_substring_no_match() {
        assert!(find_by_name_substring("random").is_none());
        assert!(find_by_name_substring("").is_none());
    }

    #[test]
    fn all_returns_every_provider() {
        let providers = super::all();
        assert!(providers.len() >= 13, "expected at least 13 providers");
        assert!(providers.iter().any(|p| p.id == "openrouter"));
        assert!(providers.iter().any(|p| p.id == "groq"));
    }

    #[test]
    fn substring_match_case_insensitive() {
        let p = find_by_name_substring("My-OpenRouter-Key").unwrap();
        assert_eq!(p.id, "openrouter");

        let p = find_by_name_substring("GROQ_KEY").unwrap();
        assert_eq!(p.id, "groq");
    }
}
