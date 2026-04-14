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

/// Find a provider whose id or display name matches the input as a substring
/// in either direction (case-insensitive). Matches both "my-openrouter-key"
/// (id contained in input) and "google" (input contained in id or name).
pub fn find_by_name_substring(input: &str) -> Option<&KnownProvider> {
    if input.is_empty() {
        return None;
    }
    let input_lower = input.to_ascii_lowercase();
    KNOWN_PROVIDERS.iter().find(|p| {
        let id_lower = p.id.to_ascii_lowercase();
        let name_lower = p.name.to_ascii_lowercase();
        id_lower.contains(&input_lower)
            || name_lower.contains(&input_lower)
            || input_lower.contains(&id_lower)
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
