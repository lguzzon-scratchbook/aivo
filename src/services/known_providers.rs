//! Central registry of known AI providers with auto-fill base URLs.
//!
//! A static `&[KnownProvider]` compiled into the binary.
//! Used by `keys add` for name-based URL auto-detection.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KnownProvider {
    pub name: &'static str,
    pub base_url: &'static str,
}

/// All known providers, ordered so that more specific names come first
/// (e.g. "openrouter" before "openai") to avoid substring false-positives.
static KNOWN_PROVIDERS: &[KnownProvider] = &[
    KnownProvider {
        name: "openrouter",
        base_url: "https://openrouter.ai/api/v1",
    },
    KnownProvider {
        name: "vercel",
        base_url: "https://ai-gateway.vercel.sh/v1",
    },
    KnownProvider {
        name: "fireworks",
        base_url: "https://api.fireworks.ai/inference/v1",
    },
    KnownProvider {
        name: "minimax",
        base_url: "https://api.minimax.io/anthropic",
    },
    KnownProvider {
        name: "deepseek",
        base_url: "https://api.deepseek.com/v1",
    },
    KnownProvider {
        name: "moonshot",
        base_url: "https://api.moonshot.ai/v1",
    },
    KnownProvider {
        name: "anthropic",
        base_url: "https://api.anthropic.com",
    },
    KnownProvider {
        name: "openai",
        base_url: "https://api.openai.com",
    },
    KnownProvider {
        name: "qwen",
        base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    KnownProvider {
        name: "zai",
        base_url: "https://api.z.ai/v1",
    },
    KnownProvider {
        name: "groq",
        base_url: "https://api.groq.com/openai/v1",
    },
    KnownProvider {
        name: "xai",
        base_url: "https://api.x.ai/v1",
    },
    KnownProvider {
        name: "mistral",
        base_url: "https://api.mistral.ai/v1",
    },
];

/// Find a provider whose name appears as a substring in the input
/// (case-insensitive). Used by `keys add` for auto-detecting base URLs from
/// key names like "my-openrouter-key".
pub fn find_by_name_substring(input: &str) -> Option<&'static KnownProvider> {
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
