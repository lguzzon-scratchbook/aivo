use std::collections::HashMap;

use crate::services::provider_protocol::{ProviderProtocol, detect_provider_protocol};
use crate::services::session_store::ApiKey;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Copilot,
    Ollama,
    OpenRouter,
    CloudflareAi,
    AnthropicCompatible,
    GoogleNative,
    OpenAiCompatible,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelListingStrategy {
    Copilot,
    Ollama,
    Google,
    Anthropic,
    CloudflareSearch,
    OpenAiCompatible,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServeFlags {
    pub is_copilot: bool,
    pub is_openrouter: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderQuirks {
    pub model_prefix: Option<&'static str>,
    pub requires_reasoning_content: bool,
    pub max_tokens_cap: Option<u64>,
}

impl ProviderQuirks {
    pub fn for_base_url(base_url: &str) -> Self {
        let model_prefix = if cloudflare_ai_base(base_url).is_some() {
            Some("@cf/")
        } else {
            None
        };
        let requires_reasoning_content = base_url.contains("moonshot.cn")
            || base_url.contains("moonshot.ai")
            || base_url.contains("deepseek.com");
        let max_tokens_cap = if base_url.contains("deepseek.com") {
            Some(8192)
        } else {
            None
        };
        Self {
            model_prefix,
            requires_reasoning_content,
            max_tokens_cap,
        }
    }

    pub fn inject(&self, env: &mut HashMap<String, String>, prefix: &str) {
        if let Some(pfx) = self.model_prefix {
            env.insert(format!("{prefix}_MODEL_PREFIX"), pfx.to_string());
        }
        if self.requires_reasoning_content {
            env.insert(format!("{prefix}_REQUIRE_REASONING"), "1".to_string());
        }
        if let Some(cap) = self.max_tokens_cap {
            env.insert(format!("{prefix}_MAX_TOKENS_CAP"), cap.to_string());
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderProfile {
    pub kind: ProviderKind,
    pub default_protocol: ProviderProtocol,
    pub quirks: ProviderQuirks,
    pub model_listing_strategy: ModelListingStrategy,
    pub serve_flags: ServeFlags,
}

pub fn is_copilot_base(base_url: &str) -> bool {
    base_url == "copilot"
}

pub fn is_ollama_base(base_url: &str) -> bool {
    base_url == "ollama"
}

pub fn is_openrouter_base(base_url: &str) -> bool {
    base_url.contains("openrouter")
}

pub fn is_direct_openai_base(base_url: &str) -> bool {
    base_url
        .trim_end_matches('/')
        .to_ascii_lowercase()
        .contains("api.openai.com")
}

pub fn cloudflare_ai_base(base_url: &str) -> Option<String> {
    let parsed = reqwest::Url::parse(base_url).ok()?;
    let host = parsed.host_str()?;
    if !host.contains("cloudflare.com") {
        return None;
    }

    let mut base = base_url.trim_end_matches('/').to_string();
    if base.ends_with("/v1/chat/completions") {
        base.truncate(base.len() - "/v1/chat/completions".len());
    } else if base.ends_with("/chat/completions") {
        base.truncate(base.len() - "/chat/completions".len());
    } else if base.ends_with("/v1") {
        base.truncate(base.len() - "/v1".len());
    }

    if !base.ends_with("/ai") {
        if let Some(idx) = base.find("/ai/") {
            base.truncate(idx + "/ai".len());
        } else {
            return None;
        }
    }

    Some(base)
}

pub fn provider_profile_for_base_url(base_url: &str) -> ProviderProfile {
    let quirks = ProviderQuirks::for_base_url(base_url);
    if is_copilot_base(base_url) {
        return ProviderProfile {
            kind: ProviderKind::Copilot,
            default_protocol: ProviderProtocol::Openai,
            quirks,
            model_listing_strategy: ModelListingStrategy::Copilot,
            serve_flags: ServeFlags {
                is_copilot: true,
                is_openrouter: false,
            },
        };
    }

    if is_ollama_base(base_url) {
        return ProviderProfile {
            kind: ProviderKind::Ollama,
            default_protocol: ProviderProtocol::Openai,
            quirks,
            model_listing_strategy: ModelListingStrategy::Ollama,
            serve_flags: ServeFlags {
                is_copilot: false,
                is_openrouter: false,
            },
        };
    }

    if is_openrouter_base(base_url) {
        return ProviderProfile {
            kind: ProviderKind::OpenRouter,
            default_protocol: ProviderProtocol::Openai,
            quirks,
            model_listing_strategy: ModelListingStrategy::OpenAiCompatible,
            serve_flags: ServeFlags {
                is_copilot: false,
                is_openrouter: true,
            },
        };
    }

    if cloudflare_ai_base(base_url).is_some() {
        return ProviderProfile {
            kind: ProviderKind::CloudflareAi,
            default_protocol: ProviderProtocol::Openai,
            quirks,
            model_listing_strategy: ModelListingStrategy::CloudflareSearch,
            serve_flags: ServeFlags {
                is_copilot: false,
                is_openrouter: false,
            },
        };
    }

    match detect_provider_protocol(base_url) {
        ProviderProtocol::Anthropic => ProviderProfile {
            kind: ProviderKind::AnthropicCompatible,
            default_protocol: ProviderProtocol::Anthropic,
            quirks,
            model_listing_strategy: ModelListingStrategy::Anthropic,
            serve_flags: ServeFlags {
                is_copilot: false,
                is_openrouter: false,
            },
        },
        ProviderProtocol::Google => ProviderProfile {
            kind: ProviderKind::GoogleNative,
            default_protocol: ProviderProtocol::Google,
            quirks,
            model_listing_strategy: ModelListingStrategy::Google,
            serve_flags: ServeFlags {
                is_copilot: false,
                is_openrouter: false,
            },
        },
        ProviderProtocol::Openai | ProviderProtocol::ResponsesApi => ProviderProfile {
            kind: ProviderKind::OpenAiCompatible,
            default_protocol: ProviderProtocol::Openai,
            quirks,
            model_listing_strategy: ModelListingStrategy::OpenAiCompatible,
            serve_flags: ServeFlags {
                is_copilot: false,
                is_openrouter: false,
            },
        },
    }
}

pub fn provider_profile_for_key(key: &ApiKey) -> ProviderProfile {
    provider_profile_for_base_url(&key.base_url)
}

#[cfg(test)]
mod tests {
    use super::{
        ModelListingStrategy, ProviderKind, cloudflare_ai_base, provider_profile_for_base_url,
    };
    use crate::services::provider_protocol::ProviderProtocol;

    #[test]
    fn classifies_copilot() {
        let profile = provider_profile_for_base_url("copilot");
        assert_eq!(profile.kind, ProviderKind::Copilot);
        assert_eq!(profile.default_protocol, ProviderProtocol::Openai);
        assert_eq!(
            profile.model_listing_strategy,
            ModelListingStrategy::Copilot
        );
        assert!(profile.serve_flags.is_copilot);
        assert!(!profile.serve_flags.is_openrouter);
    }

    #[test]
    fn classifies_ollama() {
        let profile = provider_profile_for_base_url("ollama");
        assert_eq!(profile.kind, ProviderKind::Ollama);
        assert_eq!(profile.default_protocol, ProviderProtocol::Openai);
        assert_eq!(profile.model_listing_strategy, ModelListingStrategy::Ollama);
        assert!(!profile.serve_flags.is_copilot);
        assert!(!profile.serve_flags.is_openrouter);
    }

    #[test]
    fn classifies_openrouter() {
        let profile = provider_profile_for_base_url("https://openrouter.ai/api/v1");
        assert_eq!(profile.kind, ProviderKind::OpenRouter);
        assert_eq!(profile.default_protocol, ProviderProtocol::Openai);
        assert_eq!(
            profile.model_listing_strategy,
            ModelListingStrategy::OpenAiCompatible
        );
        assert!(profile.serve_flags.is_openrouter);
    }

    #[test]
    fn classifies_anthropic_compatible_endpoints() {
        let profile = provider_profile_for_base_url("https://api.minimax.io/anthropic/v1");
        assert_eq!(profile.kind, ProviderKind::AnthropicCompatible);
        assert_eq!(profile.default_protocol, ProviderProtocol::Anthropic);
        assert_eq!(
            profile.model_listing_strategy,
            ModelListingStrategy::Anthropic
        );
    }

    #[test]
    fn classifies_google_native_endpoints() {
        let profile =
            provider_profile_for_base_url("https://generativelanguage.googleapis.com/v1beta");
        assert_eq!(profile.kind, ProviderKind::GoogleNative);
        assert_eq!(profile.default_protocol, ProviderProtocol::Google);
        assert_eq!(profile.model_listing_strategy, ModelListingStrategy::Google);
    }

    #[test]
    fn classifies_cloudflare_and_applies_prefix_quirk() {
        let profile = provider_profile_for_base_url(
            "https://api.cloudflare.com/client/v4/accounts/abc/ai/v1",
        );
        assert_eq!(profile.kind, ProviderKind::CloudflareAi);
        assert_eq!(
            profile.model_listing_strategy,
            ModelListingStrategy::CloudflareSearch
        );
        assert_eq!(profile.quirks.model_prefix, Some("@cf/"));
    }

    #[test]
    fn classifies_generic_openai_compatible_endpoints() {
        let profile = provider_profile_for_base_url("https://api.example.com/v1");
        assert_eq!(profile.kind, ProviderKind::OpenAiCompatible);
        assert_eq!(profile.default_protocol, ProviderProtocol::Openai);
        assert_eq!(
            profile.model_listing_strategy,
            ModelListingStrategy::OpenAiCompatible
        );
    }

    #[test]
    fn resolves_provider_quirks() {
        let moonshot = provider_profile_for_base_url("https://api.moonshot.cn/v1");
        assert!(moonshot.quirks.requires_reasoning_content);
        assert_eq!(moonshot.quirks.max_tokens_cap, None);

        let deepseek = provider_profile_for_base_url("https://api.deepseek.com/v1");
        assert!(deepseek.quirks.requires_reasoning_content);
        assert_eq!(deepseek.quirks.max_tokens_cap, Some(8192));
    }

    #[test]
    fn normalizes_cloudflare_ai_root() {
        assert_eq!(
            cloudflare_ai_base("https://api.cloudflare.com/client/v4/accounts/abc/ai/v1"),
            Some("https://api.cloudflare.com/client/v4/accounts/abc/ai".to_string())
        );
        assert_eq!(
            cloudflare_ai_base("https://api.cloudflare.com/client/v4/accounts/abc/ai"),
            Some("https://api.cloudflare.com/client/v4/accounts/abc/ai".to_string())
        );
        assert_eq!(cloudflare_ai_base("https://api.openai.com/v1"), None);
    }
}
