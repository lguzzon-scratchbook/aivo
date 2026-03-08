#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderProtocol {
    Openai,
    Anthropic,
    Google,
}

impl ProviderProtocol {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Openai => "openai",
            Self::Anthropic => "anthropic",
            Self::Google => "google",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "openai" => Some(Self::Openai),
            "anthropic" => Some(Self::Anthropic),
            "google" => Some(Self::Google),
            _ => None,
        }
    }
}

pub fn normalize_protocol_base(base_url: &str) -> &str {
    let trimmed = base_url.trim_end_matches('/');
    trimmed.strip_suffix("/v1").unwrap_or(trimmed)
}

pub fn is_anthropic_endpoint(base_url: &str) -> bool {
    let normalized = normalize_protocol_base(base_url).to_ascii_lowercase();
    normalized.contains("api.anthropic.com") || normalized.ends_with("/anthropic")
}

pub fn is_google_endpoint(base_url: &str) -> bool {
    let normalized = normalize_protocol_base(base_url).to_ascii_lowercase();
    normalized.contains("generativelanguage.googleapis.com")
}

pub fn detect_provider_protocol(base_url: &str) -> ProviderProtocol {
    if is_anthropic_endpoint(base_url) {
        ProviderProtocol::Anthropic
    } else if is_google_endpoint(base_url) {
        ProviderProtocol::Google
    } else {
        ProviderProtocol::Openai
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_anthropic_endpoint_variants() {
        assert_eq!(
            detect_provider_protocol("https://api.minimax.io/anthropic"),
            ProviderProtocol::Anthropic
        );
        assert_eq!(
            detect_provider_protocol("https://api.minimax.io/anthropic/v1"),
            ProviderProtocol::Anthropic
        );
    }

    #[test]
    fn detects_google_endpoint_variants() {
        assert_eq!(
            detect_provider_protocol("https://generativelanguage.googleapis.com/v1beta"),
            ProviderProtocol::Google
        );
    }

    #[test]
    fn defaults_to_openai_for_other_endpoints() {
        assert_eq!(
            detect_provider_protocol("https://openrouter.ai/api/v1"),
            ProviderProtocol::Openai
        );
    }
}
