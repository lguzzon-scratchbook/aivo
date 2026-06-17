//! Centralized model resolution logic for provider routing.
//!
//! Consolidates model selection, protocol inference, and gateway handling
//! previously scattered across `model_names`, `provider_profile`, and routers.

use crate::services::model_names::{infer_model_protocol, normalize_claude_version};
use crate::services::provider_protocol::ProviderProtocol;

fn model_family(p: ProviderProtocol) -> ProviderProtocol {
    match p {
        ProviderProtocol::ResponsesApi => ProviderProtocol::Openai,
        other => other,
    }
}

fn default_model_for_protocol(protocol: ProviderProtocol) -> &'static str {
    match protocol {
        ProviderProtocol::Openai | ProviderProtocol::ResponsesApi => "gpt-4o",
        ProviderProtocol::Anthropic => "claude-sonnet-4-5",
        ProviderProtocol::Google => "gemini-2.5-pro",
    }
}

pub fn is_gateway_style_endpoint(base_url: &str) -> bool {
    let lower = base_url.trim().to_ascii_lowercase();
    lower.contains("/endpoint") || lower.contains("gateway")
}

fn should_preserve_cross_protocol_model(
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

/// Select the appropriate model for a provider attempt.
///
/// Handles explicit overrides, gateway cross-preservation, and fallback defaults.
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

pub(crate) fn select_model_for_protocol(
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

/// Infer the appropriate provider name (e.g., "anthropic", "openai") from a model string.
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

/// Normalizes a Claude model name for providers that expect dots instead of hyphens.
pub fn normalize_claude_for_provider(base_url: &str, model: &str) -> String {
    if base_url.to_ascii_lowercase().contains("openrouter") {
        if !model.starts_with("claude-") || model.starts_with("anthropic/") {
            return model.to_string();
        }
        format!("anthropic/{}", normalize_claude_version(model))
    } else {
        model.to_string()
    }
}
