/**
 * EnvironmentInjector service for preparing tool-specific environment variables.
 * Maps API keys to the correct environment variables per AI tool.
 */
use std::collections::HashMap;

use serde_json::{json, Map, Value};

use crate::services::session_store::ApiKey;

/// EnvironmentInjector prepares tool-specific environment variables for AI tools
#[derive(Debug, Clone, Default)]
pub struct EnvironmentInjector;

impl EnvironmentInjector {
    /// Creates a new EnvironmentInjector
    pub fn new() -> Self {
        Self
    }

    /// Prepares environment variables for Claude CLI
    ///
    /// Sets ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY from the key.
    /// Disables nonessential traffic.
    /// When model is provided, sets ANTHROPIC_MODEL and related env vars for Claude Code routing.
    /// For OpenRouter, uses Claude Code Router if available, otherwise applies model name transformation.
    pub fn for_claude(&self, key: &ApiKey, model: Option<&str>) -> HashMap<String, String> {
        let mut env = HashMap::new();

        // For OpenRouter, use the built-in router (needs model name transformation + API proxying)
        if key.base_url.contains("openrouter") {
            // Placeholder URL - AI launcher overwrites with the actual random port after binding
            env.insert(
                "ANTHROPIC_BASE_URL".to_string(),
                "http://127.0.0.1:0".to_string(),
            );
            // Router will handle the OpenRouter API key transformation
            env.insert("ANTHROPIC_AUTH_TOKEN".to_string(), key.key.to_string());
            // Signal to start the router
            env.insert("AIVO_USE_ROUTER".to_string(), "1".to_string());
            env.insert("AIVO_ROUTER_API_KEY".to_string(), key.key.to_string());
            env.insert("AIVO_ROUTER_BASE_URL".to_string(), key.base_url.to_string());
        } else {
            // Direct connection to provider
            // Claude Code appends /v1/messages itself, so strip any trailing /v1 to avoid doubling
            let base_url = key.base_url.trim_end_matches('/');
            let base_url = base_url.strip_suffix("/v1").unwrap_or(base_url);
            env.insert("ANTHROPIC_BASE_URL".to_string(), base_url.to_string());
            env.insert("ANTHROPIC_AUTH_TOKEN".to_string(), key.key.to_string());
        }

        env.insert("ANTHROPIC_API_KEY".to_string(), String::new());
        env.insert(
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC".to_string(),
            "1".to_string(),
        );

        if let Some(model) = model {
            // Router handles OpenRouter model names transformation internally
            // Send the model name as-is; the router will transform it
            env.insert("ANTHROPIC_MODEL".to_string(), model.to_string());
            env.insert("ANTHROPIC_SMALL_FAST_MODEL".to_string(), model.to_string());
            env.insert(
                "ANTHROPIC_DEFAULT_HAIKU_MODEL".to_string(),
                model.to_string(),
            );
            env.insert(
                "ANTHROPIC_DEFAULT_SONNET_MODEL".to_string(),
                model.to_string(),
            );
            env.insert(
                "ANTHROPIC_DEFAULT_OPUS_MODEL".to_string(),
                model.to_string(),
            );
            env.insert("ANTHROPIC_REASONING_MODEL".to_string(), model.to_string());
        }

        env
    }

    /// Prepares environment variables for Codex CLI
    ///
    /// For non-OpenAI providers, activates the CodexRouter to strip unsupported
    /// built-in tool types (computer_use, file_search, etc.) before forwarding.
    /// For official OpenAI (api.openai.com), connects directly.
    pub fn for_codex(&self, key: &ApiKey, model: Option<&str>) -> HashMap<String, String> {
        let mut env = HashMap::new();

        if !key.base_url.contains("api.openai.com") {
            // Non-OpenAI provider: use CodexRouter to strip unsupported tool types
            // Placeholder URL - AI launcher overwrites with the actual random port after binding
            env.insert(
                "OPENAI_BASE_URL".to_string(),
                "http://127.0.0.1:0".to_string(),
            );
            env.insert("OPENAI_API_KEY".to_string(), key.key.to_string());
            env.insert("AIVO_USE_CODEX_ROUTER".to_string(), "1".to_string());
            env.insert("AIVO_CODEX_ROUTER_API_KEY".to_string(), key.key.to_string());
            env.insert(
                "AIVO_CODEX_ROUTER_BASE_URL".to_string(),
                key.base_url.clone(),
            );
        } else {
            // Official OpenAI: direct connection, no proxy needed
            env.insert("OPENAI_BASE_URL".to_string(), key.base_url.clone());
            env.insert("OPENAI_API_KEY".to_string(), key.key.to_string());
        }

        if let Some(model) = model {
            env.insert("CODEX_MODEL".to_string(), model.to_string());
            env.insert("OPENAI_DEFAULT_MODEL".to_string(), model.to_string());
            env.insert("CODEX_MODEL_DEFAULT".to_string(), model.to_string());
        }

        env
    }

    /// Prepares environment variables for Gemini CLI
    ///
    /// Sets GOOGLE_GEMINI_BASE_URL and GEMINI_API_KEY from the key.
    /// For non-Google endpoints, activates GeminiRouter to convert native Gemini
    /// API format to OpenAI chat completions format.
    /// When model is provided, sets GEMINI_MODEL.
    pub fn for_gemini(&self, key: &ApiKey, model: Option<&str>) -> HashMap<String, String> {
        let mut env = HashMap::new();

        if key.base_url.contains("generativelanguage.googleapis.com") {
            // Native Google endpoint: connect directly
            env.insert("GOOGLE_GEMINI_BASE_URL".to_string(), key.base_url.clone());
            env.insert("GEMINI_API_KEY".to_string(), key.key.to_string());
        } else {
            // Non-Google provider: use GeminiRouter to convert Gemini API → OpenAI format
            // Placeholder URL — AI launcher overwrites with actual random port after binding
            env.insert(
                "GOOGLE_GEMINI_BASE_URL".to_string(),
                "http://127.0.0.1:0".to_string(),
            );
            env.insert("GEMINI_API_KEY".to_string(), key.key.to_string());
            env.insert("AIVO_USE_GEMINI_ROUTER".to_string(), "1".to_string());
            env.insert(
                "AIVO_GEMINI_ROUTER_API_KEY".to_string(),
                key.key.to_string(),
            );
            env.insert(
                "AIVO_GEMINI_ROUTER_BASE_URL".to_string(),
                key.base_url.clone(),
            );
        }

        if let Some(model) = model {
            env.insert("GEMINI_MODEL".to_string(), model.to_string());
        }

        env
    }

    /// Prepares environment variables for OpenCode CLI.
    ///
    /// Uses OPENCODE_CONFIG_CONTENT to inject an inline OpenCode config
    /// so aivo can provide base URL and API key without writing config files.
    pub fn for_opencode(
        &self,
        key: &ApiKey,
        model: Option<&str>,
        discovered_models: Option<&[String]>,
    ) -> HashMap<String, String> {
        let mut env = HashMap::new();

        let mut provider = Map::new();
        provider.insert("npm".to_string(), json!("@ai-sdk/openai-compatible"));
        provider.insert("name".to_string(), json!("aivo"));
        provider.insert(
            "options".to_string(),
            json!({
                "baseURL": key.base_url.as_str(),
                "apiKey": key.key.as_str(),
            }),
        );

        let mut model_ids: Vec<String> = discovered_models
            .map(|models| {
                models
                    .iter()
                    .map(|m| strip_aivo_prefix(m).to_string())
                    .collect()
            })
            .unwrap_or_default();

        if let Some(model) = model {
            let model_name = strip_aivo_prefix(model).to_string();
            if !model_ids.contains(&model_name) {
                model_ids.push(model_name);
            }
        }

        model_ids.sort();
        model_ids.dedup();
        if !model_ids.is_empty() {
            let mut models = Map::new();
            for model_id in model_ids {
                models.insert(model_id.clone(), json!({ "name": model_id }));
            }
            provider.insert("models".to_string(), Value::Object(models));
        }

        let mut providers = Map::new();
        providers.insert("aivo".to_string(), Value::Object(provider));

        let mut config = Map::new();
        config.insert(
            "$schema".to_string(),
            json!("https://opencode.ai/config.json"),
        );
        config.insert("provider".to_string(), Value::Object(providers));

        if let Some(model) = model {
            config.insert(
                "model".to_string(),
                json!(format!("aivo/{}", strip_aivo_prefix(model))),
            );
        }

        env.insert(
            "OPENCODE_CONFIG_CONTENT".to_string(),
            Value::Object(config).to_string(),
        );
        env
    }

    /// Merges tool-specific environment variables with the current process environment
    ///
    /// Tool environment variables take precedence over existing process.env values.
    /// Manual environment variables take precedence over tool variables.
    pub fn merge(
        &self,
        tool_env: &HashMap<String, String>,
        manual_env: Option<&HashMap<String, String>>,
        debug: bool,
    ) -> HashMap<String, String> {
        // Start with current environment
        let mut merged: HashMap<String, String> = std::env::vars().collect();

        // Add tool environment (overrides current env)
        for (key, value) in tool_env {
            merged.insert(key.clone(), value.clone());
        }

        // Add manual environment (overrides tool env)
        if let Some(manual) = manual_env {
            for (key, value) in manual {
                merged.insert(key.clone(), value.clone());
            }
        }

        // Debug output if requested
        if debug {
            eprintln!("[aivo] Injecting environment variables:");
            let mut keys: Vec<_> = tool_env.keys().collect();
            keys.sort();
            for key in keys {
                let value = &tool_env[key];
                let display = redact_env_value(key, value);
                eprintln!("  {}={}", key, display);
            }

            if let Some(manual) = manual_env {
                if !manual.is_empty() {
                    eprintln!("[aivo] Manual environment overrides:");
                    let mut keys: Vec<_> = manual.keys().collect();
                    keys.sort();
                    for key in keys {
                        let value = &manual[key];
                        let display = redact_env_value(key, value);
                        eprintln!("  {}={}", key, display);
                    }
                }
            }
        }

        merged
    }
}

fn strip_aivo_prefix(model: &str) -> &str {
    model.strip_prefix("aivo/").unwrap_or(model)
}

fn redact_env_value(key: &str, value: &str) -> String {
    if key == "OPENCODE_CONFIG_CONTENT" {
        return "<redacted>".to_string();
    }

    if key.contains("TOKEN") || key.contains("KEY") {
        if value.len() > 12 {
            format!("{}...{}", &value[..8], &value[value.len() - 4..])
        } else {
            "***".to_string()
        }
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key() -> ApiKey {
        ApiKey::new(
            "a1b2".to_string(),
            "test-key".to_string(),
            "http://localhost:8080".to_string(),
            "sk-test-key-12345".to_string(),
        )
    }

    #[test]
    fn test_for_claude() {
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let env = injector.for_claude(&key, None);

        assert_eq!(
            env.get("ANTHROPIC_BASE_URL"),
            Some(&"http://localhost:8080".to_string())
        );
        assert_eq!(env.get("ANTHROPIC_API_KEY"), Some(&String::new()));
        assert_eq!(
            env.get("ANTHROPIC_AUTH_TOKEN"),
            Some(&"sk-test-key-12345".to_string())
        );
        assert_eq!(
            env.get("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"),
            Some(&"1".to_string())
        );
    }

    #[test]
    fn test_for_claude_with_model() {
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let env = injector.for_claude(&key, Some("claude-3-opus"));

        assert_eq!(
            env.get("ANTHROPIC_MODEL"),
            Some(&"claude-3-opus".to_string())
        );
        assert_eq!(
            env.get("ANTHROPIC_SMALL_FAST_MODEL"),
            Some(&"claude-3-opus".to_string())
        );
        assert_eq!(
            env.get("ANTHROPIC_DEFAULT_HAIKU_MODEL"),
            Some(&"claude-3-opus".to_string())
        );
        assert_eq!(
            env.get("ANTHROPIC_DEFAULT_SONNET_MODEL"),
            Some(&"claude-3-opus".to_string())
        );
        assert_eq!(
            env.get("ANTHROPIC_DEFAULT_OPUS_MODEL"),
            Some(&"claude-3-opus".to_string())
        );
        assert_eq!(
            env.get("ANTHROPIC_REASONING_MODEL"),
            Some(&"claude-3-opus".to_string())
        );
    }

    #[test]
    fn test_for_claude_openrouter_model_transformation() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://openrouter.ai/api/v1".to_string();
        let env = injector.for_claude(&key, Some("claude-haiku-4-5"));

        // With built-in router: model names pass through unchanged
        // Router handles transformation
        assert_eq!(
            env.get("ANTHROPIC_MODEL"),
            Some(&"claude-haiku-4-5".to_string())
        );
        // Router should be started
        assert_eq!(env.get("AIVO_USE_ROUTER"), Some(&"1".to_string()));
        // Base URL is a placeholder; AI launcher overwrites with actual port after binding
        assert_eq!(
            env.get("ANTHROPIC_BASE_URL"),
            Some(&"http://127.0.0.1:0".to_string())
        );
    }

    #[test]
    fn test_for_claude_openrouter_sonnet_model() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://openrouter.ai/api/v1".to_string();
        let env = injector.for_claude(&key, Some("claude-sonnet-4-6"));

        // Model name passes through unchanged - router will transform it
        assert_eq!(
            env.get("ANTHROPIC_MODEL"),
            Some(&"claude-sonnet-4-6".to_string())
        );
        // Verify router configuration is set
        assert_eq!(env.get("AIVO_ROUTER_API_KEY"), Some(&key.key.to_string()));
    }

    #[test]
    fn test_for_claude_openrouter_opus_model() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://openrouter.ai/api/v1".to_string();
        let env = injector.for_claude(&key, Some("claude-opus-4-6"));

        // Model passes through unchanged - router transforms
        assert_eq!(
            env.get("ANTHROPIC_MODEL"),
            Some(&"claude-opus-4-6".to_string())
        );
    }

    #[test]
    fn test_for_claude_openrouter_future_models() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://openrouter.ai/api/v1".to_string();

        // All models pass through unchanged - router handles transformation
        let env = injector.for_claude(&key, Some("claude-some-model-5-10"));
        assert_eq!(
            env.get("ANTHROPIC_MODEL"),
            Some(&"claude-some-model-5-10".to_string())
        );
    }

    #[test]
    fn test_for_claude_non_claude_model_no_transformation() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://openrouter.ai/api/v1".to_string();
        let env = injector.for_claude(&key, Some("gpt-4o"));

        // Non-Claude models should not be transformed
        assert_eq!(env.get("ANTHROPIC_MODEL"), Some(&"gpt-4o".to_string()));
    }

    #[test]
    fn test_router_integration_example() {
        // The built-in router is always used for OpenRouter
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://openrouter.ai/api/v1".to_string();
        let env = injector.for_claude(&key, Some("claude-sonnet-4-6"));

        // Placeholder; AI launcher overwrites with the actual random port after binding
        assert_eq!(
            env.get("ANTHROPIC_BASE_URL"),
            Some(&"http://127.0.0.1:0".to_string())
        );
        // Model name passes through unchanged - router transforms it
        assert_eq!(
            env.get("ANTHROPIC_MODEL"),
            Some(&"claude-sonnet-4-6".to_string())
        );
        // Router configuration is set
        assert_eq!(env.get("AIVO_USE_ROUTER"), Some(&"1".to_string()));
    }

    #[test]
    fn test_for_codex_non_openai_uses_router() {
        // test_key() uses http://localhost:8080 (non-OpenAI) → router enabled
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let env = injector.for_codex(&key, None);

        // Placeholder; AI launcher overwrites with actual port after binding
        assert_eq!(
            env.get("OPENAI_BASE_URL"),
            Some(&"http://127.0.0.1:0".to_string())
        );
        assert_eq!(
            env.get("OPENAI_API_KEY"),
            Some(&"sk-test-key-12345".to_string())
        );
        assert_eq!(env.get("AIVO_USE_CODEX_ROUTER"), Some(&"1".to_string()));
        assert_eq!(
            env.get("AIVO_CODEX_ROUTER_API_KEY"),
            Some(&"sk-test-key-12345".to_string())
        );
        assert_eq!(
            env.get("AIVO_CODEX_ROUTER_BASE_URL"),
            Some(&"http://localhost:8080".to_string())
        );
    }

    #[test]
    fn test_for_codex_official_openai_direct() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.openai.com/v1".to_string();
        let env = injector.for_codex(&key, None);

        // Direct connection: no router, use actual base URL
        assert_eq!(
            env.get("OPENAI_BASE_URL"),
            Some(&"https://api.openai.com/v1".to_string())
        );
        assert_eq!(
            env.get("OPENAI_API_KEY"),
            Some(&"sk-test-key-12345".to_string())
        );
        assert!(env.get("AIVO_USE_CODEX_ROUTER").is_none());
    }

    #[test]
    fn test_for_codex_with_model() {
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let env = injector.for_codex(&key, Some("o3"));

        assert_eq!(env.get("CODEX_MODEL"), Some(&"o3".to_string()));
        assert_eq!(env.get("OPENAI_DEFAULT_MODEL"), Some(&"o3".to_string()));
        assert_eq!(env.get("CODEX_MODEL_DEFAULT"), Some(&"o3".to_string()));
    }

    #[test]
    fn test_for_codex_vercel_uses_router() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://ai-gateway.vercel.sh/v1".to_string();
        let env = injector.for_codex(&key, None);

        assert_eq!(env.get("AIVO_USE_CODEX_ROUTER"), Some(&"1".to_string()));
        assert_eq!(
            env.get("AIVO_CODEX_ROUTER_BASE_URL"),
            Some(&"https://ai-gateway.vercel.sh/v1".to_string())
        );
    }

    #[test]
    fn test_for_codex_openrouter_uses_router() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://openrouter.ai/api/v1".to_string();
        let env = injector.for_codex(&key, None);

        assert_eq!(env.get("AIVO_USE_CODEX_ROUTER"), Some(&"1".to_string()));
        assert_eq!(
            env.get("AIVO_CODEX_ROUTER_BASE_URL"),
            Some(&"https://openrouter.ai/api/v1".to_string())
        );
    }

    #[test]
    fn test_for_gemini() {
        let injector = EnvironmentInjector::new();
        let key = test_key(); // base_url = http://localhost:8080 (non-Google → router)
        let env = injector.for_gemini(&key, None);

        // Non-Google URL: placeholder is used, router env vars are set
        assert_eq!(
            env.get("GOOGLE_GEMINI_BASE_URL"),
            Some(&"http://127.0.0.1:0".to_string())
        );
        assert_eq!(
            env.get("GEMINI_API_KEY"),
            Some(&"sk-test-key-12345".to_string())
        );
        assert!(env.get("GEMINI_MODEL").is_none());
    }

    #[test]
    fn test_for_gemini_with_model() {
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let env = injector.for_gemini(&key, Some("google/gemini-2.0-flash"));
        assert_eq!(
            env.get("GEMINI_MODEL"),
            Some(&"google/gemini-2.0-flash".to_string())
        );
    }

    #[test]
    fn test_for_gemini_native_google_no_router() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://generativelanguage.googleapis.com/".to_string();
        let env = injector.for_gemini(&key, None);
        assert!(env.get("AIVO_USE_GEMINI_ROUTER").is_none());
        assert_eq!(
            env.get("GOOGLE_GEMINI_BASE_URL"),
            Some(&"https://generativelanguage.googleapis.com/".to_string())
        );
    }

    #[test]
    fn test_for_gemini_non_google_uses_router() {
        let injector = EnvironmentInjector::new();
        let key = test_key(); // base_url = http://localhost:8080 (non-Google)
        let env = injector.for_gemini(&key, None);
        assert_eq!(env.get("AIVO_USE_GEMINI_ROUTER"), Some(&"1".to_string()));
        assert_eq!(
            env.get("AIVO_GEMINI_ROUTER_API_KEY"),
            Some(&"sk-test-key-12345".to_string())
        );
        assert_eq!(
            env.get("AIVO_GEMINI_ROUTER_BASE_URL"),
            Some(&"http://localhost:8080".to_string())
        );
        // Placeholder — launcher overwrites with actual port
        assert_eq!(
            env.get("GOOGLE_GEMINI_BASE_URL"),
            Some(&"http://127.0.0.1:0".to_string())
        );
    }

    #[test]
    fn test_for_gemini_vercel_uses_router() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://ai-gateway.vercel.sh/v1".to_string();
        let env = injector.for_gemini(&key, None);
        assert_eq!(env.get("AIVO_USE_GEMINI_ROUTER"), Some(&"1".to_string()));
        assert_eq!(
            env.get("AIVO_GEMINI_ROUTER_BASE_URL"),
            Some(&"https://ai-gateway.vercel.sh/v1".to_string())
        );
    }

    #[test]
    fn test_for_opencode() {
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let env = injector.for_opencode(&key, None, None);

        let config: Value =
            serde_json::from_str(env.get("OPENCODE_CONFIG_CONTENT").unwrap()).unwrap();
        assert_eq!(config["$schema"], "https://opencode.ai/config.json");
        assert_eq!(
            config["provider"]["aivo"]["npm"],
            "@ai-sdk/openai-compatible"
        );
        assert_eq!(config["provider"]["aivo"]["name"], "aivo");
        assert_eq!(
            config["provider"]["aivo"]["options"]["baseURL"],
            "http://localhost:8080"
        );
        assert_eq!(
            config["provider"]["aivo"]["options"]["apiKey"],
            "sk-test-key-12345"
        );
        assert!(config.get("model").is_none());
    }

    #[test]
    fn test_for_opencode_with_model() {
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let env = injector.for_opencode(&key, Some("gpt-5"), None);

        let config: Value =
            serde_json::from_str(env.get("OPENCODE_CONFIG_CONTENT").unwrap()).unwrap();
        assert_eq!(config["model"], "aivo/gpt-5");
        assert_eq!(
            config["provider"]["aivo"]["models"]["gpt-5"]["name"],
            "gpt-5"
        );
    }

    #[test]
    fn test_for_opencode_with_prefixed_model() {
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let env = injector.for_opencode(&key, Some("aivo/gpt-5"), None);

        let config: Value =
            serde_json::from_str(env.get("OPENCODE_CONFIG_CONTENT").unwrap()).unwrap();
        assert_eq!(config["model"], "aivo/gpt-5");
        assert_eq!(
            config["provider"]["aivo"]["models"]["gpt-5"]["name"],
            "gpt-5"
        );
    }

    #[test]
    fn test_for_opencode_with_discovered_models() {
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let discovered = vec!["gpt-4o".to_string(), "claude-sonnet-4".to_string()];
        let env = injector.for_opencode(&key, None, Some(&discovered));

        let config: Value =
            serde_json::from_str(env.get("OPENCODE_CONFIG_CONTENT").unwrap()).unwrap();
        assert!(config.get("model").is_none());
        assert_eq!(
            config["provider"]["aivo"]["models"]["claude-sonnet-4"]["name"],
            "claude-sonnet-4"
        );
        assert_eq!(
            config["provider"]["aivo"]["models"]["gpt-4o"]["name"],
            "gpt-4o"
        );
    }

    #[test]
    fn test_for_opencode_with_model_and_discovered_models() {
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let discovered = vec!["gpt-4o".to_string(), "claude-sonnet-4".to_string()];
        let env = injector.for_opencode(&key, Some("gpt-5"), Some(&discovered));

        let config: Value =
            serde_json::from_str(env.get("OPENCODE_CONFIG_CONTENT").unwrap()).unwrap();
        assert_eq!(config["model"], "aivo/gpt-5");
        assert_eq!(
            config["provider"]["aivo"]["models"]["gpt-5"]["name"],
            "gpt-5"
        );
        assert_eq!(
            config["provider"]["aivo"]["models"]["gpt-4o"]["name"],
            "gpt-4o"
        );
        assert_eq!(
            config["provider"]["aivo"]["models"]["claude-sonnet-4"]["name"],
            "claude-sonnet-4"
        );
    }

    #[test]
    fn test_merge() {
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let tool_env = injector.for_claude(&key, None);
        let merged = injector.merge(&tool_env, None, false);

        // Should contain all the tool env vars
        assert!(merged.contains_key("ANTHROPIC_BASE_URL"));
        assert!(merged.contains_key("ANTHROPIC_API_KEY"));
    }
}
