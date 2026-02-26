/**
 * EnvironmentInjector service for preparing tool-specific environment variables.
 * Maps API keys to the correct environment variables per AI tool.
 */
use std::collections::HashMap;

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
    pub fn for_claude(&self, key: &ApiKey, model: Option<&str>) -> HashMap<String, String> {
        let mut env = HashMap::new();
        env.insert("ANTHROPIC_BASE_URL".to_string(), key.base_url.clone());
        env.insert("ANTHROPIC_API_KEY".to_string(), String::new());
        env.insert("ANTHROPIC_AUTH_TOKEN".to_string(), key.key.to_string());
        env.insert(
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC".to_string(),
            "1".to_string(),
        );

        if let Some(model) = model {
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
    /// Sets OPENAI_BASE_URL and OPENAI_API_KEY from the key.
    pub fn for_codex(&self, key: &ApiKey, model: Option<&str>) -> HashMap<String, String> {
        let mut env = HashMap::new();
        env.insert("OPENAI_BASE_URL".to_string(), key.base_url.clone());
        env.insert("OPENAI_API_KEY".to_string(), key.key.to_string());

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
    pub fn for_gemini(&self, key: &ApiKey) -> HashMap<String, String> {
        let mut env = HashMap::new();
        env.insert("GOOGLE_GEMINI_BASE_URL".to_string(), key.base_url.clone());
        env.insert("GEMINI_API_KEY".to_string(), key.key.to_string());

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
                let display = if key.contains("TOKEN") || key.contains("KEY") {
                    if value.len() > 12 {
                        format!("{}...{}", &value[..8], &value[value.len() - 4..])
                    } else {
                        "***".to_string()
                    }
                } else {
                    value.clone()
                };
                eprintln!("  {}={}", key, display);
            }

            if let Some(manual) = manual_env {
                if !manual.is_empty() {
                    eprintln!("[aivo] Manual environment overrides:");
                    let mut keys: Vec<_> = manual.keys().collect();
                    keys.sort();
                    for key in keys {
                        let value = &manual[key];
                        let display = if key.contains("TOKEN") || key.contains("KEY") {
                            if value.len() > 12 {
                                format!("{}...{}", &value[..8], &value[value.len() - 4..])
                            } else {
                                "***".to_string()
                            }
                        } else {
                            value.clone()
                        };
                        eprintln!("  {}={}", key, display);
                    }
                }
            }
        }

        merged
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
        assert_eq!(
            env.get("ANTHROPIC_API_KEY"),
            Some(&String::new())
        );
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
    fn test_for_codex() {
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let env = injector.for_codex(&key, None);

        assert_eq!(
            env.get("OPENAI_BASE_URL"),
            Some(&"http://localhost:8080".to_string())
        );
        assert_eq!(
            env.get("OPENAI_API_KEY"),
            Some(&"sk-test-key-12345".to_string())
        );
        // No CLOUDHARB_API_KEY
        assert!(env.get("CLOUDHARB_API_KEY").is_none());
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
    fn test_for_gemini() {
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let env = injector.for_gemini(&key);

        assert_eq!(
            env.get("GOOGLE_GEMINI_BASE_URL"),
            Some(&"http://localhost:8080".to_string())
        );
        assert_eq!(
            env.get("GEMINI_API_KEY"),
            Some(&"sk-test-key-12345".to_string())
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
