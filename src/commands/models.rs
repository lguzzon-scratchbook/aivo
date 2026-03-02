/**
 * ModelsCommand handler for listing available models from the active provider.
 * Calls /v1/models (OpenAI-compatible) or /v1beta/models (Google Gemini).
 */
use anyhow::Result;
use reqwest::Client;
use serde::Deserialize;

use crate::commands::normalize_base_url;
use crate::errors::ExitCode;
use crate::services::models_cache::ModelsCache;
use crate::services::session_store::{ApiKey, SessionStore};
use crate::style;

pub struct ModelsCommand {
    session_store: SessionStore,
    cache: ModelsCache,
}

#[derive(Deserialize)]
struct OpenAIModelsResponse {
    data: Vec<OpenAIModel>,
}

#[derive(Deserialize)]
struct OpenAIModel {
    id: String,
}

#[derive(Deserialize)]
struct GeminiModelsResponse {
    models: Vec<GeminiModel>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiModel {
    name: String,
    #[serde(default)]
    supported_generation_methods: Vec<String>,
}

impl ModelsCommand {
    pub fn new(session_store: SessionStore, cache: ModelsCache) -> Self {
        Self {
            session_store,
            cache,
        }
    }

    pub async fn execute(&self, key_override: Option<ApiKey>, refresh: bool) -> ExitCode {
        match self.execute_internal(key_override, refresh).await {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                ExitCode::UserError
            }
        }
    }

    async fn execute_internal(
        &self,
        key_override: Option<ApiKey>,
        refresh: bool,
    ) -> Result<ExitCode> {
        let key = match key_override {
            Some(k) => k,
            None => match self.session_store.get_active_key().await? {
                Some(k) => k,
                None => {
                    eprintln!(
                        "{} No API key configured. Run 'aivo keys add' first.",
                        style::red("Error:")
                    );
                    return Ok(ExitCode::AuthError);
                }
            },
        };

        let client = Client::new();
        let mut models = fetch_models_cached(&client, &key, &self.cache, refresh).await?;
        models.sort();

        eprintln!(
            "{} {} models via {}",
            style::success_symbol(),
            models.len(),
            style::dim(&key.base_url)
        );
        for model in &models {
            println!("{}", model);
        }

        Ok(ExitCode::Success)
    }

    pub fn print_help() {
        println!("{} aivo models", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim("List available models from the active API key's provider.")
        );
        println!(
            "{}",
            style::dim(
                "Calls /v1/models for OpenAI-compatible providers, or /v1beta/models for Google."
            )
        );
        println!();
        println!("{}", style::bold("Options:"));
        println!(
            "  {}  {}",
            style::cyan("-k, --key <id|name>"),
            style::dim("Select API key by ID or name")
        );
        println!(
            "  {}        {}",
            style::cyan("-r, --refresh"),
            style::dim("Bypass cache and fetch fresh model list")
        );
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo models"));
        println!("  {}", style::dim("aivo models --key openrouter"));
        println!("  {}", style::dim("aivo models --refresh"));
    }
}

/// Returns just the scheme + host + port of a URL, e.g. "https://api.example.com".
/// Used to probe the root when the base URL includes a path segment like /endpoint.
fn url_origin(url: &str) -> Option<String> {
    let parsed = reqwest::Url::parse(url).ok()?;
    let mut origin = format!("{}://{}", parsed.scheme(), parsed.host_str()?);
    if let Some(port) = parsed.port() {
        origin.push_str(&format!(":{}", port));
    }
    Some(origin)
}

/// Returns true if the model is suitable for text chat.
/// Filters out embedding models and image/audio-only generation models.
pub(crate) fn is_text_chat_model(id: &str) -> bool {
    let lower = id.to_lowercase();
    // Embedding models
    if lower.contains("embed") {
        return false;
    }
    // Image generation, TTS, and speech recognition
    if lower.starts_with("dall-e")
        || lower.starts_with("tts-")
        || lower.starts_with("whisper-")
        || lower.contains("-image")
    {
        return false;
    }
    true
}

pub(crate) async fn fetch_models(client: &Client, key: &ApiKey) -> Result<Vec<String>> {
    let base = normalize_base_url(&key.base_url);

    if key.base_url.contains("generativelanguage.googleapis.com") {
        let url = format!("{}/v1beta/models?key={}", base, key.key.as_str());
        let response = client
            .get(&url)
            .header("User-Agent", format!("aivo/{}", crate::version::VERSION))
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("API returned {} — {}", status, body);
        }

        let resp: GeminiModelsResponse = response.json().await?;
        // Keep only models that support text generation; filter out embed-only models
        Ok(resp
            .models
            .into_iter()
            .filter(|m| {
                m.supported_generation_methods
                    .iter()
                    .any(|method| method == "generateContent")
            })
            .map(|m| {
                m.name
                    .strip_prefix("models/")
                    .unwrap_or(&m.name)
                    .to_string()
            })
            .collect())
    } else {
        // Build candidate URLs. When the base URL has a path segment (e.g.
        // "https://api.example.com/endpoint"), the bare origin is tried first
        // because the /v1/models endpoint is typically at the root, not under
        // the chat-completions path prefix.
        let model_endpoints = |b: &str| [format!("{}/v1/models", b), format!("{}/models", b)];
        let mut candidates = Vec::new();
        if let Some(origin) = url_origin(base) {
            if origin != base {
                candidates.extend(model_endpoints(&origin));
            }
        }
        candidates.extend(model_endpoints(base));
        let auth = format!("Bearer {}", key.key.as_str());
        let user_agent = format!("aivo/{}", crate::version::VERSION);

        let mut last_err = String::new();
        for url in &candidates {
            let response = client
                .get(url)
                .header("Authorization", &auth)
                .header("User-Agent", &user_agent)
                .send()
                .await?;

            if response.status().is_success() {
                let body = response.text().await.unwrap_or_default();
                match serde_json::from_str::<OpenAIModelsResponse>(&body) {
                    Ok(resp) => {
                        return Ok(resp
                            .data
                            .into_iter()
                            .map(|m| m.id)
                            .filter(|id| is_text_chat_model(id))
                            .collect())
                    }
                    Err(e) => {
                        last_err = format!("Invalid models response from {}: {}", url, e);
                        continue;
                    }
                }
            }

            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            last_err = format!("API returned {} from {} — {}", status, url, body);
        }

        anyhow::bail!("{}", last_err)
    }
}

/// Cache-aware wrapper around `fetch_models`.
/// Returns cached result if present and not expired (unless `bypass_cache` is true).
/// On cache miss, fetches from the network and writes the result to the cache.
pub(crate) async fn fetch_models_cached(
    client: &Client,
    key: &ApiKey,
    cache: &ModelsCache,
    bypass_cache: bool,
) -> Result<Vec<String>> {
    if !bypass_cache {
        if let Some(cached) = cache.get(&key.base_url).await {
            return Ok(cached);
        }
    }
    let models = fetch_models(client, key).await?;
    cache.set(&key.base_url, models.clone()).await;
    Ok(models)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::models_cache::ModelsCache;
    use tempfile::TempDir;

    fn make_key(url: &str) -> ApiKey {
        use zeroize::Zeroizing;
        ApiKey {
            id: "1".to_string(),
            name: "test".to_string(),
            base_url: url.to_string(),
            key: Zeroizing::new("sk-test".to_string()),
            created_at: "2026-01-01".to_string(),
        }
    }

    #[test]
    fn test_is_text_chat_model_keeps_chat_models() {
        assert!(is_text_chat_model("gpt-4o"));
        assert!(is_text_chat_model("gpt-4o-mini"));
        assert!(is_text_chat_model("claude-sonnet-4-6"));
        assert!(is_text_chat_model("gpt-3.5-turbo"));
        assert!(is_text_chat_model("o1"));
        assert!(is_text_chat_model("o3-mini"));
        assert!(is_text_chat_model("gpt-4o-audio-preview"));
        assert!(is_text_chat_model("gemini-1.5-pro"));
        assert!(is_text_chat_model("gemini-2.0-flash"));
    }

    #[test]
    fn test_is_text_chat_model_filters_embeddings() {
        assert!(!is_text_chat_model("text-embedding-3-small"));
        assert!(!is_text_chat_model("text-embedding-3-large"));
        assert!(!is_text_chat_model("text-embedding-ada-002"));
        assert!(!is_text_chat_model("embedding-001"));
        assert!(!is_text_chat_model("text-embeddings-inference"));
    }

    #[test]
    fn test_is_text_chat_model_filters_image_and_audio() {
        assert!(!is_text_chat_model("dall-e-2"));
        assert!(!is_text_chat_model("dall-e-3"));
        assert!(!is_text_chat_model("tts-1"));
        assert!(!is_text_chat_model("tts-1-hd"));
        assert!(!is_text_chat_model("whisper-1"));
        assert!(!is_text_chat_model("gpt-image-1"));
        assert!(!is_text_chat_model("google/gemini-3.1-flash-image-preview"));
    }

    #[tokio::test]
    async fn cached_models_returned_without_network() {
        let dir = TempDir::new().unwrap();
        let cache = ModelsCache::with_path(dir.path().join("models-cache.json"));
        let models = vec!["model-a".to_string()];
        cache.set("https://api.example.com", models.clone()).await;

        let key = make_key("https://api.example.com");
        let client = reqwest::Client::new();
        // With a valid cache, fetch_models_cached should return cached list
        // without making a network call (network call would fail with this fake key)
        let result = fetch_models_cached(&client, &key, &cache, false).await;
        assert_eq!(result.unwrap(), models);
    }

    #[tokio::test]
    async fn bypass_cache_ignores_warm_cache() {
        let dir = TempDir::new().unwrap();
        let cache = ModelsCache::with_path(dir.path().join("models-cache.json"));
        // Seed cache with stale data
        cache
            .set("https://api.example.com", vec!["stale-model".to_string()])
            .await;

        let key = make_key("https://api.example.com");
        let client = reqwest::Client::new();
        // With bypass_cache=true, the function should NOT return the cached value.
        // It will try a network call (which will fail with a fake key) — that's fine,
        // we just verify it didn't return the cached stale data.
        let result = fetch_models_cached(&client, &key, &cache, true).await;
        // Network call will fail (fake key) — result should be Err, not the stale cached value
        assert!(
            result.is_err(),
            "Expected network error, not cached stale data"
        );
    }
}
