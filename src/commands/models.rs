/**
 * ModelsCommand handler for listing available models from the active provider.
 * Calls provider-specific model listing endpoints (OpenAI, Gemini, Cloudflare).
 */
use anyhow::Result;
use reqwest::Client;
use serde::Deserialize;
use std::collections::HashSet;

use crate::commands::normalize_base_url;
use crate::errors::ExitCode;
use crate::services::copilot_auth::{
    COPILOT_EDITOR_VERSION, COPILOT_INTEGRATION_ID, CopilotTokenManager,
};
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

#[derive(Deserialize)]
struct CloudflareModelsResponse {
    #[serde(default)]
    result: Vec<CloudflareModel>,
    result_info: Option<CloudflareResultInfo>,
}

#[derive(Deserialize)]
struct CloudflareModel {
    id: String,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
struct CloudflareResultInfo {
    total_pages: Option<u32>,
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
                "Calls /v1/models (OpenAI-compatible), /v1beta/models (Google), or /ai/models/search (Cloudflare)."
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

/// Returns a Cloudflare Workers AI base URL ending in `/ai` when applicable.
fn cloudflare_ai_base(base_url: &str) -> Option<String> {
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

/// Copilot's Claude/OpenAI chat routing uses the chat completions API.
/// Exclude clearly responses-only Codex models that the endpoint rejects.
fn is_copilot_chat_model(id: &str) -> bool {
    is_text_chat_model(id) && !id.to_lowercase().contains("codex")
}

fn cloudflare_model_name(model: CloudflareModel) -> String {
    model
        .name
        .filter(|name| !name.trim().is_empty())
        .unwrap_or(model.id)
}

pub(crate) async fn fetch_models(client: &Client, key: &ApiKey) -> Result<Vec<String>> {
    let base = normalize_base_url(&key.base_url);

    if key.base_url == "copilot" {
        // Copilot: exchange GitHub token for Copilot token, then fetch models
        let tm = CopilotTokenManager::new(key.key.as_str().to_string());
        let (copilot_token, api_endpoint) = tm.get_token().await?;
        let url = format!("{}/models", api_endpoint.trim_end_matches('/'));
        let response = client
            .get(&url)
            .header("Authorization", format!("Bearer {}", copilot_token))
            .header("Editor-Version", COPILOT_EDITOR_VERSION)
            .header("Copilot-Integration-Id", COPILOT_INTEGRATION_ID)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Copilot API returned {} — {}", status, body);
        }

        let resp: OpenAIModelsResponse = response.json().await?;
        Ok(resp
            .data
            .into_iter()
            .map(|m| m.id)
            .filter(|id| is_copilot_chat_model(id))
            .collect())
    } else if key.base_url.contains("generativelanguage.googleapis.com") {
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
    } else if let Some(cloudflare_base) = cloudflare_ai_base(base) {
        let auth = format!("Bearer {}", key.key.as_str());
        let user_agent = format!("aivo/{}", crate::version::VERSION);
        let mut page = 1u32;
        let mut seen = HashSet::new();
        let mut models = Vec::new();

        loop {
            let url = format!(
                "{}/models/search?hide_experimental=true&page={}&per_page=100",
                cloudflare_base, page
            );
            let response = client
                .get(&url)
                .header("Authorization", &auth)
                .header("User-Agent", &user_agent)
                .send()
                .await?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                anyhow::bail!("API returned {} from {} — {}", status, url, body);
            }

            let resp: CloudflareModelsResponse = response.json().await?;
            for model in resp.result.into_iter().map(cloudflare_model_name) {
                if seen.insert(model.clone()) && is_text_chat_model(&model) {
                    models.push(model);
                }
            }

            let total_pages = resp
                .result_info
                .and_then(|info| info.total_pages)
                .unwrap_or(page);
            if page >= total_pages {
                break;
            }
            page += 1;
        }

        Ok(models)
    } else {
        // Build candidate URLs. When the base URL has a path segment (e.g.
        // "https://api.example.com/endpoint"), the bare origin is tried first
        // because the /v1/models endpoint is typically at the root, not under
        // the chat-completions path prefix.
        let model_endpoints = |b: &str| [format!("{}/v1/models", b), format!("{}/models", b)];
        let mut candidates = Vec::new();
        if let Some(origin) = url_origin(base)
            && origin != base
        {
            candidates.extend(model_endpoints(&origin));
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
                            .collect());
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

/// Fetches the model list (cache-first) with a spinner for network fetches,
/// filtered to text-chat models only. Used by chat and run commands for the
/// interactive model picker.
pub(crate) async fn fetch_models_for_select(
    client: &Client,
    key: &ApiKey,
    cache: &ModelsCache,
) -> Vec<String> {
    let (spinning, spinner_handle) = style::start_spinner(None);
    let list = fetch_models_cached(client, key, cache, false)
        .await
        .unwrap_or_default();
    style::stop_spinner(&spinning);
    let _ = spinner_handle.await;
    list
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
    if !bypass_cache && let Some(cached) = cache.get(&key.base_url).await {
        return Ok(cached);
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

    #[test]
    fn test_is_copilot_chat_model_filters_codex_models() {
        assert!(is_copilot_chat_model("gpt-4o"));
        assert!(is_copilot_chat_model("claude-sonnet-4"));
        assert!(!is_copilot_chat_model("gpt-5.1-codex-mini"));
        assert!(!is_copilot_chat_model("gpt-5.3-codex"));
        assert!(!is_copilot_chat_model("openai/gpt-5.1-codex-mini"));
    }

    #[test]
    fn cloudflare_ai_base_normalizes_v1_suffix() {
        assert_eq!(
            cloudflare_ai_base("https://api.cloudflare.com/client/v4/accounts/abc/ai/v1"),
            Some("https://api.cloudflare.com/client/v4/accounts/abc/ai".to_string())
        );
    }

    #[test]
    fn cloudflare_ai_base_accepts_ai_root() {
        assert_eq!(
            cloudflare_ai_base("https://api.cloudflare.com/client/v4/accounts/abc/ai"),
            Some("https://api.cloudflare.com/client/v4/accounts/abc/ai".to_string())
        );
    }

    #[test]
    fn cloudflare_ai_base_rejects_non_cloudflare() {
        assert_eq!(cloudflare_ai_base("https://api.openai.com/v1"), None);
    }

    #[test]
    fn cloudflare_model_name_prefers_name_over_id() {
        let model: CloudflareModel = serde_json::from_str(
            r#"{"id":"01564c52-8717-47dc-8efd-907a2ca18301","name":"@cf/meta/llama-3.1-8b-instruct"}"#,
        )
        .unwrap();
        assert_eq!(
            cloudflare_model_name(model),
            "@cf/meta/llama-3.1-8b-instruct".to_string()
        );
    }

    #[test]
    fn cloudflare_model_name_falls_back_to_id() {
        let model: CloudflareModel =
            serde_json::from_str(r#"{"id":"01564c52-8717-47dc-8efd-907a2ca18301"}"#).unwrap();
        assert_eq!(
            cloudflare_model_name(model),
            "01564c52-8717-47dc-8efd-907a2ca18301".to_string()
        );
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
