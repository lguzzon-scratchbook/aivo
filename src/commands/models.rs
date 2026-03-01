/**
 * ModelsCommand handler for listing available models from the active provider.
 * Calls /v1/models (OpenAI-compatible) or /v1beta/models (Google Gemini).
 */
use anyhow::Result;
use reqwest::Client;
use serde::Deserialize;

use crate::commands::normalize_base_url;
use crate::errors::ExitCode;
use crate::services::session_store::{ApiKey, SessionStore};
use crate::style;

pub struct ModelsCommand {
    session_store: SessionStore,
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
struct GeminiModel {
    name: String,
}

impl ModelsCommand {
    pub fn new(session_store: SessionStore) -> Self {
        Self { session_store }
    }

    pub async fn execute(&self, key_override: Option<ApiKey>) -> ExitCode {
        match self.execute_internal(key_override).await {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                ExitCode::UserError
            }
        }
    }

    async fn execute_internal(&self, key_override: Option<ApiKey>) -> Result<ExitCode> {
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
        let mut models = fetch_models(&client, &key).await?;
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
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo models"));
        println!("  {}", style::dim("aivo models --key openrouter"));
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
        // Gemini model names are like "models/gemini-1.5-pro"; strip the prefix
        Ok(resp
            .models
            .into_iter()
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
                let resp: OpenAIModelsResponse = response.json().await?;
                return Ok(resp.data.into_iter().map(|m| m.id).collect());
            }

            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            last_err = format!("API returned {} — {}", status, body);
        }

        anyhow::bail!("{}", last_err)
    }
}
