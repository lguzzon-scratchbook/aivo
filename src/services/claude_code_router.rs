/**
 * Built-in Claude Code Router service
 *
 * Acts as an HTTP proxy that intercepts Claude Code requests and routes them
 * to OpenRouter, handling all necessary API transformations.
 */
use anyhow::Result;
use serde_json::Value;
use std::sync::Arc;

use crate::services::http_utils::{self, router_http_client};
use crate::services::model_names::transform_model_for_provider;

#[derive(Clone)]
pub struct RouterConfig {
    pub openrouter_base_url: String,
    pub openrouter_api_key: String,
}

pub struct ClaudeCodeRouter {
    config: RouterConfig,
}

impl ClaudeCodeRouter {
    pub fn new(config: RouterConfig) -> Self {
        Self { config }
    }

    /// Binds to a random available port and starts the router in the background.
    /// Returns the actual port number so callers can set ANTHROPIC_BASE_URL.
    pub async fn start_background(&self) -> Result<(u16, tokio::task::JoinHandle<Result<()>>)> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let port = listener.local_addr()?.port();
        let config = self.config.clone();
        let handle = tokio::spawn(async move { run_router(listener, config).await });
        Ok((port, handle))
    }
}

async fn run_router(listener: tokio::net::TcpListener, config: RouterConfig) -> Result<()> {
    let config = Arc::new(config);

    loop {
        let (mut socket, _) = listener.accept().await?;
        let config = config.clone();

        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;

            let request_bytes = match http_utils::read_full_request(&mut socket).await {
                Ok(b) => b,
                Err(_) => return,
            };

            let request = String::from_utf8_lossy(&request_bytes);

            let response = if request.contains("POST /v1/messages") {
                match handle_messages_raw(&request, &config).await {
                    Ok(r) => r,
                    Err(_) => http_utils::http_error_response(500, "Internal Server Error"),
                }
            } else if request.starts_with("POST /v1/chat/completions") {
                match handle_chat_completions_raw(&request, &config).await {
                    Ok(r) => r,
                    Err(_) => http_utils::http_error_response(500, "Internal Server Error"),
                }
            } else {
                http_utils::http_response(404, "application/json", "{\"error\":\"Not found\"}")
            };

            let _ = socket.write_all(response.as_bytes()).await;
        });
    }
}

async fn handle_messages_raw(request: &str, config: &Arc<RouterConfig>) -> Result<String> {
    let body_str = http_utils::extract_request_body(request)?;

    let mut body: Value = serde_json::from_str(body_str)?;

    if let Some(model) = body.get_mut("model")
        && let Some(model_str) = model.as_str()
    {
        *model = Value::String(transform_model_for_provider(
            &config.openrouter_base_url,
            model_str,
        ));
    }

    let client = router_http_client();
    let base = config.openrouter_base_url.trim_end_matches('/');
    let url = if base.ends_with("/v1") {
        format!("{}/messages", base)
    } else {
        format!("{}/v1/messages", base)
    };

    let response = client
        .post(&url)
        .header(
            "Authorization",
            format!("Bearer {}", config.openrouter_api_key),
        )
        .header("Content-Type", "application/json")
        .header("anthropic-version", "2023-06-01")
        .json(&body)
        .send()
        .await?;

    let status_code = response.status().as_u16();
    let response_body = response.text().await?;

    Ok(http_utils::http_json_response(status_code, &response_body))
}

async fn handle_chat_completions_raw(request: &str, config: &Arc<RouterConfig>) -> Result<String> {
    let body_str = http_utils::extract_request_body(request)?;

    let mut body: Value = serde_json::from_str(body_str)?;

    if let Some(model) = body.get_mut("model")
        && let Some(model_str) = model.as_str()
    {
        *model = Value::String(transform_model_for_provider(
            &config.openrouter_base_url,
            model_str,
        ));
    }

    let client = router_http_client();
    let base = config.openrouter_base_url.trim_end_matches('/');
    let url = if base.ends_with("/v1") {
        format!("{}/chat/completions", base)
    } else {
        format!("{}/v1/chat/completions", base)
    };

    let response = client
        .post(&url)
        .header(
            "Authorization",
            format!("Bearer {}", config.openrouter_api_key),
        )
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await?;

    let status_code = response.status().as_u16();
    let response_body = response.text().await?;

    Ok(http_utils::http_json_response(status_code, &response_body))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::model_names::{normalize_claude_version, transform_model_for_openrouter};

    #[test]
    fn test_transform_openrouter_adds_prefix_and_normalizes() {
        let url = "https://openrouter.ai/api/v1";
        assert_eq!(
            transform_model_for_provider(url, "claude-sonnet-4-6"),
            "anthropic/claude-sonnet-4.6"
        );
        assert_eq!(
            transform_model_for_provider(url, "claude-opus-4-6"),
            "anthropic/claude-opus-4.6"
        );
        assert_eq!(
            transform_model_for_provider(url, "claude-haiku-4-5"),
            "anthropic/claude-haiku-4.5"
        );
    }

    #[test]
    fn test_transform_openrouter_date_suffix_preserved() {
        assert_eq!(
            transform_model_for_provider(
                "https://openrouter.ai/api/v1",
                "claude-haiku-4-5-20251001"
            ),
            "anthropic/claude-haiku-4-5-20251001"
        );
    }

    #[test]
    fn test_transform_other_provider_passthrough() {
        // Non-OpenRouter providers: model names pass through unchanged
        assert_eq!(
            transform_model_for_provider("https://ai-gateway.vercel.sh/v1", "claude-sonnet-4-6"),
            "claude-sonnet-4-6"
        );
        assert_eq!(
            transform_model_for_provider("https://api.example.com/v1", "claude-opus-4-6"),
            "claude-opus-4-6"
        );
    }

    #[test]
    fn test_transform_already_prefixed() {
        assert_eq!(
            transform_model_for_openrouter("anthropic/claude-sonnet-4.6"),
            "anthropic/claude-sonnet-4.6"
        );
    }

    #[test]
    fn test_transform_non_claude_model() {
        assert_eq!(transform_model_for_openrouter("gpt-4o"), "gpt-4o");
    }

    #[test]
    fn test_normalize_claude_version() {
        assert_eq!(
            normalize_claude_version("claude-sonnet-4-6"),
            "claude-sonnet-4.6"
        );
        assert_eq!(
            normalize_claude_version("claude-haiku-4-5-20251001"),
            "claude-haiku-4-5-20251001"
        );
    }

    #[test]
    fn test_extract_request_body_normal() {
        let req =
            "POST /v1/messages HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"key\":\"val\"}";
        assert_eq!(
            http_utils::extract_request_body(req).unwrap(),
            "{\"key\":\"val\"}"
        );
    }

    #[test]
    fn test_extract_request_body_missing_separator_returns_error() {
        let req = "POST /v1/messages HTTP/1.1";
        assert!(http_utils::extract_request_body(req).is_err());
    }

    #[test]
    fn test_extract_request_body_short_request_no_panic() {
        // A request shorter than 4 bytes must not panic
        assert!(http_utils::extract_request_body("AB").is_err());
    }
}
