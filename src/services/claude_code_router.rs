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

enum RouterResponse {
    Buffered {
        status: u16,
        content_type: String,
        body: Vec<u8>,
    },
    Streaming {
        status: u16,
        content_type: String,
        upstream: reqwest::Response,
    },
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
            let path = http_utils::extract_request_path(&request);
            let path = path.split('?').next().unwrap_or("");
            let method_is_post = request.starts_with("POST ");

            let result = if method_is_post && is_count_tokens_path(path) {
                handle_count_tokens_raw(&request, &config).await
            } else if method_is_post && is_messages_path(path) {
                handle_messages_raw(&request, &config).await
            } else if method_is_post && is_chat_completions_path(path) {
                handle_chat_completions_raw(&request, &config).await
            } else {
                let not_found =
                    http_utils::http_response(404, "application/json", "{\"error\":\"Not found\"}");
                let _ = socket.write_all(not_found.as_bytes()).await;
                return;
            };

            match result {
                Ok(resp) => {
                    let _ = write_router_response(&mut socket, resp).await;
                }
                Err(_) => {
                    let err = http_utils::http_error_response(500, "Internal Server Error");
                    let _ = socket.write_all(err.as_bytes()).await;
                }
            }
        });
    }
}

fn is_messages_path(path: &str) -> bool {
    matches!(path, "/v1/messages" | "/messages")
}

fn is_chat_completions_path(path: &str) -> bool {
    matches!(path, "/v1/chat/completions" | "/chat/completions")
}

fn is_count_tokens_path(path: &str) -> bool {
    matches!(path, "/v1/messages/count_tokens" | "/messages/count_tokens")
}

fn build_endpoint_url(base_url: &str, endpoint: &str) -> String {
    let base = base_url.trim_end_matches('/');
    if base.ends_with("/v1") {
        format!("{}/{}", base, endpoint.trim_start_matches('/'))
    } else {
        format!("{}/v1/{}", base, endpoint.trim_start_matches('/'))
    }
}

fn maybe_transform_model(body: &mut Value, base_url: &str) {
    if let Some(model) = body.get_mut("model")
        && let Some(model_str) = model.as_str()
    {
        *model = Value::String(transform_model_for_provider(base_url, model_str));
    }
}

fn status_reason_phrase(status: u16) -> &'static str {
    reqwest::StatusCode::from_u16(status)
        .ok()
        .and_then(|s| s.canonical_reason())
        .unwrap_or("OK")
}

async fn classify_upstream_response(response: reqwest::Response) -> Result<RouterResponse> {
    let status = response.status().as_u16();
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();

    if content_type.contains("text/event-stream") {
        Ok(RouterResponse::Streaming {
            status,
            content_type,
            upstream: response,
        })
    } else {
        let body = response.bytes().await?.to_vec();
        Ok(RouterResponse::Buffered {
            status,
            content_type,
            body,
        })
    }
}

async fn write_router_response(
    socket: &mut tokio::net::TcpStream,
    response: RouterResponse,
) -> Result<()> {
    use tokio::io::AsyncWriteExt;

    match response {
        RouterResponse::Buffered {
            status,
            content_type,
            body,
        } => {
            let headers = format!(
                "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                status,
                status_reason_phrase(status),
                content_type,
                body.len()
            );
            socket.write_all(headers.as_bytes()).await?;
            socket.write_all(&body).await?;
        }
        RouterResponse::Streaming {
            status,
            content_type,
            mut upstream,
        } => {
            let headers = format!(
                "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
                status,
                status_reason_phrase(status),
                content_type
            );
            socket.write_all(headers.as_bytes()).await?;

            while let Some(chunk) = upstream.chunk().await? {
                if chunk.is_empty() {
                    continue;
                }
                let chunk_len = format!("{:X}\r\n", chunk.len());
                socket.write_all(chunk_len.as_bytes()).await?;
                socket.write_all(&chunk).await?;
                socket.write_all(b"\r\n").await?;
            }

            socket.write_all(b"0\r\n\r\n").await?;
        }
    }
    Ok(())
}

async fn handle_messages_raw(request: &str, config: &Arc<RouterConfig>) -> Result<RouterResponse> {
    let body_str = http_utils::extract_request_body(request)?;

    let mut body: Value = serde_json::from_str(body_str)?;
    maybe_transform_model(&mut body, &config.openrouter_base_url);

    let client = router_http_client();
    let url = build_endpoint_url(&config.openrouter_base_url, "messages");

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

    classify_upstream_response(response).await
}

async fn handle_count_tokens_raw(
    request: &str,
    config: &Arc<RouterConfig>,
) -> Result<RouterResponse> {
    let body_str = http_utils::extract_request_body(request)?;

    let mut body: Value = serde_json::from_str(body_str)?;
    maybe_transform_model(&mut body, &config.openrouter_base_url);

    let client = router_http_client();
    let url = build_endpoint_url(&config.openrouter_base_url, "messages/count_tokens");

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

    classify_upstream_response(response).await
}

async fn handle_chat_completions_raw(
    request: &str,
    config: &Arc<RouterConfig>,
) -> Result<RouterResponse> {
    let body_str = http_utils::extract_request_body(request)?;

    let mut body: Value = serde_json::from_str(body_str)?;
    maybe_transform_model(&mut body, &config.openrouter_base_url);

    let client = router_http_client();
    let url = build_endpoint_url(&config.openrouter_base_url, "chat/completions");

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

    classify_upstream_response(response).await
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

    #[test]
    fn test_endpoint_path_matching() {
        assert!(is_messages_path("/v1/messages"));
        assert!(is_chat_completions_path("/v1/chat/completions"));
        assert!(is_count_tokens_path("/v1/messages/count_tokens"));
        assert!(!is_messages_path("/v1/messages/count_tokens"));
    }

    #[test]
    fn test_build_endpoint_url() {
        assert_eq!(
            build_endpoint_url("https://openrouter.ai/api/v1", "messages"),
            "https://openrouter.ai/api/v1/messages"
        );
        assert_eq!(
            build_endpoint_url("https://openrouter.ai/api", "chat/completions"),
            "https://openrouter.ai/api/v1/chat/completions"
        );
        assert_eq!(
            build_endpoint_url("https://openrouter.ai/api/v1/", "messages/count_tokens"),
            "https://openrouter.ai/api/v1/messages/count_tokens"
        );
    }
}
