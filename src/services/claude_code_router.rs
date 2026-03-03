/**
 * Built-in Claude Code Router service
 *
 * Acts as an HTTP proxy that intercepts Claude Code requests and routes them
 * to OpenRouter, handling all necessary API transformations.
 */
use anyhow::Result;
use serde_json::Value;
use std::sync::Arc;

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

            let request_bytes = match read_full_request(&mut socket).await {
                Ok(b) => b,
                Err(_) => return,
            };

            let request = String::from_utf8_lossy(&request_bytes);

            let response = if request.contains("POST /v1/messages") {
                match handle_messages_raw(&request, &config).await {
                    Ok(r) => r,
                    Err(_) => "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 21\r\n\r\nInternal Server Error".to_string(),
                }
            } else if request.starts_with("POST /v1/chat/completions") {
                match handle_chat_completions_raw(&request, &config).await {
                    Ok(r) => r,
                    Err(_) => "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 21\r\n\r\nInternal Server Error".to_string(),
                }
            } else {
                "HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\n\r\nNot found".to_string()
            };

            let _ = socket.write_all(response.as_bytes()).await;
        });
    }
}

/// Reads a complete HTTP request: headers + full body (using Content-Length)
async fn read_full_request(socket: &mut tokio::net::TcpStream) -> Result<Vec<u8>> {
    use tokio::io::AsyncReadExt;

    let mut buf = Vec::with_capacity(16384);
    let mut tmp = vec![0u8; 4096];

    loop {
        let n = socket.read(&mut tmp).await?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);

        if let Some(header_end) = find_header_end(&buf) {
            let headers = String::from_utf8_lossy(&buf[..header_end]);
            let content_length = parse_content_length(&headers).unwrap_or(0);
            let body_read = buf.len() - (header_end + 4);

            if body_read < content_length {
                let remaining = content_length - body_read;
                let mut body_buf = vec![0u8; remaining];
                socket.read_exact(&mut body_buf).await?;
                buf.extend_from_slice(&body_buf);
            }
            break;
        }
    }

    Ok(buf)
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn parse_content_length(headers: &str) -> Option<usize> {
    headers
        .lines()
        .find(|l| l.to_lowercase().starts_with("content-length:"))
        .and_then(|l| l.split(':').nth(1))
        .and_then(|v| v.trim().parse().ok())
}

async fn handle_messages_raw(request: &str, config: &Arc<RouterConfig>) -> Result<String> {
    let body_str = extract_request_body(request)?;

    let mut body: Value = serde_json::from_str(body_str)?;

    if let Some(model) = body.get_mut("model")
        && let Some(model_str) = model.as_str()
    {
        *model = Value::String(transform_model(&config.openrouter_base_url, model_str));
    }

    let client = reqwest::Client::new();
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

    Ok(format!(
        "HTTP/1.1 {} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        status_code,
        response_body.len(),
        response_body
    ))
}

async fn handle_chat_completions_raw(request: &str, config: &Arc<RouterConfig>) -> Result<String> {
    let body_str = extract_request_body(request)?;

    let mut body: Value = serde_json::from_str(body_str)?;

    if let Some(model) = body.get_mut("model")
        && let Some(model_str) = model.as_str()
    {
        *model = Value::String(transform_model(&config.openrouter_base_url, model_str));
    }

    let client = reqwest::Client::new();
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

    Ok(format!(
        "HTTP/1.1 {} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        status_code,
        response_body.len(),
        response_body
    ))
}

/// Extracts the HTTP request body (everything after the blank line separator).
/// Returns an error for malformed requests that are missing `\r\n\r\n`.
fn extract_request_body(request: &str) -> Result<&str> {
    let pos = request
        .find("\r\n\r\n")
        .ok_or_else(|| anyhow::anyhow!("malformed HTTP request: missing header separator"))?;
    Ok(request[pos + 4..].trim_end_matches('\0').trim())
}

/// Transforms model names based on the provider
fn transform_model(base_url: &str, model: &str) -> String {
    if base_url.contains("openrouter") {
        transform_model_for_openrouter(model)
    } else {
        model.to_string()
    }
}

/// Transforms model names from Claude format to OpenRouter format:
/// - Adds anthropic/ prefix
/// - Converts version hyphens to dots (4-6 -> 4.6), but preserves date suffixes
fn transform_model_for_openrouter(model: &str) -> String {
    if !model.starts_with("claude-") || model.starts_with("anthropic/") {
        return model.to_string();
    }
    format!("anthropic/{}", normalize_claude_version(model))
}

/// Converts claude-sonnet-4-6 -> claude-sonnet-4.6
/// Leaves date suffixes intact: claude-haiku-4-5-20251001 stays as-is
fn normalize_claude_version(model: &str) -> String {
    if let Some(last_hyphen_pos) = model.rfind('-') {
        let after_last_hyphen = &model[last_hyphen_pos + 1..];

        // Date suffix (8 digits): keep as-is
        if after_last_hyphen.len() == 8 && after_last_hyphen.chars().all(|c| c.is_ascii_digit()) {
            return model.to_string();
        }

        // Version number: convert the separating hyphen to a dot
        if after_last_hyphen
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_digit())
            && let Some(second_last_hyphen) = model[..last_hyphen_pos].rfind('-')
            && model[second_last_hyphen + 1..last_hyphen_pos]
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_digit())
        {
            let mut result = model.to_string();
            result.replace_range(last_hyphen_pos..=last_hyphen_pos, ".");
            return result;
        }
    }
    model.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_openrouter_adds_prefix_and_normalizes() {
        let url = "https://openrouter.ai/api/v1";
        assert_eq!(
            transform_model(url, "claude-sonnet-4-6"),
            "anthropic/claude-sonnet-4.6"
        );
        assert_eq!(
            transform_model(url, "claude-opus-4-6"),
            "anthropic/claude-opus-4.6"
        );
        assert_eq!(
            transform_model(url, "claude-haiku-4-5"),
            "anthropic/claude-haiku-4.5"
        );
    }

    #[test]
    fn test_transform_openrouter_date_suffix_preserved() {
        assert_eq!(
            transform_model("https://openrouter.ai/api/v1", "claude-haiku-4-5-20251001"),
            "anthropic/claude-haiku-4-5-20251001"
        );
    }

    #[test]
    fn test_transform_other_provider_passthrough() {
        // Non-OpenRouter providers: model names pass through unchanged
        assert_eq!(
            transform_model("https://ai-gateway.vercel.sh/v1", "claude-sonnet-4-6"),
            "claude-sonnet-4-6"
        );
        assert_eq!(
            transform_model("https://api.example.com/v1", "claude-opus-4-6"),
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
        assert_eq!(extract_request_body(req).unwrap(), "{\"key\":\"val\"}");
    }

    #[test]
    fn test_extract_request_body_missing_separator_returns_error() {
        let req = "POST /v1/messages HTTP/1.1";
        assert!(extract_request_body(req).is_err());
    }

    #[test]
    fn test_extract_request_body_short_request_no_panic() {
        // A request shorter than 4 bytes must not panic
        assert!(extract_request_body("AB").is_err());
    }
}
