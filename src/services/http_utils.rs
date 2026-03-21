//! Shared HTTP utilities for all built-in routers.
//!
//! Provides common functions for reading HTTP requests from raw TCP streams,
//! parsing headers, extracting bodies, and formatting responses.
//! Used by: anthropic_router, anthropic_to_openai_router, copilot_router,
//! responses_to_chat_router, gemini_router.

use anyhow::Result;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;

use crate::constants::CONTENT_TYPE_JSON;
use std::future::Future;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::services::copilot_auth::{
    COPILOT_EDITOR_VERSION, COPILOT_INTEGRATION_ID, COPILOT_OPENAI_INTENT, CopilotTokenManager,
};

const MAX_REQUEST_HEADER_BYTES: usize = 64 * 1024;
const MAX_REQUEST_BODY_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug)]
pub enum RequestReadError {
    Io(std::io::Error),
    HeaderTooLarge,
    BodyTooLarge { limit: usize },
    UnsupportedTransferEncoding,
    IncompleteHeaders,
}

impl std::fmt::Display for RequestReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error while reading HTTP request: {err}"),
            Self::HeaderTooLarge => write!(
                f,
                "HTTP request headers exceed {} bytes",
                MAX_REQUEST_HEADER_BYTES
            ),
            Self::BodyTooLarge { limit } => {
                write!(f, "HTTP request body exceeds {limit} bytes")
            }
            Self::UnsupportedTransferEncoding => {
                write!(f, "unsupported HTTP Transfer-Encoding: chunked")
            }
            Self::IncompleteHeaders => write!(f, "incomplete HTTP request headers"),
        }
    }
}

impl std::error::Error for RequestReadError {}

impl From<std::io::Error> for RequestReadError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

/// Reads a complete HTTP request from a TCP stream: headers + full body (using Content-Length).
pub async fn read_full_request(
    socket: &mut tokio::net::TcpStream,
) -> std::result::Result<Vec<u8>, RequestReadError> {
    use tokio::io::AsyncReadExt;

    let mut buf = Vec::with_capacity(65536); // 64KB initial capacity
    let mut tmp = vec![0u8; 16384]; // 16KB read buffer

    loop {
        let n = socket.read(&mut tmp).await?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);

        if let Some(expected_len) = inspect_request_buffer(&mut buf)? {
            while buf.len() < expected_len {
                let remaining = expected_len - buf.len();
                let mut body_buf = vec![0u8; remaining.min(tmp.len())];
                socket.read_exact(&mut body_buf).await?;
                buf.extend_from_slice(&body_buf);
            }
            break;
        }
    }

    if inspect_request_buffer(&mut buf)?.is_none() && !buf.is_empty() {
        return Err(RequestReadError::IncompleteHeaders);
    }

    Ok(buf)
}

/// Binds a router listener to a random localhost port and returns the listener and port.
pub async fn bind_local_listener() -> Result<(tokio::net::TcpListener, u16)> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    Ok((listener, port))
}

/// Builds an authorized POST request for OpenAI-compatible upstreams.
///
/// In Copilot mode, exchanges the GitHub token for a short-lived Copilot token
/// and targets the Copilot chat completions endpoint. Otherwise, posts directly
/// to `target_url` with standard bearer auth.
pub async fn authorized_openai_post(
    client: &reqwest::Client,
    target_url: &str,
    api_key: &str,
    copilot_token_manager: Option<&CopilotTokenManager>,
) -> Result<reqwest::RequestBuilder> {
    if let Some(tm) = copilot_token_manager {
        let (token, api_endpoint) = tm.get_token().await?;
        let copilot_url = format!("{}/chat/completions", api_endpoint.trim_end_matches('/'));
        Ok(client
            .post(&copilot_url)
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", CONTENT_TYPE_JSON)
            .header("Editor-Version", COPILOT_EDITOR_VERSION)
            .header("Copilot-Integration-Id", COPILOT_INTEGRATION_ID)
            .header("Openai-Intent", COPILOT_OPENAI_INTENT))
    } else {
        Ok(client
            .post(target_url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", CONTENT_TYPE_JSON))
    }
}

/// Runs a raw TCP HTTP router whose handler returns a complete text HTTP response.
pub async fn run_text_router<State, Handler, Fut>(
    listener: tokio::net::TcpListener,
    state: Arc<State>,
    handler: Handler,
) -> Result<()>
where
    State: Send + Sync + 'static,
    Handler: Fn(String, Arc<State>) -> Fut + Clone + Send + Sync + 'static,
    Fut: Future<Output = String> + Send + 'static,
{
    let semaphore = Arc::new(tokio::sync::Semaphore::new(100));

    loop {
        let (mut socket, _) = listener.accept().await?;
        let state = state.clone();
        let handler = handler.clone();
        let permit = semaphore.clone().acquire_owned().await.unwrap();

        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;

            let _permit = permit;
            let read_result = tokio::time::timeout(
                std::time::Duration::from_secs(30),
                read_full_request(&mut socket),
            )
            .await;

            let request_bytes = match read_result {
                Ok(Ok(b)) => b,
                Ok(Err(err)) => {
                    let response = http_request_read_error_response(&err);
                    let _ = socket.write_all(response.as_bytes()).await;
                    return;
                }
                Err(_) => {
                    let _ = socket
                        .write_all(http_error_response(408, "Request read timed out").as_bytes())
                        .await;
                    return;
                }
            };
            let request = String::from_utf8_lossy(&request_bytes).into_owned();
            let response = handler(request, state).await;
            let _ = socket.write_all(response.as_bytes()).await;
        });
    }
}

/// Finds the end of HTTP headers (the position of the first `\r\n\r\n`).
pub fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

pub fn header_value<'a>(headers: &'a str, name: &str) -> Option<&'a str> {
    headers.lines().find_map(|line| {
        let (header_name, value) = line.split_once(':')?;
        if header_name.trim().eq_ignore_ascii_case(name) {
            Some(value.trim())
        } else {
            None
        }
    })
}

/// Parses Content-Length from HTTP headers (case-insensitive).
pub fn parse_content_length(headers: &str) -> Option<usize> {
    header_value(headers, "content-length").and_then(|v| v.parse().ok())
}

fn has_chunked_transfer_encoding(headers: &str) -> bool {
    header_value(headers, "transfer-encoding")
        .map(|value| {
            value
                .split(',')
                .any(|encoding| encoding.trim().eq_ignore_ascii_case("chunked"))
        })
        .unwrap_or(false)
}

fn inspect_request_buffer(
    buf: &mut Vec<u8>,
) -> std::result::Result<Option<usize>, RequestReadError> {
    let Some(header_end) = find_header_end(buf) else {
        if buf.len() > MAX_REQUEST_HEADER_BYTES {
            return Err(RequestReadError::HeaderTooLarge);
        }
        return Ok(None);
    };

    let header_bytes = header_end + 4;
    if header_bytes > MAX_REQUEST_HEADER_BYTES {
        return Err(RequestReadError::HeaderTooLarge);
    }

    let headers = String::from_utf8_lossy(&buf[..header_end]);
    if has_chunked_transfer_encoding(&headers) {
        return Err(RequestReadError::UnsupportedTransferEncoding);
    }

    let content_length = parse_content_length(&headers).unwrap_or(0);
    if content_length > MAX_REQUEST_BODY_BYTES {
        return Err(RequestReadError::BodyTooLarge {
            limit: MAX_REQUEST_BODY_BYTES,
        });
    }

    let expected_len = header_bytes + content_length;
    if buf.len() > expected_len {
        buf.truncate(expected_len);
    }

    Ok(Some(expected_len))
}

/// Extracts the HTTP request body (everything after the blank line separator).
/// Returns an error for malformed requests that are missing `\r\n\r\n`.
pub fn extract_request_body(request: &str) -> Result<&str> {
    let pos = request
        .find("\r\n\r\n")
        .ok_or_else(|| anyhow::anyhow!("malformed HTTP request: missing header separator"))?;
    Ok(request[pos + 4..].trim_end_matches('\0').trim())
}

/// Extracts request headers that are safe to forward upstream.
///
/// This preserves custom routing metadata sent by tool clients (for example
/// `x-provider`) while excluding hop-by-hop transport headers and headers that
/// the router intentionally manages itself, such as auth and content length.
pub fn extract_passthrough_headers(request: &str) -> Result<HeaderMap> {
    let header_end = find_header_end(request.as_bytes())
        .ok_or_else(|| anyhow::anyhow!("malformed HTTP request: missing header separator"))?;
    let headers = &request[..header_end];
    let mut out = HeaderMap::new();

    for line in headers.lines().skip(1) {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim();
        if !should_passthrough_header(name) {
            continue;
        }
        let Ok(name) = HeaderName::from_bytes(name.as_bytes()) else {
            continue;
        };
        let Ok(value) = HeaderValue::from_str(value.trim()) else {
            continue;
        };
        out.append(name, value);
    }

    Ok(out)
}

fn should_passthrough_header(name: &str) -> bool {
    let lower = name.trim().to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "host"
            | "connection"
            | "content-length"
            | "content-type"
            | "transfer-encoding"
            | "te"
            | "trailer"
            | "proxy-authorization"
            | "proxy-connection"
            | "authorization"
            | "user-agent"
            | "accept-encoding"
            | "api-key"
            | "x-api-key"
            | "x-goog-api-key"
    ) {
        return false;
    }

    lower.starts_with("x-")
        || lower == "anthropic-version"
        || lower == "anthropic-beta"
        || lower.starts_with("anthropic-")
}

/// Extracts the HTTP request path from the first line (e.g., "POST /v1/messages HTTP/1.1" → "/v1/messages").
pub fn extract_request_path(request: &str) -> String {
    let first_line = request.lines().next().unwrap_or("");
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    if parts.len() >= 2 {
        parts[1].to_string()
    } else {
        "/".to_string()
    }
}

/// Returns true when the request is an HTTP POST whose path matches one of `paths`.
pub fn is_post_path(request: &str, paths: &[&str]) -> bool {
    if !request.starts_with("POST ") {
        return false;
    }
    let path = extract_request_path(request);
    let normalized_path = path.split('?').next().unwrap_or(path.as_str());
    paths.contains(&normalized_path)
}

/// Extracts the effective Content-Type from an upstream response.
pub fn response_content_type(response: &reqwest::Response) -> String {
    response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or(CONTENT_TYPE_JSON)
        .to_string()
}

/// Returns the standard HTTP reason phrase for common status codes.
fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        301 => "Moved Permanently",
        302 => "Found",
        304 => "Not Modified",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        413 => "Payload Too Large",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => {
            if status < 300 {
                "OK"
            } else if status < 400 {
                "Redirect"
            } else if status < 500 {
                "Client Error"
            } else {
                "Server Error"
            }
        }
    }
}

/// Returns the pre-formatted CORS header lines (without trailing \r\n\r\n).
pub fn cors_header_block() -> &'static str {
    "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\nAccess-Control-Max-Age: 86400"
}

/// Formats the HTTP response head (status line + headers) without the body.
pub fn http_response_head(status: u16, content_type: &str, content_length: usize) -> String {
    http_response_head_with_extra(status, content_type, content_length, "")
}

/// Formats extra headers as a block to append before the final \r\n\r\n.
fn format_extra_headers(extra: &str) -> String {
    if extra.is_empty() {
        String::new()
    } else {
        format!("\r\n{}", extra)
    }
}

/// Formats the HTTP response head with extra headers injected before the final \r\n\r\n.
pub fn http_response_head_with_extra(
    status: u16,
    content_type: &str,
    content_length: usize,
    extra: &str,
) -> String {
    format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close{}\r\n\r\n",
        status,
        reason_phrase(status),
        content_type,
        content_length,
        format_extra_headers(extra)
    )
}

/// Formats the HTTP response head for chunked transfer encoding.
pub fn http_chunked_response_head(status: u16, content_type: &str) -> String {
    http_chunked_response_head_with_extra(status, content_type, "")
}

/// Formats the chunked HTTP response head with extra headers injected.
pub fn http_chunked_response_head_with_extra(
    status: u16,
    content_type: &str,
    extra: &str,
) -> String {
    format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nTransfer-Encoding: chunked\r\nConnection: close{}\r\n\r\n",
        status,
        reason_phrase(status),
        content_type,
        format_extra_headers(extra)
    )
}

/// Formats an HTTP response with the correct status line, Content-Type, and body.
pub fn http_response(status: u16, content_type: &str, body: &str) -> String {
    format!(
        "{}{}",
        http_response_head(status, content_type, body.len()),
        body
    )
}

/// Converts a buffered upstream response into a raw HTTP response string.
pub async fn buffered_reqwest_to_http_response(response: reqwest::Response) -> Result<String> {
    let status = response.status().as_u16();
    let content_type = response_content_type(&response);
    let body = response.bytes().await?;
    let body = String::from_utf8_lossy(&body);
    Ok(http_response(status, &content_type, &body))
}

/// Formats a JSON error response with the correct HTTP status line.
pub fn http_json_response(status: u16, body: &str) -> String {
    http_response(status, CONTENT_TYPE_JSON, body)
}

/// Formats a JSON error response body with an error message.
pub fn http_error_response(status: u16, message: &str) -> String {
    let body = serde_json::json!({"error": {"message": message}}).to_string();
    http_response(status, CONTENT_TYPE_JSON, &body)
}

pub fn http_request_read_error_response(error: &RequestReadError) -> String {
    match error {
        RequestReadError::HeaderTooLarge | RequestReadError::BodyTooLarge { .. } => {
            http_error_response(413, &error.to_string())
        }
        RequestReadError::UnsupportedTransferEncoding => {
            http_error_response(400, &error.to_string())
        }
        RequestReadError::IncompleteHeaders => http_error_response(400, &error.to_string()),
        RequestReadError::Io(_) => http_error_response(400, &error.to_string()),
    }
}

/// Constructs a target URL, avoiding `/v1` duplication when base already ends with `/v1`.
pub fn build_target_url(base_url: &str, path: &str) -> String {
    let base = base_url.trim_end_matches('/');
    let effective_path = if base.ends_with("/v1") && path.starts_with("/v1/") {
        &path[3..]
    } else {
        path
    };
    format!("{}/{}", base, effective_path.trim_start_matches('/'))
}

/// Constructs a /v1/chat/completions URL, avoiding /v1/v1 duplication.
pub fn build_chat_completions_url(base_url: &str) -> String {
    let base = base_url.trim_end_matches('/');
    if base.ends_with("/v1") {
        format!("{}/chat/completions", base)
    } else {
        format!("{}/v1/chat/completions", base)
    }
}

/// Returns the current Unix timestamp in seconds.
pub fn current_unix_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Parses a JSON value as a `u64`, accepting both JSON numbers and numeric strings.
pub fn parse_token_u64(v: &Value) -> Option<u64> {
    v.as_u64()
        .or_else(|| v.as_str().and_then(|s| s.trim().parse::<u64>().ok()))
}

/// Returns the SSE payload for a `data:` line.
/// Accepts both `data: {...}` and `data:{...}`.
pub fn sse_data_payload(line: &str) -> Option<&str> {
    line.strip_prefix("data:").map(str::trim_start)
}

/// Creates a `reqwest::Client` with a configurable overall timeout.
/// If `secs` is 0, no overall timeout is applied.
pub fn router_http_client_with_timeout(secs: u64) -> reqwest::Client {
    let mut builder = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .pool_max_idle_per_host(10)
        .tcp_keepalive(std::time::Duration::from_secs(60));
    if secs > 0 {
        builder = builder.timeout(std::time::Duration::from_secs(secs));
    }
    builder.build().unwrap_or_else(|_| reqwest::Client::new())
}

/// Creates a `reqwest::Client` with connection pooling for router use.
/// Enables keep-alive for connection reuse across requests.
pub fn router_http_client() -> reqwest::Client {
    router_http_client_with_timeout(300)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_find_header_end() {
        let buf = b"POST /v1 HTTP/1.1\r\nHost: localhost\r\n\r\nbody";
        assert_eq!(find_header_end(buf), Some(34));
    }

    #[test]
    fn test_find_header_end_none() {
        let buf = b"POST /v1 HTTP/1.1\r\nHost: localhost";
        assert_eq!(find_header_end(buf), None);
    }

    #[test]
    fn test_parse_content_length() {
        let headers = "POST /v1 HTTP/1.1\r\nContent-Length: 42\r\nHost: localhost";
        assert_eq!(parse_content_length(headers), Some(42));
    }

    #[test]
    fn test_parse_content_length_case_insensitive() {
        let headers = "POST /v1 HTTP/1.1\r\ncontent-length: 100\r\nHost: localhost";
        assert_eq!(parse_content_length(headers), Some(100));
    }

    #[test]
    fn test_parse_content_length_missing() {
        let headers = "POST /v1 HTTP/1.1\r\nHost: localhost";
        assert_eq!(parse_content_length(headers), None);
    }

    #[test]
    fn test_has_chunked_transfer_encoding() {
        let headers = "POST /v1 HTTP/1.1\r\nTransfer-Encoding: gzip, chunked\r\nHost: localhost";
        assert!(has_chunked_transfer_encoding(headers));
    }

    #[test]
    fn test_inspect_request_buffer_rejects_large_header() {
        let mut buf = vec![b'a'; MAX_REQUEST_HEADER_BYTES + 1];
        let err = inspect_request_buffer(&mut buf).unwrap_err();
        assert!(matches!(err, RequestReadError::HeaderTooLarge));
    }

    #[test]
    fn test_inspect_request_buffer_rejects_large_body() {
        let mut buf = format!(
            "POST /v1 HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
            MAX_REQUEST_BODY_BYTES + 1
        )
        .into_bytes();
        let err = inspect_request_buffer(&mut buf).unwrap_err();
        assert!(matches!(err, RequestReadError::BodyTooLarge { .. }));
    }

    #[test]
    fn test_inspect_request_buffer_rejects_chunked_requests() {
        let mut buf =
            b"POST /v1 HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n4\r\ntest\r\n".to_vec();
        let err = inspect_request_buffer(&mut buf).unwrap_err();
        assert!(matches!(err, RequestReadError::UnsupportedTransferEncoding));
    }

    #[test]
    fn test_inspect_request_buffer_truncates_to_content_length() {
        let mut buf =
            b"POST /v1 HTTP/1.1\r\nContent-Length: 4\r\n\r\ntestEXTRA_TRAILING_BYTES".to_vec();
        let expected_len = inspect_request_buffer(&mut buf).unwrap().unwrap();
        assert_eq!(expected_len, buf.len());
        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            "POST /v1 HTTP/1.1\r\nContent-Length: 4\r\n\r\ntest"
        );
    }

    #[test]
    fn test_extract_request_body() {
        let req =
            "POST /v1/messages HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"key\":\"val\"}";
        assert_eq!(extract_request_body(req).unwrap(), "{\"key\":\"val\"}");
    }

    #[test]
    fn test_extract_request_body_missing_separator() {
        let req = "POST /v1/messages HTTP/1.1";
        assert!(extract_request_body(req).is_err());
    }

    #[test]
    fn test_extract_request_body_short() {
        assert!(extract_request_body("AB").is_err());
    }

    #[test]
    fn test_extract_passthrough_headers_keeps_custom_provider_headers() {
        let req = concat!(
            "POST /v1/messages HTTP/1.1\r\n",
            "Host: localhost:8080\r\n",
            "Authorization: Bearer local-token\r\n",
            "x-api-key: upstream-token\r\n",
            "Content-Type: application/json\r\n",
            "x-provider: anthropic\r\n",
            "x-vercel-ai-gateway-team: team_123\r\n",
            "anthropic-beta: prompt-caching-2024-07-31\r\n",
            "\r\n",
            "{}"
        );

        let headers = extract_passthrough_headers(req).unwrap();
        assert_eq!(
            headers.get("x-provider").and_then(|v| v.to_str().ok()),
            Some("anthropic")
        );
        assert_eq!(
            headers
                .get("x-vercel-ai-gateway-team")
                .and_then(|v| v.to_str().ok()),
            Some("team_123")
        );
        assert_eq!(
            headers.get("anthropic-beta").and_then(|v| v.to_str().ok()),
            Some("prompt-caching-2024-07-31")
        );
        assert!(headers.get("authorization").is_none());
        assert!(headers.get("x-api-key").is_none());
        assert!(headers.get("content-type").is_none());
    }

    #[test]
    fn test_extract_passthrough_headers_requires_header_separator() {
        assert!(extract_passthrough_headers("POST /v1/messages HTTP/1.1").is_err());
    }

    #[test]
    fn test_extract_request_path() {
        let req = "POST /v1/messages HTTP/1.1\r\nHost: localhost";
        assert_eq!(extract_request_path(req), "/v1/messages");
    }

    #[test]
    fn test_extract_request_path_empty() {
        assert_eq!(extract_request_path(""), "/");
    }

    #[test]
    fn test_is_post_path_matches_supported_path() {
        let req = "POST /v1/messages HTTP/1.1\r\nHost: localhost";
        assert!(is_post_path(req, &["/v1/messages", "/messages"]));
    }

    #[test]
    fn test_is_post_path_ignores_query_string() {
        let req = "POST /v1/messages?beta=true HTTP/1.1\r\nHost: localhost";
        assert!(is_post_path(req, &["/v1/messages", "/messages"]));
    }

    #[test]
    fn test_is_post_path_rejects_wrong_method_or_path() {
        let get_req = "GET /v1/messages HTTP/1.1\r\nHost: localhost";
        let other_req = "POST /health HTTP/1.1\r\nHost: localhost";
        assert!(!is_post_path(get_req, &["/v1/messages"]));
        assert!(!is_post_path(other_req, &["/v1/messages"]));
    }

    #[test]
    fn test_reason_phrase() {
        assert_eq!(reason_phrase(200), "OK");
        assert_eq!(reason_phrase(400), "Bad Request");
        assert_eq!(reason_phrase(404), "Not Found");
        assert_eq!(reason_phrase(500), "Internal Server Error");
    }

    #[test]
    fn test_http_response_format() {
        let resp = http_response(200, CONTENT_TYPE_JSON, "{\"ok\":true}");
        assert!(resp.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(resp.contains("Content-Type: application/json"));
        assert!(resp.ends_with("{\"ok\":true}"));
    }

    #[test]
    fn test_http_response_head_format() {
        let head = http_response_head(200, CONTENT_TYPE_JSON, 11);
        assert_eq!(
            head,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\nConnection: close\r\n\r\n"
        );
    }

    #[test]
    fn test_http_chunked_response_head_format() {
        let head = http_chunked_response_head(200, "text/event-stream");
        assert_eq!(
            head,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
        );
    }

    #[test]
    fn test_http_response_error_status() {
        let resp = http_response(500, CONTENT_TYPE_JSON, "{\"error\":true}");
        assert!(resp.starts_with("HTTP/1.1 500 Internal Server Error\r\n"));
    }

    #[test]
    fn test_http_error_response() {
        let resp = http_error_response(404, "Not found");
        assert!(resp.contains("404 Not Found"));
        assert!(resp.contains("Not found"));
    }

    #[test]
    fn test_http_request_read_error_response_uses_413_for_size_limits() {
        let resp = http_request_read_error_response(&RequestReadError::BodyTooLarge { limit: 123 });
        assert!(resp.starts_with("HTTP/1.1 413 Payload Too Large\r\n"));
    }

    #[test]
    fn test_build_target_url_with_v1() {
        assert_eq!(
            build_target_url("https://api.example.com/v1", "/v1/chat/completions"),
            "https://api.example.com/v1/chat/completions"
        );
    }

    #[test]
    fn test_build_target_url_without_v1() {
        assert_eq!(
            build_target_url("https://api.example.com", "/v1/chat/completions"),
            "https://api.example.com/v1/chat/completions"
        );
    }

    #[test]
    fn test_build_chat_completions_url() {
        assert_eq!(
            build_chat_completions_url("https://ai-gateway.vercel.sh/v1"),
            "https://ai-gateway.vercel.sh/v1/chat/completions"
        );
        assert_eq!(
            build_chat_completions_url("https://example.com"),
            "https://example.com/v1/chat/completions"
        );
    }

    #[test]
    fn test_build_target_url_trailing_slash() {
        assert_eq!(
            build_target_url("https://example.com/v1/", "/chat/completions"),
            "https://example.com/v1/chat/completions"
        );
    }

    #[test]
    fn test_build_chat_completions_url_trailing_slash() {
        assert_eq!(
            build_chat_completions_url("https://example.com/v1/"),
            "https://example.com/v1/chat/completions"
        );
    }

    #[test]
    fn test_reason_phrase_uncommon_status_codes() {
        assert_eq!(reason_phrase(201), "Created");
        assert_eq!(reason_phrase(204), "No Content");
        assert_eq!(reason_phrase(301), "Moved Permanently");
        assert_eq!(reason_phrase(302), "Found");
        assert_eq!(reason_phrase(304), "Not Modified");
        assert_eq!(reason_phrase(405), "Method Not Allowed");
        assert_eq!(reason_phrase(408), "Request Timeout");
        assert_eq!(reason_phrase(413), "Payload Too Large");
        assert_eq!(reason_phrase(429), "Too Many Requests");
        assert_eq!(reason_phrase(502), "Bad Gateway");
        assert_eq!(reason_phrase(503), "Service Unavailable");
        assert_eq!(reason_phrase(504), "Gateway Timeout");
    }

    #[test]
    fn test_reason_phrase_unknown_ranges() {
        assert_eq!(reason_phrase(299), "OK");
        assert_eq!(reason_phrase(399), "Redirect");
        assert_eq!(reason_phrase(499), "Client Error");
        assert_eq!(reason_phrase(599), "Server Error");
    }

    #[test]
    fn test_http_error_response_json_structure() {
        let resp = http_error_response(422, "Validation failed");
        assert!(resp.contains("422"));
        assert!(resp.contains("Validation failed"));
        assert!(resp.contains("application/json"));
    }

    #[test]
    fn test_http_request_read_error_response_header_too_large() {
        let resp = http_request_read_error_response(&RequestReadError::HeaderTooLarge);
        assert!(resp.starts_with("HTTP/1.1 413"));
    }

    #[test]
    fn test_http_request_read_error_response_unsupported_encoding() {
        let resp = http_request_read_error_response(&RequestReadError::UnsupportedTransferEncoding);
        assert!(resp.starts_with("HTTP/1.1 400"));
        assert!(resp.contains("chunked"));
    }

    #[test]
    fn test_http_request_read_error_response_incomplete_headers() {
        let resp = http_request_read_error_response(&RequestReadError::IncompleteHeaders);
        assert!(resp.starts_with("HTTP/1.1 400"));
    }

    #[test]
    fn test_parse_content_length_invalid_value() {
        let headers = "POST /v1 HTTP/1.1\r\nContent-Length: not_a_number\r\n";
        assert_eq!(parse_content_length(headers), None);
    }

    #[test]
    fn test_sse_data_payload_with_space() {
        assert_eq!(
            sse_data_payload("data: {\"ok\":true}"),
            Some("{\"ok\":true}")
        );
    }

    #[test]
    fn test_sse_data_payload_without_space() {
        assert_eq!(
            sse_data_payload("data:{\"ok\":true}"),
            Some("{\"ok\":true}")
        );
    }

    #[test]
    fn test_sse_data_payload_non_data_line() {
        assert_eq!(sse_data_payload("event: message"), None);
        assert_eq!(sse_data_payload(""), None);
    }

    #[test]
    fn test_parse_token_u64_number() {
        assert_eq!(parse_token_u64(&json!(42)), Some(42));
    }

    #[test]
    fn test_parse_token_u64_string() {
        assert_eq!(parse_token_u64(&json!("100")), Some(100));
    }

    #[test]
    fn test_parse_token_u64_invalid_string() {
        assert_eq!(parse_token_u64(&json!("not_a_number")), None);
    }

    #[test]
    fn test_parse_token_u64_null() {
        assert_eq!(parse_token_u64(&json!(null)), None);
    }

    #[test]
    fn test_extract_request_path_single_word() {
        assert_eq!(extract_request_path("GET"), "/");
    }

    #[test]
    fn test_is_post_path_empty_paths() {
        let req = "POST /v1/messages HTTP/1.1\r\n";
        assert!(!is_post_path(req, &[]));
    }

    #[test]
    fn test_extract_passthrough_headers_no_custom_headers() {
        let req = "POST /v1 HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer token\r\n\r\n{}";
        let headers = extract_passthrough_headers(req).unwrap();
        assert!(headers.is_empty());
    }
}
