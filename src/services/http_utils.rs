//! Shared HTTP utilities for all built-in routers.
//!
//! Provides common functions for reading HTTP requests from raw TCP streams,
//! parsing headers, extracting bodies, and formatting responses.
//! Used by: claude_code_router, copilot_router, codex_router, gemini_router.

use anyhow::Result;

/// Reads a complete HTTP request from a TCP stream: headers + full body (using Content-Length).
pub async fn read_full_request(socket: &mut tokio::net::TcpStream) -> Result<Vec<u8>> {
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

/// Finds the end of HTTP headers (the position of the first `\r\n\r\n`).
pub fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

/// Parses Content-Length from HTTP headers (case-insensitive).
pub fn parse_content_length(headers: &str) -> Option<usize> {
    headers
        .lines()
        .find(|l| l.to_lowercase().starts_with("content-length:"))
        .and_then(|l| l.split(':').nth(1))
        .and_then(|v| v.trim().parse().ok())
}

/// Extracts the HTTP request body (everything after the blank line separator).
/// Returns an error for malformed requests that are missing `\r\n\r\n`.
pub fn extract_request_body(request: &str) -> Result<&str> {
    let pos = request
        .find("\r\n\r\n")
        .ok_or_else(|| anyhow::anyhow!("malformed HTTP request: missing header separator"))?;
    Ok(request[pos + 4..].trim_end_matches('\0').trim())
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
        429 => "Too Many Requests",
        500 => "Internal Server Error",
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

/// Formats an HTTP response with the correct status line, Content-Type, and body.
pub fn http_response(status: u16, content_type: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        status,
        reason_phrase(status),
        content_type,
        body.len(),
        body
    )
}

/// Formats a JSON error response with the correct HTTP status line.
pub fn http_json_response(status: u16, body: &str) -> String {
    http_response(status, "application/json", body)
}

/// Formats a JSON error response body with an error message.
pub fn http_error_response(status: u16, message: &str) -> String {
    let body = serde_json::json!({"error": {"message": message}}).to_string();
    http_response(status, "application/json", &body)
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

/// Creates a `reqwest::Client` with sensible default timeouts for router use.
pub fn router_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300)) // 5 minute overall timeout
        .connect_timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_extract_request_path() {
        let req = "POST /v1/messages HTTP/1.1\r\nHost: localhost";
        assert_eq!(extract_request_path(req), "/v1/messages");
    }

    #[test]
    fn test_extract_request_path_empty() {
        assert_eq!(extract_request_path(""), "/");
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
        let resp = http_response(200, "application/json", "{\"ok\":true}");
        assert!(resp.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(resp.contains("Content-Type: application/json"));
        assert!(resp.ends_with("{\"ok\":true}"));
    }

    #[test]
    fn test_http_response_error_status() {
        let resp = http_response(500, "application/json", "{\"error\":true}");
        assert!(resp.starts_with("HTTP/1.1 500 Internal Server Error\r\n"));
    }

    #[test]
    fn test_http_error_response() {
        let resp = http_error_response(404, "Not found");
        assert!(resp.contains("404 Not Found"));
        assert!(resp.contains("Not found"));
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
}
