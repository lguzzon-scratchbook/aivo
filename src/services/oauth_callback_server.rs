//! Shared single-shot HTTP callback server for OAuth flows.
//!
//! Provides the raw socket machinery used by both the Codex (fixed-port) and
//! Gemini (ephemeral-port) OAuth callback modules. Each caller re-exports only
//! what its public API needs, keeping the callers as thin ~10-line wrappers.
//!
//! ## Architecture
//!
//! ```text
//!                          ┌──────────────────────────────┐
//!                          │   oauth_callback_server      │
//!                          │                              │
//!                          │  • wait_for_callback(bind)   │
//!                          │  • wait_for_callback_on(bnd) │
//!                          │  • bind_loopback()           │
//!                          │  • extract_callback_params   │
//!                          │  • PortUnavailable           │
//!                          │  • LoopbackBinding           │
//!                          └──────┬───────────────────────┘
//!                                  │
//!                   ┌──────────────┼──────────────┐
//!                   ▼                            ▼
//!     ┌────────────────────────┐    ┌────────────────────────┐
//!     │ codex_oauth_callback   │    │ gemini_oauth_callback  │
//!     │                        │    │                        │
//!     │ wait_for_callback(e,t) │    │ bind_loopback()        │
//!     │ CallbackOutcome        │    │ wait_for_callback(b,t) │
//!     │ PortUnavailable        │    │ LoopbackBinding        │
//!     └────────────────────────┘    └────────────────────────┘
//! ```

use anyhow::{Context, Result, anyhow};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Port-in-use failure. Codex callers catch this to fall back to manual paste.
#[derive(Debug)]
pub struct PortUnavailable;

impl std::fmt::Display for PortUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OAuth callback port is in use; falling back to manual paste")
    }
}

impl std::error::Error for PortUnavailable {}

// ---------------------------------------------------------------------------
// Loopback binding (ephemeral port support)
// ---------------------------------------------------------------------------

/// A bound loopback listener and its OS-assigned port.
///
/// Created by [`bind_loopback`] so callers can embed the port in an OAuth
/// redirect URI before the callback arrives.
pub struct LoopbackBinding {
    pub(crate) listener: TcpListener,
    port: u16,
}

impl LoopbackBinding {
    /// The OS-assigned port number.
    pub fn port(&self) -> u16 {
        self.port
    }
}

/// Bind a new `TcpListener` on `127.0.0.1:0` and return the binding.
///
/// The OS assigns an ephemeral port; call [`LoopbackBinding::port`] to
/// retrieve it.
pub async fn bind_loopback() -> Result<LoopbackBinding> {
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .await
        .context("bind OAuth callback listener on 127.0.0.1:0")?;
    let port = listener.local_addr().context("resolve bound port")?.port();
    Ok(LoopbackBinding { listener, port })
}

// ---------------------------------------------------------------------------
// Public API — wait for OAuth callback
// ---------------------------------------------------------------------------

/// Bind `bind_addr`, then wait for one valid callback on `callback_path`.
///
/// # Parameters
///
/// - `bind_addr` — local address to bind (e.g. `127.0.0.1:1455`).
/// - `callback_path` — path to match (e.g. `/auth/callback`).
/// - `expected_state` — anti-CSRF state to verify.
/// - `timeout` — how long to wait before giving up.
/// - `brand` — human-readable brand name for the success HTML page.
///
/// # Returns
///
/// A `HashMap` with at least `"code"` and optionally `"state"` populated from
/// the callback query parameters.
///
/// # Errors
///
/// Returns [`PortUnavailable`] (wrapped in `anyhow`) when `bind_addr` is
/// already in use. Returns a generic error on state mismatch, missing `code`,
/// provider error param, or timeout.
pub async fn wait_for_callback(
    bind_addr: SocketAddr,
    callback_path: &str,
    expected_state: &str,
    timeout: Duration,
    brand: &str,
) -> Result<HashMap<String, String>> {
    let listener = TcpListener::bind(bind_addr).await.map_err(|e| {
        if matches!(
            e.kind(),
            std::io::ErrorKind::AddrInUse | std::io::ErrorKind::PermissionDenied
        ) {
            anyhow!(PortUnavailable)
        } else {
            anyhow!(e).context("bind OAuth callback listener")
        }
    })?;

    tokio::time::timeout(timeout, accept_one(listener, callback_path, expected_state, brand))
        .await
        .map_err(|_| anyhow!("timed out waiting for OAuth callback"))?
}

/// Like [`wait_for_callback`] but accepts a pre-bound [`LoopbackBinding`] so
/// the caller can embed the port in an authorize URL before listening.
pub async fn wait_for_callback_on(
    binding: LoopbackBinding,
    callback_path: &str,
    expected_state: &str,
    timeout: Duration,
    brand: &str,
) -> Result<HashMap<String, String>> {
    tokio::time::timeout(timeout, accept_one(binding.listener, callback_path, expected_state, brand))
        .await
        .map_err(|_| anyhow!("timed out waiting for OAuth callback"))?
}

// ---------------------------------------------------------------------------
// Internal: accept loop
// ---------------------------------------------------------------------------

async fn accept_one(
    listener: TcpListener,
    callback_path: &str,
    expected_state: &str,
    brand: &str,
) -> Result<HashMap<String, String>> {
    loop {
        let (mut stream, _) = listener.accept().await.context("accept OAuth callback")?;
        let request_line = match read_request_line(&mut stream).await {
            Ok(line) => line,
            Err(_) => {
                let _ = stream.shutdown().await;
                continue;
            }
        };

        let path_and_query = parse_request_target(&request_line);

        if !path_and_query.starts_with(callback_path) {
            respond(&mut stream, 404, "text/plain; charset=utf-8", b"not found").await;
            continue;
        }

        let query = path_and_query.split_once('?').map(|(_, q)| q).unwrap_or("");
        let (code, state, error) = extract_callback_params(query);

        if let Some(err) = error {
            respond(
                &mut stream,
                400,
                "text/plain; charset=utf-8",
                format!("OAuth error: {err}").as_bytes(),
            )
            .await;
            return Err(anyhow!("OAuth provider returned error: {err}"));
        }

        if state.as_deref() != Some(expected_state) {
            respond(
                &mut stream,
                400,
                "text/plain; charset=utf-8",
                b"state mismatch",
            )
            .await;
            return Err(anyhow!("OAuth callback state mismatch"));
        }

        let code_val = code.ok_or_else(|| anyhow!("OAuth callback missing `code`"))?;

        let html = success_html(brand);
        respond(
            &mut stream,
            200,
            "text/html; charset=utf-8",
            html.as_bytes(),
        )
        .await;

        let mut params = HashMap::new();
        params.insert("code".to_string(), code_val);
        if let Some(s) = state {
            params.insert("state".to_string(), s);
        }
        return Ok(params);
    }
}

// ---------------------------------------------------------------------------
// Low-level I/O helpers
// ---------------------------------------------------------------------------

/// Read up to the first CRLF / LF (bounded to 8 KiB).
async fn read_request_line(stream: &mut TcpStream) -> Result<String> {
    let mut buf = [0u8; 8192];
    let mut total = 0usize;
    loop {
        let n = stream.read(&mut buf[total..]).await?;
        if n == 0 {
            break;
        }
        total += n;
        if let Some(end) = find_line_end(&buf[..total]) {
            return Ok(String::from_utf8_lossy(&buf[..end]).into_owned());
        }
        if total == buf.len() {
            break;
        }
    }
    Err(anyhow!("request line missing or too long"))
}

fn find_line_end(bytes: &[u8]) -> Option<usize> {
    for i in 0..bytes.len() {
        if bytes[i] == b'\n' {
            let end = if i > 0 && bytes[i - 1] == b'\r' {
                i - 1
            } else {
                i
            };
            return Some(end);
        }
    }
    None
}

fn parse_request_target(request_line: &str) -> &str {
    // "GET /auth/callback?code=...&state=... HTTP/1.1"
    let mut parts = request_line.split_whitespace();
    let _method = parts.next();
    parts.next().unwrap_or("")
}

/// Extract `(code, state, error)` from a URL-encoded query string.
///
/// Exposed as `pub(crate)` for callers that implement manual-paste fallback
/// (see `codex_oauth::manual_paste_prompt`).
pub(crate) fn extract_callback_params(
    query: &str,
) -> (Option<String>, Option<String>, Option<String>) {
    let mut code = None;
    let mut state = None;
    let mut error = None;
    for pair in query.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (k, v) = match pair.split_once('=') {
            Some(kv) => kv,
            None => (pair, ""),
        };
        let decoded = crate::services::percent_codec::decode(v);
        match k {
            "code" => code = Some(decoded),
            "state" => state = Some(decoded),
            "error" => error = Some(decoded),
            "error_description" if error.is_none() => error = Some(decoded),
            _ => {}
        }
    }
    (code, state, error)
}

async fn respond(stream: &mut TcpStream, status: u16, content_type: &str, body: &[u8]) {
    let status_text = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "",
    };
    let head = format!(
        "HTTP/1.1 {status} {status_text}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         X-Frame-Options: DENY\r\n\
         X-Content-Type-Options: nosniff\r\n\
         Cache-Control: no-store\r\n\
         \r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes()).await;
    let _ = stream.write_all(body).await;
    let _ = stream.shutdown().await;
}

fn success_html(brand: &str) -> String {
    format!(
        r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>aivo — signed in</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           background: #0b0b0e; color: #e5e7eb; display: flex; align-items: center;
           justify-content: center; height: 100vh; margin: 0; }}
    .card {{ text-align: center; padding: 2rem 3rem; border: 1px solid #2a2a31;
            border-radius: 12px; background: #141418; }}
    h1 {{ margin: 0 0 .5rem; font-size: 1.25rem; }}
    p {{ margin: 0; color: #9ca3af; font-size: .95rem; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Signed in to {brand}.</h1>
    <p>You can close this tab and return to your terminal.</p>
  </div>
</body>
</html>"#
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_request_target() {
        assert_eq!(
            parse_request_target("GET /auth/callback?code=abc&state=xyz HTTP/1.1"),
            "/auth/callback?code=abc&state=xyz"
        );
        assert_eq!(parse_request_target("GET / HTTP/1.1"), "/");
    }

    #[test]
    fn extracts_code_and_state() {
        let (code, state, err) = extract_callback_params("code=abc&state=xyz");
        assert_eq!(code.as_deref(), Some("abc"));
        assert_eq!(state.as_deref(), Some("xyz"));
        assert!(err.is_none());
    }

    #[test]
    fn decodes_percent_encoded_code() {
        let (code, _, _) = extract_callback_params("code=a%2Bb%3Dc&state=s");
        assert_eq!(code.as_deref(), Some("a+b=c"));
    }

    #[test]
    fn propagates_error_param() {
        let (code, _, err) = extract_callback_params("error=access_denied");
        assert!(code.is_none());
        assert_eq!(err.as_deref(), Some("access_denied"));
    }

    #[test]
    fn tolerates_empty_query() {
        let (code, state, err) = extract_callback_params("");
        assert!(code.is_none() && state.is_none() && err.is_none());
    }

    #[test]
    fn find_line_end_crlf_and_lf() {
        assert_eq!(find_line_end(b"GET /x\r\n"), Some(6));
        assert_eq!(find_line_end(b"GET /x\n"), Some(6));
        assert_eq!(find_line_end(b"no newline"), None);
    }

    #[tokio::test]
    async fn bind_loopback_assigns_nonzero_port() {
        let b = bind_loopback().await.unwrap();
        assert!(b.port() > 0);
    }
}
