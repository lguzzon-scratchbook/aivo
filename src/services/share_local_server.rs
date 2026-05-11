//! Tiny HTTP/1.1 server that serves a single `SharePayload` for one share
//! session. Hand-rolled on `tokio::net::TcpListener`, mirroring the pattern
//! used by `serve_router.rs`. Reuses `http_utils` for request reading and
//! response formatting.
//!
//! Routes:
//! - `GET /state` — combined `{ meta, payload }` JSON, with ETag for cheap polling.
//! - `HEAD /state` — ETag-only freshness check (returns 304 when matched).
//!
//! In `--live` mode the payload is re-resolved + re-redacted on each
//! `GET /state` call so the recipient sees ongoing updates. Snapshot mode
//! serves the cached bytes verbatim.

use std::sync::Arc;

use anyhow::Result;
use serde_json::json;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::sync::RwLock;

use crate::services::device_fingerprint::hex_sha256;
use crate::services::http_utils::{
    self, http_error_response, http_response_head_with_extra, read_full_request,
};
use crate::services::share_payload::SharePayload;
use crate::services::share_redact::{RedactCtx, redact};
use crate::services::share_resolver::{ResolverContext, resolve_session};

/// CORS allowlist origin for the share viewer. Matches `DEFAULT_SHARE_BASE_URL`
/// at the protocol+host level.
const VIEWER_ORIGIN: &str = "https://s.getaivo.dev";

/// What the local server has loaded. The cached snapshot bytes are pre-
/// serialized so a hot polling loop doesn't re-format JSON on every request.
pub struct LiveState {
    /// Source of truth for re-resolving in live mode.
    session_id: String,
    /// Cached redacted payload (also held in `cached_bytes`'s parsed form).
    snapshot: SharePayload,
    cached_bytes: Vec<u8>,
    cached_etag: String,
    live: bool,
    redact_ctx: RedactCtx,
    resolver_ctx: Option<Arc<ResolverContext>>,
}

impl LiveState {
    fn from_snapshot(
        session_id: String,
        mut snapshot: SharePayload,
        live: bool,
        redact_ctx: RedactCtx,
        resolver_ctx: Option<Arc<ResolverContext>>,
    ) -> Self {
        snapshot.meta.live = live;
        let bytes = serde_json::to_vec(&snapshot).expect("serialize SharePayload");
        let etag = etag_for(&bytes);
        Self {
            session_id,
            snapshot,
            cached_bytes: bytes,
            cached_etag: etag,
            live,
            redact_ctx,
            resolver_ctx,
        }
    }

    /// Re-resolve and re-redact in live mode; cache the result. Snapshot
    /// mode is a no-op.
    async fn refresh_if_live(&mut self) -> Result<()> {
        if !self.live {
            return Ok(());
        }
        let Some(resolver) = self.resolver_ctx.clone() else {
            return Ok(());
        };
        let resolved = resolve_session(&self.session_id, &resolver).await?;
        let mut payload = resolved.payload;
        payload.meta.live = true;
        let (mut redacted, _) = redact(payload, &self.redact_ctx);
        redacted.meta.served_at = chrono::Utc::now();
        self.cached_bytes = serde_json::to_vec(&redacted)?;
        self.cached_etag = etag_for(&self.cached_bytes);
        self.snapshot = redacted;
        Ok(())
    }
}

fn etag_for(bytes: &[u8]) -> String {
    let hex = hex_sha256(bytes);
    // 16-char prefix is plenty of entropy for change detection; full sha256
    // would be overkill for a polled JSON blob.
    format!("\"{}\"", &hex[..16.min(hex.len())])
}

/// Bind a local listener and run the share server until `shutdown` fires or
/// the listener errors. Returns the bound port (when host is `127.0.0.1:0`)
/// alongside the join handle so the caller can print the URL and await
/// shutdown via Ctrl+C.
pub async fn start_local_server(
    bind_addr: &str,
    state: LiveState,
    shutdown: Arc<tokio::sync::Notify>,
) -> Result<(u16, tokio::task::JoinHandle<()>)> {
    let listener = TcpListener::bind(bind_addr).await?;
    let port = listener.local_addr()?.port();
    let state = Arc::new(RwLock::new(state));
    let handle = tokio::spawn(run_loop(listener, state, shutdown));
    Ok((port, handle))
}

/// Public constructor so the share command can build a state without the
/// resolver context (snapshot-only) or with it (live mode).
pub fn build_state(
    session_id: String,
    snapshot: SharePayload,
    live: bool,
    redact_ctx: RedactCtx,
    resolver_ctx: Option<Arc<ResolverContext>>,
) -> LiveState {
    LiveState::from_snapshot(session_id, snapshot, live, redact_ctx, resolver_ctx)
}

async fn run_loop(
    listener: TcpListener,
    state: Arc<RwLock<LiveState>>,
    shutdown: Arc<tokio::sync::Notify>,
) {
    loop {
        let accept = tokio::select! {
            result = listener.accept() => result,
            _ = shutdown.notified() => return,
        };
        let Ok((mut socket, _peer)) = accept else {
            continue;
        };
        let state = state.clone();
        tokio::spawn(async move {
            let request_bytes = match tokio::time::timeout(
                std::time::Duration::from_secs(30),
                read_full_request(&mut socket),
            )
            .await
            {
                Ok(Ok(b)) => b,
                _ => {
                    let _ = socket
                        .write_all(http_error_response(400, "bad request").as_bytes())
                        .await;
                    return;
                }
            };
            let request = String::from_utf8_lossy(&request_bytes).into_owned();
            let response = handle_request(&request, &state).await;
            let _ = socket.write_all(&response).await;
        });
    }
}

async fn handle_request(request: &str, state: &Arc<RwLock<LiveState>>) -> Vec<u8> {
    let path = http_utils::extract_request_path(request);
    let path_no_query = path.split('?').next().unwrap_or(&path);
    let method_is_head = request.starts_with("HEAD ");
    let method_is_get = request.starts_with("GET ") || method_is_head;
    let method_is_options = request.starts_with("OPTIONS ");

    if method_is_options {
        return cors_preflight();
    }

    if !method_is_get {
        return http_error_response(405, "method not allowed").into_bytes();
    }

    match path_no_query {
        "/state" => serve_state(request, state, method_is_head).await,
        _ => http_error_response(404, "not found").into_bytes(),
    }
}

async fn serve_state(request: &str, state: &Arc<RwLock<LiveState>>, head_only: bool) -> Vec<u8> {
    // Snapshot mode is the common case and never mutates — keep concurrent
    // polls non-serialized by skipping the write lock entirely.
    if state.read().await.live {
        let mut guard = state.write().await;
        if let Err(err) = guard.refresh_if_live().await {
            return http_error_response(500, &format!("refresh failed: {err}")).into_bytes();
        }
    }
    let guard = state.read().await;

    let if_none_match = http_utils::header_value(request, "If-None-Match");
    if if_none_match == Some(guard.cached_etag.as_str()) {
        let head = http_response_head_with_extra(304, "application/json", 0, &cors_extra());
        return head.into_bytes();
    }

    // Combined `{ meta, payload }` body. `cached_bytes` is the payload JSON;
    // splice it in raw to avoid re-serializing on every poll.
    let meta_json = serde_json::to_vec(&meta_value(&guard)).expect("serialize meta");
    let mut body = Vec::with_capacity(meta_json.len() + guard.cached_bytes.len() + 24);
    body.extend_from_slice(b"{\"meta\":");
    body.extend_from_slice(&meta_json);
    body.extend_from_slice(b",\"payload\":");
    body.extend_from_slice(&guard.cached_bytes);
    body.extend_from_slice(b"}");

    let extra = format!(
        "ETag: {}\r\nCache-Control: no-store\r\n{}",
        guard.cached_etag,
        cors_extra()
    );
    let head = http_response_head_with_extra(
        200,
        "application/json",
        body.len(),
        extra.trim_end_matches("\r\n"),
    );
    let mut out = head.into_bytes();
    if !head_only {
        out.extend_from_slice(&body);
    }
    out
}

fn meta_value(state: &LiveState) -> serde_json::Value {
    json!({
        "source_cli": state.snapshot.source_cli,
        "session_id": state.snapshot.session_id,
        "model": state.snapshot.model,
        "project": state.snapshot.project,
        "created_at": state.snapshot.created_at,
        "updated_at": state.snapshot.updated_at,
        "live": state.live,
        "message_count": state.snapshot.messages.len(),
        "size_bytes": state.cached_bytes.len(),
        "schema_version": state.snapshot.schema_version,
    })
}

fn cors_extra() -> String {
    format!(
        "Access-Control-Allow-Origin: {}\r\nAccess-Control-Allow-Methods: GET, HEAD, OPTIONS\r\nAccess-Control-Allow-Headers: If-None-Match",
        VIEWER_ORIGIN
    )
}

fn cors_preflight() -> Vec<u8> {
    let extra = format!("{}\r\nAccess-Control-Max-Age: 86400", cors_extra());
    let head = http_response_head_with_extra(204, "text/plain", 0, &extra);
    head.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::share_payload::{
        ContentBlock, ProjectInfo, SHARE_SCHEMA_VERSION, ShareMessage, SharePayload,
    };
    use std::time::Duration;
    use tokio::io::AsyncReadExt;
    use tokio::net::TcpStream;

    fn fake_payload() -> SharePayload {
        SharePayload {
            schema_version: SHARE_SCHEMA_VERSION.into(),
            source_cli: "amp".into(),
            session_id: "T-test".into(),
            project: ProjectInfo {
                root: Some("~/work/aivo".into()),
                name: Some("aivo".into()),
            },
            model: Some("claude-sonnet-4-5".into()),
            created_at: None,
            updated_at: None,
            messages: vec![ShareMessage {
                role: "user".into(),
                timestamp: None,
                model: None,
                reasoning: None,
                content: vec![ContentBlock::Text {
                    text: "hello".into(),
                }],
            }],
            meta: SharePayload::new_meta(false),
        }
    }

    async fn http_get(port: u16, path: &str) -> (u16, Vec<u8>) {
        http_request(port, "GET", path, None).await
    }

    async fn http_request(
        port: u16,
        method: &str,
        path: &str,
        if_none_match: Option<&str>,
    ) -> (u16, Vec<u8>) {
        let mut sock = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        let mut req =
            format!("{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n");
        if let Some(etag) = if_none_match {
            req.push_str(&format!("If-None-Match: {etag}\r\n"));
        }
        req.push_str("\r\n");
        tokio::io::AsyncWriteExt::write_all(&mut sock, req.as_bytes())
            .await
            .unwrap();
        let mut buf = Vec::new();
        sock.read_to_end(&mut buf).await.unwrap();
        let head_end = buf.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
        let head = std::str::from_utf8(&buf[..head_end]).unwrap();
        let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
        let body = buf[head_end + 4..].to_vec();
        (status, body)
    }

    async fn read_etag(port: u16, path: &str) -> String {
        let mut sock = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        let req = format!("HEAD {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");
        tokio::io::AsyncWriteExt::write_all(&mut sock, req.as_bytes())
            .await
            .unwrap();
        let mut buf = Vec::new();
        sock.read_to_end(&mut buf).await.unwrap();
        let head = String::from_utf8_lossy(&buf);
        for line in head.lines() {
            if let Some(rest) = line.strip_prefix("ETag: ") {
                return rest.to_string();
            }
        }
        panic!("no ETag in HEAD response: {head}");
    }

    #[tokio::test]
    async fn serves_state_and_head() {
        let state = build_state(
            "T-test".into(),
            fake_payload(),
            false,
            RedactCtx::default(),
            None,
        );
        let shutdown = Arc::new(tokio::sync::Notify::new());
        let (port, _h) = start_local_server("127.0.0.1:0", state, shutdown.clone())
            .await
            .unwrap();

        let (status, body) = http_get(port, "/state").await;
        assert_eq!(status, 200);
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed["meta"]["source_cli"], "amp");
        assert_eq!(parsed["meta"]["message_count"], 1);
        assert_eq!(parsed["meta"]["live"], false);
        assert_eq!(parsed["payload"]["source_cli"], "amp");
        assert_eq!(parsed["payload"]["session_id"], "T-test");

        // HEAD /state → empty body, ETag header present.
        let etag = read_etag(port, "/state").await;
        assert!(etag.starts_with('"') && etag.ends_with('"'));

        // 304 path: same etag → no body.
        let (status, body) = http_request(port, "GET", "/state", Some(&etag)).await;
        assert_eq!(status, 304);
        assert!(body.is_empty());

        shutdown.notify_waiters();
        // give the loop a tick to break out
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    #[tokio::test]
    async fn returns_404_for_unknown_routes_and_405_for_post() {
        let state = build_state(
            "T-test".into(),
            fake_payload(),
            false,
            RedactCtx::default(),
            None,
        );
        let shutdown = Arc::new(tokio::sync::Notify::new());
        let (port, _h) = start_local_server("127.0.0.1:0", state, shutdown.clone())
            .await
            .unwrap();

        let (status, _) = http_get(port, "/nope").await;
        assert_eq!(status, 404);

        let (status, _) = http_request(port, "POST", "/state", None).await;
        assert_eq!(status, 405);

        shutdown.notify_waiters();
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    #[tokio::test]
    async fn cors_preflight_returns_204() {
        let state = build_state(
            "T-test".into(),
            fake_payload(),
            false,
            RedactCtx::default(),
            None,
        );
        let shutdown = Arc::new(tokio::sync::Notify::new());
        let (port, _h) = start_local_server("127.0.0.1:0", state, shutdown.clone())
            .await
            .unwrap();
        let (status, _) = http_request(port, "OPTIONS", "/state", None).await;
        assert_eq!(status, 204);
        shutdown.notify_waiters();
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}
