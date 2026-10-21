//! Vision fallback: describe images with a cheap vision model so text-only
//! models can "see" them, via the hosted gateway `/v1/describe` (device-signed
//! and quota'd, mirroring `/v1/search`) or the user's own vision key.

use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::time::Duration;

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::services::session_store::ApiKey;

/// Where a describe call goes, resolved (and the custom key decrypted) at
/// dispatch so the spawned turn just calls it.
#[derive(Clone)]
pub enum DescriberSource {
    Gateway,
    OwnKey { model: String, key: Box<ApiKey> },
}

const DESCRIBE_PROMPT: &str = "Transcribe this image for a reader who cannot see it. \
First: all readable text, verbatim, preserving structure (headings, lists, tables, code). \
Then: a concise layout/visual description — UI elements and arrangement, charts with axes \
and approximate values, colors only where meaningful. No commentary.";

/// Decoded-size cap; larger images fail locally, never uploaded.
const MAX_IMAGE_BYTES: usize = 5 * 1024 * 1024;

const TIMEOUT_SECS: u64 = 60;

/// Latched once describe is known-exhausted this session (quota/auth/config),
/// so the shim falls back to the plain refusal instead of re-hitting the gateway.
pub static DESCRIBE_EXHAUSTED: AtomicBool = AtomicBool::new(false);

/// Held by tests that flip the process-global latch or the endpoint env var, so
/// parallel threads can't observe each other's window. Async because both
/// holders await mid-critical-section.
#[cfg(test)]
pub static TEST_DESCRIBE_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

pub fn describe_exhausted() -> bool {
    DESCRIBE_EXHAUSTED.load(Relaxed)
}

/// The own-key path rides the caller's loopback serve, so usage is accounted
/// under "code" like any other turn request.
pub async fn describe(
    src: &DescriberSource,
    client: &reqwest::Client,
    base: &str,
    auth: &str,
    data_url: &str,
) -> Result<String, String> {
    let Some((mime, data)) = parse_data_url(data_url).filter(|(mime, _)| !mime.is_empty()) else {
        return Err("image attachment isn't a base64 data URL".to_string());
    };
    if image_too_large(data) {
        return Err(format!(
            "image exceeds the {} MB describe limit",
            MAX_IMAGE_BYTES / (1024 * 1024)
        ));
    }
    match src {
        DescriberSource::Gateway => describe_via_gateway(mime, data).await,
        DescriberSource::OwnKey { model, .. } => {
            describe_via_key(client, base, auth, model, data_url).await
        }
    }
}

/// Non-200 → (actionable message, whether to latch the session exhausted).
fn classify_describe_error(status: u16) -> (String, bool) {
    match status {
        401 => (
            "image describe needs sign-in — run `aivo login`".to_string(),
            true,
        ),
        403 => (
            "image describe isn't available on your plan".to_string(),
            true,
        ),
        429 => ("today's image-describe quota is used up".to_string(), true),
        503 => ("image describe isn't configured".to_string(), true),
        413 => ("image too large for describe".to_string(), false),
        400 => ("image couldn't be decoded for describe".to_string(), false),
        502 => ("image describe is temporarily down".to_string(), false),
        _ => (format!("image describe failed (HTTP {status})"), false),
    }
}

/// Cap check on base64 length — no decode.
fn image_too_large(base64: &str) -> bool {
    base64.len() / 4 * 3 > MAX_IMAGE_BYTES
}

#[derive(Serialize)]
struct GatewayBody<'a> {
    image: &'a str,
    media_type: &'a str,
}

/// Latches `DESCRIBE_EXHAUSTED` on persistent failures so callers can stop
/// offering the shim this session.
async fn describe_via_gateway(media_type: &str, base64: &str) -> Result<String, String> {
    if describe_exhausted() {
        return Err("image describe is unavailable for the rest of this session".to_string());
    }
    // `AIVO_DESCRIBE_ENDPOINT` (tests, local wrangler) points at loopback, which
    // an HTTP(S)_PROXY env would swallow.
    let override_endpoint = std::env::var("AIVO_DESCRIBE_ENDPOINT")
        .ok()
        .filter(|s| !s.trim().is_empty());
    let mut builder = crate::services::http_utils::aivo_http_client_builder()
        .timeout(Duration::from_secs(TIMEOUT_SECS));
    if override_endpoint.is_some() {
        builder = builder.no_proxy();
    }
    let client = builder
        .build()
        .map_err(|e| format!("build http client: {e}"))?;
    let endpoint = override_endpoint
        .unwrap_or_else(|| format!("{}/v1/describe", crate::constants::AIVO_GATEWAY_URL));
    // Device-signed (same auth as chat); the gateway holds the keys + quota.
    let builder = client.post(endpoint).json(&GatewayBody {
        image: base64,
        media_type,
    });
    let resp = crate::services::device_fingerprint::with_device_headers(builder)
        .send()
        .await
        .map_err(|e| format!("couldn't reach image describe ({e})"))?;
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    if status.is_success() {
        let description = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|v| {
                v.get("description")?
                    .as_str()
                    .map(str::trim)
                    .map(String::from)
            })
            .unwrap_or_default();
        if description.is_empty() {
            return Err("describe returned no text".to_string());
        }
        return Ok(description);
    }
    let (message, latch) = classify_describe_error(status.as_u16());
    if latch {
        DESCRIBE_EXHAUSTED.store(true, Relaxed);
    }
    Err(message)
}

async fn describe_via_key(
    client: &reqwest::Client,
    base: &str,
    auth: &str,
    model: &str,
    data_url: &str,
) -> Result<String, String> {
    let request = crate::agent::protocol::ChatRequest {
        model: model.to_string(),
        messages: vec![serde_json::json!({
            "role": "user",
            "content": [
                {"type": "text", "text": DESCRIBE_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        })],
        tools: vec![],
        extra: serde_json::Map::new(),
    };
    let mut sink = |_: crate::agent::serve_client::StreamDelta| {};
    let call = crate::agent::serve_client::complete(client, base, Some(auth), &request, &mut sink);
    match tokio::time::timeout(Duration::from_secs(TIMEOUT_SECS), call).await {
        Err(_) => Err(format!("image describe via {model} timed out")),
        Ok(Err(e)) => Err(format!(
            "image describe via {model} failed: {e} — re-pick the describer in /config \
(Vision fallback → custom)"
        )),
        Ok(Ok(msg)) => {
            let text = msg.content.unwrap_or_default().trim().to_string();
            if text.is_empty() {
                Err(format!("describer {model} returned no text"))
            } else {
                Ok(text)
            }
        }
    }
}

/// Stable per-image cache key: sha256 of the base64 payload, first 16 bytes hex.
pub fn image_hash(base64: &str) -> String {
    let digest = Sha256::digest(base64.as_bytes());
    digest[..16].iter().map(|b| format!("{b:02x}")).collect()
}

/// `data:<mime>[;param…];base64,<payload>` → (mime, payload). The mime may be
/// empty (`data:;base64,…`), so callers that need one apply their own default.
pub fn parse_data_url(url: &str) -> Option<(&str, &str)> {
    let (meta, payload) = url.strip_prefix("data:")?.split_once(',')?;
    let is_base64 = |p: &&str| p.eq_ignore_ascii_case("base64");
    if payload.is_empty() || !meta.split(';').any(|p| is_base64(&p)) {
        return None;
    }
    Some((
        meta.split(';').find(|p| !is_base64(p)).unwrap_or(""),
        payload,
    ))
}

/// Substitution wrapper, mirroring `format_text_attachment_content`.
pub fn format_described_image(desc: &str) -> String {
    format!("[Image] (described for a text-only model)\n{desc}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Latching statuses are covered by the pure classify table — hitting them
    /// here would race other tests through the process-global latch.
    #[tokio::test]
    async fn gateway_round_trip_against_fake_server() {
        let _guard = TEST_DESCRIBE_LOCK.lock().await;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            for body in [
                r#"{"description":"a red Submit button"}"#,
                r#"{"description":""}"#,
            ] {
                let (mut sock, _) = listener.accept().unwrap();
                let mut buf = [0u8; 4096];
                let _ = sock.read(&mut buf);
                let _ = sock.write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                );
            }
        });
        unsafe {
            std::env::set_var(
                "AIVO_DESCRIBE_ENDPOINT",
                format!("http://{addr}/v1/describe"),
            );
        }
        let ok = describe_via_gateway("image/png", "aGVsbG8=").await;
        let empty = describe_via_gateway("image/png", "aGVsbG8=").await;
        unsafe {
            std::env::remove_var("AIVO_DESCRIBE_ENDPOINT");
        }
        assert_eq!(ok.unwrap(), "a red Submit button");
        empty.expect_err("empty description is a failure");
        assert!(!describe_exhausted(), "empty description must not latch");
    }

    #[test]
    fn classify_latches_only_persistent_statuses() {
        for (status, latch) in [
            (401, true),
            (403, true),
            (429, true),
            (503, true),
            (413, false),
            (400, false),
            (502, false),
            (500, false),
        ] {
            let (message, got) = classify_describe_error(status);
            assert_eq!(got, latch, "status {status}");
            assert!(!message.is_empty());
        }
    }

    #[test]
    fn image_hash_is_stable_and_distinct() {
        let a = image_hash("aGVsbG8=");
        assert_eq!(a, image_hash("aGVsbG8="));
        assert_ne!(a, image_hash("d29ybGQ="));
        assert_eq!(a.len(), 32);
    }

    #[test]
    fn parse_data_url_roundtrip() {
        assert_eq!(
            parse_data_url("data:image/png;base64,aGVsbG8="),
            Some(("image/png", "aGVsbG8="))
        );
        // Parameterized meta: the params must not leak into the media type.
        assert_eq!(
            parse_data_url("data:image/svg+xml;charset=utf-8;BASE64,QUJD"),
            Some(("image/svg+xml", "QUJD"))
        );
        assert_eq!(parse_data_url("data:;base64,x"), Some(("", "x")));
        assert!(parse_data_url("data:image/png;base64,").is_none());
        assert!(parse_data_url("data:text/plain,hello").is_none());
        assert!(parse_data_url("https://example.com/a.png").is_none());
    }

    #[test]
    fn size_cap_uses_decoded_length() {
        assert!(!image_too_large("aGVsbG8="));
        let big = "A".repeat(MAX_IMAGE_BYTES / 3 * 4 + 8);
        assert!(image_too_large(&big));
    }

    #[test]
    fn described_image_wrapper() {
        assert_eq!(
            format_described_image("a red button"),
            "[Image] (described for a text-only model)\na red button"
        );
    }
}
