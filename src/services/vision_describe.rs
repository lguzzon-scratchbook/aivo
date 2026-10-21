//! Vision fallback: describe images with a cheap vision model so text-only
//! models can "see" them, through one of the user's own configured keys.

use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::services::session_store::ApiKey;

/// Where a describe call goes, resolved (and the key decrypted) at dispatch so
/// the spawned turn just calls it.
#[derive(Clone)]
pub enum DescriberSource {
    OwnKey { model: String, key: Box<ApiKey> },
}

const DESCRIBE_PROMPT: &str = "Transcribe this image for a reader who cannot see it. \
First: all readable text, verbatim, preserving structure (headings, lists, tables, code). \
Then: a concise layout/visual description — UI elements and arrangement, charts with axes \
and approximate values, colors only where meaningful. No commentary.";

/// Decoded-size cap; larger images fail locally, never uploaded.
const MAX_IMAGE_BYTES: usize = 5 * 1024 * 1024;

const TIMEOUT_SECS: u64 = 60;

/// The own-key path rides the caller's loopback serve, so usage is accounted
/// under "code" like any other turn request.
pub async fn describe(
    src: &DescriberSource,
    client: &reqwest::Client,
    base: &str,
    auth: &str,
    data_url: &str,
) -> Result<String, String> {
    let Some((_mime, data)) = parse_data_url(data_url).filter(|(mime, _)| !mime.is_empty()) else {
        return Err("image attachment isn't a base64 data URL".to_string());
    };
    if image_too_large(data) {
        return Err(format!(
            "image exceeds the {} MB describe limit",
            MAX_IMAGE_BYTES / (1024 * 1024)
        ));
    }
    let DescriberSource::OwnKey { model, .. } = src;
    describe_via_key(client, base, auth, model, data_url).await
}

/// Cap check on base64 length — no decode.
fn image_too_large(base64: &str) -> bool {
    base64.len() / 4 * 3 > MAX_IMAGE_BYTES
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
