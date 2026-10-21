//! Image generation for `generate_image`, through one of the user's own keys:
//! a chat-wire image model or an `/v1/images/generations` provider.

use std::time::Duration;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde_json::Value;

use crate::services::session_store::ApiKey;

/// Resolved (and the key decrypted) at dispatch, so the turn just calls it.
#[derive(Clone)]
pub enum GeneratorSource {
    OwnKey {
        model: String,
        key: Box<ApiKey>,
    },
    /// `n`-flagged model (gpt-image-2, imagen): never answers on the chat wire.
    ImagesApi {
        model: String,
        key: Box<ApiKey>,
    },
}

impl GeneratorSource {
    /// What the agent and its errors call the generator.
    pub fn label(&self) -> &str {
        match self {
            Self::OwnKey { model, .. } | Self::ImagesApi { model, .. } => model,
        }
    }

    /// The model the turn's loopback serve must carry; the Images-API path
    /// talks to its endpoint directly and needs none.
    pub fn upstream_model(&self) -> Option<&str> {
        match self {
            Self::ImagesApi { .. } => None,
            Self::OwnKey { model, .. } => Some(model),
        }
    }
}

/// Plain OpenAI-compatible HTTP providers only — sentinels, native
/// Anthropic/Gemini, OAuth, and ACP keys have no `/v1/images/generations`.
pub fn key_serves_images_api(key: &ApiKey) -> bool {
    use crate::services::provider_profile as profile;
    use crate::services::provider_protocol::{ProviderProtocol, detect_provider_protocol};
    !key.is_any_oauth()
        && !key.is_cursor_acp()
        && !key.is_copilot()
        && !profile::is_ollama_base(&key.base_url)
        && detect_provider_protocol(&key.base_url) == ProviderProtocol::Openai
}

const TIMEOUT_SECS: u64 = 180;

/// `data:` URLs — the shape `save_data_url_images` takes, so both sources
/// share one save path.
pub async fn generate(
    src: &GeneratorSource,
    client: &reqwest::Client,
    base: &str,
    auth: &str,
    prompt: &str,
) -> Result<Vec<String>, String> {
    match src {
        GeneratorSource::OwnKey { model, .. } => {
            generate_via_key(client, base, auth, model, prompt).await
        }
        GeneratorSource::ImagesApi { model, key } => {
            generate_via_images_api(client, key, model, prompt).await
        }
    }
    .and_then(|(images, text)| {
        if images.is_empty() {
            // Text-only reply (refusal/clarification) — surface it so the agent can react.
            let who = src.label();
            return Err(if text.trim().is_empty() {
                format!("{who} returned no image")
            } else {
                format!("{who} returned no image: {text}")
            });
        }
        Ok(images)
    })
}

/// The own-key path rides the caller's loopback serve, so usage is accounted
/// under "code" like any other turn request.
async fn generate_via_key(
    client: &reqwest::Client,
    base: &str,
    auth: &str,
    model: &str,
    prompt: &str,
) -> Result<(Vec<String>, String), String> {
    // OpenRouter emits images only with this opt-in; the Gemini bridge maps it
    // to `responseModalities`.
    let mut extra = serde_json::Map::new();
    extra.insert(
        "modalities".to_string(),
        serde_json::json!(["image", "text"]),
    );
    let request = crate::agent::protocol::ChatRequest {
        model: model.to_string(),
        messages: vec![serde_json::json!({"role": "user", "content": prompt})],
        tools: vec![],
        extra,
    };
    let mut sink = |_: crate::agent::serve_client::StreamDelta| {};
    let call = crate::agent::serve_client::complete(client, base, Some(auth), &request, &mut sink);
    match tokio::time::timeout(Duration::from_secs(TIMEOUT_SECS), call).await {
        Err(_) => Err(format!("image generation via {model} timed out")),
        Ok(Err(e)) => Err(format!("image generation via {model} failed: {e}")),
        Ok(Ok(msg)) => Ok((msg.images, msg.content.unwrap_or_default())),
    }
}

/// Direct POST, off the loopback serve — these providers bill per image,
/// not per token.
async fn generate_via_images_api(
    client: &reqwest::Client,
    key: &ApiKey,
    model: &str,
    prompt: &str,
) -> Result<(Vec<String>, String), String> {
    let url =
        crate::services::http_utils::build_target_url(&key.base_url, "/v1/images/generations");
    let send = crate::services::opencode_session::with_session_header(
        client
            .post(&url)
            .bearer_auth(key.key.as_str())
            .json(&serde_json::json!({ "model": model, "prompt": prompt })),
        &url,
        None,
    )
    .send();
    let resp = match tokio::time::timeout(Duration::from_secs(TIMEOUT_SECS), send).await {
        Err(_) => return Err(format!("image generation via {model} timed out")),
        Ok(Err(e)) => return Err(format!("image generation via {model} failed: {e}")),
        Ok(Ok(resp)) => resp,
    };
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(format!(
            "image generation via {model} failed (HTTP {}): {}",
            status.as_u16(),
            upstream_error_message(&text),
        ));
    }
    let parsed: Value =
        serde_json::from_str(&text).map_err(|_| format!("{model} returned malformed JSON"))?;
    let entries = parsed.get("data").and_then(Value::as_array);
    let mut urls = Vec::new();
    for entry in entries.into_iter().flatten() {
        if let Some(b64) = nonempty_str(entry, "b64_json") {
            urls.push(format!("data:{};base64,{b64}", sniff_b64_image_mime(b64)));
        } else if let Some(remote) = nonempty_str(entry, "url") {
            // Hosted URLs expire in minutes — inline now.
            urls.push(fetch_image_as_data_url(client, remote).await?);
        }
    }
    Ok((urls, String::new()))
}

fn nonempty_str<'a>(v: &'a Value, field: &str) -> Option<&'a str> {
    v.get(field)?.as_str().filter(|s| !s.is_empty())
}

fn upstream_error_message(body: &str) -> String {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| {
            v.pointer("/error/message")
                .or_else(|| v.pointer("/message"))?
                .as_str()
                .map(str::to_string)
        })
        .unwrap_or_else(|| {
            body.chars()
                .take(200)
                .collect::<String>()
                .trim()
                .to_string()
        })
}

/// Responses carry bare base64 with no media type — sniff the decoded prefix.
fn sniff_b64_image_mime(b64: &str) -> &'static str {
    let prefix: String = b64
        .chars()
        .filter(|c| !c.is_whitespace())
        .take(24)
        .collect();
    let prefix = &prefix[..prefix.len() - prefix.len() % 4];
    BASE64
        .decode(prefix)
        .map(|bytes| sniff_image_mime(&bytes))
        .unwrap_or("image/png")
}

fn sniff_image_mime(bytes: &[u8]) -> &'static str {
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        "image/jpeg"
    } else if bytes.starts_with(b"GIF8") {
        "image/gif"
    } else if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        "image/webp"
    } else {
        "image/png"
    }
}

async fn fetch_image_as_data_url(client: &reqwest::Client, url: &str) -> Result<String, String> {
    let fetch = async {
        let resp = client
            .get(url)
            .send()
            .await
            .map_err(|e| format!("couldn't fetch the generated image ({e})"))?;
        if !resp.status().is_success() {
            return Err(format!(
                "couldn't fetch the generated image (HTTP {})",
                resp.status().as_u16()
            ));
        }
        let mime = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.split(';').next().unwrap_or(s).trim().to_string())
            .filter(|s| s.starts_with("image/"));
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| format!("couldn't read the generated image ({e})"))?;
        let mime = mime.unwrap_or_else(|| sniff_image_mime(&bytes).to_string());
        Ok(format!("data:{mime};base64,{}", BASE64.encode(&bytes)))
    };
    match tokio::time::timeout(Duration::from_secs(TIMEOUT_SECS), fetch).await {
        Err(_) => Err("fetching the generated image timed out".to_string()),
        Ok(result) => result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn images_api_source_needs_no_upstream() {
        let src = GeneratorSource::ImagesApi {
            model: "gpt-image-2".into(),
            key: Box::new(test_key("https://api.example.com/v1")),
        };
        assert_eq!(src.label(), "gpt-image-2");
        assert_eq!(src.upstream_model(), None);
    }

    fn test_key(base: &str) -> ApiKey {
        ApiKey::new_with_protocol("id".into(), "k".into(), base.into(), None, "sk-test".into())
    }

    #[test]
    fn images_api_key_gate_admits_plain_openai_wire_only() {
        assert!(key_serves_images_api(&test_key(
            "https://api.acme.example/endpoint"
        )));
        assert!(key_serves_images_api(&test_key(
            "https://ai-gateway.vercel.sh/v1"
        )));
        assert!(!key_serves_images_api(&test_key("copilot")));
        assert!(!key_serves_images_api(&test_key("ollama")));
        assert!(!key_serves_images_api(&test_key(
            "https://generativelanguage.googleapis.com"
        )));
        assert!(!key_serves_images_api(&test_key(
            "https://api.anthropic.com"
        )));
    }

    #[test]
    fn sniffs_common_image_magics_defaulting_png() {
        assert_eq!(sniff_b64_image_mime("iVBORw0KGgoAAAANSUhEUg"), "image/png");
        assert_eq!(sniff_b64_image_mime("/9j/4AAQSkZJRg"), "image/jpeg");
        assert_eq!(sniff_b64_image_mime("R0lGODlhAQABAAAA"), "image/gif");
        assert_eq!(sniff_b64_image_mime("not!!valid@@base64"), "image/png");
        assert_eq!(sniff_b64_image_mime(""), "image/png");
    }

    #[test]
    fn upstream_error_message_prefers_error_envelope() {
        assert_eq!(
            upstream_error_message(r#"{"error":{"message":"prompt rejected"}}"#),
            "prompt rejected"
        );
        assert_eq!(
            upstream_error_message(r#"{"message":"The requested operation is unsupported."}"#),
            "The requested operation is unsupported."
        );
        assert_eq!(
            upstream_error_message("<html>bad gateway</html>"),
            "<html>bad gateway</html>"
        );
    }

    #[tokio::test]
    async fn images_api_round_trip_against_fake_server() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let jpeg = [0xFFu8, 0xD8, 0xFF, 0xE0];
        let generations =
            format!(r#"{{"data":[{{"b64_json":"iVBORw0KGgo="}},{{"url":"http://{addr}/img"}}]}}"#);
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            let responses: [(u16, &str, Vec<u8>); 3] = [
                (200, "application/json", generations.into_bytes()),
                (200, "image/jpeg", jpeg.to_vec()),
                (
                    400,
                    "application/json",
                    br#"{"error":{"message":"prompt rejected"}}"#.to_vec(),
                ),
            ];
            for (status, ctype, body) in responses {
                let (mut sock, _) = listener.accept().unwrap();
                let mut buf = [0u8; 4096];
                let _ = sock.read(&mut buf);
                let _ = sock.write_all(
                    format!(
                        "HTTP/1.1 {status} X\r\ncontent-type: {ctype}\r\nconnection: close\r\ncontent-length: {}\r\n\r\n",
                        body.len()
                    )
                    .as_bytes(),
                );
                let _ = sock.write_all(&body);
            }
        });
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let key = test_key(&format!("http://{addr}"));

        let (urls, text) = generate_via_images_api(&client, &key, "gpt-image-2", "a cat")
            .await
            .unwrap();
        assert!(text.is_empty());
        assert_eq!(
            urls,
            vec![
                "data:image/png;base64,iVBORw0KGgo=".to_string(),
                format!("data:image/jpeg;base64,{}", BASE64.encode(jpeg)),
            ]
        );

        let err = generate_via_images_api(&client, &key, "gpt-image-2", "a cat")
            .await
            .unwrap_err();
        assert!(
            err.contains("HTTP 400") && err.contains("prompt rejected"),
            "{err}"
        );
    }
}
