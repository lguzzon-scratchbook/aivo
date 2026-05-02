//! Image generation service.
//!
//! Handles OpenAI-compatible `/v1/images/generations` and Google's
//! `generativelanguage.googleapis.com` surfaces (Gemini-native multimodal
//! image models via `:generateContent`, Imagen via `:predict`).
//!
//! Output UX (path parsing, overwrite policy, atomic writes, error
//! extraction) lives in [`super::media_io`] and is shared with `video_gen`
//! and `audio_gen`. This module re-exports the pieces image callers reach
//! for so existing imports keep working.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::services::http_utils::router_http_client;
use crate::services::media_io::{align_extension, atomic_write, extract_error_message};
use crate::services::provider_protocol::{ProviderProtocol, detect_provider_protocol};
use crate::services::session_store::ApiKey;

/// Options for a single image generation request.
#[derive(Debug, Clone)]
pub struct ImageRequest {
    pub prompt: String,
    pub model: String,
    pub size: Option<String>,
    pub quality: Option<String>,
}

/// One saved image (or URL, when `--url` was set).
#[derive(Debug, Clone)]
pub struct ImageArtifact {
    /// Path the file was written to, or `None` when `url_only` is set.
    pub path: Option<PathBuf>,
    /// Provider URL (OpenAI sometimes returns a URL, sometimes base64).
    pub url: Option<String>,
    /// Size of the written file in bytes (0 when `url_only`).
    pub bytes: u64,
}

#[derive(Debug, Deserialize)]
struct OpenAIImageResponse {
    data: Vec<OpenAIImageItem>,
}

#[derive(Debug, Deserialize)]
struct OpenAIImageItem {
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    b64_json: Option<String>,
}

/// Maps an HTTP `Content-Type` header to a file extension for images.
/// Falls back to `"png"` for anything unrecognized — OpenAI's default.
pub fn ext_from_content_type(ct: Option<&str>) -> String {
    match ct.map(|c| {
        c.split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase()
    }) {
        Some(ref s) if s == "image/jpeg" || s == "image/jpg" => "jpg".into(),
        Some(ref s) if s == "image/webp" => "webp".into(),
        Some(ref s) if s == "image/gif" => "gif".into(),
        _ => "png".into(),
    }
}

/// Translate the CLI's `-s` argument to a Google `aspectRatio`. Accepts
/// either OpenAI-style `WxH` (mapped to the closest Google ratio) or a
/// pass-through `W:H` form. Returns `None` for absent or unrecognized
/// values — callers treat that as "let the server pick its default" rather
/// than guessing.
fn aspect_ratio_for_size(size: Option<&str>) -> Option<String> {
    let raw = size?.trim();
    if raw.contains(':') {
        return Some(raw.to_string());
    }
    match raw {
        "1024x1024" => Some("1:1".into()),
        "1792x1024" => Some("16:9".into()),
        "1024x1792" => Some("9:16".into()),
        _ => None,
    }
}

/// True for Imagen models (use `:predict`). Gemini multimodal image models
/// (`gemini-*-image*`) use `:generateContent` instead and are the default
/// path for anything that isn't recognized as Imagen.
fn is_imagen_model(model: &str) -> bool {
    model.trim().to_ascii_lowercase().starts_with("imagen-")
}

/// Build `{base}/v1beta/models/{model}:{verb}`. Tolerates a trailing slash
/// or a `/v1beta` suffix already present on the stored `base_url`, so users
/// who pasted either form get the same endpoint.
fn google_endpoint(base_url: &str, model: &str, verb: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    let root = trimmed.strip_suffix("/v1beta").unwrap_or(trimmed);
    format!("{root}/v1beta/models/{model}:{verb}")
}

/// Build the JSON body for Google's Gemini-native image generation
/// (`:generateContent`). Always sets `responseModalities` to request both
/// text and image; emits `imageConfig.aspectRatio` only when the user's
/// `-s` arg resolves to a known ratio, otherwise lets the server default.
fn build_gemini_image_body(prompt: &str, size: Option<&str>) -> Value {
    let mut generation_config = json!({
        "responseModalities": ["TEXT", "IMAGE"],
    });
    if let Some(ratio) = aspect_ratio_for_size(size) {
        generation_config["imageConfig"] = json!({ "aspectRatio": ratio });
    }
    json!({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    })
}

/// Build the JSON body for Google's Imagen `:predict` REST call. Always
/// emits `instances[0].prompt` and a default `parameters.sampleCount = 1`.
/// `aspect_ratio_for_size` translates the user's `-s` arg when it
/// resolves; `quality` of `hd` or `high` (case-insensitive) maps to
/// `imageSize: "2K"`, otherwise the server's 1K default is used.
fn build_imagen_body(prompt: &str, size: Option<&str>, quality: Option<&str>) -> Value {
    let mut parameters = json!({ "sampleCount": 1 });
    if let Some(ratio) = aspect_ratio_for_size(size) {
        parameters["aspectRatio"] = Value::String(ratio);
    }
    if matches!(
        quality
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("hd") | Some("high")
    ) {
        parameters["imageSize"] = Value::String("2K".into());
    }
    json!({
        "instances": [{"prompt": prompt}],
        "parameters": parameters,
    })
}

/// Decode a Gemini-native (`:generateContent`) image response. Walks
/// `candidates[0].content.parts[]` and returns the first image part's
/// `(decoded_bytes, mime_type)`. Tolerates both snake_case (`inline_data`,
/// `mime_type`) and camelCase (`inlineData`, `mimeType`) — Google's REST
/// surface emits both depending on context.
fn decode_gemini_image_response(body: &Value) -> Result<(Vec<u8>, Option<String>)> {
    let parts = body
        .get("candidates")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("content"))
        .and_then(|c| c.get("parts"))
        .and_then(|p| p.as_array())
        .ok_or_else(|| anyhow!("Google response missing candidates[0].content.parts"))?;

    for part in parts {
        let inline = part.get("inline_data").or_else(|| part.get("inlineData"));
        let Some(inline) = inline else { continue };
        let data = inline
            .get("data")
            .and_then(|d| d.as_str())
            .ok_or_else(|| anyhow!("Google inline_data/inlineData missing 'data' field"))?;
        let mime = inline
            .get("mime_type")
            .or_else(|| inline.get("mimeType"))
            .and_then(|m| m.as_str())
            .map(str::to_string);
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(data)
            .context("failed to decode base64 image payload from Google response")?;
        return Ok((bytes, mime));
    }
    bail!("Google response contained no image (no inline_data/inlineData part)")
}

/// Decode an Imagen `:predict` response. Reads `predictions[0]` and
/// returns `(decoded_bytes, mime_type)`. Bails when there are no
/// predictions; the field name is `bytesBase64Encoded` per Google's
/// documented Imagen REST shape.
fn decode_imagen_response(body: &Value) -> Result<(Vec<u8>, Option<String>)> {
    let prediction = body
        .get("predictions")
        .and_then(|p| p.as_array())
        .and_then(|arr| arr.first())
        .ok_or_else(|| anyhow!("Imagen response had no predictions"))?;

    let data = prediction
        .get("bytesBase64Encoded")
        .and_then(|d| d.as_str())
        .ok_or_else(|| anyhow!("Imagen prediction missing 'bytesBase64Encoded'"))?;
    let mime = prediction
        .get("mimeType")
        .and_then(|m| m.as_str())
        .map(str::to_string);
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data)
        .context("failed to decode base64 from Imagen response")?;
    Ok((bytes, mime))
}

/// Top-level generation entry point. Picks the protocol from `key.base_url`
/// and dispatches accordingly. `path` is the pre-resolved, overwrite-applied
/// target path (or `None` when `url_only` is set). When `pinned_extension`
/// is false, the caller chose the extension (e.g. the default `.png` for
/// `OutputTarget::Default`) and we may swap it to the server's actual
/// content-type suffix silently. When true, the user's extension is honored.
///
/// When `url_only` is true, skips the download step and only returns the URL
/// (fails for base64-only responses).
pub async fn generate(
    key: &ApiKey,
    request: &ImageRequest,
    path: Option<&Path>,
    pinned_extension: bool,
    url_only: bool,
) -> Result<ImageArtifact> {
    let protocol = detect_provider_protocol(&key.base_url);
    match protocol {
        ProviderProtocol::Openai | ProviderProtocol::ResponsesApi => {
            generate_openai(key, request, path, pinned_extension, url_only).await
        }
        ProviderProtocol::Google => {
            generate_google(key, request, path, pinned_extension, url_only).await
        }
        ProviderProtocol::Anthropic => {
            bail!("Anthropic does not support image generation")
        }
    }
}

async fn generate_openai(
    key: &ApiKey,
    request: &ImageRequest,
    path: Option<&Path>,
    pinned_extension: bool,
    url_only: bool,
) -> Result<ImageArtifact> {
    let base = key.base_url.trim_end_matches('/');
    // Accept both "https://api.example.com" and "https://api.example.com/v1".
    let url = if base.ends_with("/v1") {
        format!("{base}/images/generations")
    } else {
        format!("{base}/v1/images/generations")
    };

    let mut body = json!({
        "model": request.model,
        "prompt": request.prompt,
    });
    if let Some(s) = &request.size {
        body["size"] = Value::String(s.clone());
    }
    if let Some(q) = &request.quality {
        body["quality"] = Value::String(q.clone());
    }
    if url_only {
        body["response_format"] = Value::String("url".into());
    }

    let client = router_http_client();
    let response = client
        .post(&url)
        .bearer_auth(key.key.as_str())
        .json(&body)
        .send()
        .await
        .with_context(|| format!("image request to {url} failed"))?;

    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        let detail = extract_error_message(&text).unwrap_or_else(|| text.clone());
        bail!("image generation failed ({}): {}", status.as_u16(), detail);
    }

    let parsed: OpenAIImageResponse = response
        .json()
        .await
        .context("failed to decode image response")?;

    let item = parsed
        .data
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("provider returned no images"))?;

    if url_only {
        let url = item
            .url
            .ok_or_else(|| anyhow!("--url requested but provider returned base64 only"))?;
        return Ok(ImageArtifact {
            path: None,
            url: Some(url),
            bytes: 0,
        });
    }

    let (bytes, maybe_url, ext_hint) = if let Some(b64) = item.b64_json {
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&b64)
            .context("failed to decode base64 image payload")?;
        (decoded, None, None::<String>)
    } else if let Some(u) = item.url {
        let resp = client
            .get(&u)
            .send()
            .await
            .with_context(|| format!("downloading image from {u} failed"))?;
        let status = resp.status();
        if !status.is_success() {
            let host = reqwest::Url::parse(&u)
                .ok()
                .and_then(|parsed| parsed.host_str().map(str::to_string))
                .unwrap_or_else(|| u.clone());
            bail!(
                "image download failed: {} returned HTTP {} — the signed URL may have expired",
                host,
                status.as_u16()
            );
        }
        let ct = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|h| h.to_str().ok())
            .map(str::to_string);
        let body = resp.bytes().await.context("reading image body failed")?;
        (body.to_vec(), Some(u), ct)
    } else {
        bail!("provider response missing both url and b64_json");
    };

    let Some(target_path) = path else {
        return Ok(ImageArtifact {
            path: None,
            url: maybe_url,
            bytes: bytes.len() as u64,
        });
    };

    let server_ext = ext_hint.as_deref().map(|c| ext_from_content_type(Some(c)));
    let final_path = align_extension(target_path, server_ext.as_deref(), pinned_extension);
    let written = atomic_write(&final_path, &bytes)?;
    Ok(ImageArtifact {
        path: Some(final_path),
        url: maybe_url,
        bytes: written,
    })
}

async fn generate_google(
    key: &ApiKey,
    request: &ImageRequest,
    path: Option<&Path>,
    pinned_extension: bool,
    url_only: bool,
) -> Result<ImageArtifact> {
    if url_only {
        bail!("--url is not supported for Google: the API returns base64 only");
    }

    let imagen = is_imagen_model(&request.model);
    let verb = if imagen { "predict" } else { "generateContent" };
    let url = google_endpoint(&key.base_url, &request.model, verb);

    let body = if imagen {
        build_imagen_body(
            &request.prompt,
            request.size.as_deref(),
            request.quality.as_deref(),
        )
    } else {
        build_gemini_image_body(&request.prompt, request.size.as_deref())
    };

    let client = router_http_client();
    let response = client
        .post(&url)
        .header("x-goog-api-key", key.key.as_str())
        .json(&body)
        .send()
        .await
        .with_context(|| format!("image request to {url} failed"))?;

    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        let detail = extract_error_message(&text).unwrap_or_else(|| text.clone());
        bail!("image generation failed ({}): {}", status.as_u16(), detail);
    }

    let parsed: Value = response
        .json()
        .await
        .context("failed to decode Google image response")?;

    let (bytes, mime) = if imagen {
        decode_imagen_response(&parsed)?
    } else {
        decode_gemini_image_response(&parsed)?
    };

    let Some(target_path) = path else {
        return Ok(ImageArtifact {
            path: None,
            url: None,
            bytes: bytes.len() as u64,
        });
    };

    let server_ext = mime.as_deref().map(|m| ext_from_content_type(Some(m)));
    let final_path = align_extension(target_path, server_ext.as_deref(), pinned_extension);
    let written = atomic_write(&final_path, &bytes)?;
    Ok(ImageArtifact {
        path: Some(final_path),
        url: None,
        bytes: written,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ext_from_content_type_maps_known_types() {
        assert_eq!(ext_from_content_type(Some("image/jpeg")), "jpg");
        assert_eq!(ext_from_content_type(Some("image/jpg")), "jpg");
        assert_eq!(ext_from_content_type(Some("image/webp")), "webp");
        assert_eq!(ext_from_content_type(Some("image/gif")), "gif");
        assert_eq!(ext_from_content_type(Some("image/png")), "png");
        assert_eq!(ext_from_content_type(Some("application/json")), "png");
        assert_eq!(ext_from_content_type(None), "png");
    }

    #[test]
    fn ext_from_content_type_ignores_charset_suffix() {
        assert_eq!(ext_from_content_type(Some("image/jpeg; charset=x")), "jpg");
    }

    #[test]
    fn aspect_ratio_for_size_maps_common_openai_sizes() {
        assert_eq!(aspect_ratio_for_size(Some("1024x1024")), Some("1:1".into()));
        assert_eq!(
            aspect_ratio_for_size(Some("1792x1024")),
            Some("16:9".into())
        );
        assert_eq!(
            aspect_ratio_for_size(Some("1024x1792")),
            Some("9:16".into())
        );
    }

    #[test]
    fn aspect_ratio_for_size_passes_through_ratio_form() {
        assert_eq!(aspect_ratio_for_size(Some("16:9")), Some("16:9".into()));
        assert_eq!(aspect_ratio_for_size(Some("3:4")), Some("3:4".into()));
    }

    #[test]
    fn aspect_ratio_for_size_none_when_absent_or_unknown() {
        assert_eq!(aspect_ratio_for_size(None), None);
        assert_eq!(aspect_ratio_for_size(Some("512x768")), None);
        assert_eq!(aspect_ratio_for_size(Some("garbage")), None);
    }

    #[test]
    fn is_imagen_model_matches_imagen_prefix() {
        assert!(is_imagen_model("imagen-4.0-generate-001"));
        assert!(is_imagen_model("imagen-4.0-ultra-generate-001"));
        assert!(is_imagen_model("imagen-4.0-fast-generate-001"));
    }

    #[test]
    fn is_imagen_model_rejects_gemini_or_other() {
        assert!(!is_imagen_model("gemini-2.5-flash-image"));
        assert!(!is_imagen_model("gemini-3-pro-image-preview"));
        assert!(!is_imagen_model("gpt-image-1"));
        assert!(!is_imagen_model(""));
    }

    #[test]
    fn google_endpoint_uses_v1beta_models_path() {
        assert_eq!(
            google_endpoint(
                "https://generativelanguage.googleapis.com",
                "gemini-2.5-flash-image",
                "generateContent",
            ),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent"
        );
    }

    #[test]
    fn google_endpoint_strips_trailing_slash_and_v1beta_suffix() {
        let bare = google_endpoint(
            "https://generativelanguage.googleapis.com/",
            "imagen-4.0-generate-001",
            "predict",
        );
        let with_v1beta = google_endpoint(
            "https://generativelanguage.googleapis.com/v1beta",
            "imagen-4.0-generate-001",
            "predict",
        );
        let with_trailing = google_endpoint(
            "https://generativelanguage.googleapis.com/v1beta/",
            "imagen-4.0-generate-001",
            "predict",
        );
        let expected = "https://generativelanguage.googleapis.com/v1beta/models/imagen-4.0-generate-001:predict";
        assert_eq!(bare, expected);
        assert_eq!(with_v1beta, expected);
        assert_eq!(with_trailing, expected);
    }

    #[test]
    fn build_gemini_image_body_includes_prompt_and_response_modalities() {
        let body = build_gemini_image_body("a red panda", None);
        assert_eq!(
            body["contents"][0]["parts"][0]["text"],
            serde_json::Value::String("a red panda".into())
        );
        assert_eq!(
            body["generationConfig"]["responseModalities"],
            serde_json::json!(["TEXT", "IMAGE"])
        );
        assert!(body["generationConfig"].get("imageConfig").is_none());
    }

    #[test]
    fn build_gemini_image_body_emits_aspect_ratio_when_size_known() {
        let body = build_gemini_image_body("x", Some("1792x1024"));
        assert_eq!(
            body["generationConfig"]["imageConfig"]["aspectRatio"],
            serde_json::Value::String("16:9".into())
        );
    }

    #[test]
    fn build_gemini_image_body_skips_aspect_ratio_for_unknown_size() {
        let body = build_gemini_image_body("x", Some("512x512"));
        assert!(body["generationConfig"].get("imageConfig").is_none());
    }

    #[test]
    fn build_imagen_body_includes_prompt_and_default_sample_count() {
        let body = build_imagen_body("a red panda", None, None);
        assert_eq!(
            body["instances"][0]["prompt"],
            serde_json::Value::String("a red panda".into())
        );
        assert_eq!(
            body["parameters"]["sampleCount"],
            serde_json::Value::from(1u64)
        );
        assert!(body["parameters"].get("aspectRatio").is_none());
    }

    #[test]
    fn build_imagen_body_sets_aspect_ratio_when_resolvable() {
        let body = build_imagen_body("x", Some("1024x1792"), None);
        assert_eq!(
            body["parameters"]["aspectRatio"],
            serde_json::Value::String("9:16".into())
        );
    }

    #[test]
    fn build_imagen_body_maps_quality_to_image_size() {
        let body_hd = build_imagen_body("x", None, Some("hd"));
        assert_eq!(
            body_hd["parameters"]["imageSize"],
            serde_json::Value::String("2K".into())
        );
        let body_high = build_imagen_body("x", None, Some("high"));
        assert_eq!(
            body_high["parameters"]["imageSize"],
            serde_json::Value::String("2K".into())
        );
        let body_low = build_imagen_body("x", None, Some("low"));
        assert!(body_low["parameters"].get("imageSize").is_none());
        let body_caps = build_imagen_body("x", None, Some(" HD "));
        assert_eq!(
            body_caps["parameters"]["imageSize"],
            serde_json::Value::String("2K".into())
        );
    }

    #[test]
    fn decode_gemini_image_response_extracts_inline_data_snake_case() {
        let body = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "ok"},
                        {"inline_data": {"mime_type": "image/png", "data": "aGVsbG8="}}
                    ]
                }
            }]
        });
        let (bytes, mime) = decode_gemini_image_response(&body).unwrap();
        assert_eq!(bytes, b"hello");
        assert_eq!(mime.as_deref(), Some("image/png"));
    }

    #[test]
    fn decode_gemini_image_response_handles_camel_case() {
        let body = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [
                        {"inlineData": {"mimeType": "image/jpeg", "data": "aGVsbG8="}}
                    ]
                }
            }]
        });
        let (bytes, mime) = decode_gemini_image_response(&body).unwrap();
        assert_eq!(bytes, b"hello");
        assert_eq!(mime.as_deref(), Some("image/jpeg"));
    }

    #[test]
    fn decode_gemini_image_response_errors_when_no_image_part() {
        let body = serde_json::json!({
            "candidates": [{"content": {"parts": [{"text": "no image for you"}]}}]
        });
        let err = decode_gemini_image_response(&body).unwrap_err();
        assert!(err.to_string().to_lowercase().contains("no image"));
    }

    #[test]
    fn decode_imagen_response_extracts_bytes_base64_encoded() {
        let body = serde_json::json!({
            "predictions": [
                {"bytesBase64Encoded": "aGVsbG8=", "mimeType": "image/png"}
            ]
        });
        let (bytes, mime) = decode_imagen_response(&body).unwrap();
        assert_eq!(bytes, b"hello");
        assert_eq!(mime.as_deref(), Some("image/png"));
    }

    #[test]
    fn decode_imagen_response_errors_when_predictions_empty() {
        let body = serde_json::json!({"predictions": []});
        let err = decode_imagen_response(&body).unwrap_err();
        assert!(err.to_string().to_lowercase().contains("no predictions"));
    }

    #[tokio::test]
    async fn generate_google_rejects_url_only_with_clear_message() {
        let key = ApiKey::new_with_protocol(
            "test".into(),
            "test".into(),
            "https://generativelanguage.googleapis.com".into(),
            None,
            "fake".into(),
        );
        let request = ImageRequest {
            prompt: "x".into(),
            model: "imagen-4.0-generate-001".into(),
            size: None,
            quality: None,
        };
        let err = generate_google(&key, &request, None, false, true)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("--url"), "got: {msg}");
        assert!(msg.contains("base64"), "got: {msg}");
    }
}
