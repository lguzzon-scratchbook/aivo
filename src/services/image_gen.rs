//! Image generation service.
//!
//! Handles OpenAI-compatible `/v1/images/generations` and Google's
//! `generativelanguage.googleapis.com` surfaces (Gemini-native multimodal
//! image models via `:generateContent`, Imagen via `:predict`), plus the
//! output path UX (default/exact/directory/template forms), overwrite
//! policy, and atomic file writes.

use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine;
use chrono::Utc;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::services::http_utils::router_http_client;
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

/// How `-o`/`--output` was specified.
#[derive(Debug, Clone)]
pub enum OutputTarget {
    /// No `-o` given — use default timestamped filename in CWD.
    Default,
    /// `-o path` where the path ends with `/` or is an existing directory.
    Directory(PathBuf),
    /// `-o path` pointing at a specific file.
    File(PathBuf),
    /// `-o "tmpl.png"` — a template with `{ts}`/`{model}` tokens.
    Template(String),
}

impl OutputTarget {
    /// Parse the raw `-o` argument. Returns `Default` when `arg` is `None`.
    pub fn parse(arg: Option<&str>) -> Self {
        let raw = match arg {
            None => return Self::Default,
            Some(r) => r,
        };

        if raw.contains('{') && raw.contains('}') {
            return Self::Template(raw.to_string());
        }

        let path = PathBuf::from(raw);
        let looks_like_dir =
            raw.ends_with('/') || raw.ends_with(std::path::MAIN_SEPARATOR) || path.is_dir();
        if looks_like_dir {
            Self::Directory(path)
        } else {
            Self::File(path)
        }
    }

    /// True when the user's `-o` value pins a file extension. For Default
    /// and Directory the extension was chosen by us (currently "png"), so
    /// the caller is free to swap in the server's actual content-type.
    pub fn pins_extension(&self) -> bool {
        match self {
            Self::Default | Self::Directory(_) => false,
            Self::File(p) => p.extension().is_some(),
            Self::Template(s) => Path::new(s).extension().is_some(),
        }
    }
}

/// Resolves the concrete output path before any API call is made. Collision
/// resolution happens after the response, once the extension is known, via
/// [`apply_overwrite_policy`].
pub fn resolve_output_path(target: &OutputTarget, model: &str, ext: &str) -> Result<PathBuf> {
    let ts = Utc::now().format("%Y%m%d-%H%M%S").to_string();

    match target {
        OutputTarget::Default => Ok(PathBuf::from(format!("./aivo-{ts}.{ext}"))),
        OutputTarget::Directory(dir) => {
            verify_writable_dir(dir)?;
            Ok(dir.join(format!("aivo-{ts}.{ext}")))
        }
        OutputTarget::File(path) => {
            if let Some(parent) = parent_if_nonempty(path) {
                verify_writable_dir(parent)?;
            }
            let (stem, real_ext) = split_stem_ext(path, ext);
            let dir = path.parent().unwrap_or(Path::new("."));
            Ok(dir.join(format!("{stem}.{real_ext}")))
        }
        OutputTarget::Template(tmpl) => {
            let expanded = expand_template(tmpl, &ts, model);
            let path = PathBuf::from(&expanded);
            if let Some(parent) = parent_if_nonempty(&path) {
                verify_writable_dir(parent)?;
            }
            Ok(path)
        }
    }
}

/// Returns `path.parent()` only when it's a non-empty path. `Path::parent()`
/// returns `Some("")` for bare filenames like `cat.png`, which isn't useful
/// for directory checks.
fn parent_if_nonempty(path: &Path) -> Option<&Path> {
    path.parent().filter(|p| !p.as_os_str().is_empty())
}

fn expand_template(tmpl: &str, ts: &str, model: &str) -> String {
    tmpl.replace("{ts}", ts)
        .replace("{model}", &sanitize_model(model))
}

/// Replaces filesystem-hostile characters in a model id with underscores.
pub fn sanitize_model(model: &str) -> String {
    model
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            c => c,
        })
        .collect()
}

fn split_stem_ext(path: &Path, default_ext: &str) -> (String, String) {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("aivo")
        .to_string();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| default_ext.to_string());
    (stem, ext)
}

fn verify_writable_dir(dir: &Path) -> Result<()> {
    if !dir.exists() {
        bail!(
            "directory '{}' does not exist (create it first, or omit -o)",
            dir.display()
        );
    }
    if !dir.is_dir() {
        bail!("'{}' is not a directory", dir.display());
    }
    // Probe writability with a metadata check; a true write test would race.
    let meta = fs::metadata(dir)
        .with_context(|| format!("cannot access directory '{}'", dir.display()))?;
    if meta.permissions().readonly() {
        bail!("cannot write to '{}': permission denied", dir.display());
    }
    Ok(())
}

/// Decision made by [`apply_overwrite_policy`] for a single file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OverwriteDecision {
    /// Write at this exact path (replacing any existing file, or writing a
    /// fresh file, or a `-1`/`-2`/… auto-suffix when the user chose skip).
    Write(PathBuf),
    /// Abort the whole run (non-TTY / JSON / explicit "no").
    Abort,
}

/// Controls how [`apply_overwrite_policy`] resolves existing-file collisions.
#[derive(Debug, Clone, Copy)]
pub struct OverwritePolicy {
    pub force: bool,
    pub interactive: bool,
}

impl OverwritePolicy {
    pub fn from_flags(force: bool, json_mode: bool) -> Self {
        Self {
            force,
            interactive: !json_mode && io::stdin().is_terminal() && io::stderr().is_terminal(),
        }
    }
}

/// Decides what to do with a single intended target path.
///
/// When `force` is set, always returns `Write`. When the path doesn't exist,
/// also `Write`. Otherwise prompts (if `interactive`) or aborts.
pub fn apply_overwrite_policy(
    path: &Path,
    policy: OverwritePolicy,
    prompt_answer: Option<char>,
) -> OverwriteDecision {
    if !path.exists() {
        return OverwriteDecision::Write(path.to_path_buf());
    }
    if policy.force {
        return OverwriteDecision::Write(path.to_path_buf());
    }
    if !policy.interactive {
        return OverwriteDecision::Abort;
    }
    let answer = prompt_answer.unwrap_or('n');
    match answer {
        'y' | 'Y' | 'a' | 'A' => OverwriteDecision::Write(path.to_path_buf()),
        's' | 'S' => {
            // Auto-suffix until we find a free path.
            OverwriteDecision::Write(next_free_path(path))
        }
        _ => OverwriteDecision::Abort,
    }
}

/// Finds the next free path by appending `-1`, `-2`, … before the extension.
pub fn next_free_path(path: &Path) -> PathBuf {
    let parent = path.parent().unwrap_or(Path::new("."));
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("aivo");
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("png");
    for i in 1..=9999 {
        let candidate = parent.join(format!("{stem}-{i}.{ext}"));
        if !candidate.exists() {
            return candidate;
        }
    }
    // Extreme fallback — unlikely to ever hit.
    parent.join(format!("{stem}-overflow.{ext}"))
}

/// Prompts the user once about overwriting `path`. Returns the raw char the
/// user entered (lowercased) or `'n'` on empty/EOF.
pub fn prompt_overwrite(path: &Path) -> char {
    eprint!(
        "File '{}' exists. Overwrite? [y/N/s(kip-and-suffix)]: ",
        path.display()
    );
    let _ = io::stderr().flush();
    let mut buf = String::new();
    if io::stdin().read_line(&mut buf).is_err() {
        return 'n';
    }
    buf.trim()
        .chars()
        .next()
        .map(|c| c.to_ascii_lowercase())
        .unwrap_or('n')
}

/// Infers a file extension from an HTTP `Content-Type` header. Falls back to
/// `"png"` for anything unrecognized — OpenAI's default.
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

/// Writes bytes atomically via a `.part` sibling + rename. Cleans up the
/// partial file on failure so a failed run never leaves a half-written
/// file at the final name.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<u64> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let stem = path.file_name().and_then(|s| s.to_str()).unwrap_or("aivo");
    let suffix: u32 = rand::random();
    let tmp = parent.join(format!(".{stem}.aivo-tmp-{suffix:x}.part"));

    let write_result = (|| -> Result<u64> {
        let mut f = fs::File::create(&tmp)
            .with_context(|| format!("cannot create temp file '{}'", tmp.display()))?;
        f.write_all(bytes)
            .with_context(|| format!("write failed for '{}'", tmp.display()))?;
        f.sync_all().ok();
        drop(f);
        replace_with_temp_file(&tmp, path)?;
        Ok(bytes.len() as u64)
    })();

    if write_result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    write_result
}

fn replace_with_temp_file(tmp: &Path, path: &Path) -> Result<()> {
    #[cfg(windows)]
    if path.exists() {
        fs::remove_file(path)
            .with_context(|| format!("cannot replace existing file '{}'", path.display()))?;
    }

    fs::rename(tmp, path)
        .with_context(|| format!("rename '{}' -> '{}' failed", tmp.display(), path.display()))
}

// ---- Provider-level image generation ----

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

    let final_path = align_extension(target_path, ext_hint.as_deref(), pinned_extension);
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

    let final_path = align_extension(target_path, mime.as_deref(), pinned_extension);
    let written = atomic_write(&final_path, &bytes)?;
    Ok(ImageArtifact {
        path: Some(final_path),
        url: None,
        bytes: written,
    })
}

/// When the user didn't pin an extension, silently swap to the server's
/// actual content-type. When they did, honor their choice. Falls through
/// unchanged when the server didn't report a content-type.
fn align_extension(path: &Path, content_type: Option<&str>, pinned: bool) -> PathBuf {
    if pinned {
        return path.to_path_buf();
    }
    let Some(ct) = content_type else {
        return path.to_path_buf();
    };
    let server_ext = ext_from_content_type(Some(ct));
    let current_ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    if current_ext == server_ext {
        return path.to_path_buf();
    }
    path.with_extension(server_ext)
}

fn extract_error_message(body: &str) -> Option<String> {
    let v: Value = serde_json::from_str(body).ok()?;
    v.get("error")
        .and_then(|e| e.get("message"))
        .and_then(|m| m.as_str())
        .map(str::to_string)
        .or_else(|| {
            v.get("message")
                .and_then(|m| m.as_str())
                .map(str::to_string)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn parse_none_returns_default() {
        assert!(matches!(OutputTarget::parse(None), OutputTarget::Default));
    }

    #[test]
    fn parse_file_path() {
        let t = OutputTarget::parse(Some("cat.png"));
        assert!(matches!(t, OutputTarget::File(_)));
    }

    #[test]
    fn parse_trailing_slash_is_directory() {
        let t = OutputTarget::parse(Some("out/"));
        assert!(matches!(t, OutputTarget::Directory(_)));
    }

    #[test]
    fn parse_template_with_braces() {
        let t = OutputTarget::parse(Some("cat-{n}.png"));
        assert!(matches!(t, OutputTarget::Template(_)));
    }

    #[test]
    fn resolve_default() {
        let path = resolve_output_path(&OutputTarget::Default, "gpt-image-1", "png").unwrap();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.starts_with("aivo-"));
        assert!(name.ends_with(".png"));
    }

    #[test]
    fn resolve_exact_file() {
        let t = OutputTarget::File(PathBuf::from("cat.png"));
        let path = resolve_output_path(&t, "gpt-image-1", "png").unwrap();
        assert_eq!(path.to_string_lossy(), "cat.png");
    }

    #[test]
    fn resolve_directory_existing() {
        let tmp = TempDir::new().unwrap();
        let t = OutputTarget::Directory(tmp.path().to_path_buf());
        let path = resolve_output_path(&t, "gpt-image-1", "png").unwrap();
        assert!(path.starts_with(tmp.path()));
    }

    #[test]
    fn resolve_directory_missing_errors() {
        let t = OutputTarget::Directory(PathBuf::from("/definitely/does/not/exist/aivo-test"));
        let err = resolve_output_path(&t, "gpt-image-1", "png").unwrap_err();
        assert!(err.to_string().contains("does not exist"));
    }

    #[test]
    fn resolve_template_with_model_token() {
        let t = OutputTarget::Template("{model}.png".into());
        let path = resolve_output_path(&t, "gpt-image-1", "png").unwrap();
        assert_eq!(path.to_string_lossy(), "gpt-image-1.png");
    }

    #[test]
    fn resolve_template_with_ts_token() {
        let t = OutputTarget::Template("shot-{ts}.png".into());
        let path = resolve_output_path(&t, "gpt-image-1", "png").unwrap();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.starts_with("shot-"));
        assert!(name.ends_with(".png"));
    }

    #[test]
    fn resolve_template_checks_parent_directory() {
        let t = OutputTarget::Template("/definitely/does/not/exist/cat.png".into());
        let err = resolve_output_path(&t, "gpt-image-1", "png").unwrap_err();
        assert!(err.to_string().contains("does not exist"));
    }

    #[test]
    fn parent_if_nonempty_skips_bare_filename() {
        // `Path::parent()` returns Some("") for a bare filename — the helper
        // filters that out so callers don't run directory checks against "".
        assert!(parent_if_nonempty(Path::new("cat.png")).is_none());
        assert_eq!(
            parent_if_nonempty(Path::new("out/cat.png")),
            Some(Path::new("out"))
        );
        assert_eq!(
            parent_if_nonempty(Path::new("/abs/dir/cat.png")),
            Some(Path::new("/abs/dir"))
        );
    }

    #[test]
    fn sanitize_model_replaces_slashes_and_colons() {
        assert_eq!(sanitize_model("org/model"), "org_model");
        assert_eq!(sanitize_model("name:tag"), "name_tag");
        assert_eq!(sanitize_model("normal-name_1"), "normal-name_1");
    }

    #[test]
    fn pins_extension_distinguishes_user_vs_auto() {
        assert!(!OutputTarget::Default.pins_extension());
        assert!(!OutputTarget::Directory(PathBuf::from("out")).pins_extension());
        assert!(OutputTarget::File(PathBuf::from("cat.png")).pins_extension());
        assert!(!OutputTarget::File(PathBuf::from("cat")).pins_extension());
        assert!(OutputTarget::Template("{model}-{ts}.png".into()).pins_extension());
        assert!(!OutputTarget::Template("{model}-{ts}".into()).pins_extension());
    }

    #[test]
    fn align_extension_swaps_when_unpinned_and_mismatched() {
        assert_eq!(
            align_extension(Path::new("aivo.png"), Some("image/jpeg"), false),
            PathBuf::from("aivo.jpg")
        );
        assert_eq!(
            align_extension(Path::new("out/aivo.png"), Some("image/webp"), false),
            PathBuf::from("out/aivo.webp")
        );
    }

    #[test]
    fn align_extension_keeps_user_pinned_path() {
        assert_eq!(
            align_extension(Path::new("cat.png"), Some("image/jpeg"), true),
            PathBuf::from("cat.png")
        );
    }

    #[test]
    fn align_extension_noop_when_matched_or_unknown_ct() {
        assert_eq!(
            align_extension(Path::new("aivo.png"), Some("image/png"), false),
            PathBuf::from("aivo.png")
        );
        assert_eq!(
            align_extension(Path::new("aivo.png"), None, false),
            PathBuf::from("aivo.png")
        );
    }

    #[test]
    fn next_free_path_finds_suffix() {
        let tmp = TempDir::new().unwrap();
        let existing = tmp.path().join("cat.png");
        fs::write(&existing, b"x").unwrap();
        let free = next_free_path(&existing);
        assert_eq!(free, tmp.path().join("cat-1.png"));

        fs::write(tmp.path().join("cat-1.png"), b"x").unwrap();
        assert_eq!(next_free_path(&existing), tmp.path().join("cat-2.png"));
    }

    #[test]
    fn ext_from_content_type_maps_known_types() {
        assert_eq!(ext_from_content_type(Some("image/jpeg")), "jpg");
        assert_eq!(ext_from_content_type(Some("image/jpg")), "jpg");
        assert_eq!(ext_from_content_type(Some("image/webp")), "webp");
        assert_eq!(ext_from_content_type(Some("image/gif")), "gif");
        assert_eq!(ext_from_content_type(Some("image/png")), "png");
        assert_eq!(ext_from_content_type(Some("application/json")), "png"); // fallback
        assert_eq!(ext_from_content_type(None), "png");
    }

    #[test]
    fn ext_from_content_type_ignores_charset_suffix() {
        assert_eq!(ext_from_content_type(Some("image/jpeg; charset=x")), "jpg");
    }

    #[test]
    fn overwrite_policy_force_writes() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("a.png");
        fs::write(&path, b"old").unwrap();
        let d = apply_overwrite_policy(
            &path,
            OverwritePolicy {
                force: true,
                interactive: false,
            },
            None,
        );
        assert_eq!(d, OverwriteDecision::Write(path));
    }

    #[test]
    fn overwrite_policy_nontty_aborts() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("a.png");
        fs::write(&path, b"old").unwrap();
        let d = apply_overwrite_policy(
            &path,
            OverwritePolicy {
                force: false,
                interactive: false,
            },
            None,
        );
        assert_eq!(d, OverwriteDecision::Abort);
    }

    #[test]
    fn overwrite_policy_missing_file_writes() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("nope.png");
        let d = apply_overwrite_policy(
            &path,
            OverwritePolicy {
                force: false,
                interactive: false,
            },
            None,
        );
        assert_eq!(d, OverwriteDecision::Write(path));
    }

    #[test]
    fn overwrite_policy_yes_answer_writes() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("a.png");
        fs::write(&path, b"old").unwrap();
        let d = apply_overwrite_policy(
            &path,
            OverwritePolicy {
                force: false,
                interactive: true,
            },
            Some('y'),
        );
        assert_eq!(d, OverwriteDecision::Write(path));
    }

    #[test]
    fn overwrite_policy_no_answer_aborts() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("a.png");
        fs::write(&path, b"old").unwrap();
        let d = apply_overwrite_policy(
            &path,
            OverwritePolicy {
                force: false,
                interactive: true,
            },
            Some('n'),
        );
        assert_eq!(d, OverwriteDecision::Abort);
    }

    #[test]
    fn overwrite_policy_skip_finds_free_path() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("a.png");
        fs::write(&path, b"old").unwrap();
        let d = apply_overwrite_policy(
            &path,
            OverwritePolicy {
                force: false,
                interactive: true,
            },
            Some('s'),
        );
        match d {
            OverwriteDecision::Write(p) => {
                assert_eq!(p, tmp.path().join("a-1.png"));
            }
            _ => panic!("expected Write with suffixed path"),
        }
    }

    #[test]
    fn atomic_write_produces_final_file_only() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("out.png");
        let n = atomic_write(&path, b"hello").unwrap();
        assert_eq!(n, 5);
        assert_eq!(fs::read(&path).unwrap(), b"hello");
        // No stray tmp files left behind
        let dir_entries: Vec<_> = fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(dir_entries.len(), 1);
        assert_eq!(dir_entries[0], "out.png");
    }

    #[test]
    fn atomic_write_replaces_existing_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("out.png");
        fs::write(&path, b"old").unwrap();
        let n = atomic_write(&path, b"new bytes").unwrap();
        assert_eq!(n, 9);
        assert_eq!(fs::read(&path).unwrap(), b"new bytes");
    }

    #[test]
    fn extract_error_message_reads_openai_shape() {
        let body = r#"{"error":{"message":"bad prompt","type":"invalid"}}"#;
        assert_eq!(extract_error_message(body).as_deref(), Some("bad prompt"));
    }

    #[test]
    fn extract_error_message_reads_flat_shape() {
        let body = r#"{"message":"rate limited"}"#;
        assert_eq!(extract_error_message(body).as_deref(), Some("rate limited"));
    }

    #[test]
    fn extract_error_message_returns_none_for_plain_text() {
        assert!(extract_error_message("not json").is_none());
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
        // Unknown WxH falls through to None rather than guessing.
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
        // Without a size hint, no imageConfig block is emitted (let server default).
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
        // Case-insensitive + trimmed: locks in `to_ascii_lowercase` + `trim`.
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

    #[tokio::test]
    async fn generate_google_rejects_url_only_with_clear_message() {
        // The Google APIs return base64 only — there is no signed-URL form.
        // Verify the dispatcher rejects --url before any HTTP is attempted.
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
