//! Video generation service.
//!
//! Video providers run async on every backend we support today: OpenAI's
//! Sora (`/v1/videos`), Google Veo (`:predictLongRunning`), xAI Grok
//! Imagine (`/v1/videos/generations`), and Vercel AI Gateway
//! (`/v4/ai/video-model`, single-event SSE). Each accepts a prompt,
//! returns a job/operation ID (or an SSE stream), and requires a polling
//! loop until the result is ready. We submit, print the job ID to stderr
//! (so the user can recover after Ctrl+C), poll on a fixed interval until
//! the configured timeout, then download the bytes.
//!
//! For recovery, [`attach`] skips submit and jumps straight into the
//! polling loop against an existing job ID. Vercel's path doesn't expose
//! a job ID and rejects `--job-id` cleanly.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::time::{Instant, sleep};

use crate::services::http_utils::{router_http_client, router_http_client_with_timeout};
use crate::services::media_io::{align_extension, atomic_write, extract_error_message};
use crate::services::provider_protocol::{ProviderProtocol, detect_provider_protocol};
use crate::services::session_store::ApiKey;

/// One video generation request.
#[derive(Debug, Clone)]
pub struct VideoRequest {
    pub prompt: String,
    pub model: String,
    /// `WxH` (e.g. `1280x720`) or `W:H` (e.g. `16:9`). Server interprets.
    pub size: Option<String>,
    pub seconds: Option<u32>,
    pub seed: Option<u64>,
}

/// One generated video.
#[derive(Debug, Clone)]
pub struct VideoArtifact {
    pub path: Option<PathBuf>,
    pub url: Option<String>,
    pub bytes: u64,
    /// Provider-side job/operation ID. `Some` for routes that expose one
    /// (Sora `video_…`, Veo `operations/…`, xAI `request_id`); `None` for
    /// Vercel's single-SSE path which has no job concept and rejects
    /// `--job-id` recovery upfront.
    pub job_id: Option<String>,
}

/// Knobs for the polling loop.
#[derive(Debug, Clone, Copy)]
pub struct PollOptions {
    pub timeout: Duration,
    pub interval: Duration,
}

impl Default for PollOptions {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(600),
            interval: Duration::from_secs(5),
        }
    }
}

/// Maps an HTTP `Content-Type` header to a video file extension. Falls
/// back to `"mp4"` for unrecognized values — Sora's default.
pub fn ext_from_content_type(ct: Option<&str>) -> String {
    match ct.map(|c| {
        c.split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase()
    }) {
        Some(ref s) if s == "video/mp4" => "mp4".into(),
        Some(ref s) if s == "video/webm" => "webm".into(),
        Some(ref s) if s == "video/quicktime" => "mov".into(),
        Some(ref s) if s == "video/x-matroska" || s == "video/mkv" => "mkv".into(),
        _ => "mp4".into(),
    }
}

/// Submit a fresh job and poll until completion.
pub async fn generate(
    key: &ApiKey,
    request: &VideoRequest,
    path: Option<&Path>,
    pinned_extension: bool,
    poll: PollOptions,
) -> Result<VideoArtifact> {
    if is_vercel_gateway(&key.base_url) {
        // Vercel AI Gateway is OpenAI-protocol for chat but ships video
        // through a separate /v4/ai/video-model endpoint with a single SSE
        // event (no job ID, no polling). Route it before the protocol
        // dispatch so the OpenAI branch doesn't 404 on /v1/videos.
        return vercel_generate(key, request, path, pinned_extension, poll).await;
    }
    if is_xai_endpoint(&key.base_url) {
        // xAI is OpenAI-protocol for chat / images but uses a different
        // path for video — `/v1/videos/generations` (note `/generations`
        // suffix), not Sora's `/v1/videos`. Different request body shape
        // (`aspect_ratio` + `resolution` as separate fields) and a custom
        // poll response (`{status, video.url}`).
        let request_id = xai_submit(key, request).await?;
        announce_job_id(&request_id);
        return xai_poll_and_download(key, &request_id, path, pinned_extension, poll).await;
    }
    let protocol = detect_provider_protocol(&key.base_url);
    match protocol {
        ProviderProtocol::Openai | ProviderProtocol::ResponsesApi => {
            let job_id = openai_submit(key, request).await?;
            announce_job_id(&job_id);
            openai_poll_and_download(key, &job_id, path, pinned_extension, poll).await
        }
        ProviderProtocol::Google => {
            let op_name = google_submit(key, request).await?;
            announce_job_id(&op_name);
            google_poll_and_download(key, &op_name, path, pinned_extension, poll).await
        }
        ProviderProtocol::Anthropic => bail!("Anthropic does not support video generation"),
    }
}

/// Skip the submit step; pick up an existing job and poll/download.
/// Use this to recover after a Ctrl+C: rerun with `--job-id <id>` and the
/// CLI will attach to the in-flight job instead of submitting a fresh one.
pub async fn attach(
    key: &ApiKey,
    job_id: &str,
    path: Option<&Path>,
    pinned_extension: bool,
    poll: PollOptions,
) -> Result<VideoArtifact> {
    if is_vercel_gateway(&key.base_url) {
        bail!(
            "Vercel AI Gateway video generation is synchronous (single SSE) and \
             returns no job ID — `--job-id` isn't supported here. Re-run the \
             original `aivo video <prompt>` command instead."
        );
    }
    if is_xai_endpoint(&key.base_url) {
        return xai_poll_and_download(key, job_id, path, pinned_extension, poll).await;
    }
    let protocol = detect_provider_protocol(&key.base_url);
    match protocol {
        ProviderProtocol::Openai | ProviderProtocol::ResponsesApi => {
            openai_poll_and_download(key, job_id, path, pinned_extension, poll).await
        }
        ProviderProtocol::Google => {
            // Google's operation IDs come back as `operations/xxx` on submit.
            // Accept either form so users can paste either.
            let op = if job_id.contains('/') {
                job_id.to_string()
            } else {
                format!("operations/{job_id}")
            };
            google_poll_and_download(key, &op, path, pinned_extension, poll).await
        }
        ProviderProtocol::Anthropic => bail!("Anthropic does not support video generation"),
    }
}

fn announce_job_id(job_id: &str) {
    use std::io::IsTerminal;
    if std::io::stderr().is_terminal() {
        eprintln!(
            "  job id: {} (recover with --job-id <id> if interrupted)",
            job_id
        );
    } else {
        // Non-TTY (CI, --json piped through jq) — keep stderr machine-grep-able.
        eprintln!("aivo: video job_id={}", job_id);
    }
}

// ── OpenAI Sora ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct OpenAIVideoJob {
    id: String,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    error: Option<Value>,
}

async fn openai_submit(key: &ApiKey, request: &VideoRequest) -> Result<String> {
    let base = key.base_url.trim_end_matches('/');
    let url = if base.ends_with("/v1") {
        format!("{base}/videos")
    } else {
        format!("{base}/v1/videos")
    };

    let mut body = json!({
        "model": request.model,
        "prompt": request.prompt,
    });
    if let Some(s) = &request.size {
        body["size"] = Value::String(s.clone());
    }
    if let Some(secs) = request.seconds {
        // The Sora API has historically accepted `seconds` as either
        // string or int — send as int and let the server coerce.
        body["seconds"] = json!(secs);
    }
    if let Some(seed) = request.seed {
        body["seed"] = json!(seed);
    }

    let client = router_http_client();
    let response = client
        .post(&url)
        .bearer_auth(key.key.as_str())
        .json(&body)
        .send()
        .await
        .with_context(|| format!("video submit to {url} failed"))?;

    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        let detail = extract_error_message(&text).unwrap_or_else(|| text.clone());
        bail!("video submit failed ({}): {}", status.as_u16(), detail);
    }

    let job: OpenAIVideoJob = response
        .json()
        .await
        .context("failed to decode video submit response")?;
    Ok(job.id)
}

async fn openai_poll_and_download(
    key: &ApiKey,
    job_id: &str,
    path: Option<&Path>,
    pinned_extension: bool,
    poll: PollOptions,
) -> Result<VideoArtifact> {
    let base = key.base_url.trim_end_matches('/');
    let job_url = if base.ends_with("/v1") {
        format!("{base}/videos/{job_id}")
    } else {
        format!("{base}/v1/videos/{job_id}")
    };
    let content_url = format!("{job_url}/content");

    let client = router_http_client();
    let started = Instant::now();
    loop {
        if started.elapsed() >= poll.timeout {
            bail!(
                "video polling timed out after {}s (job is still running — recover with `aivo video --job-id {}`)",
                poll.timeout.as_secs(),
                job_id,
            );
        }

        let response = client
            .get(&job_url)
            .bearer_auth(key.key.as_str())
            .send()
            .await
            .with_context(|| format!("polling {job_url} failed"))?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            let detail = extract_error_message(&text).unwrap_or_else(|| text.clone());
            bail!("video poll failed ({}): {}", status.as_u16(), detail);
        }

        let job: OpenAIVideoJob = response
            .json()
            .await
            .context("failed to decode video poll response")?;
        let state = job.status.as_deref().unwrap_or("").to_ascii_lowercase();
        if matches!(state.as_str(), "completed" | "succeeded" | "done") {
            break;
        }
        if matches!(
            state.as_str(),
            "failed" | "error" | "cancelled" | "canceled"
        ) {
            let detail = job
                .error
                .as_ref()
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .map(str::to_string)
                .unwrap_or_else(|| state.clone());
            bail!("video generation {} ({}): {}", state, job_id, detail);
        }

        sleep(poll.interval).await;
    }

    // Status: completed. Download the content.
    let resp = client
        .get(&content_url)
        .bearer_auth(key.key.as_str())
        .send()
        .await
        .with_context(|| format!("downloading video from {content_url} failed"))?;
    let dl_status = resp.status();
    if !dl_status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        let detail = extract_error_message(&text).unwrap_or_else(|| text.clone());
        bail!(
            "video content download failed ({}): {}",
            dl_status.as_u16(),
            detail
        );
    }
    let ct = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|h| h.to_str().ok())
        .map(str::to_string);
    let body = resp.bytes().await.context("reading video body failed")?;
    let bytes = body.to_vec();

    let Some(target_path) = path else {
        return Ok(VideoArtifact {
            path: None,
            url: None,
            bytes: bytes.len() as u64,
            job_id: Some(job_id.to_string()),
        });
    };

    let server_ext = ct.as_deref().map(|c| ext_from_content_type(Some(c)));
    let final_path = align_extension(target_path, server_ext.as_deref(), pinned_extension);
    let written = atomic_write(&final_path, &bytes)?;
    Ok(VideoArtifact {
        path: Some(final_path),
        url: None,
        bytes: written,
        job_id: Some(job_id.to_string()),
    })
}

// ── Google Veo ──────────────────────────────────────────────────────────

async fn google_submit(key: &ApiKey, request: &VideoRequest) -> Result<String> {
    let trimmed = key.base_url.trim_end_matches('/');
    let root = trimmed.strip_suffix("/v1beta").unwrap_or(trimmed);
    let url = format!(
        "{root}/v1beta/models/{model}:predictLongRunning",
        model = request.model,
    );

    let mut parameters = json!({});
    if let Some(s) = &request.size
        && let Some(ratio) = aspect_ratio_for_size(s.as_str())
    {
        parameters["aspectRatio"] = Value::String(ratio);
    }
    if let Some(secs) = request.seconds {
        parameters["durationSeconds"] = json!(secs);
    }
    if let Some(seed) = request.seed {
        parameters["seed"] = json!(seed);
    }

    let body = json!({
        "instances": [{"prompt": request.prompt}],
        "parameters": parameters,
    });

    let client = router_http_client();
    let response = client
        .post(&url)
        .header("x-goog-api-key", key.key.as_str())
        .json(&body)
        .send()
        .await
        .with_context(|| format!("video submit to {url} failed"))?;

    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        let detail = extract_error_message(&text).unwrap_or_else(|| text.clone());
        bail!("video submit failed ({}): {}", status.as_u16(), detail);
    }

    let parsed: Value = response
        .json()
        .await
        .context("failed to decode Veo submit response")?;
    let name = parsed
        .get("name")
        .and_then(|n| n.as_str())
        .ok_or_else(|| anyhow!("Veo submit response missing 'name'"))?;
    Ok(name.to_string())
}

async fn google_poll_and_download(
    key: &ApiKey,
    op_name: &str,
    path: Option<&Path>,
    pinned_extension: bool,
    poll: PollOptions,
) -> Result<VideoArtifact> {
    let trimmed = key.base_url.trim_end_matches('/');
    let root = trimmed.strip_suffix("/v1beta").unwrap_or(trimmed);
    // op_name is `operations/xxx`; the GET URL is `{root}/v1beta/{op_name}`.
    let op_url = format!("{root}/v1beta/{op_name}");

    let client = router_http_client();
    let started = Instant::now();

    let response_value: Value = loop {
        if started.elapsed() >= poll.timeout {
            bail!(
                "video polling timed out after {}s (operation is still running — recover with `aivo video --job-id {}`)",
                poll.timeout.as_secs(),
                op_name,
            );
        }

        let response = client
            .get(&op_url)
            .header("x-goog-api-key", key.key.as_str())
            .send()
            .await
            .with_context(|| format!("polling {op_url} failed"))?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            let detail = extract_error_message(&text).unwrap_or_else(|| text.clone());
            bail!("video poll failed ({}): {}", status.as_u16(), detail);
        }

        let parsed: Value = response
            .json()
            .await
            .context("failed to decode Veo poll response")?;
        let done = parsed
            .get("done")
            .and_then(|d| d.as_bool())
            .unwrap_or(false);
        if done {
            if let Some(error) = parsed.get("error") {
                let msg = error
                    .get("message")
                    .and_then(|m| m.as_str())
                    .map(str::to_string)
                    .unwrap_or_else(|| error.to_string());
                bail!("video generation failed: {}", msg);
            }
            break parsed;
        }

        sleep(poll.interval).await;
    };

    let video_uri = extract_veo_video_uri(&response_value)?;
    let resp = client
        .get(&video_uri)
        .header("x-goog-api-key", key.key.as_str())
        .send()
        .await
        .with_context(|| format!("downloading video from {video_uri} failed"))?;
    let dl_status = resp.status();
    if !dl_status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        let detail = extract_error_message(&text).unwrap_or_else(|| text.clone());
        bail!("video download failed ({}): {}", dl_status.as_u16(), detail);
    }
    let ct = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|h| h.to_str().ok())
        .map(str::to_string);
    let body = resp.bytes().await.context("reading video body failed")?;
    let bytes = body.to_vec();

    let Some(target_path) = path else {
        return Ok(VideoArtifact {
            path: None,
            url: Some(video_uri),
            bytes: bytes.len() as u64,
            job_id: Some(op_name.to_string()),
        });
    };

    let server_ext = ct.as_deref().map(|c| ext_from_content_type(Some(c)));
    let final_path = align_extension(target_path, server_ext.as_deref(), pinned_extension);
    let written = atomic_write(&final_path, &bytes)?;
    Ok(VideoArtifact {
        path: Some(final_path),
        url: Some(video_uri),
        bytes: written,
        job_id: Some(op_name.to_string()),
    })
}

/// Veo's done operation has wandered between two response shapes across API
/// generations. Try the new `generateVideoResponse.generatedSamples[0]
/// .video.uri` path first, then fall back to the older `predictions[0]
/// .videoUri`. Both have appeared in current Google docs.
fn extract_veo_video_uri(response: &Value) -> Result<String> {
    // The actual generation result lives under `response` for long-running
    // operations.
    let inner = response.get("response").unwrap_or(response);

    if let Some(uri) = inner
        .get("generateVideoResponse")
        .and_then(|r| r.get("generatedSamples"))
        .and_then(|s| s.as_array())
        .and_then(|arr| arr.first())
        .and_then(|sample| sample.get("video"))
        .and_then(|v| v.get("uri"))
        .and_then(|u| u.as_str())
    {
        return Ok(uri.to_string());
    }

    if let Some(uri) = inner
        .get("predictions")
        .and_then(|p| p.as_array())
        .and_then(|arr| arr.first())
        .and_then(|pred| pred.get("videoUri").or_else(|| pred.get("video_uri")))
        .and_then(|u| u.as_str())
    {
        return Ok(uri.to_string());
    }

    bail!("Veo response did not contain a generated video URI")
}

// ── Vercel AI Gateway ───────────────────────────────────────────────────
//
// This contract is reverse-engineered from `vercel/ai`'s
// `packages/gateway/src/gateway-video-model.ts` and `gateway-provider.ts`.
// Vercel marks the SDK function `experimental_generateVideo` "experimental"
// and does not publish an HTTP API — they may change the wire shape
// without notice. We pin to it because there's no other way to use
// Vercel's gateway for video, and aivo users routing through Vercel would
// otherwise get a dead-end 404 on `/v1/videos`.

/// Vercel's global gateway protocol version, sent on every request
/// regardless of modality. Source:
/// `packages/gateway/src/gateway-provider.ts:174` defines
/// `AI_GATEWAY_PROTOCOL_VERSION = '0.0.1'` and `:198` injects it as
/// `ai-gateway-protocol-version`. Omitting it produces `400 Unsupported
/// gateway protocol version` — that's the per-modality spec version's
/// older sibling, easy to miss because it lives in the provider file
/// rather than the video-model file.
const VERCEL_GATEWAY_PROTOCOL_VERSION: &str = "0.0.1";

/// Vercel's per-modality video-model spec version. Source:
/// `packages/gateway/src/gateway-video-model.ts:196`. Bumped from `3` to
/// `4` on 2026-03-17 (commit `73848413`, `v3 -> v4 spec usage for ai@7
/// beta`); no further bumps as of 2026-05.
const VERCEL_VIDEO_MODEL_SPEC_VERSION: &str = "4";

/// Token prefix mimicking what Node's undici + AI SDK produce. We tried
/// just `aivo/<v> ai-sdk/gateway/4.0.0` first; Seedance still failed.
/// `ai-cli` (which works) sends a UA starting literally with `node ` and
/// containing the `ai-sdk/provider-utils/<v>` token too — Vercel or the
/// upstream may be doing structural matching. We mirror that prefix while
/// still identifying as aivo via a trailing `runtime/aivo/<version>`
/// segment. If this is the wrong angle, `AIVO_DEBUG=1` will surface what
/// we're actually sending so we can compare against `NODE_DEBUG=http`
/// output from `ai-cli`.
const VERCEL_USER_AGENT_PREFIX: &str = "node ai-sdk/gateway/4.0.0 ai-sdk/provider-utils/5.0.0";

/// Returns true when the key's base URL points at Vercel's AI Gateway.
/// Vercel routes video through a dedicated `/v4/ai/video-model` endpoint
/// rather than the OpenAI-shaped `/v1/...` namespace, so we have to
/// detect it before the protocol-based dispatch.
///
/// Setting `AIVO_VERCEL_BASE_URL` forces the Vercel branch regardless of
/// the key's stored URL — useful when routing through a local echo
/// server to capture wire bytes for diagnostics.
fn is_vercel_gateway(base_url: &str) -> bool {
    if std::env::var_os("AIVO_VERCEL_BASE_URL").is_some() {
        return true;
    }
    base_url.contains("ai-gateway.vercel.sh")
}

/// Builds `<scheme>://<host>/v4/ai/video-model` from the user's stored
/// base URL. Falls back to the canonical Vercel host on parse failure so
/// detection-by-substring stays useful even with a malformed entry.
///
/// `AIVO_VERCEL_BASE_URL` overrides the host entirely — point it at e.g.
/// `http://127.0.0.1:8888/v4/ai` to redirect through a local capture
/// server. We append `/video-model` only if the override doesn't already
/// have it.
fn vercel_video_url(base_url: &str) -> String {
    if let Ok(override_url) = std::env::var("AIVO_VERCEL_BASE_URL") {
        let trimmed = override_url.trim_end_matches('/');
        return if trimmed.ends_with("/video-model") {
            trimmed.to_string()
        } else {
            format!("{trimmed}/video-model")
        };
    }
    if let Ok(parsed) = reqwest::Url::parse(base_url.trim_end_matches('/'))
        && let Some(host) = parsed.host_str()
    {
        return format!("{}://{host}/v4/ai/video-model", parsed.scheme());
    }
    "https://ai-gateway.vercel.sh/v4/ai/video-model".to_string()
}

/// Body schema for Vercel's video-model endpoint. Field declaration
/// order is **load-bearing**: `serde_json::to_string` writes fields in
/// this exact order on the wire, and ByteDance's upstream Seedance
/// pipeline silently fails generation when the JSON keys aren't in
/// `prompt, n, aspectRatio, …` order (the order JavaScript's
/// `JSON.stringify` produces from `ai-cli`'s object literal). Switching
/// from `serde_json::json!({...})` to a derived `Serialize` struct is
/// the fix — `json!` builds a `BTreeMap`-backed `Value` that sorts keys
/// alphabetically (`aspectRatio, duration, n, prompt`), and at least one
/// step in ByteDance's stack treats that as a different request even
/// though the parsed JSON is semantically identical.
///
/// Source for the canonical order:
/// `vercel/ai/packages/gateway/src/gateway-video-model.ts:60-71`.
#[derive(Serialize)]
struct VercelVideoBody<'a> {
    prompt: &'a str,
    n: u32,
    #[serde(rename = "aspectRatio", skip_serializing_if = "Option::is_none")]
    aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    resolution: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
}

/// Builds the request body. Splits a `--size` value into `aspectRatio`
/// (for `W:H` form like `16:9`) or `resolution` (for `WxH` form like
/// `1920x1080`). Only one of the two is ever populated.
fn build_vercel_video_body(request: &VideoRequest) -> VercelVideoBody<'_> {
    let mut aspect_ratio = None;
    let mut resolution = None;
    if let Some(s) = &request.size {
        let trimmed = s.trim();
        if trimmed.contains(':') {
            aspect_ratio = Some(trimmed.to_string());
        } else if trimmed.contains('x') {
            resolution = Some(trimmed.to_string());
        }
    }
    VercelVideoBody {
        prompt: request.prompt.as_str(),
        n: 1,
        aspect_ratio,
        resolution,
        duration: request.seconds,
        seed: request.seed,
    }
}

/// Parses Vercel's single SSE event. The gateway holds the connection
/// open for the entire generation, then emits one `data: {...}` event
/// and closes. Multi-line `data:` accumulation per the SSE spec is
/// implemented but Vercel emits a single one in practice.
fn parse_vercel_sse_event(body: &str) -> Result<Value> {
    let mut data = String::new();
    for line in body.lines() {
        if line.is_empty() {
            // Blank line terminates an event; first event wins.
            if !data.is_empty() {
                break;
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(rest.strip_prefix(' ').unwrap_or(rest));
        }
        // `event:` / `id:` / `retry:` lines are ignored — we only care
        // about the JSON payload Vercel ships in `data:`.
    }
    if data.is_empty() {
        bail!("Vercel SSE response had no data event");
    }
    serde_json::from_str(&data).context("parsing Vercel SSE event JSON")
}

async fn vercel_generate(
    key: &ApiKey,
    request: &VideoRequest,
    path: Option<&Path>,
    pinned_extension: bool,
    poll: PollOptions,
) -> Result<VideoArtifact> {
    let url = vercel_video_url(&key.base_url);
    let body = build_vercel_video_body(request);
    // Serialize directly — `serde_json::to_string` honors the struct's
    // field declaration order. Going through `Value` (e.g. `json!({})`)
    // would alphabetize the keys via `BTreeMap` and trip ByteDance's
    // upstream order check.
    let body_str = serde_json::to_string(&body).context("failed to serialize Vercel video body")?;

    // Vercel holds the SSE connection open for the entire generation
    // (potentially several minutes). Use a client whose overall timeout
    // matches `--timeout` plus a small buffer for the bytes download.
    let client_timeout = poll.timeout.as_secs().saturating_add(60);
    let client = router_http_client_with_timeout(client_timeout);

    let user_agent = format!(
        "{VERCEL_USER_AGENT_PREFIX} runtime/aivo/{}",
        env!("CARGO_PKG_VERSION")
    );
    // Single source of truth for the header set. The debug dump and the
    // request builder both iterate this list — keeps them from drifting,
    // since a missing header on Vercel video usually means a silent
    // upstream failure with a generic 500. ByteDance Seedance's CDN
    // rejects requests with an empty `user-agent` (reqwest sends none by
    // default; ai-cli works because Node/undici always fills one in), so
    // we set it explicitly. `x-title` is attribution that ai-cli sends.
    let headers: [(&str, &str); 7] = [
        ("accept", "text/event-stream"),
        (
            "ai-gateway-protocol-version",
            VERCEL_GATEWAY_PROTOCOL_VERSION,
        ),
        ("ai-gateway-auth-method", "api-key"),
        ("ai-model-id", &request.model),
        (
            "ai-video-model-specification-version",
            VERCEL_VIDEO_MODEL_SPEC_VERSION,
        ),
        ("user-agent", &user_agent),
        ("x-title", "aivo"),
    ];

    let debug = std::env::var_os("AIVO_DEBUG").is_some();
    if debug {
        eprintln!("[aivo debug] POST {url}");
        eprintln!("[aivo debug] header authorization: Bearer ***");
        for (k, v) in &headers {
            eprintln!("[aivo debug] header {k}: {v}");
        }
        eprintln!("[aivo debug] body: {body_str}");
    }

    let mut builder = client
        .post(&url)
        .bearer_auth(key.key.as_str())
        .header("content-type", "application/json");
    for (k, v) in &headers {
        builder = builder.header(*k, *v);
    }
    let response = builder
        .body(body_str)
        .send()
        .await
        .with_context(|| format!("Vercel video request to {url} failed"))?;
    if debug {
        eprintln!("[aivo debug] response status: {}", response.status());
        for (k, v) in response.headers() {
            eprintln!(
                "[aivo debug] response header {}: {}",
                k,
                v.to_str().unwrap_or("<non-ascii>")
            );
        }
    }

    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        let detail = extract_error_message(&text).unwrap_or_else(|| text.clone());
        bail!("Vercel gateway HTTP {} — {}", status.as_u16(), detail);
    }

    let body_text = response
        .text()
        .await
        .context("reading Vercel SSE body failed")?;
    if debug {
        eprintln!("[aivo debug] response body:\n{body_text}");
    }
    let event = parse_vercel_sse_event(&body_text)?;

    let event_type = event.get("type").and_then(|t| t.as_str()).unwrap_or("");
    if event_type == "error" {
        // SSE error events come from the upstream provider relayed by
        // Vercel — the gateway itself returned 200, but the actual
        // generation failed. Label it that way so users don't chase
        // gateway/auth bugs when the model upstream is the problem.
        let msg = event
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown error");
        let upstream = event
            .get("statusCode")
            .and_then(|c| c.as_u64())
            .map(|c| format!(" (upstream status {c})"))
            .unwrap_or_default();
        bail!("upstream video provider failed{}: {}", upstream, msg);
    }
    if event_type != "result" {
        bail!("Vercel SSE returned unexpected event type: {event_type}");
    }

    let videos = event
        .get("videos")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("Vercel result event missing 'videos' array"))?;
    let first = videos
        .first()
        .ok_or_else(|| anyhow!("Vercel returned no videos"))?;

    let (bytes, source_url, mime) = decode_vercel_video(&client, first).await?;

    let Some(target_path) = path else {
        return Ok(VideoArtifact {
            path: None,
            url: source_url,
            bytes: bytes.len() as u64,
            // Vercel has no real job id — surface the model so JSON
            // callers still have something correlatable.
            job_id: None,
        });
    };

    let server_ext = mime.as_deref().map(|c| ext_from_content_type(Some(c)));
    let final_path = align_extension(target_path, server_ext.as_deref(), pinned_extension);
    let written = atomic_write(&final_path, &bytes)?;
    Ok(VideoArtifact {
        path: Some(final_path),
        url: source_url,
        bytes: written,
        job_id: None,
    })
}

/// Resolve a Vercel `videos[i]` entry into raw bytes + source URL +
/// MIME. Two shapes per `gateway-video-model.ts:218-229`:
///   * `{type:"url", url, mediaType}` → GET the URL
///   * `{type:"base64", data, mediaType}` → decode inline
async fn decode_vercel_video(
    client: &reqwest::Client,
    video: &Value,
) -> Result<(Vec<u8>, Option<String>, Option<String>)> {
    let kind = video
        .get("type")
        .and_then(|t| t.as_str())
        .ok_or_else(|| anyhow!("Vercel video entry missing 'type'"))?;
    match kind {
        "url" => {
            let url = video
                .get("url")
                .and_then(|u| u.as_str())
                .ok_or_else(|| anyhow!("Vercel video.url missing 'url' field"))?
                .to_string();
            let mime = video
                .get("mediaType")
                .and_then(|m| m.as_str())
                .map(str::to_string);
            let resp = client
                .get(&url)
                .send()
                .await
                .with_context(|| format!("Vercel video download from {url} failed"))?;
            let dl_status = resp.status();
            if !dl_status.is_success() {
                bail!(
                    "Vercel video download failed: HTTP {} (signed URL may have expired)",
                    dl_status.as_u16()
                );
            }
            let bytes = resp
                .bytes()
                .await
                .context("reading Vercel video body failed")?
                .to_vec();
            Ok((bytes, Some(url), mime))
        }
        "base64" => {
            let data = video
                .get("data")
                .and_then(|d| d.as_str())
                .ok_or_else(|| anyhow!("Vercel video.base64 missing 'data' field"))?;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(data)
                .context("failed to decode Vercel base64 video payload")?;
            let mime = video
                .get("mediaType")
                .and_then(|m| m.as_str())
                .map(str::to_string);
            Ok((bytes, None, mime))
        }
        other => bail!("Vercel video has unexpected source type: {other}"),
    }
}

// ── xAI Grok Imagine ────────────────────────────────────────────────────
//
// Public docs: https://docs.x.ai/docs/guides/video-generation
//
// Submit:  POST /v1/videos/generations  (note the `/generations` suffix —
//          this is *different* from Sora's `/v1/videos`)
// Poll:    GET  /v1/videos/{request_id}
// Auth:    standard `Authorization: Bearer`
// Sync:    async; submit returns `{request_id, ...}` immediately; poll
//          until `status` is `done` / `succeeded` / `completed`.
// Body fields aren't OpenAI-shaped: `aspect_ratio` (W:H string) and
// `resolution` ("480p" or "720p") are separate, and there's no `n`/`seed`.

/// Returns true for any xAI-hosted base URL.
fn is_xai_endpoint(base_url: &str) -> bool {
    base_url.contains("api.x.ai")
}

/// Body schema for xAI's video submit. Field order isn't load-bearing on
/// xAI (unlike Vercel/ByteDance), but we use `Serialize`-derived structs
/// throughout this module for consistency and to keep `skip_serializing_if`
/// trivial.
#[derive(Serialize)]
struct XaiVideoBody<'a> {
    model: &'a str,
    prompt: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    resolution: Option<String>,
}

fn build_xai_video_body(request: &VideoRequest) -> XaiVideoBody<'_> {
    let (aspect_ratio, resolution) = xai_size_split(request.size.as_deref());
    XaiVideoBody {
        model: request.model.as_str(),
        prompt: request.prompt.as_str(),
        duration: request.seconds,
        aspect_ratio,
        resolution,
    }
}

/// Splits the user's `-s` value into xAI's `aspect_ratio` + `resolution`
/// pair. xAI accepts these as orthogonal fields:
///   * `aspect_ratio`: `1:1` / `16:9` / `9:16` / `4:3` / `3:4` / `3:2` / `2:3`
///   * `resolution`: `480p` or `720p`
///
/// W:H form maps directly to `aspect_ratio` (resolution stays None →
/// server picks 720p default). Known WxH presets map to both. Anything
/// else returns `(None, None)` so we don't fight the server's defaults.
fn xai_size_split(size: Option<&str>) -> (Option<String>, Option<String>) {
    let Some(raw) = size.map(str::trim) else {
        return (None, None);
    };
    if raw.is_empty() {
        return (None, None);
    }
    if raw.contains(':') {
        return (Some(raw.to_string()), None);
    }
    match raw {
        "1280x720" | "1920x1080" => (Some("16:9".into()), Some("720p".into())),
        "720x1280" | "1080x1920" => (Some("9:16".into()), Some("720p".into())),
        "854x480" => (Some("16:9".into()), Some("480p".into())),
        "480x854" => (Some("9:16".into()), Some("480p".into())),
        "1024x1024" | "1080x1080" => (Some("1:1".into()), Some("720p".into())),
        _ => (None, None),
    }
}

async fn xai_submit(key: &ApiKey, request: &VideoRequest) -> Result<String> {
    let base = key.base_url.trim_end_matches('/');
    let url = if base.ends_with("/v1") {
        format!("{base}/videos/generations")
    } else {
        format!("{base}/v1/videos/generations")
    };

    let body = build_xai_video_body(request);
    let body_str = serde_json::to_string(&body).context("failed to serialize xAI video body")?;

    let client = router_http_client();
    let response = client
        .post(&url)
        .bearer_auth(key.key.as_str())
        .header("content-type", "application/json")
        .body(body_str)
        .send()
        .await
        .with_context(|| format!("xAI video submit to {url} failed"))?;

    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        let detail = extract_error_message(&text).unwrap_or_else(|| text.clone());
        bail!("video submit failed ({}): {}", status.as_u16(), detail);
    }

    let parsed: Value = response
        .json()
        .await
        .context("failed to decode xAI video submit response")?;
    // xAI's published response uses `request_id`. Some SDK versions read
    // `id` defensively; do the same in case of future changes.
    let id = parsed
        .get("request_id")
        .or_else(|| parsed.get("id"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("xAI submit response missing request_id"))?;
    Ok(id.to_string())
}

async fn xai_poll_and_download(
    key: &ApiKey,
    request_id: &str,
    path: Option<&Path>,
    pinned_extension: bool,
    poll: PollOptions,
) -> Result<VideoArtifact> {
    let base = key.base_url.trim_end_matches('/');
    let job_url = if base.ends_with("/v1") {
        format!("{base}/videos/{request_id}")
    } else {
        format!("{base}/v1/videos/{request_id}")
    };

    let client = router_http_client();
    let started = Instant::now();
    let video_url: String = loop {
        if started.elapsed() >= poll.timeout {
            bail!(
                "xAI video polling timed out after {}s (job is still running — recover with `aivo video --job-id {}`)",
                poll.timeout.as_secs(),
                request_id,
            );
        }

        let response = client
            .get(&job_url)
            .bearer_auth(key.key.as_str())
            .send()
            .await
            .with_context(|| format!("polling {job_url} failed"))?;

        let http_status = response.status();
        if !http_status.is_success() {
            let text = response.text().await.unwrap_or_default();
            let detail = extract_error_message(&text).unwrap_or_else(|| text.clone());
            bail!(
                "xAI video poll failed ({}): {}",
                http_status.as_u16(),
                detail
            );
        }

        let parsed: Value = response
            .json()
            .await
            .context("failed to decode xAI video poll response")?;
        let state = parsed
            .get("status")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        match state.as_str() {
            "done" | "succeeded" | "completed" => {
                let url = parsed
                    .get("video")
                    .and_then(|v| v.get("url"))
                    .and_then(|u| u.as_str())
                    .ok_or_else(|| anyhow!("xAI completed response missing video.url"))?;
                break url.to_string();
            }
            "failed" | "error" | "cancelled" | "canceled" => {
                let detail = parsed
                    .get("error")
                    .and_then(|e| e.get("message").or(Some(e)))
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| state.clone());
                bail!(
                    "xAI video generation {} ({}): {}",
                    state,
                    request_id,
                    detail.trim_matches('"')
                );
            }
            _ => {} // pending / processing — keep polling
        }

        sleep(poll.interval).await;
    };

    let resp = client
        .get(&video_url)
        .send()
        .await
        .with_context(|| format!("downloading xAI video from {video_url} failed"))?;
    let dl_status = resp.status();
    if !dl_status.is_success() {
        bail!(
            "xAI video download failed: HTTP {} (signed URL may have expired — \
             xAI URLs are valid for ~24 hours)",
            dl_status.as_u16()
        );
    }
    let ct = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|h| h.to_str().ok())
        .map(str::to_string);
    let bytes = resp
        .bytes()
        .await
        .context("reading xAI video body failed")?
        .to_vec();

    let Some(target_path) = path else {
        return Ok(VideoArtifact {
            path: None,
            url: Some(video_url),
            bytes: bytes.len() as u64,
            job_id: Some(request_id.to_string()),
        });
    };

    let server_ext = ct.as_deref().map(|c| ext_from_content_type(Some(c)));
    let final_path = align_extension(target_path, server_ext.as_deref(), pinned_extension);
    let written = atomic_write(&final_path, &bytes)?;
    Ok(VideoArtifact {
        path: Some(final_path),
        url: Some(video_url),
        bytes: written,
        job_id: Some(request_id.to_string()),
    })
}

/// Translate a CLI `-s` value to a Google `aspectRatio`. Same shape as
/// `image_gen` — accepts `WxH` (mapped to a known ratio) or pass-through
/// `W:H`. Returns `None` for anything we don't recognize so the server's
/// default kicks in instead of guessing.
fn aspect_ratio_for_size(size: &str) -> Option<String> {
    let raw = size.trim();
    if raw.contains(':') {
        return Some(raw.to_string());
    }
    match raw {
        "1280x720" | "1920x1080" => Some("16:9".into()),
        "720x1280" | "1080x1920" => Some("9:16".into()),
        "1024x1024" => Some("1:1".into()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ext_from_content_type_maps_video_types() {
        assert_eq!(ext_from_content_type(Some("video/mp4")), "mp4");
        assert_eq!(ext_from_content_type(Some("video/webm")), "webm");
        assert_eq!(ext_from_content_type(Some("video/quicktime")), "mov");
        assert_eq!(ext_from_content_type(Some("video/x-matroska")), "mkv");
    }

    #[test]
    fn ext_from_content_type_falls_back_to_mp4() {
        assert_eq!(ext_from_content_type(None), "mp4");
        assert_eq!(ext_from_content_type(Some("application/json")), "mp4");
    }

    #[test]
    fn ext_from_content_type_ignores_charset_suffix() {
        assert_eq!(ext_from_content_type(Some("video/mp4; charset=x")), "mp4");
    }

    #[test]
    fn aspect_ratio_for_size_maps_common_video_sizes() {
        assert_eq!(aspect_ratio_for_size("1280x720"), Some("16:9".into()));
        assert_eq!(aspect_ratio_for_size("1920x1080"), Some("16:9".into()));
        assert_eq!(aspect_ratio_for_size("720x1280"), Some("9:16".into()));
        assert_eq!(aspect_ratio_for_size("1080x1920"), Some("9:16".into()));
        assert_eq!(aspect_ratio_for_size("1024x1024"), Some("1:1".into()));
    }

    #[test]
    fn aspect_ratio_for_size_passes_through_ratio_form() {
        assert_eq!(aspect_ratio_for_size("16:9"), Some("16:9".into()));
        assert_eq!(aspect_ratio_for_size("9:16"), Some("9:16".into()));
        assert_eq!(aspect_ratio_for_size("3:4"), Some("3:4".into()));
    }

    #[test]
    fn aspect_ratio_for_size_returns_none_for_unknown() {
        assert_eq!(aspect_ratio_for_size("garbage"), None);
        assert_eq!(aspect_ratio_for_size("512x768"), None);
    }

    #[test]
    fn extract_veo_video_uri_handles_new_shape() {
        let body = json!({
            "done": true,
            "response": {
                "generateVideoResponse": {
                    "generatedSamples": [
                        {"video": {"uri": "https://example.com/veo/abc.mp4"}}
                    ]
                }
            }
        });
        assert_eq!(
            extract_veo_video_uri(&body).unwrap(),
            "https://example.com/veo/abc.mp4"
        );
    }

    #[test]
    fn extract_veo_video_uri_handles_predictions_shape() {
        let body = json!({
            "done": true,
            "response": {
                "predictions": [
                    {"videoUri": "https://example.com/veo/legacy.mp4"}
                ]
            }
        });
        assert_eq!(
            extract_veo_video_uri(&body).unwrap(),
            "https://example.com/veo/legacy.mp4"
        );
    }

    #[test]
    fn extract_veo_video_uri_handles_snake_case_predictions() {
        let body = json!({
            "done": true,
            "response": {
                "predictions": [
                    {"video_uri": "https://example.com/veo/snake.mp4"}
                ]
            }
        });
        assert_eq!(
            extract_veo_video_uri(&body).unwrap(),
            "https://example.com/veo/snake.mp4"
        );
    }

    #[test]
    fn extract_veo_video_uri_errors_on_missing_uri() {
        let body = json!({"done": true, "response": {}});
        let err = extract_veo_video_uri(&body).unwrap_err();
        assert!(err.to_string().to_lowercase().contains("uri"), "got: {err}");
    }

    #[test]
    fn poll_options_default_is_10min_5s_interval() {
        let p = PollOptions::default();
        assert_eq!(p.timeout, Duration::from_secs(600));
        assert_eq!(p.interval, Duration::from_secs(5));
    }

    #[test]
    fn is_vercel_gateway_detects_canonical_host() {
        assert!(is_vercel_gateway("https://ai-gateway.vercel.sh/v1"));
        assert!(is_vercel_gateway("https://ai-gateway.vercel.sh"));
        assert!(is_vercel_gateway("https://ai-gateway.vercel.sh/v4/ai"));
        assert!(!is_vercel_gateway("https://api.openai.com/v1"));
        assert!(!is_vercel_gateway(
            "https://generativelanguage.googleapis.com"
        ));
    }

    #[test]
    fn vercel_video_url_rebases_to_v4_ai_video_model() {
        // Whatever path the user has on file (chat at /v1, or just the
        // host), the video endpoint lives at the absolute /v4/ai path.
        assert_eq!(
            vercel_video_url("https://ai-gateway.vercel.sh/v1"),
            "https://ai-gateway.vercel.sh/v4/ai/video-model"
        );
        assert_eq!(
            vercel_video_url("https://ai-gateway.vercel.sh"),
            "https://ai-gateway.vercel.sh/v4/ai/video-model"
        );
        assert_eq!(
            vercel_video_url("https://ai-gateway.vercel.sh/v1/"),
            "https://ai-gateway.vercel.sh/v4/ai/video-model"
        );
    }

    #[test]
    fn build_vercel_video_body_routes_size_into_resolution_or_aspect_ratio() {
        // WxH → resolution, W:H → aspectRatio, neither field touched if absent.
        let req = VideoRequest {
            prompt: "x".into(),
            model: "bytedance/seedance-2.0".into(),
            size: Some("1920x1080".into()),
            seconds: Some(8),
            seed: Some(42),
        };
        let body = build_vercel_video_body(&req);
        assert_eq!(body.prompt, "x");
        assert_eq!(body.n, 1);
        assert_eq!(body.resolution.as_deref(), Some("1920x1080"));
        assert!(body.aspect_ratio.is_none());
        assert_eq!(body.duration, Some(8));
        assert_eq!(body.seed, Some(42));

        let ratio_req = VideoRequest {
            size: Some("16:9".into()),
            ..req.clone()
        };
        let body = build_vercel_video_body(&ratio_req);
        assert_eq!(body.aspect_ratio.as_deref(), Some("16:9"));
        assert!(body.resolution.is_none());

        let bare_req = VideoRequest {
            size: None,
            seconds: None,
            seed: None,
            ..req
        };
        let body = build_vercel_video_body(&bare_req);
        assert!(body.aspect_ratio.is_none());
        assert!(body.resolution.is_none());
        assert!(body.duration.is_none());
        assert!(body.seed.is_none());
    }

    #[test]
    fn vercel_video_body_serializes_in_struct_declaration_order() {
        // Pin the wire byte order: prompt, n, aspectRatio, [resolution],
        // duration, [seed]. JS object literals preserve insertion order
        // through `JSON.stringify`, and ByteDance's upstream Seedance
        // pipeline silently fails generation when keys arrive in the
        // alphabetical order that `serde_json::Value` produces. Regressing
        // this would re-introduce the original aivo-vs-ai-cli split.
        let req = VideoRequest {
            prompt: "a black cat".into(),
            model: "bytedance/seedance-2.0".into(),
            size: Some("16:9".into()),
            seconds: Some(5),
            seed: None,
        };
        let body = build_vercel_video_body(&req);
        let s = serde_json::to_string(&body).unwrap();
        assert_eq!(
            s,
            r#"{"prompt":"a black cat","n":1,"aspectRatio":"16:9","duration":5}"#
        );
    }

    #[test]
    fn vercel_video_body_omits_optionals_when_unset() {
        let req = VideoRequest {
            prompt: "x".into(),
            model: "bytedance/seedance-2.0".into(),
            size: None,
            seconds: None,
            seed: None,
        };
        let body = build_vercel_video_body(&req);
        let s = serde_json::to_string(&body).unwrap();
        assert_eq!(s, r#"{"prompt":"x","n":1}"#);
    }

    #[test]
    fn parse_vercel_sse_event_reads_single_data_line() {
        let body = "data: {\"type\":\"result\",\"videos\":[]}\n\n";
        let v = parse_vercel_sse_event(body).unwrap();
        assert_eq!(v["type"], json!("result"));
    }

    #[test]
    fn parse_vercel_sse_event_handles_no_space_after_data_colon() {
        // Some SSE producers emit `data:{...}` without the conventional
        // space; the spec allows it.
        let body = "data:{\"type\":\"result\",\"videos\":[]}\n\n";
        let v = parse_vercel_sse_event(body).unwrap();
        assert_eq!(v["type"], json!("result"));
    }

    #[test]
    fn parse_vercel_sse_event_concatenates_multi_line_data() {
        // SSE spec: multiple `data:` lines in the same event get joined
        // by `\n` before parsing.
        let body = "data: {\"type\":\"result\",\n\
                    data: \"videos\":[]}\n\n";
        let v = parse_vercel_sse_event(body).unwrap();
        assert_eq!(v["type"], json!("result"));
    }

    #[test]
    fn parse_vercel_sse_event_errors_on_empty_body() {
        let err = parse_vercel_sse_event("\n\n").unwrap_err();
        assert!(err.to_string().contains("no data event"), "got: {err}");
    }

    #[tokio::test]
    async fn attach_rejects_vercel_with_clear_message() {
        // No job IDs on Vercel — `--job-id` should error with a hint
        // rather than blindly trying to GET /v1/videos/<id>.
        let key = ApiKey::new_with_protocol(
            "test".into(),
            "vercel".into(),
            "https://ai-gateway.vercel.sh/v1".into(),
            None,
            "fake".into(),
        );
        let err = attach(&key, "anything", None, false, PollOptions::default())
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Vercel"), "got: {msg}");
        assert!(msg.contains("--job-id"), "got: {msg}");
    }

    #[test]
    fn is_xai_endpoint_detects_canonical_host() {
        assert!(is_xai_endpoint("https://api.x.ai/v1"));
        assert!(is_xai_endpoint("https://api.x.ai"));
        assert!(!is_xai_endpoint("https://api.openai.com/v1"));
        assert!(!is_xai_endpoint("https://ai-gateway.vercel.sh/v1"));
    }

    #[test]
    fn xai_size_split_routes_ratio_form_to_aspect_ratio_only() {
        // W:H input → aspect_ratio populated, resolution left None so xAI
        // picks its own default (720p).
        assert_eq!(xai_size_split(Some("16:9")), (Some("16:9".into()), None));
        assert_eq!(xai_size_split(Some("9:16")), (Some("9:16".into()), None));
        assert_eq!(xai_size_split(Some("1:1")), (Some("1:1".into()), None));
    }

    #[test]
    fn xai_size_split_maps_known_wxh_presets_to_both_fields() {
        // 720p presets — width:height → 16:9 / 9:16, height = 720 → "720p".
        assert_eq!(
            xai_size_split(Some("1280x720")),
            (Some("16:9".into()), Some("720p".into()))
        );
        assert_eq!(
            xai_size_split(Some("720x1280")),
            (Some("9:16".into()), Some("720p".into()))
        );
        // 480p presets.
        assert_eq!(
            xai_size_split(Some("854x480")),
            (Some("16:9".into()), Some("480p".into()))
        );
        assert_eq!(
            xai_size_split(Some("480x854")),
            (Some("9:16".into()), Some("480p".into()))
        );
        // Square.
        assert_eq!(
            xai_size_split(Some("1024x1024")),
            (Some("1:1".into()), Some("720p".into()))
        );
    }

    #[test]
    fn xai_size_split_returns_none_for_unknown_or_empty() {
        // Unknown WxH falls back to (None, None) so the server's defaults
        // kick in instead of us guessing wrong.
        assert_eq!(xai_size_split(Some("512x768")), (None, None));
        assert_eq!(xai_size_split(Some("garbage")), (None, None));
        assert_eq!(xai_size_split(Some("   ")), (None, None));
        assert_eq!(xai_size_split(None), (None, None));
    }

    #[test]
    fn build_xai_video_body_carries_model_prompt_and_split_size() {
        let req = VideoRequest {
            prompt: "a cat".into(),
            model: "grok-imagine-video".into(),
            size: Some("1280x720".into()),
            seconds: Some(5),
            seed: Some(42), // intentionally ignored — xAI doesn't accept seed
        };
        let body = build_xai_video_body(&req);
        assert_eq!(body.model, "grok-imagine-video");
        assert_eq!(body.prompt, "a cat");
        assert_eq!(body.duration, Some(5));
        assert_eq!(body.aspect_ratio.as_deref(), Some("16:9"));
        assert_eq!(body.resolution.as_deref(), Some("720p"));

        // Verify wire shape: `seed` and `n` must NOT appear (xAI rejects them).
        let s = serde_json::to_string(&body).unwrap();
        assert!(!s.contains("\"seed\""), "got: {s}");
        assert!(!s.contains("\"n\":"), "got: {s}");
    }

    #[test]
    fn build_xai_video_body_omits_optionals_when_unset() {
        let req = VideoRequest {
            prompt: "x".into(),
            model: "grok-imagine-video".into(),
            size: None,
            seconds: None,
            seed: None,
        };
        let body = build_xai_video_body(&req);
        let s = serde_json::to_string(&body).unwrap();
        // Only `model` and `prompt` should appear; everything else is
        // skipped so xAI applies its defaults (8s, 720p, 16:9).
        assert_eq!(s, r#"{"model":"grok-imagine-video","prompt":"x"}"#);
    }

    #[tokio::test]
    async fn anthropic_protocol_bails_with_clear_message() {
        let key = ApiKey::new_with_protocol(
            "test".into(),
            "test".into(),
            "https://api.anthropic.com/v1".into(),
            None,
            "fake".into(),
        );
        let request = VideoRequest {
            prompt: "x".into(),
            model: "sora-2".into(),
            size: None,
            seconds: None,
            seed: None,
        };
        let err = generate(&key, &request, None, false, PollOptions::default())
            .await
            .unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("anthropic"),
            "got: {err}"
        );
    }
}
