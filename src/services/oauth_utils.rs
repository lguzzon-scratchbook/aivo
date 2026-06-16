//! Shared OAuth utilities used across provider OAuth implementations.
//!
//! Provides:
//! - `PkcePair`: PKCE code verifier/challenge pair.
//! - `generate_state()`: random state string for CSRF protection.
//! - `redact_oauth_body()`: sanitize OAuth response bodies for logging.
//! - `OAuthCredential` trait: common interface for persisted credential types.

use anyhow::{Context, Result};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use rand::RngCore;
use reqwest::Response;
use serde::Serialize;
use serde::de::DeserializeOwned;
use sha2::{Digest, Sha256};

/// PKCE pair for a single authorize flow. `verifier` is never logged or
/// serialized — it lives only in memory for the duration of the flow.
pub struct PkcePair {
    pub verifier: String,
    pub challenge: String,
}

impl PkcePair {
    pub fn generate() -> Self {
        // 32 random bytes → 43 URL-safe base64 chars (no padding). RFC 7636
        // requires 43-128 chars of [A-Z a-z 0-9 -._~]; URL_SAFE_NO_PAD uses
        // the "-._~" alphabet subset, which satisfies the spec.
        let mut buf = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut buf);
        let verifier = URL_SAFE_NO_PAD.encode(buf);
        let digest = Sha256::digest(verifier.as_bytes());
        let challenge = URL_SAFE_NO_PAD.encode(digest);
        Self {
            verifier,
            challenge,
        }
    }
}

/// 32-hex-char state (16 random bytes). Matches codex-multi-auth.
pub fn generate_state() -> String {
    let mut buf = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut buf);
    buf.iter().fold(String::with_capacity(32), |mut acc, b| {
        use std::fmt::Write;
        let _ = write!(acc, "{:02x}", b);
        acc
    })
}

/// Redacts common OAuth secrets before logging an upstream response body.
pub fn redact_oauth_body(body: &str) -> String {
    // Cheap but effective: mask the values after known token keys. Advances
    // past each replacement so we never re-match the same occurrence.
    let mut out = body.to_string();
    for key in [
        "access_token",
        "refresh_token",
        "id_token",
        "code",
        "code_verifier",
    ] {
        let needle = format!("\"{}\"", key);
        let mut cursor = 0usize;
        while let Some(rel_idx) = out[cursor..].find(&needle) {
            let idx = cursor + rel_idx;
            let after_key = idx + needle.len();
            let rest = &out[after_key..];
            let Some(colon) = rest.find(':') else { break };
            let Some(open) = rest[colon..].find('"') else {
                cursor = after_key;
                continue;
            };
            let Some(close_rel) = rest[colon + open + 1..].find('"') else {
                cursor = after_key;
                continue;
            };
            let start = after_key + colon + open + 1;
            let end = start + close_rel;
            out.replace_range(start..end, "<redacted>");
            // Skip past the replacement so we don't rescan the same key.
            cursor = start + "<redacted>".len();
        }
    }
    out
}

/// Checks an OAuth HTTP response for success. On failure, redacts secrets
/// and returns an error with the context description.
///
/// Replaces the 4 inline copies of this check across `codex_oauth.rs` and
/// `gemini_oauth.rs`.
pub async fn oauth_http_check(response: Response, context: &str) -> Result<Response> {
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!(
            "{} failed ({}): {}",
            context,
            status.as_u16(),
            redact_oauth_body(&body)
        );
    }
    Ok(response)
}

/// Common interface for persisted OAuth credential types.
///
/// Implementors must be `Serialize + DeserializeOwned` (satisfied by
/// `#[derive(Serialize, Deserialize)]`). Default implementations serialize
/// to / from JSON, which is then encrypted through the `ApiKey.key` slot.
///
/// # Expiry / refresh
///
/// `is_expired` and `refresh` are per-provider; `ensure_fresh` provides a
/// default that calls both in the common expiry-check order.
pub trait OAuthCredential: Serialize + DeserializeOwned {
    fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).context("serialize OAuth credential")
    }

    fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).context("parse OAuth credential JSON")
    }

    /// Returns true if the access token is expired (with skew).
    fn is_expired(&self, skew_secs: i64) -> bool;

    /// Refreshes the access token in place.
    async fn refresh(&mut self) -> Result<()>;

    /// Ensures the token is fresh, refreshing if near expiry.
    /// Returns true if a refresh actually happened.
    async fn ensure_fresh(&mut self, skew_secs: i64) -> Result<bool> {
        if self.is_expired(skew_secs) {
            self.refresh().await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}
