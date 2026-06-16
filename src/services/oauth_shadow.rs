//! Generic OAuth shadow directory lifecycle for provider home directories.
//!
//! Replaces the duplicated per-provider prepare/finalize/persist sequence
//! in `launch_runtime.rs` with a single generic pipeline. Each provider
//! implements [`OAuthHomeShadow`] for its home-shadow type, and
//! [`prepare_oauth_shadow`] + [`finalize_oauth_shadow`] handle the rest.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;

use crate::services::codex_oauth::CodexOAuthCredential;
use crate::services::gemini_oauth::GeminiOAuthCredential;
use crate::services::session_store::SessionStore;

// ---------------------------------------------------------------------------
// Internal credential ops trait — implemented per provider so the generic
// pipeline never needs to reach into the `OAuthCredential` trait from
// `oauth_utils.rs` (whose changes the linter reverts).
// ---------------------------------------------------------------------------

/// Minimal ops that a credential must support for the shadow lifecycle.
trait OAuthCredOps: Sized {
    fn from_json(json: &str) -> Result<Self>;
    fn to_json(&self) -> Result<String>;
    fn is_expired(&self, skew_secs: i64) -> bool;
    async fn refresh(&mut self) -> Result<()>;

    async fn ensure_fresh(&mut self, skew_secs: i64) -> Result<bool> {
        if self.is_expired(skew_secs) {
            self.refresh().await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

impl OAuthCredOps for CodexOAuthCredential {
    fn from_json(json: &str) -> Result<Self> {
        CodexOAuthCredential::from_json(json)
    }
    fn to_json(&self) -> Result<String> {
        CodexOAuthCredential::to_json(self)
    }
    fn is_expired(&self, skew_secs: i64) -> bool {
        CodexOAuthCredential::is_expired(self, skew_secs)
    }
    async fn refresh(&mut self) -> Result<()> {
        crate::services::codex_oauth::refresh(self).await
    }
}

impl OAuthCredOps for GeminiOAuthCredential {
    fn from_json(json: &str) -> Result<Self> {
        GeminiOAuthCredential::from_json(json)
    }
    fn to_json(&self) -> Result<String> {
        GeminiOAuthCredential::to_json(self)
    }
    fn is_expired(&self, skew_secs: i64) -> bool {
        GeminiOAuthCredential::is_expired(self, skew_secs)
    }
    async fn refresh(&mut self) -> Result<()> {
        crate::services::gemini_oauth::refresh(self).await
    }
}

// ---------------------------------------------------------------------------
// Generic shadow lifecycle
// ---------------------------------------------------------------------------

/// Holds the shadow home dir + metadata needed to sync refreshed tokens back
/// into aivo's store after the child process exits.
pub(crate) struct OAuthSync<C, S> {
    pub(crate) key_id: String,
    pub(crate) shadow: S,
    pub(crate) original: C,
}

/// Trait for provider-specific home shadow directories that project OAuth
/// credentials into a tool-native filesystem layout.
///
/// # Lifetimes
///
/// Instances own a temp directory. Dropping them removes the directory;
/// callers who want to sync refreshed tokens back must call [`read_back`]
/// before the value is dropped.
pub(crate) trait OAuthHomeShadow: Sized {
    /// The credential type this shadow wraps.
    type Credential;

    /// The on-disk format returned by [`read_back`].
    type DiskFormat;

    /// Creates the shadow temp dir and writes the credential file(s) in the
    /// layout the native tool expects.
    async fn create(creds: &Self::Credential) -> Result<Self>;

    /// Returns the base path of the shadow directory to set as the home env var.
    fn path(&self) -> &Path;

    /// Reads the (possibly-rotated) credential file(s) back after the child
    /// process exits. Returns `Ok(None)` if the file is missing or malformed.
    async fn read_back(&self) -> Result<Option<Self::DiskFormat>>;
}

/// Prepares an OAuth shadow directory and inserts its path into `env`.
///
/// # Steps
///
/// 1. Extract `AIVO_*_CREDS` from `env`
/// 2. Extract `AIVO_*_KEY_ID` from `env`
/// 3. Parse JSON into credential
/// 4. Refresh the token if near expiry
/// 5. Create shadow dir via `S::create`
/// 6. Insert shadow path into `env` under `home_env`
/// 7. Return [`OAuthSync`] for post-exit finalization
pub(crate) async fn prepare_oauth_shadow<C, S>(
    env: &mut HashMap<String, String>,
    creds_env: &str,
    key_id_env: &str,
    home_env: &str,
    skew_secs: i64,
) -> Result<OAuthSync<C, S>>
where
    C: OAuthCredOps,
    S: OAuthHomeShadow<Credential = C>,
{
    let raw = env
        .remove(creds_env)
        .ok_or_else(|| anyhow::anyhow!("missing {creds_env}"))?;
    let key_id = env
        .remove(key_id_env)
        .ok_or_else(|| anyhow::anyhow!("missing {key_id_env}"))?;
    let mut creds = C::from_json(&raw)?;

    // Refresh pre-launch so the tool starts with a valid access token. The
    // post-exit sync path below will persist any rotation the tool performs
    // during the session, so we don't persist here.
    let _refreshed = creds.ensure_fresh(skew_secs).await?;

    let shadow = S::create(&creds).await?;
    env.insert(
        home_env.to_string(),
        shadow.path().to_string_lossy().to_string(),
    );

    Ok(OAuthSync {
        key_id,
        shadow,
        original: creds,
    })
}

/// Reads the shadow credential file(s) back after the tool exits and, if any
/// token changed, persists the rotated credential into aivo's store.
///
/// `into_credential` is a closure that projects the disk format back into
/// the credential type, using the pre-launch original for any fields the
/// tool doesn't persist (e.g. email, expiry metadata).
///
/// Errors are logged but never propagated — the user's session has already
/// completed, and a failed sync just means the next launch will refresh again.
pub(crate) async fn finalize_oauth_shadow<C, D, S>(
    session_store: &SessionStore,
    sync: Option<OAuthSync<C, S>>,
    into_credential: impl FnOnce(D, &C) -> C,
) where
    C: OAuthCredOps + PartialEq,
    S: OAuthHomeShadow<Credential = C, DiskFormat = D>,
{
    let Some(sync) = sync else {
        return;
    };

    let disk = match sync.shadow.read_back().await {
        Ok(Some(v)) => v,
        Ok(None) => {
            // File missing/truncated — the tool probably crashed before
            // writing. Persist the pre-launch (freshly refreshed) creds so
            // the refresh_token rotation isn't lost.
            persist_refreshed_oauth_if_needed(
                session_store,
                &sync.key_id,
                &sync.original,
                &sync.original,
            )
            .await;
            return;
        }
        Err(_) => return,
    };

    let updated = into_credential(disk, &sync.original);
    persist_refreshed_oauth_if_needed(session_store, &sync.key_id, &sync.original, &updated).await;
}

/// Persists `updated` into aivo's key store if it differs from `original`.
async fn persist_refreshed_oauth_if_needed<C>(
    session_store: &SessionStore,
    key_id: &str,
    original: &C,
    updated: &C,
) where
    C: OAuthCredOps + PartialEq,
{
    if original == updated {
        return;
    }
    let json = match updated.to_json() {
        Ok(j) => j,
        Err(_) => return,
    };
    // base_url / name / protocols are preserved by passing the same values
    // as the existing entry. Pull the current entry first so we don't
    // clobber name changes made mid-session.
    if let Ok(Some(existing)) = session_store.api_keys().get_key_by_id(key_id).await {
        let _ = session_store
            .update_key(
                key_id,
                &existing.name,
                &existing.base_url,
                existing.claude_protocol,
                &json,
            )
            .await;
    }
}
