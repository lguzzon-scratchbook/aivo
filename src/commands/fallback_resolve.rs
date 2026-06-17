/// Shared fallback resolution for CLI commands.
///
/// Provides a single function that resolves a fallback alias to an ordered
/// list of (key, target) pairs by matching provider names to stored API keys.
/// Used by run.rs, chat.rs, and start.rs to avoid duplicating the key-matching
/// iteration logic.
use std::time::Instant;

use anyhow::{Context, Result};

use crate::services::fallback::ProviderModelPair;
use crate::services::session_store::{ApiKey, SessionStore};

pub fn check_timeout(start: Instant, timeout_ms: Option<u64>) -> anyhow::Result<()> {
    if let Some(timeout) = timeout_ms {
        let elapsed = start.elapsed().as_millis() as u64;
        if elapsed >= timeout {
            anyhow::bail!("Fallback timeout exceeded ({}ms)", timeout);
        }
    }
    Ok(())
}

/// Resolve a fallback alias to an ordered list of (key, target) pairs.
///
/// For each target (provider:model) in the fallback definition, finds a key
/// matching the provider name (by key name, short_id, or id). Falls back to
/// `original_key` when no key matches by name.
///
/// Returns `Err` with "Unknown fallback alias" when the fallback_id doesn't
/// exist, or "All fallback targets exhausted" when no target has a matching key.
pub async fn resolve_fallback_targets(
    session_store: &SessionStore,
    fallback_id: &str,
    original_key: Option<&ApiKey>,
) -> Result<(Vec<(ApiKey, ProviderModelPair)>, Option<u64>)> {
    let targets = session_store
        .get_fallback_targets(fallback_id)
        .await
        .with_context(|| format!("Unknown fallback alias '{}'", fallback_id))?;

    let all_keys = session_store.get_keys().await?;
    let mut resolved: Vec<(ApiKey, ProviderModelPair)> = Vec::new();

    for target in targets.iter() {
        let key = all_keys
            .iter()
            .find(|k| {
                k.name == target.provider
                    || k.short_id() == target.provider
                    || k.id == target.provider
            })
            .cloned()
            .or_else(|| original_key.cloned());

        if let Some(key) = key {
            resolved.push((key, target.clone()));
        }
    }

    if resolved.is_empty() {
        anyhow::bail!("All fallback targets for '{}' exhausted", fallback_id);
    }

    let fallbacks = session_store.get_fallbacks().await?;
    let timeout_ms = fallbacks.get(fallback_id).and_then(|f| f.timeout_ms);

    Ok((resolved, timeout_ms))
}
