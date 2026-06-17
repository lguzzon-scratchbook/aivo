use std::io::{self, IsTerminal};
use std::process;

use crate::commands;
use crate::errors::ExitCode;
use crate::services::api_key_store::ApiKeyStore;
use crate::services::key_compat::KeyCompatContext;
use crate::services::last_selection::{LastSelectionStore, SelectionScope};
use crate::services::session_store::{ApiKey, LastSelection};
use crate::style;

#[allow(clippy::large_enum_variant)]
pub(crate) enum KeyResolution {
    Selected(ApiKey),
    Cancelled,
    MissingAuth,
}

pub(crate) enum KeyLookupMode {
    RequireActiveOrPrompt,
    PreferActiveAllowNone,
}

/// Which last-used record drives the key picker default. `Image` is isolated
/// from the chat/run record so an image session can't pre-select an
/// image-only key for the next chat session and vice-versa.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LastSelectionView {
    Default,
    Image,
}

impl LastSelectionView {
    /// Reads the last-used record for this view, falling back to the default
    /// scope so a fresh image session can still inherit a chat-resolved key
    /// (the picker downstream can swap if it isn't compatible).
    async fn read(&self, last_sel: &LastSelectionStore) -> Option<LastSelection> {
        match self {
            LastSelectionView::Default => {
                last_sel.get(SelectionScope::Default).await.ok().flatten()
            }
            LastSelectionView::Image => match last_sel.get(SelectionScope::Image).await {
                Ok(Some(sel)) => Some(sel),
                _ => last_sel.get(SelectionScope::Default).await.ok().flatten(),
            },
        }
    }
}

pub(crate) fn key_or_exit(result: anyhow::Result<KeyResolution>) -> Option<ApiKey> {
    match result {
        Ok(KeyResolution::Selected(key)) => Some(key),
        Ok(KeyResolution::Cancelled) => process::exit(ExitCode::Success.code()),
        Ok(KeyResolution::MissingAuth) => process::exit(ExitCode::AuthError.code()),
        Err(e) => {
            eprintln!("{} {}", style::red("Error:"), e);
            process::exit(ExitCode::UserError.code());
        }
    }
}

pub(crate) async fn resolve_key_override(
    api_keys: &ApiKeyStore,
    last_sel: &LastSelectionStore,
    key_flag: Option<&str>,
    mode: KeyLookupMode,
    compat: KeyCompatContext,
) -> anyhow::Result<KeyResolution> {
    resolve_key_override_scoped(
        api_keys,
        last_sel,
        key_flag,
        mode,
        compat,
        LastSelectionView::Default,
    )
    .await
}

/// Image-aware variant: prefers `last_image_selection` for picker defaults
/// before falling back to the shared `last_selection`. Used by `aivo image`.
pub(crate) async fn resolve_image_key_override(
    api_keys: &ApiKeyStore,
    last_sel: &LastSelectionStore,
    key_flag: Option<&str>,
    mode: KeyLookupMode,
    compat: KeyCompatContext,
) -> anyhow::Result<KeyResolution> {
    resolve_key_override_scoped(
        api_keys,
        last_sel,
        key_flag,
        mode,
        compat,
        LastSelectionView::Image,
    )
    .await
}

async fn resolve_key_override_scoped(
    api_keys: &ApiKeyStore,
    last_sel: &LastSelectionStore,
    key_flag: Option<&str>,
    mode: KeyLookupMode,
    compat: KeyCompatContext,
    view: LastSelectionView,
) -> anyhow::Result<KeyResolution> {
    match key_flag {
        Some("") => prompt_temporary_key_override(api_keys, last_sel, compat, view).await,
        Some(key_id_or_name) => resolve_by_id_or_name_or_pick(api_keys, key_id_or_name).await,
        None => match mode {
            KeyLookupMode::RequireActiveOrPrompt => {
                match resolve_active_key_or_prompt(api_keys, last_sel, compat, view).await {
                    Some(key) => Ok(KeyResolution::Selected(key)),
                    None => Ok(KeyResolution::MissingAuth),
                }
            }
            KeyLookupMode::PreferActiveAllowNone => {
                // Try last-used selection first (scoped to the view).
                if let Some(last_sel_rec) = view.read(last_sel).await
                    && let Ok(Some(key)) = api_keys.get_key_by_id(&last_sel_rec.key_id).await
                {
                    return Ok(KeyResolution::Selected(key));
                }
                match api_keys.get_active_key().await? {
                    Some(key) => Ok(KeyResolution::Selected(key)),
                    None => Ok(KeyResolution::MissingAuth),
                }
            }
        },
    }
}

/// Resolves `key_id_or_name` to a single key. Shows a picker on ambiguous
/// name matches when a terminal is available; falls back to the low-level
/// error otherwise so scripts/CI still get a clear failure.
pub(crate) async fn resolve_by_id_or_name_or_pick(
    api_keys: &ApiKeyStore,
    key_id_or_name: &str,
) -> anyhow::Result<KeyResolution> {
    let matches = api_keys.find_keys_by_id_or_name(key_id_or_name).await?;
    match matches.len() {
        0 => {
            // Delegate to the existing error path for consistent messaging.
            Err(api_keys
                .resolve_key_by_id_or_name(key_id_or_name)
                .await
                .expect_err("empty matches must produce a not-found error"))
        }
        1 => Ok(KeyResolution::Selected(matches.into_iter().next().unwrap())),
        _ => {
            if !io::stderr().is_terminal() {
                return Err(api_keys
                    .resolve_key_by_id_or_name(key_id_or_name)
                    .await
                    .expect_err("ambiguous matches must produce an error"));
            }
            eprintln!(
                "{} Multiple keys match {}:",
                style::yellow("Note:"),
                style::cyan(key_id_or_name)
            );
            let prompt = format!("Select key '{}'", key_id_or_name);
            match commands::keys::prompt_pick_key_without_activation(&matches, &[], &prompt, 0)? {
                Some(key) => Ok(KeyResolution::Selected(key)),
                None => Ok(KeyResolution::Cancelled),
            }
        }
    }
}

async fn prompt_temporary_key_override(
    api_keys: &ApiKeyStore,
    last_sel: &LastSelectionStore,
    compat: KeyCompatContext,
    view: LastSelectionView,
) -> anyhow::Result<KeyResolution> {
    let all_keys = api_keys.get_keys().await?;
    if all_keys.is_empty() {
        eprintln!("{} No API keys configured.", style::yellow("Note:"));
        eprintln!();
        eprintln!("  Run {} to add one.", style::cyan("aivo keys add"));
        return Ok(KeyResolution::MissingAuth);
    }
    if !io::stderr().is_terminal() {
        anyhow::bail!(
            "Cannot open key picker without a terminal. Run in a terminal or pass --key <id|name>."
        );
    }

    let last_sel_key_id = view.read(last_sel).await.map(|s| s.key_id);
    let active_key_id = api_keys
        .get_active_key_info()
        .await
        .ok()
        .flatten()
        .map(|k| k.id);
    let default_idx = last_sel_key_id
        .as_ref()
        .and_then(|id| all_keys.iter().position(|key| &key.id == id))
        .or_else(|| {
            active_key_id
                .as_ref()
                .and_then(|id| all_keys.iter().position(|key| &key.id == id))
        })
        .unwrap_or(0);

    let annotations = compat.annotations_for(&all_keys);
    match commands::keys::prompt_pick_key_without_activation(
        &all_keys,
        &annotations,
        "Select a key",
        default_idx,
    )? {
        Some(key) => Ok(KeyResolution::Selected(key)),
        None => Ok(KeyResolution::Cancelled),
    }
}

async fn resolve_active_key_or_prompt(
    api_keys: &ApiKeyStore,
    last_sel: &LastSelectionStore,
    compat: KeyCompatContext,
    view: LastSelectionView,
) -> Option<ApiKey> {
    // Try last-used selection first (scoped to the view).
    if let Some(last_sel_rec) = view.read(last_sel).await
        && let Ok(Some(key)) = api_keys.get_key_by_id(&last_sel_rec.key_id).await
    {
        return Some(key);
    }
    // Then active key
    if let Ok(Some(key)) = api_keys.get_active_key().await {
        return Some(key);
    }

    let all_keys = match api_keys.get_keys().await {
        Ok(keys) => keys,
        Err(e) => {
            eprintln!("{} {}", style::red("Error:"), e);
            return None;
        }
    };

    if all_keys.is_empty() {
        eprintln!("{} No API keys configured.", style::yellow("Note:"));
        eprintln!();
        eprintln!("  Run {} to add one.", style::cyan("aivo keys add"));
        return None;
    }

    eprintln!(
        "{} No active API key. Select one to continue:",
        style::yellow("Note:")
    );
    eprintln!();

    if !io::stderr().is_terminal() {
        eprintln!(
            "{} Cannot open key picker without a terminal. Run in a terminal or activate a key first.",
            style::red("Error:")
        );
        return None;
    }

    let annotations = compat.annotations_for(&all_keys);
    match commands::keys::prompt_select_key(api_keys, &all_keys, &annotations, "Select a key", 0)
        .await
    {
        Ok(Some(key)) => {
            eprintln!();
            Some(key)
        }
        Ok(None) => {
            eprintln!("{}", style::dim("Cancelled."));
            None
        }
        Err(e) => {
            eprintln!("{} {}", style::red("Error:"), e);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{KeyCompatContext, KeyLookupMode, KeyResolution, resolve_key_override};
    use crate::services::api_key_store::ApiKeyStore;
    use crate::services::last_selection::LastSelectionStore;
    use crate::services::session_store::{ConfigContext, SessionStore};
    use tempfile::TempDir;

    fn setup() -> (ApiKeyStore, LastSelectionStore, SessionStore, TempDir) {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("config.json");
        let config_dir = dir.path().to_path_buf();
        let ctx = ConfigContext {
            config_path,
            config_dir,
        };
        let api_keys = ApiKeyStore { ctx: ctx.clone() };
        let last_sel = LastSelectionStore { ctx };
        let store = SessionStore::with_path(dir.path().join("config.json"));
        (api_keys, last_sel, store, dir)
    }

    #[tokio::test]
    async fn prefer_active_allow_none_returns_active_key() {
        let (api_keys, last_sel, _store, _dir) = setup();
        let id = api_keys
            .add_key_with_protocol(
                "openrouter",
                "https://openrouter.ai/api/v1",
                None,
                "sk-test",
            )
            .await
            .unwrap();
        api_keys.set_active_key(&id).await.unwrap();

        let resolved = resolve_key_override(
            &api_keys,
            &last_sel,
            None,
            KeyLookupMode::PreferActiveAllowNone,
            KeyCompatContext::None,
        )
        .await;

        match resolved.unwrap() {
            KeyResolution::Selected(key) => assert_eq!(key.id, id),
            _ => panic!("expected selected key"),
        }
    }

    #[tokio::test]
    async fn prefer_active_allow_none_returns_missing_auth_without_keys() {
        let (api_keys, last_sel, _store, _dir) = setup();

        let resolved = resolve_key_override(
            &api_keys,
            &last_sel,
            None,
            KeyLookupMode::PreferActiveAllowNone,
            KeyCompatContext::None,
        )
        .await;

        assert!(matches!(resolved.unwrap(), KeyResolution::MissingAuth));
    }

    #[tokio::test]
    async fn prefer_active_allow_none_returns_starter_after_ensure() {
        let (api_keys, last_sel, store, _dir) = setup();

        let (starter, _) = store.ensure_starter_key().await.unwrap();
        api_keys.set_active_key(&starter.id).await.unwrap();

        let resolved = resolve_key_override(
            &api_keys,
            &last_sel,
            None,
            KeyLookupMode::PreferActiveAllowNone,
            KeyCompatContext::None,
        )
        .await;

        match resolved.unwrap() {
            KeyResolution::Selected(key) => {
                assert_eq!(key.name, crate::constants::AIVO_STARTER_KEY_NAME);
                assert_eq!(key.base_url, crate::constants::AIVO_STARTER_SENTINEL);
            }
            _ => panic!("expected starter key"),
        }
    }

    #[tokio::test]
    async fn ensure_starter_key_creates_and_is_idempotent() {
        let (api_keys, _last_sel, store, _dir) = setup();

        let (key, is_new) = store
            .ensure_starter_key()
            .await
            .expect("should create starter key");
        assert_eq!(key.name, crate::constants::AIVO_STARTER_KEY_NAME);
        assert_eq!(key.base_url, crate::constants::AIVO_STARTER_SENTINEL);
        assert!(is_new);

        // Verify chat model was pre-set
        let model = api_keys.get_chat_model(&key.id).await.unwrap();
        assert_eq!(
            model,
            Some(crate::constants::AIVO_STARTER_MODEL.to_string())
        );

        // Calling again returns the same key (idempotent)
        let (key2, is_new2) = store
            .ensure_starter_key()
            .await
            .expect("should return existing starter key");
        assert_eq!(key.id, key2.id);
        assert!(!is_new2);

        // Only one key exists
        let all_keys = api_keys.get_keys().await.unwrap();
        assert_eq!(all_keys.len(), 1);
    }
}
