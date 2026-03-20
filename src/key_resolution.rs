use std::io::{self, IsTerminal};
use std::process;

use crate::commands;
use crate::errors::ExitCode;
use crate::services::session_store::{ApiKey, SessionStore};
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
    session_store: &SessionStore,
    key_flag: Option<&str>,
    mode: KeyLookupMode,
) -> anyhow::Result<KeyResolution> {
    match key_flag {
        Some("") => prompt_temporary_key_override(session_store).await,
        Some(key_id_or_name) => Ok(KeyResolution::Selected(
            session_store
                .resolve_key_by_id_or_name(key_id_or_name)
                .await?,
        )),
        None => match mode {
            KeyLookupMode::RequireActiveOrPrompt => {
                match resolve_active_key_or_prompt(session_store).await {
                    Some(key) => Ok(KeyResolution::Selected(key)),
                    None => Ok(KeyResolution::MissingAuth),
                }
            }
            KeyLookupMode::PreferActiveAllowNone => match session_store.get_active_key().await? {
                Some(key) => Ok(KeyResolution::Selected(key)),
                None => Ok(KeyResolution::MissingAuth),
            },
        },
    }
}

async fn prompt_temporary_key_override(
    session_store: &SessionStore,
) -> anyhow::Result<KeyResolution> {
    let all_keys = session_store.get_keys().await?;
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

    let default_idx = session_store
        .get_active_key_info()
        .await?
        .and_then(|active_key| all_keys.iter().position(|key| key.id == active_key.id))
        .unwrap_or(0);

    match commands::keys::prompt_pick_key_without_activation(
        &all_keys,
        "Select a key",
        default_idx,
    )? {
        Some(key) => Ok(KeyResolution::Selected(key)),
        None => Ok(KeyResolution::Cancelled),
    }
}

async fn resolve_active_key_or_prompt(session_store: &SessionStore) -> Option<ApiKey> {
    if let Ok(Some(key)) = session_store.get_active_key().await {
        return Some(key);
    }

    let all_keys = match session_store.get_keys().await {
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

    match commands::keys::prompt_select_key(session_store, &all_keys, "Select a key", 0).await {
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
    use super::{KeyLookupMode, KeyResolution, resolve_key_override};
    use crate::services::session_store::SessionStore;
    use tempfile::TempDir;

    fn temp_store() -> (TempDir, SessionStore) {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        (temp_dir, SessionStore::with_path(config_path))
    }

    #[tokio::test]
    async fn prefer_active_allow_none_returns_active_key() {
        let (_temp_dir, store) = temp_store();
        let id = store
            .add_key_with_protocol(
                "openrouter",
                "https://openrouter.ai/api/v1",
                None,
                "sk-test",
            )
            .await
            .unwrap();
        store.set_active_key(&id).await.unwrap();

        let resolved =
            resolve_key_override(&store, None, KeyLookupMode::PreferActiveAllowNone).await;

        match resolved.unwrap() {
            KeyResolution::Selected(key) => assert_eq!(key.id, id),
            _ => panic!("expected selected key"),
        }
    }

    #[tokio::test]
    async fn prefer_active_allow_none_returns_missing_auth_without_active_key() {
        let (_temp_dir, store) = temp_store();

        let resolved =
            resolve_key_override(&store, None, KeyLookupMode::PreferActiveAllowNone).await;

        assert!(matches!(resolved.unwrap(), KeyResolution::MissingAuth));
    }
}
