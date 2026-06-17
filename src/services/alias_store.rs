use anyhow::Result;
use std::collections::HashMap;

use crate::services::session_store::ConfigContext;

/// Standalone store for model alias operations.
///
/// Aliases map short names (e.g. "fast") to full model names
/// (e.g. "claude-haiku-4-5"). Reads and writes the `aliases`
/// field of StoredConfig.
#[derive(Debug, Clone)]
pub struct AliasStore {
    pub(crate) ctx: ConfigContext,
}

impl AliasStore {
    /// Returns all model aliases.
    pub async fn get_aliases(&self) -> Result<HashMap<String, String>> {
        let config = self.ctx.load().await?;
        Ok(config.aliases)
    }

    /// Sets a model alias. Returns the previous value if it existed.
    pub async fn set_alias(&self, name: String, model: String) -> Result<Option<String>> {
        let _lock = self.ctx.acquire_config_lock()?;
        let mut config = self.ctx.load().await?;
        let prev = config.aliases.insert(name, model);
        self.ctx.save_raw(&config).await?;
        Ok(prev)
    }

    /// Removes a model alias. Returns the removed value if it existed.
    pub async fn remove_alias(&self, name: &str) -> Result<Option<String>> {
        let _lock = self.ctx.acquire_config_lock()?;
        let mut config = self.ctx.load().await?;
        let removed = config.aliases.remove(name);
        if removed.is_some() {
            self.ctx.save_raw(&config).await?;
        }
        Ok(removed)
    }

    /// Resolves a model name through aliases, with cycle detection.
    /// Returns the final resolved model name.
    pub async fn resolve_alias(&self, model: &str) -> Result<String> {
        let aliases = self.get_aliases().await?;
        let mut current = model.to_string();
        let mut seen = std::collections::HashSet::new();
        while let Some(target) = aliases.get(&current) {
            if !seen.insert(current.clone()) {
                anyhow::bail!("circular alias detected: {}", model);
            }
            current = target.clone();
        }
        Ok(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::session_store::{ConfigContext, StoredConfig};
    use tempfile::TempDir;

    fn make_store(temp_dir: &TempDir) -> AliasStore {
        let config_path = temp_dir.path().join("config.json");
        let config_dir = temp_dir.path().to_path_buf();
        AliasStore {
            ctx: ConfigContext {
                config_path,
                config_dir,
            },
        }
    }

    #[tokio::test]
    async fn get_aliases_empty() {
        let dir = TempDir::new().unwrap();
        let store = make_store(&dir);
        assert!(store.get_aliases().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn set_and_get_aliases() {
        let dir = TempDir::new().unwrap();
        let store = make_store(&dir);

        assert!(store.get_aliases().await.unwrap().is_empty());

        store
            .set_alias("fast".to_string(), "claude-haiku".to_string())
            .await
            .unwrap();
        let aliases = store.get_aliases().await.unwrap();
        assert_eq!(aliases.get("fast").unwrap(), "claude-haiku");
    }

    #[tokio::test]
    async fn remove_alias_returns_old_value() {
        let dir = TempDir::new().unwrap();
        let store = make_store(&dir);

        store
            .set_alias("fast".to_string(), "haiku".to_string())
            .await
            .unwrap();
        let removed = store.remove_alias("fast").await.unwrap();
        assert_eq!(removed, Some("haiku".to_string()));

        let removed_again = store.remove_alias("fast").await.unwrap();
        assert_eq!(removed_again, None);
    }

    #[tokio::test]
    async fn resolve_alias_follows_chain() {
        let dir = TempDir::new().unwrap();
        let store = make_store(&dir);

        store
            .set_alias("quick".to_string(), "fast".to_string())
            .await
            .unwrap();
        store
            .set_alias("fast".to_string(), "claude-haiku".to_string())
            .await
            .unwrap();

        let resolved = store.resolve_alias("quick").await.unwrap();
        assert_eq!(resolved, "claude-haiku");
    }

    #[tokio::test]
    async fn resolve_alias_detects_cycle() {
        let dir = TempDir::new().unwrap();
        let store = make_store(&dir);

        store
            .set_alias("a".to_string(), "b".to_string())
            .await
            .unwrap();
        store
            .set_alias("b".to_string(), "a".to_string())
            .await
            .unwrap();

        let result = store.resolve_alias("a").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn resolve_alias_passthrough_non_alias() {
        let dir = TempDir::new().unwrap();
        let store = make_store(&dir);

        let resolved = store.resolve_alias("claude-sonnet-4-6").await.unwrap();
        assert_eq!(resolved, "claude-sonnet-4-6");
    }

    #[tokio::test]
    async fn set_overwrites_existing() {
        let dir = TempDir::new().unwrap();
        let store = make_store(&dir);

        store
            .set_alias("fast".to_string(), "haiku".to_string())
            .await
            .unwrap();
        store
            .set_alias("fast".to_string(), "sonnet".to_string())
            .await
            .unwrap();

        let aliases = store.get_aliases().await.unwrap();
        assert_eq!(aliases.get("fast").unwrap(), "sonnet");
    }
}
