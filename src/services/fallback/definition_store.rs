use anyhow::Result;
use std::collections::HashMap;

use crate::services::session_store::ConfigContext;

/// Standalone store for fallback definition CRUD.
///
/// Reads/writes the `fallbacks` field of StoredConfig. Does NOT manage
/// the cached FallbackManager (OnceLock) — that stays on SessionStore
/// since it needs mutable cached state.
#[derive(Debug, Clone)]
pub struct FallbackDefinitionStore {
    pub(crate) ctx: ConfigContext,
}

impl FallbackDefinitionStore {
    /// Returns fallback definitions from config.
    pub async fn get_fallbacks(
        &self,
    ) -> Result<HashMap<String, crate::services::fallback::FallbackDefinition>> {
        let config = self.ctx.load().await?;
        Ok(config.fallbacks)
    }

    /// Check if a name is a registered fallback alias.
    pub async fn is_fallback(&self, name: &str) -> Result<bool> {
        let fallbacks = self.get_fallbacks().await?;
        Ok(fallbacks.contains_key(name))
    }

    /// Persist the full fallback definitions map to config.
    /// Used by the fallback CLI command for set/rm operations.
    /// Note: does NOT invalidate the cached FallbackManager on SessionStore.
    /// The CLI is short-lived so stale cache is acceptable.
    pub async fn set_fallback(
        &self,
        fallbacks: &HashMap<String, crate::services::fallback::FallbackDefinition>,
    ) -> Result<()> {
        let _lock = self.ctx.acquire_config_lock()?;
        let mut config = self.ctx.load().await?;
        config.fallbacks = fallbacks.clone();
        self.ctx.save_raw(&config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::fallback::FallbackDefinition;
    use crate::services::session_store::ConfigContext;
    use tempfile::TempDir;

    fn make_store(temp_dir: &TempDir) -> FallbackDefinitionStore {
        let config_path = temp_dir.path().join("config.json");
        let config_dir = temp_dir.path().to_path_buf();
        FallbackDefinitionStore {
            ctx: ConfigContext {
                config_path,
                config_dir,
            },
        }
    }

    #[tokio::test]
    async fn get_fallbacks_empty() {
        let dir = TempDir::new().unwrap();
        let store = make_store(&dir);
        assert!(store.get_fallbacks().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn set_and_get_fallback() {
        let dir = TempDir::new().unwrap();
        let store = make_store(&dir);

        let mut fallbacks = HashMap::new();
        fallbacks.insert(
            "my-fallback".to_string(),
            FallbackDefinition {
                id: "my-fallback".to_string(),
                description: None,
                timeout_ms: None,
                sequence: vec![],
            },
        );
        store.set_fallback(&fallbacks).await.unwrap();

        let loaded = store.get_fallbacks().await.unwrap();
        assert!(loaded.contains_key("my-fallback"));
    }

    #[tokio::test]
    async fn is_fallback_checks_registration() {
        let dir = TempDir::new().unwrap();
        let store = make_store(&dir);

        assert!(!store.is_fallback("nonexistent").await.unwrap());

        let mut fallbacks = HashMap::new();
        fallbacks.insert(
            "my-fallback".to_string(),
            FallbackDefinition {
                id: "my-fallback".to_string(),
                description: None,
                timeout_ms: None,
                sequence: vec![],
            },
        );
        store.set_fallback(&fallbacks).await.unwrap();

        assert!(store.is_fallback("my-fallback").await.unwrap());
    }
}
