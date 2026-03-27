use anyhow::Result;
use chrono::Utc;

use crate::services::session_store::{ApiKey, ConfigContext, DirectoryStartRecord};

fn has_valid_key(record: &DirectoryStartRecord, keys: &[ApiKey]) -> bool {
    keys.iter()
        .any(|key| key.id == record.key_id && key.base_url == record.base_url)
}

#[derive(Debug, Clone)]
pub(crate) struct DirectoryStartsStore {
    pub(crate) ctx: ConfigContext,
}

impl DirectoryStartsStore {
    pub(crate) async fn get_directory_start(
        &self,
        cwd: &str,
        tool: &str,
    ) -> Result<Option<DirectoryStartRecord>> {
        let config = self.ctx.load().await?;
        let Some(tools) = config.directory_starts.get(cwd) else {
            return Ok(None);
        };
        let Some(record) = tools.get(tool).cloned() else {
            return Ok(None);
        };

        if has_valid_key(&record, &config.api_keys) {
            return Ok(Some(record));
        }

        // Stale record — remove just this tool's entry
        let _lock = self.ctx.acquire_config_lock()?;
        let mut config = self.ctx.load().await?;
        if let Some(tools) = config.directory_starts.get_mut(cwd) {
            tools.remove(tool);
            if tools.is_empty() {
                config.directory_starts.remove(cwd);
            }
        }
        self.ctx.save_raw(&config).await?;
        Ok(None)
    }

    pub(crate) async fn get_latest_directory_start(
        &self,
        cwd: &str,
    ) -> Result<Option<DirectoryStartRecord>> {
        let config = self.ctx.load().await?;
        let Some(tools) = config.directory_starts.get(cwd) else {
            return Ok(None);
        };

        let latest = tools
            .values()
            .filter(|record| has_valid_key(record, &config.api_keys))
            .max_by_key(|r| &r.updated_at)
            .cloned();

        Ok(latest)
    }

    pub(crate) async fn get_all_directory_starts(
        &self,
        cwd: &str,
    ) -> Result<Vec<DirectoryStartRecord>> {
        let config = self.ctx.load().await?;
        let Some(tools) = config.directory_starts.get(cwd) else {
            return Ok(Vec::new());
        };
        Ok(tools
            .values()
            .filter(|record| has_valid_key(record, &config.api_keys))
            .cloned()
            .collect())
    }

    pub(crate) async fn set_directory_start(
        &self,
        cwd: &str,
        key_id: &str,
        base_url: &str,
        tool: &str,
        model: Option<&str>,
    ) -> Result<()> {
        let _lock = self.ctx.acquire_config_lock()?;
        let mut config = self.ctx.load().await?;
        let tools = config.directory_starts.entry(cwd.to_string()).or_default();
        tools.insert(
            tool.to_string(),
            DirectoryStartRecord {
                key_id: key_id.to_string(),
                base_url: base_url.to_string(),
                tool: tool.to_string(),
                model: model
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(ToString::to_string),
                updated_at: Utc::now().to_rfc3339(),
            },
        );
        self.ctx.save_raw(&config).await
    }

    #[allow(dead_code)]
    pub(crate) async fn clear_directory_start(&self, cwd: &str) -> Result<bool> {
        let _lock = self.ctx.acquire_config_lock()?;
        let mut config = self.ctx.load().await?;
        let removed = config.directory_starts.remove(cwd).is_some();
        if removed {
            self.ctx.save_raw(&config).await?;
        }
        Ok(removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::session_store::{ApiKey, ConfigContext, StoredConfig};
    use tempfile::TempDir;

    fn make_store(temp_dir: &TempDir) -> DirectoryStartsStore {
        let config_path = temp_dir.path().join("config.json");
        let config_dir = temp_dir.path().to_path_buf();
        DirectoryStartsStore {
            ctx: ConfigContext {
                config_path,
                config_dir,
            },
        }
    }

    async fn write_config_with_key(store: &DirectoryStartsStore, key_id: &str, base_url: &str) {
        let config = StoredConfig {
            api_keys: vec![ApiKey::new_with_protocol(
                key_id.to_string(),
                "test".to_string(),
                base_url.to_string(),
                None,
                "sk-test".to_string(),
            )],
            ..StoredConfig::new()
        };
        let data = serde_json::to_string_pretty(&config).unwrap();
        tokio::fs::create_dir_all(&store.ctx.config_dir)
            .await
            .unwrap();
        tokio::fs::write(&store.ctx.config_path, &data)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn get_directory_start_returns_none_when_empty() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);
        write_config_with_key(&store, "key1", "http://localhost").await;

        let result = store
            .get_directory_start("/tmp/test", "claude")
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn set_and_get_directory_start_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);
        write_config_with_key(&store, "key1", "http://localhost").await;

        store
            .set_directory_start(
                "/tmp/test",
                "key1",
                "http://localhost",
                "claude",
                Some("gpt-4o"),
            )
            .await
            .unwrap();

        let record = store
            .get_directory_start("/tmp/test", "claude")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(record.key_id, "key1");
        assert_eq!(record.base_url, "http://localhost");
        assert_eq!(record.tool, "claude");
        assert_eq!(record.model.as_deref(), Some("gpt-4o"));
    }

    #[tokio::test]
    async fn set_directory_start_with_empty_model_stores_none() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);
        write_config_with_key(&store, "key1", "http://localhost").await;

        store
            .set_directory_start(
                "/tmp/test",
                "key1",
                "http://localhost",
                "claude",
                Some("  "),
            )
            .await
            .unwrap();

        let record = store
            .get_directory_start("/tmp/test", "claude")
            .await
            .unwrap()
            .unwrap();
        assert!(record.model.is_none());
    }

    #[tokio::test]
    async fn set_directory_start_with_none_model() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);
        write_config_with_key(&store, "key1", "http://localhost").await;

        store
            .set_directory_start("/tmp/test", "key1", "http://localhost", "codex", None)
            .await
            .unwrap();

        let record = store
            .get_directory_start("/tmp/test", "codex")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(record.tool, "codex");
        assert!(record.model.is_none());
    }

    #[tokio::test]
    async fn get_directory_start_prunes_stale_record() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);
        // Write config with key1
        write_config_with_key(&store, "key1", "http://localhost").await;

        // Set directory start for key1
        store
            .set_directory_start("/tmp/test", "key1", "http://localhost", "claude", None)
            .await
            .unwrap();

        // Now write a config WITHOUT key1 (simulating key deletion)
        let config = StoredConfig::new();
        let data = serde_json::to_string_pretty(&config).unwrap();
        tokio::fs::write(&store.ctx.config_path, &data)
            .await
            .unwrap();

        // Should return None and prune the stale record
        let result = store
            .get_directory_start("/tmp/test", "claude")
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn clear_directory_start_removes_record() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);
        write_config_with_key(&store, "key1", "http://localhost").await;

        store
            .set_directory_start("/tmp/test", "key1", "http://localhost", "claude", None)
            .await
            .unwrap();

        assert!(store.clear_directory_start("/tmp/test").await.unwrap());
        assert!(!store.clear_directory_start("/tmp/test").await.unwrap());

        let result = store
            .get_directory_start("/tmp/test", "claude")
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn set_directory_start_overwrites_existing() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);
        write_config_with_key(&store, "key1", "http://localhost").await;

        store
            .set_directory_start("/tmp/test", "key1", "http://localhost", "claude", None)
            .await
            .unwrap();

        store
            .set_directory_start(
                "/tmp/test",
                "key1",
                "http://localhost",
                "codex",
                Some("gpt-4o"),
            )
            .await
            .unwrap();

        let record = store
            .get_directory_start("/tmp/test", "codex")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(record.tool, "codex");
        assert_eq!(record.model.as_deref(), Some("gpt-4o"));
    }

    #[tokio::test]
    async fn get_directory_start_returns_tool_specific_record() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);
        write_config_with_key(&store, "key1", "http://localhost").await;

        store
            .set_directory_start(
                "/tmp/test",
                "key1",
                "http://localhost",
                "claude",
                Some("sonnet"),
            )
            .await
            .unwrap();
        store
            .set_directory_start(
                "/tmp/test",
                "key1",
                "http://localhost",
                "codex",
                Some("gpt-4o"),
            )
            .await
            .unwrap();

        let claude_record = store
            .get_directory_start("/tmp/test", "claude")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(claude_record.model.as_deref(), Some("sonnet"));

        let codex_record = store
            .get_directory_start("/tmp/test", "codex")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(codex_record.model.as_deref(), Some("gpt-4o"));
    }

    #[tokio::test]
    async fn get_latest_directory_start_returns_most_recent() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);
        write_config_with_key(&store, "key1", "http://localhost").await;

        store
            .set_directory_start(
                "/tmp/test",
                "key1",
                "http://localhost",
                "claude",
                Some("sonnet"),
            )
            .await
            .unwrap();

        // Small delay so codex has a later timestamp
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        store
            .set_directory_start(
                "/tmp/test",
                "key1",
                "http://localhost",
                "codex",
                Some("gpt-4o"),
            )
            .await
            .unwrap();

        let latest = store
            .get_latest_directory_start("/tmp/test")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(latest.tool, "codex");
    }

    #[tokio::test]
    async fn get_all_directory_starts_returns_all_valid() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);
        write_config_with_key(&store, "key1", "http://localhost").await;

        store
            .set_directory_start(
                "/tmp/test",
                "key1",
                "http://localhost",
                "claude",
                Some("sonnet"),
            )
            .await
            .unwrap();
        store
            .set_directory_start(
                "/tmp/test",
                "key1",
                "http://localhost",
                "codex",
                Some("gpt-4o"),
            )
            .await
            .unwrap();

        let all = store.get_all_directory_starts("/tmp/test").await.unwrap();
        assert_eq!(all.len(), 2);
    }
}
