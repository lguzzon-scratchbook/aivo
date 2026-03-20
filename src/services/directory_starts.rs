use anyhow::Result;
use chrono::Utc;

use crate::services::session_store::{ConfigContext, DirectoryStartRecord};

#[derive(Debug, Clone)]
pub(crate) struct DirectoryStartsStore {
    pub(crate) ctx: ConfigContext,
}

impl DirectoryStartsStore {
    pub(crate) async fn get_directory_start(
        &self,
        cwd: &str,
    ) -> Result<Option<DirectoryStartRecord>> {
        let config = self.ctx.load().await?;
        let Some(record) = config.directory_starts.get(cwd).cloned() else {
            return Ok(None);
        };

        let key_is_valid = config
            .api_keys
            .iter()
            .any(|key| key.id == record.key_id && key.base_url == record.base_url);
        if key_is_valid {
            return Ok(Some(record));
        }

        // Stale record — re-acquire exclusive lock, reload, remove, save.
        let _lock = self.ctx.acquire_config_lock()?;
        let mut config = self.ctx.load().await?;
        config.directory_starts.remove(cwd);
        self.ctx.save_raw(&config).await?;
        Ok(None)
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
        config.directory_starts.insert(
            cwd.to_string(),
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
