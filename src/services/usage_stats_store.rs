use anyhow::Result;

use crate::services::session_store::ConfigContext;

#[derive(Debug, Clone)]
pub(crate) struct UsageStatsStore {
    pub(crate) ctx: ConfigContext,
}

impl UsageStatsStore {
    pub(crate) async fn record_selection(
        &self,
        key_id: &str,
        tool: &str,
        model: Option<&str>,
    ) -> Result<()> {
        let _lock = self.ctx.acquire_config_lock()?;
        let mut config = self.ctx.load().await?;
        config.stats.record_selection(key_id, tool, model);
        self.ctx.save_raw(&config).await
    }

    pub(crate) async fn record_tokens(
        &self,
        key_id: &str,
        model: Option<&str>,
        prompt_tokens: u64,
        completion_tokens: u64,
        cache_read_input_tokens: u64,
        cache_creation_input_tokens: u64,
    ) -> Result<()> {
        let _lock = self.ctx.acquire_config_lock()?;
        let mut config = self.ctx.load().await?;
        config.stats.record_tokens(
            key_id,
            model,
            prompt_tokens,
            completion_tokens,
            cache_read_input_tokens,
            cache_creation_input_tokens,
        );
        self.ctx.save_raw(&config).await
    }
}
