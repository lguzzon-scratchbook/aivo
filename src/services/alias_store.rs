use anyhow::Result;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;

use crate::services::session_store::ConfigContext;

/// A legacy launcher alias from older aivo versions.
/// Maps a short name to a tool invocation with specific arguments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AliasLauncher {
    /// The AI tool to run (e.g. "claude", "codex").
    pub tool: String,
    /// Arguments to pass to the tool.
    pub args: Vec<String>,
}

/// Standalone store for model alias operations.
///
/// Handles two types of aliases stored in the `aliases` field of StoredConfig:
/// - Simple string aliases: `"fast" -> "claude-haiku-4-5"` (model name resolution)
/// - Legacy launcher aliases: `"fre" -> {"tool": "claude", "args": ["--model", "..."]}`
///   (invokes a tool with specific arguments — from older aivo versions)
#[derive(Debug, Clone)]
pub struct AliasStore {
    pub(crate) ctx: ConfigContext,
}

impl AliasStore {
    /// Returns all simple (string-valued) model aliases.
    pub async fn get_aliases(&self) -> Result<HashMap<String, String>> {
        let config = self.ctx.load().await?;
        Ok(config
            .aliases
            .iter()
            .filter_map(|(k, v)| match v {
                Value::String(s) => Some((k.clone(), s.clone())),
                _ => None,
            })
            .collect())
    }

    /// Sets a model alias. Returns the previous simple string value if it existed.
    pub async fn set_alias(&self, name: String, model: String) -> Result<Option<String>> {
        let _lock = self.ctx.acquire_config_lock()?;
        let mut config = self.ctx.load().await?;
        let prev = config.aliases.insert(name, Value::String(model));
        self.ctx.save_raw(&config).await?;
        Ok(prev.and_then(|v| match v {
            Value::String(s) => Some(s),
            _ => None,
        }))
    }

    /// Removes a model alias (both simple and launcher types).
    /// Returns the string representation of the removed value, if any.
    pub async fn remove_alias(&self, name: &str) -> Result<Option<String>> {
        let _lock = self.ctx.acquire_config_lock()?;
        let mut config = self.ctx.load().await?;
        let removed = config.aliases.remove(name);
        if removed.is_some() {
            self.ctx.save_raw(&config).await?;
        }
        Ok(removed.map(|v| alias_value_display(&v)))
    }

    /// Resolves a model name through simple aliases, with cycle detection.
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

    /// Returns all legacy launcher aliases.
    pub async fn get_alias_launchers(&self) -> Result<HashMap<String, AliasLauncher>> {
        let config = self.ctx.load().await?;
        Ok(config
            .aliases
            .iter()
            .filter_map(|(k, v)| {
                serde_json::from_value::<AliasLauncher>(v.clone())
                    .ok()
                    .map(|a| (k.clone(), a))
            })
            .collect())
    }

    /// Returns both simple and launcher aliases as displayable entries.
    /// Simple aliases show `name -> model`, launcher aliases show `name -> tool args...`.
    pub async fn get_all_display_entries(&self) -> Result<Vec<(String, String)>> {
        let config = self.ctx.load().await?;
        let mut entries: Vec<(String, String)> = Vec::new();
        for (name, value) in &config.aliases {
            let display = match value {
                Value::String(s) => s.clone(),
                other => {
                    if let Ok(launcher) =
                        serde_json::from_value::<AliasLauncher>(other.clone())
                    {
                        format!("{} {}", launcher.tool, launcher.args.join(" "))
                    } else {
                        alias_value_display(other)
                    }
                }
            };
            entries.push((name.clone(), display));
        }
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(entries)
    }
}

/// Formats an alias value for display.
fn alias_value_display(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => serde_json::to_string_pretty(other).unwrap_or_else(|_| other.to_string()),
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

    #[tokio::test]
    async fn get_aliases_filters_launcher_entries() {
        let dir = TempDir::new().unwrap();
        let store = make_store(&dir);

        let config = StoredConfig {
            aliases: HashMap::from([
                ("fast".to_string(), Value::String("claude-haiku".to_string())),
                ("fre".to_string(), serde_json::json!({"tool": "claude", "args": ["--model", "free-model"]})),
            ]),
            ..StoredConfig::new()
        };
        let data = serde_json::to_string_pretty(&config).unwrap();
        std::fs::write(&store.ctx.config_path, &data).unwrap();

        let simple = store.get_aliases().await.unwrap();
        assert_eq!(simple.len(), 1);
        assert_eq!(simple.get("fast").unwrap(), "claude-haiku");
        assert!(simple.get("fre").is_none());
    }

    #[tokio::test]
    async fn get_alias_launchers_returns_launcher_entries() {
        let dir = TempDir::new().unwrap();
        let store = make_store(&dir);

        let config = StoredConfig {
            aliases: HashMap::from([
                ("fast".to_string(), Value::String("claude-haiku".to_string())),
                ("fre".to_string(), serde_json::json!({"tool": "claude", "args": ["--model", "free-model"]})),
            ]),
            ..StoredConfig::new()
        };
        let data = serde_json::to_string_pretty(&config).unwrap();
        std::fs::write(&store.ctx.config_path, &data).unwrap();

        let launchers = store.get_alias_launchers().await.unwrap();
        assert_eq!(launchers.len(), 1);
        let fre = launchers.get("fre").unwrap();
        assert_eq!(fre.tool, "claude");
        assert_eq!(fre.args, vec!["--model", "free-model"]);
    }

    #[tokio::test]
    async fn get_all_display_entries_shows_both_types() {
        let dir = TempDir::new().unwrap();
        let store = make_store(&dir);

        let config = StoredConfig {
            aliases: HashMap::from([
                ("fast".to_string(), Value::String("claude-haiku".to_string())),
                ("fre".to_string(), serde_json::json!({"tool": "claude", "args": ["--model", "free-model"]})),
            ]),
            ..StoredConfig::new()
        };
        let data = serde_json::to_string_pretty(&config).unwrap();
        std::fs::write(&store.ctx.config_path, &data).unwrap();

        let entries = store.get_all_display_entries().await.unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, "fast");
        assert_eq!(entries[0].1, "claude-haiku");
        assert_eq!(entries[1].0, "fre");
        assert_eq!(entries[1].1, "claude --model free-model");
    }

    #[tokio::test]
    async fn remove_alias_removes_launcher_entry() {
        let dir = TempDir::new().unwrap();
        let store = make_store(&dir);

        let config = StoredConfig {
            aliases: HashMap::from([
                ("fre".to_string(), serde_json::json!({"tool": "claude", "args": ["--model", "free-model"]})),
            ]),
            ..StoredConfig::new()
        };
        let data = serde_json::to_string_pretty(&config).unwrap();
        std::fs::write(&store.ctx.config_path, &data).unwrap();

        assert!(store.get_alias_launchers().await.unwrap().contains_key("fre"));

        let removed = store.remove_alias("fre").await.unwrap();
        assert!(removed.is_some());

        assert!(store.get_alias_launchers().await.unwrap().is_empty());
    }
}
