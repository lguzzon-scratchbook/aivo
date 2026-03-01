use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

const CACHE_TTL_SECS: u64 = 86400; // 24 hours

#[derive(Serialize, Deserialize, Clone)]
struct CacheEntry {
    models: Vec<String>,
    fetched_at: u64,
}

/// Disk cache for model lists keyed by base_url.
/// Stored at ~/.config/aivo/models-cache.json as plaintext JSON.
#[derive(Debug, Clone)]
pub struct ModelsCache {
    cache_path: PathBuf,
}

impl ModelsCache {
    pub fn new() -> Self {
        let cache_path = dirs::home_dir()
            .map(|p| p.join(".config").join("aivo").join("models-cache.json"))
            .unwrap_or_else(|| PathBuf::from(".config/aivo/models-cache.json"));
        Self { cache_path }
    }

    #[cfg(test)]
    pub fn with_path(cache_path: PathBuf) -> Self {
        Self { cache_path }
    }

    /// Returns cached models for `base_url` if present and not expired.
    pub async fn get(&self, base_url: &str) -> Option<Vec<String>> {
        let data = tokio::fs::read_to_string(&self.cache_path).await.ok()?;
        let map: HashMap<String, CacheEntry> = serde_json::from_str(&data).ok()?;
        let entry = map.get(base_url)?;
        let now = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs();
        if now.saturating_sub(entry.fetched_at) < CACHE_TTL_SECS {
            Some(entry.models.clone())
        } else {
            None
        }
    }

    /// Writes models for `base_url` into the cache file.
    /// Silently ignores write errors.
    pub async fn set(&self, base_url: &str, models: Vec<String>) {
        let mut map: HashMap<String, CacheEntry> = tokio::fs::read_to_string(&self.cache_path)
            .await
            .ok()
            .and_then(|data| serde_json::from_str(&data).ok())
            .unwrap_or_default();

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        map.insert(
            base_url.to_string(),
            CacheEntry {
                models,
                fetched_at: now,
            },
        );

        if let Ok(json) = serde_json::to_string_pretty(&map) {
            if let Some(parent) = self.cache_path.parent() {
                let _ = tokio::fs::create_dir_all(parent).await;
            }
            let _ = tokio::fs::write(&self.cache_path, json).await;
        }
    }
}

impl Default for ModelsCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_cache(dir: &TempDir) -> ModelsCache {
        ModelsCache::with_path(dir.path().join("models-cache.json"))
    }

    #[tokio::test]
    async fn cache_miss_on_empty() {
        let dir = TempDir::new().unwrap();
        let cache = make_cache(&dir);
        assert!(cache.get("https://api.example.com").await.is_none());
    }

    #[tokio::test]
    async fn roundtrip_set_and_get() {
        let dir = TempDir::new().unwrap();
        let cache = make_cache(&dir);
        let models = vec!["gpt-4o".to_string(), "gpt-4o-mini".to_string()];
        cache.set("https://api.example.com", models.clone()).await;
        let got = cache.get("https://api.example.com").await.unwrap();
        assert_eq!(got, models);
    }

    #[tokio::test]
    async fn corrupt_file_returns_none() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("models-cache.json");
        tokio::fs::write(&path, b"not json {{{").await.unwrap();
        let cache = ModelsCache::with_path(path);
        assert!(cache.get("https://api.example.com").await.is_none());
    }

    #[tokio::test]
    async fn expired_entry_returns_none() {
        let dir = TempDir::new().unwrap();
        let cache = make_cache(&dir);
        // Write a cache entry with fetched_at = 0 (epoch, definitely expired)
        let entry = serde_json::json!({
            "https://api.example.com": {
                "models": ["gpt-4o"],
                "fetched_at": 0u64
            }
        });
        tokio::fs::write(
            dir.path().join("models-cache.json"),
            serde_json::to_string(&entry).unwrap(),
        )
        .await
        .unwrap();
        assert!(cache.get("https://api.example.com").await.is_none());
    }
}
