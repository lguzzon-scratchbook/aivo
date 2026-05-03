use anyhow::{Context, Result};
use chrono::Utc;
use std::path::PathBuf;

use crate::services::atomic_write::atomic_write_secure;
use crate::services::session_crypto::encrypt;
use crate::services::session_store::{
    ChatSessionState, ChatTokenWindow, ConfigContext, ConfigLockGuard, SessionIndex,
    SessionIndexEntry, SessionTokens, StoredChatMessage,
};

#[derive(Debug, Clone)]
pub(crate) struct ChatSessionStore {
    pub(crate) ctx: ConfigContext,
}

fn compute_session_title(messages: &[StoredChatMessage], model: &str) -> String {
    let last_user = messages
        .iter()
        .rev()
        .find(|m| m.role == "user" && !m.content.trim().is_empty())
        .map(|m| first_non_empty_line(&m.content));
    let fallback = messages
        .iter()
        .rev()
        .find(|m| !m.content.trim().is_empty())
        .map(|m| first_non_empty_line(&m.content));
    last_user
        .or(fallback)
        .filter(|t| !t.is_empty())
        .unwrap_or_else(|| model.to_string())
}

fn compute_session_preview(messages: &[StoredChatMessage], model: &str) -> String {
    let snippets: Vec<String> = messages
        .iter()
        .rev()
        .filter(|m| !m.content.trim().is_empty())
        .take(2)
        .map(|m| m.content.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect();
    let joined = snippets
        .into_iter()
        .rev()
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(" · ");
    if !joined.is_empty() {
        joined
    } else {
        model.to_string()
    }
}

fn first_non_empty_line(text: &str) -> String {
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("")
        .to_string()
}

impl ChatSessionStore {
    pub(crate) fn sessions_dir(&self) -> PathBuf {
        self.ctx.config_dir.join("sessions")
    }

    pub(crate) fn session_file_path(&self, session_id: &str) -> PathBuf {
        self.sessions_dir().join(format!("{session_id}.json"))
    }

    fn index_path(&self) -> PathBuf {
        self.sessions_dir().join("index.json")
    }

    fn session_lock_path(&self) -> PathBuf {
        self.sessions_dir().join("sessions.lock")
    }

    fn acquire_session_lock(&self) -> Result<ConfigLockGuard> {
        let sessions_dir = self.sessions_dir();
        std::fs::create_dir_all(&sessions_dir)
            .with_context(|| format!("Failed to create sessions directory: {:?}", sessions_dir))?;
        ConfigLockGuard::acquire(&self.session_lock_path())
    }

    async fn load_index(&self) -> Result<SessionIndex> {
        let path = self.index_path();
        match tokio::fs::read_to_string(&path).await {
            Ok(data) => serde_json::from_str(&data).context("Failed to parse session index"),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(SessionIndex::default()),
            Err(e) => Err(e).with_context(|| format!("Failed to read session index: {:?}", path)),
        }
    }

    async fn save_index(&self, index: &SessionIndex) -> Result<()> {
        let sessions_dir = self.sessions_dir();
        tokio::fs::create_dir_all(&sessions_dir)
            .await
            .with_context(|| format!("Failed to create sessions directory: {:?}", sessions_dir))?;

        let data =
            serde_json::to_string_pretty(index).context("Failed to serialize session index")?;
        atomic_write_secure(&self.index_path(), data.into_bytes()).await
    }

    async fn load_session_file(&self, session_id: &str) -> Result<ChatSessionState> {
        let path = self.session_file_path(session_id);
        let data = tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("Failed to read session file: {:?}", path))?;
        serde_json::from_str(&data).context("Failed to parse session file")
    }

    async fn save_session_file(&self, state: &ChatSessionState) -> Result<()> {
        let sessions_dir = self.sessions_dir();
        tokio::fs::create_dir_all(&sessions_dir)
            .await
            .with_context(|| format!("Failed to create sessions directory: {:?}", sessions_dir))?;

        let data = serde_json::to_string_pretty(state).context("Failed to serialize session")?;
        atomic_write_secure(
            &self.session_file_path(&state.session_id),
            data.into_bytes(),
        )
        .await
    }

    // ── Migration ─────────────────────────────────────────────────────────

    async fn migrate_sessions_if_needed(&self) -> Result<()> {
        let marker = self.sessions_dir().join(".migrated");
        if marker.exists() {
            return Ok(());
        }

        // Load config and check for legacy sessions
        let config = self.ctx.load().await?;
        if config.chat_sessions.is_empty() {
            // Write marker even if nothing to migrate
            let sessions_dir = self.sessions_dir();
            tokio::fs::create_dir_all(&sessions_dir).await?;
            tokio::fs::write(&marker, b"").await?;
            return Ok(());
        }

        let sessions_dir = self.sessions_dir();
        tokio::fs::create_dir_all(&sessions_dir).await?;

        let mut index = self.load_index().await.unwrap_or_default();

        for session in config.chat_sessions.values() {
            let file_path = self.session_file_path(&session.session_id);
            // Skip if already migrated
            if file_path.exists() {
                continue;
            }

            // Compute title/preview by decrypting
            let (title, preview) = if let Ok(messages) = session.decrypt_messages() {
                (
                    compute_session_title(&messages, &session.model),
                    compute_session_preview(&messages, &session.model),
                )
            } else {
                (session.model.clone(), String::new())
            };

            self.save_session_file(session).await?;

            // Update or insert index entry
            let pos = index
                .entries
                .iter()
                .position(|e| e.session_id == session.session_id);
            let entry = SessionIndexEntry {
                session_id: session.session_id.clone(),
                key_id: session.key_id.clone(),
                base_url: session.base_url.clone(),
                cwd: session.cwd.clone(),
                model: session.model.clone(),
                billed_model: None,
                updated_at: session.updated_at.clone(),
                created_at: session.created_at.clone(),
                title,
                preview,
                prompt_tokens: 0,
                completion_tokens: 0,
                cache_read_tokens: 0,
                cache_write_tokens: 0,
            };
            if let Some(i) = pos {
                index.entries[i] = entry;
            } else {
                index.entries.push(entry);
            }
        }

        self.save_index(&index).await?;
        tokio::fs::write(&marker, b"").await?;
        Ok(())
    }

    // ── Eviction ──────────────────────────────────────────────────────────

    async fn evict_old_sessions(&self, index: &mut SessionIndex) -> Result<()> {
        const MAX_SESSIONS_PER_SCOPE: usize = 20;
        const MAX_TOTAL_SESSIONS: usize = 100;

        let mut to_delete: Vec<String> = Vec::new();

        // Group by (key_id, cwd) and mark per-scope excess
        let mut scope_map: std::collections::HashMap<(String, String), Vec<usize>> =
            std::collections::HashMap::new();
        for (i, entry) in index.entries.iter().enumerate() {
            scope_map
                .entry((entry.key_id.clone(), entry.cwd.clone()))
                .or_default()
                .push(i);
        }
        let mut keep = vec![true; index.entries.len()];
        for indices in scope_map.values() {
            // Sort by updated_at desc (most recent first) and mark excess
            let mut sorted = indices.clone();
            sorted.sort_by(|&a, &b| {
                index.entries[b]
                    .updated_at
                    .cmp(&index.entries[a].updated_at)
            });
            for &idx in sorted.iter().skip(MAX_SESSIONS_PER_SCOPE) {
                keep[idx] = false;
                to_delete.push(index.entries[idx].session_id.clone());
            }
        }

        // Global cap: if still over limit, drop oldest across all scopes
        let remaining: Vec<usize> = keep
            .iter()
            .enumerate()
            .filter_map(|(i, &k)| if k { Some(i) } else { None })
            .collect();
        if remaining.len() > MAX_TOTAL_SESSIONS {
            let mut sorted = remaining.clone();
            sorted.sort_by(|&a, &b| {
                index.entries[b]
                    .updated_at
                    .cmp(&index.entries[a].updated_at)
            });
            for &idx in sorted.iter().skip(MAX_TOTAL_SESSIONS) {
                keep[idx] = false;
                to_delete.push(index.entries[idx].session_id.clone());
            }
        }

        // Delete session files
        for session_id in &to_delete {
            let path = self.session_file_path(session_id);
            let _ = tokio::fs::remove_file(&path).await;
        }

        // Prune index
        if !to_delete.is_empty() {
            index.entries.retain(|e| !to_delete.contains(&e.session_id));
        }

        Ok(())
    }

    // ── Rebuild index safety net ──────────────────────────────────────────

    async fn rebuild_index(&self) -> Result<SessionIndex> {
        let sessions_dir = self.sessions_dir();
        let mut read_dir = match tokio::fs::read_dir(&sessions_dir).await {
            Ok(rd) => rd,
            Err(_) => return Ok(SessionIndex::default()),
        };

        let mut entries = Vec::new();
        while let Ok(Some(entry)) = read_dir.next_entry().await {
            let path = entry.path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !name.ends_with(".json") || name == "index.json" {
                continue;
            }
            let session_id = name.trim_end_matches(".json");
            if let Ok(state) = self.load_session_file(session_id).await {
                let (title, preview) = if let Ok(messages) = state.decrypt_messages() {
                    (
                        compute_session_title(&messages, &state.model),
                        compute_session_preview(&messages, &state.model),
                    )
                } else {
                    (state.model.clone(), String::new())
                };

                entries.push(SessionIndexEntry {
                    session_id: state.session_id.clone(),
                    key_id: state.key_id.clone(),
                    base_url: state.base_url.clone(),
                    cwd: state.cwd.clone(),
                    model: state.model.clone(),
                    billed_model: None,
                    updated_at: state.updated_at.clone(),
                    created_at: state.created_at.clone(),
                    title,
                    preview,
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    cache_read_tokens: 0,
                    cache_write_tokens: 0,
                });
            }
        }

        entries.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(SessionIndex { entries })
    }

    // ── Public methods ────────────────────────────────────────────────────

    pub(crate) async fn get_chat_session(
        &self,
        session_id: &str,
    ) -> Result<Option<ChatSessionState>> {
        self.migrate_sessions_if_needed().await?;
        let path = self.session_file_path(session_id);
        if !path.exists() {
            return Ok(None);
        }
        Ok(Some(self.load_session_file(session_id).await?))
    }

    pub(crate) async fn list_chat_sessions(
        &self,
        key_id: &str,
        base_url: &str,
        cwd: &str,
    ) -> Result<Vec<SessionIndexEntry>> {
        self.migrate_sessions_if_needed().await?;
        let _lock = self.acquire_session_lock()?;

        let mut index = match self.load_index().await {
            Ok(idx) => idx,
            Err(_) => self.rebuild_index().await?,
        };

        // Validate key still exists; prune stale entries
        let key_is_valid = {
            let config = self.ctx.load().await?;
            config
                .api_keys
                .iter()
                .any(|k| k.id == key_id && k.base_url == base_url)
        };

        let mut stale_ids: Vec<String> = Vec::new();
        let mut entries: Vec<SessionIndexEntry> = Vec::new();

        for entry in &index.entries {
            if entry.key_id != key_id || entry.cwd != cwd {
                continue;
            }
            if !key_is_valid || entry.base_url != base_url {
                stale_ids.push(entry.session_id.clone());
            } else {
                entries.push(entry.clone());
            }
        }

        if !stale_ids.is_empty() {
            for session_id in &stale_ids {
                let _ = tokio::fs::remove_file(self.session_file_path(session_id)).await;
            }
            index.entries.retain(|e| !stale_ids.contains(&e.session_id));
            self.save_index(&index).await?;
        }

        entries.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(entries)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn save_chat_session_with_id(
        &self,
        key_id: &str,
        base_url: &str,
        cwd: &str,
        session_id: &str,
        model: &str,
        billed_model: Option<&str>,
        messages: &[StoredChatMessage],
        title: &str,
        preview: &str,
        tokens: SessionTokens,
    ) -> Result<()> {
        self.migrate_sessions_if_needed().await?;
        let _lock = self.acquire_session_lock()?;

        let json = serde_json::to_string(messages).context("Failed to serialize messages")?;
        let encrypted = encrypt(&json)?;
        let now = Utc::now().to_rfc3339();
        // Preserve created_at from existing session file; use now for new sessions.
        let created_at = self
            .load_session_file(session_id)
            .await
            .ok()
            .and_then(|s| {
                if s.created_at.is_empty() {
                    None
                } else {
                    Some(s.created_at)
                }
            })
            .unwrap_or_else(|| now.clone());
        let state = ChatSessionState {
            session_id: session_id.to_string(),
            key_id: key_id.to_string(),
            base_url: base_url.to_string(),
            cwd: cwd.to_string(),
            model: model.to_string(),
            messages: encrypted,
            updated_at: now.clone(),
            created_at: created_at.clone(),
        };
        self.save_session_file(&state).await?;

        let mut index = match self.load_index().await {
            Ok(idx) => idx,
            Err(_) => self.rebuild_index().await?,
        };

        let existing_pos = index
            .entries
            .iter()
            .position(|e| e.session_id == session_id);
        // A heartbeat save (no fresh turn) passes None; preserve the
        // previous turn's upstream model rather than clearing it.
        let preserved_billed = billed_model
            .map(str::to_string)
            .or_else(|| existing_pos.and_then(|pos| index.entries[pos].billed_model.clone()));

        let new_entry = SessionIndexEntry {
            session_id: session_id.to_string(),
            key_id: key_id.to_string(),
            base_url: base_url.to_string(),
            cwd: cwd.to_string(),
            model: model.to_string(),
            billed_model: preserved_billed,
            updated_at: now,
            created_at,
            title: title.to_string(),
            preview: preview.to_string(),
            prompt_tokens: tokens.prompt_tokens,
            completion_tokens: tokens.completion_tokens,
            cache_read_tokens: tokens.cache_read_tokens,
            cache_write_tokens: tokens.cache_write_tokens,
        };

        match existing_pos {
            Some(pos) => index.entries[pos] = new_entry,
            None => index.entries.push(new_entry),
        }

        self.evict_old_sessions(&mut index).await?;
        self.save_index(&index).await
    }

    pub(crate) async fn count_chat_sessions(&self) -> u64 {
        self.load_index()
            .await
            .map(|idx| idx.entries.len() as u64)
            .unwrap_or(0)
    }

    /// Walks the session index once, returning the count of entries inside
    /// the window and per-model token totals across them.
    ///
    /// Index entries written before token tracking deserialize with zero
    /// token fields, so they're skipped from `per_model` (no empty rows in
    /// the model breakdown) but still contribute to `count`.
    pub(crate) async fn aggregate_chat_window_since(
        &self,
        cutoff: chrono::DateTime<chrono::Utc>,
    ) -> ChatTokenWindow {
        let Ok(idx) = self.load_index().await else {
            return ChatTokenWindow::default();
        };
        let cutoff_str = cutoff.to_rfc3339();
        let mut window = ChatTokenWindow::default();
        for e in &idx.entries {
            if e.updated_at.as_str() < cutoff_str.as_str() {
                continue;
            }
            window.count += 1;
            let entry_tokens = SessionTokens {
                prompt_tokens: e.prompt_tokens,
                completion_tokens: e.completion_tokens,
                cache_read_tokens: e.cache_read_tokens,
                cache_write_tokens: e.cache_write_tokens,
            };
            if entry_tokens.total() == 0 {
                continue;
            }
            // Prefer billed_model so aliases (`aivo/starter` → `deepseek-v4-flash`)
            // collapse onto the same key claude-code records.
            let key = e.billed_model.clone().unwrap_or_else(|| e.model.clone());
            let model_entry = window.per_model.entry(key).or_default();
            *model_entry = model_entry.merge(entry_tokens);
        }
        window
    }

    pub(crate) async fn delete_chat_session(&self, session_id: &str) -> Result<bool> {
        self.migrate_sessions_if_needed().await?;
        let _lock = self.acquire_session_lock()?;

        let path = self.session_file_path(session_id);
        let existed = path.exists();
        if existed {
            tokio::fs::remove_file(&path)
                .await
                .with_context(|| format!("Failed to delete session file: {:?}", path))?;
        }

        let mut index = self.load_index().await.unwrap_or_default();
        let before = index.entries.len();
        index.entries.retain(|e| e.session_id != session_id);
        if index.entries.len() < before {
            self.save_index(&index).await?;
        }

        Ok(existed || before > index.entries.len())
    }

    /// Removes session files for all sessions belonging to a key.
    pub(crate) async fn remove_sessions_for_key(&self, key_id: &str) -> Result<()> {
        let _lock = self.acquire_session_lock()?;
        let mut index = self.load_index().await.unwrap_or_default();
        let to_delete: Vec<String> = index
            .entries
            .iter()
            .filter(|e| e.key_id == key_id)
            .map(|e| e.session_id.clone())
            .collect();
        for session_id in &to_delete {
            let _ = tokio::fs::remove_file(self.session_file_path(session_id)).await;
        }
        index.entries.retain(|e| e.key_id != key_id);
        if !to_delete.is_empty() {
            self.save_index(&index).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::session_store::{ApiKey, ConfigContext, StoredChatMessage, StoredConfig};
    use tempfile::TempDir;

    fn make_store(temp_dir: &TempDir) -> ChatSessionStore {
        let config_path = temp_dir.path().join("config.json");
        let config_dir = temp_dir.path().to_path_buf();
        ChatSessionStore {
            ctx: ConfigContext {
                config_path,
                config_dir,
            },
        }
    }

    async fn setup_store_with_key(temp_dir: &TempDir) -> (ChatSessionStore, String) {
        let store = make_store(temp_dir);
        let key_id = "abc".to_string();
        let base_url = "http://localhost".to_string();

        // Write a config with one key so list_chat_sessions validates it
        let config = StoredConfig {
            api_keys: vec![ApiKey::new_with_protocol(
                key_id.clone(),
                "test".to_string(),
                base_url.clone(),
                None,
                "sk-test".to_string(),
            )],
            active_key_id: Some(key_id.clone()),
            ..StoredConfig::new()
        };
        let data = serde_json::to_string_pretty(&config).unwrap();
        tokio::fs::write(&store.ctx.config_path, &data)
            .await
            .unwrap();

        (store, key_id)
    }

    fn sample_messages() -> Vec<StoredChatMessage> {
        vec![StoredChatMessage {
            role: "user".to_string(),
            content: "hello world".to_string(),
            reasoning_content: None,
            id: None,
            timestamp: None,
            attachments: None,
        }]
    }

    #[tokio::test]
    async fn get_nonexistent_session_returns_none() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);
        let result = store.get_chat_session("nonexistent").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn save_and_get_session_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let (store, key_id) = setup_store_with_key(&temp_dir).await;

        store
            .save_chat_session_with_id(
                &key_id,
                "http://localhost",
                "/tmp/test",
                "sess1",
                "gpt-4o",
                None,
                &sample_messages(),
                "hello world",
                "hello world",
                SessionTokens::default(),
            )
            .await
            .unwrap();

        let session = store.get_chat_session("sess1").await.unwrap().unwrap();
        assert_eq!(session.session_id, "sess1");
        assert_eq!(session.model, "gpt-4o");
        assert_eq!(session.key_id, key_id);

        let messages = session.decrypt_messages().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "hello world");
    }

    #[tokio::test]
    async fn delete_session_removes_file_and_index() {
        let temp_dir = TempDir::new().unwrap();
        let (store, key_id) = setup_store_with_key(&temp_dir).await;

        store
            .save_chat_session_with_id(
                &key_id,
                "http://localhost",
                "/tmp/test",
                "sess1",
                "gpt-4o",
                None,
                &sample_messages(),
                "hello",
                "hello",
                SessionTokens::default(),
            )
            .await
            .unwrap();

        let deleted = store.delete_chat_session("sess1").await.unwrap();
        assert!(deleted);

        // Session should be gone
        let session = store.get_chat_session("sess1").await.unwrap();
        assert!(session.is_none());

        // Deleting again returns false
        let deleted = store.delete_chat_session("sess1").await.unwrap();
        assert!(!deleted);
    }

    #[tokio::test]
    async fn remove_sessions_for_key_cleans_up() {
        let temp_dir = TempDir::new().unwrap();
        let (store, key_id) = setup_store_with_key(&temp_dir).await;

        // Create two sessions for same key
        for sid in &["sess1", "sess2"] {
            store
                .save_chat_session_with_id(
                    &key_id,
                    "http://localhost",
                    "/tmp/test",
                    sid,
                    "gpt-4o",
                    None,
                    &sample_messages(),
                    "title",
                    "preview",
                    SessionTokens::default(),
                )
                .await
                .unwrap();
        }

        store.remove_sessions_for_key(&key_id).await.unwrap();

        // Both should be gone
        assert!(store.get_chat_session("sess1").await.unwrap().is_none());
        assert!(store.get_chat_session("sess2").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn list_sessions_filters_by_key_and_cwd() {
        let temp_dir = TempDir::new().unwrap();
        let (store, key_id) = setup_store_with_key(&temp_dir).await;

        // Save session in /tmp/a
        store
            .save_chat_session_with_id(
                &key_id,
                "http://localhost",
                "/tmp/a",
                "sess-a",
                "gpt-4o",
                None,
                &sample_messages(),
                "title-a",
                "preview-a",
                SessionTokens::default(),
            )
            .await
            .unwrap();

        // Save session in /tmp/b
        store
            .save_chat_session_with_id(
                &key_id,
                "http://localhost",
                "/tmp/b",
                "sess-b",
                "gpt-4o",
                None,
                &sample_messages(),
                "title-b",
                "preview-b",
                SessionTokens::default(),
            )
            .await
            .unwrap();

        let sessions = store
            .list_chat_sessions(&key_id, "http://localhost", "/tmp/a")
            .await
            .unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].session_id, "sess-a");
    }

    #[tokio::test]
    async fn save_session_preserves_created_at() {
        let temp_dir = TempDir::new().unwrap();
        let (store, key_id) = setup_store_with_key(&temp_dir).await;

        store
            .save_chat_session_with_id(
                &key_id,
                "http://localhost",
                "/tmp/test",
                "sess1",
                "gpt-4o",
                None,
                &sample_messages(),
                "title",
                "preview",
                SessionTokens::default(),
            )
            .await
            .unwrap();

        let first = store.get_chat_session("sess1").await.unwrap().unwrap();
        let original_created = first.created_at.clone();

        // Save again (update)
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        store
            .save_chat_session_with_id(
                &key_id,
                "http://localhost",
                "/tmp/test",
                "sess1",
                "gpt-4o",
                None,
                &sample_messages(),
                "title2",
                "preview2",
                SessionTokens::default(),
            )
            .await
            .unwrap();

        let updated = store.get_chat_session("sess1").await.unwrap().unwrap();
        assert_eq!(updated.created_at, original_created);
        // updated_at should be different (or at least not earlier)
        assert!(updated.updated_at >= original_created);
    }

    #[tokio::test]
    async fn rebuild_index_recovers_from_corrupted_index() {
        let temp_dir = TempDir::new().unwrap();
        let (store, key_id) = setup_store_with_key(&temp_dir).await;

        // Save a session (creates index + file)
        store
            .save_chat_session_with_id(
                &key_id,
                "http://localhost",
                "/tmp/test",
                "sess1",
                "gpt-4o",
                None,
                &sample_messages(),
                "title",
                "preview",
                SessionTokens::default(),
            )
            .await
            .unwrap();

        // Corrupt the index file (triggers rebuild via load_index() Err path)
        let index_path = store.sessions_dir().join("index.json");
        tokio::fs::write(&index_path, b"not valid json {{{")
            .await
            .unwrap();

        // list_chat_sessions should rebuild the index from session files
        let sessions = store
            .list_chat_sessions(&key_id, "http://localhost", "/tmp/test")
            .await
            .unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].session_id, "sess1");
    }

    #[test]
    fn compute_session_title_uses_last_user_message() {
        let messages = vec![
            StoredChatMessage {
                role: "user".to_string(),
                content: "first question".to_string(),
                reasoning_content: None,
                id: None,
                timestamp: None,
                attachments: None,
            },
            StoredChatMessage {
                role: "assistant".to_string(),
                content: "answer".to_string(),
                reasoning_content: None,
                id: None,
                timestamp: None,
                attachments: None,
            },
            StoredChatMessage {
                role: "user".to_string(),
                content: "second question".to_string(),
                reasoning_content: None,
                id: None,
                timestamp: None,
                attachments: None,
            },
        ];
        let title = compute_session_title(&messages, "model");
        assert_eq!(title, "second question");
    }

    #[test]
    fn compute_session_title_falls_back_to_model() {
        let messages: Vec<StoredChatMessage> = vec![];
        let title = compute_session_title(&messages, "gpt-4o");
        assert_eq!(title, "gpt-4o");
    }

    #[test]
    fn compute_session_preview_joins_recent_messages() {
        let messages = vec![
            StoredChatMessage {
                role: "user".to_string(),
                content: "hello  world".to_string(),
                reasoning_content: None,
                id: None,
                timestamp: None,
                attachments: None,
            },
            StoredChatMessage {
                role: "assistant".to_string(),
                content: "hi  there".to_string(),
                reasoning_content: None,
                id: None,
                timestamp: None,
                attachments: None,
            },
        ];
        let preview = compute_session_preview(&messages, "model");
        assert_eq!(preview, "hello world · hi there");
    }

    #[test]
    fn compute_session_preview_falls_back_to_model() {
        let messages: Vec<StoredChatMessage> = vec![];
        let preview = compute_session_preview(&messages, "gpt-4o");
        assert_eq!(preview, "gpt-4o");
    }

    #[test]
    fn first_non_empty_line_skips_blank_lines() {
        assert_eq!(first_non_empty_line("\n\n  hello\nworld"), "hello");
        assert_eq!(first_non_empty_line(""), "");
        assert_eq!(first_non_empty_line("  \n  \n  "), "");
    }

    #[tokio::test]
    async fn count_chat_sessions_since_filters_by_cutoff() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);

        let now = Utc::now();
        let old = now - chrono::Duration::days(30);
        let recent = now - chrono::Duration::hours(1);

        let index = SessionIndex {
            entries: vec![
                SessionIndexEntry {
                    session_id: "old".to_string(),
                    key_id: "k".to_string(),
                    base_url: "http://localhost".to_string(),
                    cwd: "/tmp".to_string(),
                    model: "gpt-4o".to_string(),
                    billed_model: None,
                    updated_at: old.to_rfc3339(),
                    created_at: old.to_rfc3339(),
                    title: "old".to_string(),
                    preview: "old".to_string(),
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    cache_read_tokens: 0,
                    cache_write_tokens: 0,
                },
                SessionIndexEntry {
                    session_id: "recent".to_string(),
                    key_id: "k".to_string(),
                    base_url: "http://localhost".to_string(),
                    cwd: "/tmp".to_string(),
                    model: "gpt-4o".to_string(),
                    billed_model: None,
                    updated_at: recent.to_rfc3339(),
                    created_at: recent.to_rfc3339(),
                    title: "recent".to_string(),
                    preview: "recent".to_string(),
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    cache_read_tokens: 0,
                    cache_write_tokens: 0,
                },
            ],
        };
        store.save_index(&index).await.unwrap();

        assert_eq!(store.count_chat_sessions().await, 2);
        let cutoff = now - chrono::Duration::days(7);
        assert_eq!(store.aggregate_chat_window_since(cutoff).await.count, 1);
    }

    #[tokio::test]
    async fn aggregate_chat_window_since_sums_per_model_and_filters_by_cutoff() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);

        let now = Utc::now();
        let old = now - chrono::Duration::days(30);
        let recent_a = now - chrono::Duration::minutes(30);
        let recent_b = now - chrono::Duration::minutes(10);

        let mk_entry =
            |id: &str, ts: chrono::DateTime<chrono::Utc>, model: &str, p: u64, c: u64| {
                SessionIndexEntry {
                    session_id: id.to_string(),
                    key_id: "k".to_string(),
                    base_url: "http://localhost".to_string(),
                    cwd: "/tmp".to_string(),
                    model: model.to_string(),
                    billed_model: None,
                    updated_at: ts.to_rfc3339(),
                    created_at: ts.to_rfc3339(),
                    title: id.to_string(),
                    preview: id.to_string(),
                    prompt_tokens: p,
                    completion_tokens: c,
                    cache_read_tokens: 0,
                    cache_write_tokens: 0,
                }
            };

        let index = SessionIndex {
            entries: vec![
                mk_entry("old", old, "gpt-4o", 999_999, 999_999), // outside window
                mk_entry("a1", recent_a, "minimax-m2.7", 100, 200),
                mk_entry("a2", recent_a, "minimax-m2.7", 50, 70), // same model, sums
                mk_entry("b1", recent_b, "claude-opus-4.7", 10, 5),
                mk_entry("zero", recent_b, "kimi", 0, 0), // skipped: no tokens
            ],
        };
        store.save_index(&index).await.unwrap();

        let cutoff = now - chrono::Duration::hours(1);
        let window = store.aggregate_chat_window_since(cutoff).await;
        let total = window.total();

        assert_eq!(window.count, 4, "old is filtered; zero-token still counts");
        assert_eq!(total.prompt_tokens, 160);
        assert_eq!(total.completion_tokens, 275);
        assert_eq!(
            window.per_model.len(),
            2,
            "kimi entry has zero tokens; old is filtered"
        );
        let minimax = &window.per_model["minimax-m2.7"];
        assert_eq!(minimax.prompt_tokens, 150);
        assert_eq!(minimax.completion_tokens, 270);
        let claude = &window.per_model["claude-opus-4.7"];
        assert_eq!(claude.prompt_tokens, 10);
        assert_eq!(claude.completion_tokens, 5);
    }

    #[tokio::test]
    async fn aggregate_chat_window_since_prefers_billed_model_over_alias() {
        // `aivo/starter` is an alias that resolves upstream to a real model
        // like `deepseek-v4-flash`. Stats prefers the billed name so chat
        // lines up with claude-code (which records model from the upstream
        // response). Entries without `billed_model` (legacy / non-aliased
        // providers) keep using `model` as the per-model key.
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);

        let now = Utc::now();
        let recent = now - chrono::Duration::minutes(5);

        let index = SessionIndex {
            entries: vec![
                SessionIndexEntry {
                    session_id: "starter-session".to_string(),
                    key_id: "k".to_string(),
                    base_url: "http://localhost".to_string(),
                    cwd: "/tmp".to_string(),
                    model: "aivo/starter".to_string(),
                    billed_model: Some("deepseek-v4-flash".to_string()),
                    updated_at: recent.to_rfc3339(),
                    created_at: recent.to_rfc3339(),
                    title: "starter".to_string(),
                    preview: "starter".to_string(),
                    prompt_tokens: 100,
                    completion_tokens: 50,
                    cache_read_tokens: 0,
                    cache_write_tokens: 0,
                },
                SessionIndexEntry {
                    session_id: "legacy-session".to_string(),
                    key_id: "k".to_string(),
                    base_url: "http://localhost".to_string(),
                    cwd: "/tmp".to_string(),
                    model: "gpt-4o".to_string(),
                    billed_model: None,
                    updated_at: recent.to_rfc3339(),
                    created_at: recent.to_rfc3339(),
                    title: "legacy".to_string(),
                    preview: "legacy".to_string(),
                    prompt_tokens: 7,
                    completion_tokens: 3,
                    cache_read_tokens: 0,
                    cache_write_tokens: 0,
                },
            ],
        };
        store.save_index(&index).await.unwrap();

        let cutoff = now - chrono::Duration::hours(1);
        let window = store.aggregate_chat_window_since(cutoff).await;

        assert_eq!(window.per_model.len(), 2);
        let billed = &window.per_model["deepseek-v4-flash"];
        assert_eq!(billed.prompt_tokens, 100);
        assert_eq!(billed.completion_tokens, 50);
        assert!(
            !window.per_model.contains_key("aivo/starter"),
            "billed_model should replace, not duplicate, the alias"
        );
        let legacy = &window.per_model["gpt-4o"];
        assert_eq!(legacy.prompt_tokens, 7);
        assert_eq!(legacy.completion_tokens, 3);
    }

    #[tokio::test]
    async fn save_chat_session_records_and_preserves_billed_model() {
        // First save carries a billed_model from the turn; a follow-up save
        // (e.g. TUI heartbeat with no fresh turn) passes None and must keep
        // the previously-recorded billed_model on the index entry.
        let temp_dir = TempDir::new().unwrap();
        let (store, key_id) = setup_store_with_key(&temp_dir).await;

        store
            .save_chat_session_with_id(
                &key_id,
                "http://localhost",
                "/tmp/test",
                "sess1",
                "aivo/starter",
                Some("deepseek-v4-flash"),
                &sample_messages(),
                "title",
                "preview",
                SessionTokens {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                    cache_read_tokens: 0,
                    cache_write_tokens: 0,
                },
            )
            .await
            .unwrap();

        let idx = store.load_index().await.unwrap();
        let entry = idx
            .entries
            .iter()
            .find(|e| e.session_id == "sess1")
            .unwrap();
        assert_eq!(entry.billed_model.as_deref(), Some("deepseek-v4-flash"));
        assert_eq!(entry.model, "aivo/starter");

        // Heartbeat-style save without a fresh billed_model preserves it.
        store
            .save_chat_session_with_id(
                &key_id,
                "http://localhost",
                "/tmp/test",
                "sess1",
                "aivo/starter",
                None,
                &sample_messages(),
                "title",
                "preview",
                SessionTokens::default(),
            )
            .await
            .unwrap();

        let idx = store.load_index().await.unwrap();
        let entry = idx
            .entries
            .iter()
            .find(|e| e.session_id == "sess1")
            .unwrap();
        assert_eq!(entry.billed_model.as_deref(), Some("deepseek-v4-flash"));
    }
}
