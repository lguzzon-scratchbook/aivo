use anyhow::{Context, Result};
use chrono::Utc;
use std::path::PathBuf;

use crate::services::session_crypto::encrypt;
use crate::services::session_store::{
    ChatSessionState, ConfigContext, ConfigLockGuard, SessionIndex, SessionIndexEntry,
    StoredChatMessage,
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
        let path = self.index_path();
        let tmp_path = path.with_extension("json.tmp");

        tokio::fs::write(&tmp_path, &data)
            .await
            .with_context(|| format!("Failed to write temp index file: {:?}", tmp_path))?;

        tokio::fs::rename(&tmp_path, &path)
            .await
            .with_context(|| format!("Failed to rename temp index to {:?}", path))?;

        Ok(())
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
        let path = self.session_file_path(&state.session_id);
        let tmp_path = path.with_extension("json.tmp");

        tokio::fs::write(&tmp_path, &data)
            .await
            .with_context(|| format!("Failed to write temp session file: {:?}", tmp_path))?;

        tokio::fs::rename(&tmp_path, &path)
            .await
            .with_context(|| format!("Failed to rename temp session to {:?}", path))?;

        Ok(())
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
                updated_at: session.updated_at.clone(),
                created_at: session.created_at.clone(),
                title,
                preview,
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
                    updated_at: state.updated_at.clone(),
                    created_at: state.created_at.clone(),
                    title,
                    preview,
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
        messages: &[StoredChatMessage],
        title: &str,
        preview: &str,
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

        let new_entry = SessionIndexEntry {
            session_id: session_id.to_string(),
            key_id: key_id.to_string(),
            base_url: base_url.to_string(),
            cwd: cwd.to_string(),
            model: model.to_string(),
            updated_at: now,
            created_at,
            title: title.to_string(),
            preview: preview.to_string(),
        };

        if let Some(pos) = index
            .entries
            .iter()
            .position(|e| e.session_id == session_id)
        {
            index.entries[pos] = new_entry;
        } else {
            index.entries.push(new_entry);
        }

        self.evict_old_sessions(&mut index).await?;
        self.save_index(&index).await
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
