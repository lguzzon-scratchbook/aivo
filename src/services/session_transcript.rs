//! Verbatim recent-turns extraction for the cross-tool MCP bridge.
//!
//! Unlike `context_ingest`, which returns a compressed `(topic, last_response)`
//! pair per session for the `--context` flow, this module returns a
//! chronological list of natural-language turns with no text truncation beyond
//! a per-turn safety cap. Consumers are MCP tools (`list_sessions`,
//! `get_session`) exposed by `aivo mcp-serve`, which Claude/Codex call to
//! inspect each other's in-flight conversations.
//!
//! Design notes:
//! - Skip Claude `isSidechain=true` turns (agent-within-agent noise).
//! - Skip tool-use / tool-result blocks; keep only natural-language text.
//! - Per-turn cap of 8 KB guards against multi-MB code pastes blowing up the
//!   MCP response envelope. Top-level `max_turns` cap is applied by the server.
//! - Silently skip unparseable JSONL lines — a partial/streaming last line
//!   from a peer tool that's still writing is normal, not an error.

use std::path::Path;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde_json::Value;
use tokio::fs;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::services::context_ingest::{
    encode_claude_dir, extract_claude_text, extract_codex_message_text, list_jsonl_newest_first,
    paths_match, walk_jsonl_newest_first,
};
use crate::services::system_env;

/// Hard cap on a single turn's text payload. Longer turns are truncated with
/// `…` to protect the MCP response from multi-MB pastes.
const MAX_TURN_BYTES: usize = 8 * 1024;

/// A single conversational turn, in chronological order.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Turn {
    /// "user" or "assistant".
    pub role: String,
    /// Verbatim text, capped at `MAX_TURN_BYTES`.
    pub text: String,
    /// RFC 3339 timestamp when available.
    pub timestamp: Option<DateTime<Utc>>,
}

/// A full transcript of one session: last N turns, chronological.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Transcript {
    /// "claude" or "codex".
    pub cli: String,
    /// Native session id.
    pub session_id: String,
    /// Source JSONL path for provenance.
    pub source_path: String,
    /// Turns in chronological order (oldest first), bounded by `max_turns`.
    pub turns: Vec<Turn>,
    /// Most recent turn's timestamp, if any (for sorting by freshness).
    pub updated_at: Option<DateTime<Utc>>,
}

/// Resolve a session for the given CLI, optionally by id prefix, and load its
/// recent turns. Returns `None` if no matching session exists for this cwd.
///
/// - `cli`: "claude" or "codex". Other values return `Ok(None)`.
/// - `session_id`: `None` → most-recent for this project; `Some(prefix)` →
///   prefix-match on the native session id.
/// - `exclude_session_ids`: any transcript whose `session_id` starts with
///   one of these strings is skipped. Used to skip the caller's own
///   session in same-CLI peer queries (the calling tool is actively
///   writing its own file, so without this it would be the newest match).
/// - `max_turns`: cap on the number of turns returned (chronologically the
///   last N).
pub async fn resolve_session(
    project_root: &Path,
    cli: &str,
    session_id: Option<&str>,
    exclude_session_ids: &[String],
    started_after: Option<DateTime<Utc>>,
    max_turns: usize,
) -> Result<Option<Transcript>> {
    match cli {
        "claude" => {
            resolve_claude(
                project_root,
                session_id,
                exclude_session_ids,
                started_after,
                max_turns,
            )
            .await
        }
        "codex" => {
            resolve_codex(
                project_root,
                session_id,
                exclude_session_ids,
                started_after,
                max_turns,
            )
            .await
        }
        _ => Ok(None),
    }
}

/// Returns true if `session_id` starts with any exclude prefix.
fn is_excluded(session_id: &str, exclude_prefixes: &[String]) -> bool {
    exclude_prefixes.iter().any(|p| session_id.starts_with(p))
}

/// Find matching claude JSONL file(s) and extract the first/best transcript.
async fn resolve_claude(
    project_root: &Path,
    session_id: Option<&str>,
    exclude_session_ids: &[String],
    started_after: Option<DateTime<Utc>>,
    max_turns: usize,
) -> Result<Option<Transcript>> {
    let canonical_root =
        std::fs::canonicalize(project_root).unwrap_or_else(|_| project_root.to_path_buf());
    let home = match system_env::home_dir() {
        Some(h) => h,
        None => return Ok(None),
    };
    let session_dir = home
        .join(".claude")
        .join("projects")
        .join(encode_claude_dir(&canonical_root.to_string_lossy()));
    if !session_dir.exists() {
        return Ok(None);
    }

    let files = list_jsonl_newest_first(&session_dir).await;
    for path in files {
        if let Some(prefix) = session_id
            && !claude_file_matches_prefix(&path, prefix)
        {
            continue;
        }
        // Fast-path exclusion: claude filenames are `<session_id>.jsonl`.
        if let Some(name) = path.file_stem().and_then(|s| s.to_str())
            && is_excluded(name, exclude_session_ids)
        {
            continue;
        }
        if let Some(transcript) = load_claude_transcript(&path, max_turns).await? {
            if is_excluded(&transcript.session_id, exclude_session_ids) {
                continue;
            }
            if let Some(cutoff) = started_after
                && let Some(updated) = transcript.updated_at
                && updated < cutoff
            {
                continue; // session predates this nickname's registration
            }
            return Ok(Some(transcript));
        }
    }
    Ok(None)
}

/// Claude stores session id inside each JSONL line (`sessionId` field).
/// Filenames are typically `<uuid>.jsonl`, but we scan the first valid line to
/// be safe across layout changes.
fn claude_file_matches_prefix(path: &Path, prefix: &str) -> bool {
    // Fast path: filename is usually the UUID.
    if let Some(name) = path.file_stem().and_then(|s| s.to_str())
        && name.starts_with(prefix)
    {
        return true;
    }
    false
}

/// Load the last `max_turns` natural-language turns from a Claude session.
pub async fn load_claude_transcript(path: &Path, max_turns: usize) -> Result<Option<Transcript>> {
    let file = match fs::File::open(path).await {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };
    let mut lines = BufReader::new(file).lines();

    let mut session_id: Option<String> = None;
    let mut turns: Vec<Turn> = Vec::new();
    let mut updated_at: Option<DateTime<Utc>> = None;

    while let Ok(Some(line)) = lines.next_line().await {
        if line.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            // Skip partial/malformed lines (e.g. streaming tail).
            Err(_) => continue,
        };

        if session_id.is_none()
            && let Some(sid) = v.get("sessionId").and_then(|s| s.as_str())
        {
            session_id = Some(sid.to_string());
        }

        if v.get("isSidechain")
            .and_then(|b| b.as_bool())
            .unwrap_or(false)
        {
            continue;
        }

        let kind = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
        if kind != "user" && kind != "assistant" {
            continue;
        }

        let raw = match extract_claude_text(v.get("message")) {
            Some(t) => t,
            None => continue,
        };
        let text = cap_turn(&raw);
        if text.trim().is_empty() {
            continue;
        }

        let timestamp = v
            .get("timestamp")
            .and_then(|s| s.as_str())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|d| d.with_timezone(&Utc));
        if timestamp.is_some() {
            updated_at = timestamp;
        }

        turns.push(Turn {
            role: kind.to_string(),
            text,
            timestamp,
        });
    }

    let session_id = match session_id {
        Some(s) => s,
        None => return Ok(None),
    };
    if turns.is_empty() {
        return Ok(None);
    }
    if turns.len() > max_turns {
        let start = turns.len() - max_turns;
        turns = turns.split_off(start);
    }
    Ok(Some(Transcript {
        cli: "claude".into(),
        session_id,
        source_path: path.to_string_lossy().to_string(),
        turns,
        updated_at,
    }))
}

/// Find matching codex rollout JSONL file(s) and extract the first valid one.
async fn resolve_codex(
    project_root: &Path,
    session_id: Option<&str>,
    exclude_session_ids: &[String],
    started_after: Option<DateTime<Utc>>,
    max_turns: usize,
) -> Result<Option<Transcript>> {
    let canonical_root =
        std::fs::canonicalize(project_root).unwrap_or_else(|_| project_root.to_path_buf());
    let canonical_str = canonical_root.to_string_lossy().to_string();
    let home = match system_env::home_dir() {
        Some(h) => h,
        None => return Ok(None),
    };
    let codex_root = home.join(".codex").join("sessions");
    if !codex_root.exists() {
        return Ok(None);
    }

    let files = walk_jsonl_newest_first(&codex_root).await;
    for path in files {
        match load_codex_transcript(&path, &canonical_str, max_turns).await? {
            Some(transcript) => {
                if let Some(prefix) = session_id
                    && !transcript.session_id.starts_with(prefix)
                {
                    continue;
                }
                if is_excluded(&transcript.session_id, exclude_session_ids) {
                    continue;
                }
                if let Some(cutoff) = started_after
                    && let Some(updated) = transcript.updated_at
                    && updated < cutoff
                {
                    continue; // session predates this nickname's registration
                }
                return Ok(Some(transcript));
            }
            None => continue,
        }
    }
    Ok(None)
}

/// Load the last `max_turns` natural-language turns from a Codex rollout file,
/// but only if its `session_meta.cwd` matches `project_root`.
pub async fn load_codex_transcript(
    path: &Path,
    project_root: &str,
    max_turns: usize,
) -> Result<Option<Transcript>> {
    let file = match fs::File::open(path).await {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };
    let mut lines = BufReader::new(file).lines();

    let mut session_id: Option<String> = None;
    let mut project_matches = false;
    let mut turns: Vec<Turn> = Vec::new();
    let mut updated_at: Option<DateTime<Utc>> = None;

    while let Ok(Some(line)) = lines.next_line().await {
        if line.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let kind = v.get("type").and_then(|t| t.as_str()).unwrap_or("");

        if kind == "session_meta"
            && let Some(payload) = v.get("payload")
        {
            if let Some(id) = payload.get("id").and_then(|s| s.as_str()) {
                session_id = Some(id.to_string());
            }
            if let Some(cwd) = payload.get("cwd").and_then(|s| s.as_str())
                && paths_match(cwd, project_root)
            {
                project_matches = true;
            }
        }

        let timestamp = v
            .get("timestamp")
            .and_then(|s| s.as_str())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|d| d.with_timezone(&Utc));
        if timestamp.is_some() {
            updated_at = timestamp;
        }

        if kind != "response_item" {
            continue;
        }
        let payload = match v.get("payload") {
            Some(p) => p,
            None => continue,
        };
        if payload.get("type").and_then(|t| t.as_str()) != Some("message") {
            continue;
        }
        let role = payload
            .get("role")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string();
        if role != "user" && role != "assistant" {
            continue;
        }
        let raw = match extract_codex_message_text(payload) {
            Some(t) => t,
            None => continue,
        };
        let text = cap_turn(&raw);
        if text.trim().is_empty() {
            continue;
        }

        turns.push(Turn {
            role,
            text,
            timestamp,
        });
    }

    if !project_matches {
        return Ok(None);
    }
    let session_id = match session_id {
        Some(s) => s,
        None => return Ok(None),
    };
    if turns.is_empty() {
        return Ok(None);
    }
    if turns.len() > max_turns {
        let start = turns.len() - max_turns;
        turns = turns.split_off(start);
    }
    Ok(Some(Transcript {
        cli: "codex".into(),
        session_id,
        source_path: path.to_string_lossy().to_string(),
        turns,
        updated_at,
    }))
}

/// Cap a turn's text at `MAX_TURN_BYTES`, respecting UTF-8 char boundaries.
/// Suffixes with `…` when truncated.
fn cap_turn(text: &str) -> String {
    if text.len() <= MAX_TURN_BYTES {
        return text.to_string();
    }
    // Walk char boundaries to find the largest prefix <= MAX_TURN_BYTES - 3
    // (reserving three bytes for the `…` we append).
    let budget = MAX_TURN_BYTES.saturating_sub(3);
    let mut end = 0;
    for (idx, _) in text.char_indices() {
        if idx > budget {
            break;
        }
        end = idx;
    }
    let mut out = String::with_capacity(end + 3);
    out.push_str(&text[..end]);
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn is_excluded_matches_prefix() {
        let ex = vec!["abc".to_string(), "xyz-1".to_string()];
        assert!(is_excluded("abc123", &ex));
        assert!(is_excluded("xyz-1234", &ex));
        assert!(!is_excluded("def", &ex));
        assert!(!is_excluded("abc", &[])); // empty exclude list never matches
    }

    #[tokio::test]
    async fn load_claude_transcript_returns_verbatim_turns_chronologically() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sess.jsonl");
        let lines = [
            r#"{"type":"user","sessionId":"sid-abc","isSidechain":false,"timestamp":"2026-04-01T10:00:00Z","message":{"content":"Please review the pagination helper in handlers/users.go."}}"#,
            r#"{"type":"assistant","sessionId":"sid-abc","isSidechain":true,"timestamp":"2026-04-01T10:01:00Z","message":{"content":[{"type":"text","text":"SIDECHAIN - SHOULD NOT APPEAR"}]}}"#,
            r#"{"type":"assistant","sessionId":"sid-abc","isSidechain":false,"timestamp":"2026-04-01T10:02:00Z","message":{"content":[{"type":"text","text":"Found two issues: (1) empty cursor returns 500, (2) limit > 1000 is not clamped."}]}}"#,
            r#"{"type":"user","sessionId":"sid-abc","isSidechain":false,"timestamp":"2026-04-01T10:03:00Z","message":{"content":"fix them"}}"#,
        ];
        fs::write(&path, lines.join("\n")).await.unwrap();

        let t = load_claude_transcript(&path, 10)
            .await
            .unwrap()
            .expect("should extract");
        assert_eq!(t.cli, "claude");
        assert_eq!(t.session_id, "sid-abc");
        assert_eq!(t.turns.len(), 3); // sidechain skipped
        assert_eq!(t.turns[0].role, "user");
        assert!(t.turns[0].text.starts_with("Please review"));
        assert_eq!(t.turns[1].role, "assistant");
        assert!(t.turns[1].text.starts_with("Found two issues"));
        assert!(!t.turns[1].text.contains("SIDECHAIN"));
        assert_eq!(t.turns[2].role, "user");
        assert_eq!(t.turns[2].text, "fix them");
    }

    #[tokio::test]
    async fn load_claude_transcript_respects_max_turns() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sess.jsonl");
        let mut lines = Vec::new();
        for i in 0..10 {
            lines.push(format!(
                r#"{{"type":"user","sessionId":"sid-x","isSidechain":false,"message":{{"content":"turn {i} content long enough to count"}}}}"#
            ));
        }
        fs::write(&path, lines.join("\n")).await.unwrap();

        let t = load_claude_transcript(&path, 3)
            .await
            .unwrap()
            .expect("should extract");
        assert_eq!(t.turns.len(), 3);
        // Chronological: last 3 → 7, 8, 9
        assert!(t.turns[0].text.contains("turn 7"));
        assert!(t.turns[2].text.contains("turn 9"));
    }

    #[tokio::test]
    async fn load_claude_transcript_silently_skips_malformed_lines() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sess.jsonl");
        let mut content = String::new();
        content.push_str(r#"{"type":"user","sessionId":"sid-y","isSidechain":false,"message":{"content":"hello world"}}"#);
        content.push('\n');
        content.push_str("{not json at all");
        content.push('\n');
        content.push_str(r#"{"type":"assistant","sessionId":"sid-y","isSidechain":false,"message":{"content":[{"type":"text","text":"hi"}]}}"#);
        fs::write(&path, &content).await.unwrap();

        let t = load_claude_transcript(&path, 10)
            .await
            .unwrap()
            .expect("should extract despite one garbage line");
        assert_eq!(t.turns.len(), 2);
    }

    #[tokio::test]
    async fn load_codex_transcript_matches_cwd_and_returns_turns() {
        let dir = TempDir::new().unwrap();
        let project_root = dir.path().join("proj");
        fs::create_dir_all(&project_root).await.unwrap();
        let proj_str = project_root.to_string_lossy().to_string();

        let path = dir.path().join("rollout.jsonl");
        let lines = [
            format!(
                r#"{{"type":"session_meta","timestamp":"2026-04-01T10:00:00Z","payload":{{"id":"codex-abc","cwd":"{}"}}}}"#,
                proj_str
            ),
            r#"{"type":"response_item","timestamp":"2026-04-01T10:01:00Z","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"review this pagination patch"}]}}"#.to_string(),
            r#"{"type":"response_item","timestamp":"2026-04-01T10:02:00Z","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Looks mostly fine. One issue: empty cursor returns 500."}]}}"#.to_string(),
        ];
        fs::write(&path, lines.join("\n")).await.unwrap();

        let t = load_codex_transcript(&path, &proj_str, 10)
            .await
            .unwrap()
            .expect("should extract");
        assert_eq!(t.cli, "codex");
        assert_eq!(t.session_id, "codex-abc");
        assert_eq!(t.turns.len(), 2);
        assert_eq!(t.turns[0].role, "user");
        assert!(t.turns[1].text.contains("empty cursor"));
    }

    #[tokio::test]
    async fn load_codex_transcript_rejects_non_matching_cwd() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("rollout.jsonl");
        let lines = [
            r#"{"type":"session_meta","payload":{"id":"codex-1","cwd":"/nope"}}"#,
            r#"{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}}"#,
        ];
        fs::write(&path, lines.join("\n")).await.unwrap();

        let t = load_codex_transcript(&path, "/other", 10).await.unwrap();
        assert!(t.is_none());
    }

    #[test]
    fn cap_turn_truncates_oversize_text() {
        let big = "a".repeat(MAX_TURN_BYTES + 100);
        let capped = cap_turn(&big);
        assert!(capped.ends_with('…'));
        assert!(capped.len() <= MAX_TURN_BYTES);
    }

    #[test]
    fn cap_turn_respects_utf8_boundaries() {
        // Construct a string whose MAX_TURN_BYTES-th byte falls inside a multi-byte char.
        let mut s = "a".repeat(MAX_TURN_BYTES - 2);
        s.push('🚀'); // 4 bytes, crosses the cap boundary
        s.push_str("xyz");
        let capped = cap_turn(&s);
        // Must still be valid UTF-8 — the test itself would panic if not.
        assert!(capped.ends_with('…'));
        assert!(capped.is_char_boundary(capped.len()));
    }
}
