use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::fs;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::services::model_names::normalize_claude_version;
use crate::services::system_env;

/// Aggregated stats from a tool's native data files.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct GlobalToolStats {
    pub sessions: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    pub models: HashMap<String, ModelTokens>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ModelTokens {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl GlobalToolStats {
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens.saturating_add(self.output_tokens)
    }
}

// ---------------------------------------------------------------------------
// Per-file cache: stores stats per file keyed by path, with file size for
// change detection. Only files whose size changed get re-parsed.
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Default)]
struct FileEntry {
    size: u64,
    input_tokens: u64,
    output_tokens: u64,
    cache_read_tokens: u64,
    cache_write_tokens: u64,
    models: HashMap<String, (u64, u64)>, // model -> (input, output)
    has_session: bool,
}

#[derive(Serialize, Deserialize, Default)]
struct StatsCache {
    files: HashMap<String, FileEntry>,
}

/// Collect global stats for all known tools sequentially.
/// Sequential avoids progress line flickering (all tools share one stderr line).
/// Returns a map of tool name → stats (only tools with data).
pub async fn collect_all(refresh: bool) -> HashMap<String, GlobalToolStats> {
    let tools = ["claude", "codex", "gemini", "opencode", "pi"];
    let total_tools = tools.len();
    let mut result = HashMap::new();
    for (i, tool) in tools.iter().enumerate() {
        let step = Some((i + 1, total_tools));
        if let Ok(Some(stats)) = collect_with_step(tool, refresh, step).await
            && (stats.total_tokens() > 0 || stats.sessions > 0)
        {
            result.insert(tool.to_string(), stats);
        }
    }
    result
}

pub async fn collect(tool: &str, refresh: bool) -> Result<Option<GlobalToolStats>> {
    collect_with_step(tool, refresh, None).await
}

async fn collect_with_step(
    tool: &str,
    refresh: bool,
    step: Option<(usize, usize)>,
) -> Result<Option<GlobalToolStats>> {
    if !matches!(tool, "claude" | "codex" | "gemini") {
        return match tool {
            "opencode" => collect_opencode().await,
            "pi" => collect_pi().await,
            _ => Ok(None),
        };
    }

    let data_dir = match tool_data_dir(tool) {
        Some(d) if d.exists() => d,
        _ => return Ok(None),
    };

    let filter = tool_file_filter(tool);
    let cache_path = cache_path(tool);
    let mut cache = if refresh {
        StatsCache::default()
    } else {
        read_cache(&cache_path).await.unwrap_or_default()
    };

    // Walk files and collect paths + sizes
    let all_files = walk_files_with_size(&data_dir, filter).await;
    if all_files.is_empty() {
        return Ok(None);
    }

    // Find stale files (new or size changed)
    let current_paths: HashSet<&str> = all_files
        .iter()
        .map(|(p, _)| p.to_str().unwrap_or(""))
        .collect();

    let mut stale: Vec<(&Path, u64)> = Vec::new();
    for (path, size) in &all_files {
        let key = path.to_string_lossy();
        match cache.files.get(key.as_ref()) {
            Some(cached) if cached.size == *size => {} // unchanged
            _ => stale.push((path, *size)),
        }
    }

    // Remove deleted files from cache
    cache
        .files
        .retain(|k, _| current_paths.contains(k.as_str()));

    // Re-parse stale files
    if !stale.is_empty() {
        let total = stale.len();
        let parser = tool_file_parser(tool);

        let show_progress = total > 5;
        let update_interval = (total / 50).max(1);
        if show_progress {
            print_progress(0, total, step);
        }

        for (i, (path, size)) in stale.iter().enumerate() {
            if let Some(entry) = parser(path).await {
                cache.files.insert(
                    path.to_string_lossy().to_string(),
                    FileEntry {
                        size: *size,
                        ..entry
                    },
                );
            }
            if show_progress && ((i + 1) % update_interval == 0 || i + 1 == total) {
                print_progress(i + 1, total, step);
            }
        }

        if show_progress {
            eprint!("\r{:<30}\r", "");
        }
        let _ = write_cache(&cache_path, &cache).await;
    }

    // Aggregate from all cached file entries
    let stats = aggregate_cache(&cache);
    if stats.sessions == 0 && stats.total_tokens() == 0 {
        return Ok(None);
    }
    Ok(Some(stats))
}

fn aggregate_cache(cache: &StatsCache) -> GlobalToolStats {
    let mut stats = GlobalToolStats::default();

    for entry in cache.files.values() {
        stats.input_tokens += entry.input_tokens;
        stats.output_tokens += entry.output_tokens;
        stats.cache_read_tokens += entry.cache_read_tokens;
        stats.cache_write_tokens += entry.cache_write_tokens;
        if entry.has_session {
            stats.sessions += 1;
        }
        for (model, (inp, out)) in &entry.models {
            let m = stats.models.entry(model.clone()).or_default();
            m.input_tokens += inp;
            m.output_tokens += out;
        }
    }

    stats
}

// ---------------------------------------------------------------------------
// Infrastructure helpers
// ---------------------------------------------------------------------------

fn tool_data_dir(tool: &str) -> Option<PathBuf> {
    let home = system_env::home_dir()?;
    match tool {
        "claude" => Some(home.join(".claude").join("projects")),
        "codex" => Some(home.join(".codex").join("sessions")),
        "gemini" => Some(home.join(".gemini").join("tmp")),
        _ => None,
    }
}

fn cache_path(tool: &str) -> PathBuf {
    system_env::home_dir()
        .map(|p| {
            p.join(".config")
                .join("aivo")
                .join(format!("stats-cache-{tool}.json"))
        })
        .unwrap_or_else(|| PathBuf::from(format!("stats-cache-{tool}.json")))
}

fn tool_file_filter(tool: &str) -> fn(&str) -> bool {
    match tool {
        "claude" | "codex" => |name: &str| name.ends_with(".jsonl"),
        "gemini" => |name: &str| name.starts_with("session-") && name.ends_with(".json"),
        _ => |_: &str| true,
    }
}

type FileParser =
    fn(
        &Path,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<FileEntry>> + Send + '_>>;

fn tool_file_parser(tool: &str) -> FileParser {
    match tool {
        "claude" => |p| Box::pin(parse_claude_file(p)),
        "codex" => |p| Box::pin(parse_codex_file(p)),
        "gemini" => |p| Box::pin(parse_gemini_file(p)),
        _ => |_| Box::pin(async { None }),
    }
}

/// Walk directory recursively, returning matching files with their sizes.
async fn walk_files_with_size(root: &Path, filter: fn(&str) -> bool) -> Vec<(PathBuf, u64)> {
    let mut result = Vec::new();
    let mut dirs = vec![root.to_path_buf()];

    while let Some(dir) = dirs.pop() {
        let mut entries = match fs::read_dir(&dir).await {
            Ok(e) => e,
            Err(_) => continue,
        };
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if path.is_dir() {
                dirs.push(path);
            } else if let Some(name) = path.file_name().and_then(|n| n.to_str())
                && filter(name)
            {
                let size = entry.metadata().await.map(|m| m.len()).unwrap_or(0);
                result.push((path, size));
            }
        }
    }

    result
}

fn print_progress(current: usize, total: usize, step: Option<(usize, usize)>) {
    let pct = if total > 0 {
        (current * 100) / total
    } else {
        0
    };
    let step_prefix = match step {
        Some((i, n)) => format!("({i}/{n}) "),
        None => String::new(),
    };
    eprint!(
        "\r{}{} {pct:>3}%",
        step_prefix,
        crate::style::dim("reading")
    );
}

async fn read_cache(path: &Path) -> Option<StatsCache> {
    let data = fs::read_to_string(path).await.ok()?;
    serde_json::from_str(&data).ok()
}

async fn write_cache(path: &Path, cache: &StatsCache) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await?;
    }
    let data = serde_json::to_string(cache)?;
    fs::write(path, data).await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Per-file parsers — return FileEntry for a single file
// ---------------------------------------------------------------------------

/// Parse a single Claude Code JSONL file.
async fn parse_claude_file(path: &Path) -> Option<FileEntry> {
    let file = fs::File::open(path).await.ok()?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let mut entry = FileEntry::default();
    let mut seen_session = false;

    while let Ok(Some(line)) = lines.next_line().await {
        // Fast pre-filter: skip full JSON parse for non-assistant lines
        if !line.contains("\"type\":\"assistant\"") {
            continue;
        }
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let usage = match v.get("message").and_then(|m| m.get("usage")) {
            Some(u) => u,
            None => continue,
        };
        let input = usage
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let output = usage
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let cache_read = usage
            .get("cache_read_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let cache_write = usage
            .get("cache_creation_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        entry.input_tokens += input;
        entry.output_tokens += output;
        entry.cache_read_tokens += cache_read;
        entry.cache_write_tokens += cache_write;

        if !seen_session && v.get("sessionId").and_then(|s| s.as_str()).is_some() {
            seen_session = true;
            entry.has_session = true;
        }
        if let Some(model) = v
            .get("message")
            .and_then(|m| m.get("model"))
            .and_then(|m| m.as_str())
        {
            let key = normalize_model_for_display(model);
            let e = entry.models.entry(key).or_default();
            e.0 += input;
            e.1 += output;
        }
    }

    Some(entry)
}

/// Parse a single Codex JSONL file.
async fn parse_codex_file(path: &Path) -> Option<FileEntry> {
    let file = fs::File::open(path).await.ok()?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let mut last_input = 0u64;
    let mut last_output = 0u64;
    let mut last_cached = 0u64;
    let mut has_tokens = false;
    let mut model: Option<String> = None;

    while let Ok(Some(line)) = lines.next_line().await {
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        if v.get("type").and_then(|t| t.as_str()) == Some("turn_context")
            && let Some(m) = v
                .get("payload")
                .and_then(|p| p.get("model"))
                .and_then(|m| m.as_str())
        {
            model = Some(m.to_string());
        }

        if v.get("type").and_then(|t| t.as_str()) != Some("event_msg") {
            continue;
        }
        let payload = match v.get("payload") {
            Some(p) => p,
            None => continue,
        };
        if payload.get("type").and_then(|t| t.as_str()) != Some("token_count") {
            continue;
        }
        let info = match payload.get("info") {
            Some(Value::Object(_)) => payload.get("info").unwrap(),
            _ => continue,
        };
        let usage = match info.get("total_token_usage") {
            Some(u) => u,
            None => continue,
        };

        last_input = usage
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        last_output = usage
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        last_cached = usage
            .get("cached_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        has_tokens = true;
    }

    let mut entry = FileEntry {
        has_session: has_tokens,
        input_tokens: last_input,
        output_tokens: last_output,
        cache_read_tokens: last_cached,
        ..Default::default()
    };

    if has_tokens && let Some(ref m) = model {
        let key = normalize_model_for_display(m);
        entry.models.insert(key, (last_input, last_output));
    }

    Some(entry)
}

/// Parse a single Gemini session JSON file.
async fn parse_gemini_file(path: &Path) -> Option<FileEntry> {
    let content = fs::read_to_string(path).await.ok()?;
    let v: Value = serde_json::from_str(&content).ok()?;
    let messages = v.get("messages")?.as_array()?;

    let mut entry = FileEntry {
        has_session: true,
        ..Default::default()
    };

    for msg in messages {
        if msg.get("type").and_then(|t| t.as_str()) != Some("gemini") {
            continue;
        }
        let tokens = match msg.get("tokens") {
            Some(t) => t,
            None => continue,
        };

        let input = tokens.get("input").and_then(|v| v.as_u64()).unwrap_or(0);
        let output = tokens.get("output").and_then(|v| v.as_u64()).unwrap_or(0);
        let cached = tokens.get("cached").and_then(|v| v.as_u64()).unwrap_or(0);

        entry.input_tokens += input;
        entry.output_tokens += output;
        entry.cache_read_tokens += cached;

        if let Some(model) = msg.get("model").and_then(|m| m.as_str()) {
            let key = normalize_model_for_display(model);
            let e = entry.models.entry(key).or_default();
            e.0 += input;
            e.1 += output;
        }
    }

    Some(entry)
}

// ---------------------------------------------------------------------------
// Non-cached tool collectors (OpenCode via SQLite, Pi)
// ---------------------------------------------------------------------------

/// OpenCode: ~/.local/share/opencode/opencode.db (SQLite via sqlite3 CLI)
async fn collect_opencode() -> Result<Option<GlobalToolStats>> {
    let home = match system_env::home_dir() {
        Some(h) => h,
        None => return Ok(None),
    };

    let db_path = home
        .join(".local")
        .join("share")
        .join("opencode")
        .join("opencode.db");
    if !db_path.exists() {
        return Ok(None);
    }

    let query = "SELECT session_id, json_extract(data, '$.modelID'), json_extract(data, '$.tokens.input'), json_extract(data, '$.tokens.output'), json_extract(data, '$.tokens.cache.read'), json_extract(data, '$.tokens.cache.write') FROM message WHERE json_extract(data, '$.role') = 'assistant' AND json_extract(data, '$.tokens') IS NOT NULL;";

    let output = tokio::process::Command::new("sqlite3")
        .arg("-separator")
        .arg("\t")
        .arg(&db_path)
        .arg(query)
        .output()
        .await;

    let output = match output {
        Ok(o) if o.status.success() => o,
        _ => return Ok(None), // sqlite3 not found or query failed
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut stats = GlobalToolStats::default();
    let mut session_ids = HashSet::new();

    for line in stdout.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 6 {
            continue;
        }
        let session_id = cols[0];
        let model = cols[1];
        let input: u64 = cols[2].parse().unwrap_or(0);
        let output: u64 = cols[3].parse().unwrap_or(0);
        let cache_read: u64 = cols[4].parse().unwrap_or(0);
        let cache_write: u64 = cols[5].parse().unwrap_or(0);

        session_ids.insert(session_id.to_string());
        stats.input_tokens += input;
        stats.output_tokens += output;
        stats.cache_read_tokens += cache_read;
        stats.cache_write_tokens += cache_write;

        if !model.is_empty() {
            let key = normalize_model_for_display(model);
            let entry = stats.models.entry(key).or_default();
            entry.input_tokens += input;
            entry.output_tokens += output;
        }
    }

    stats.sessions = session_ids.len() as u64;
    if stats.sessions == 0 {
        return Ok(None);
    }
    Ok(Some(stats))
}

/// Pi: ~/.pi/agent/sessions/**/*.jsonl
async fn collect_pi() -> Result<Option<GlobalToolStats>> {
    let home = match system_env::home_dir() {
        Some(h) => h,
        None => return Ok(None),
    };

    let data_dir = home.join(".pi").join("agent").join("sessions");
    if !data_dir.exists() {
        return Ok(None);
    }

    let files = walk_files_with_size(&data_dir, |name| name.ends_with(".jsonl")).await;
    if files.is_empty() {
        return Ok(None);
    }

    let mut stats = GlobalToolStats::default();
    let mut session_ids = HashSet::new();

    for (path, _) in &files {
        let file = match fs::File::open(path).await {
            Ok(f) => f,
            Err(_) => continue,
        };
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        while let Ok(Some(line)) = lines.next_line().await {
            let v: Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => continue,
            };

            if v.get("type").and_then(|t| t.as_str()) == Some("session")
                && let Some(sid) = v.get("id").and_then(|s| s.as_str())
            {
                session_ids.insert(sid.to_string());
            }

            if v.get("type").and_then(|t| t.as_str()) != Some("message") {
                continue;
            }

            let usage = match v.get("message").and_then(|m| m.get("usage")) {
                Some(u) => u,
                None => continue,
            };

            let input = usage.get("input").and_then(|v| v.as_u64()).unwrap_or(0);
            let output = usage.get("output").and_then(|v| v.as_u64()).unwrap_or(0);
            let cache_read = usage.get("cacheRead").and_then(|v| v.as_u64()).unwrap_or(0);
            let cache_write = usage
                .get("cacheWrite")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);

            stats.input_tokens += input;
            stats.output_tokens += output;
            stats.cache_read_tokens += cache_read;
            stats.cache_write_tokens += cache_write;

            if let Some(model) = v
                .get("message")
                .and_then(|m| m.get("model"))
                .and_then(|m| m.as_str())
            {
                let key = normalize_model_for_display(model);
                let entry = stats.models.entry(key).or_default();
                entry.input_tokens += input;
                entry.output_tokens += output;
            }
        }
    }

    stats.sessions = session_ids.len() as u64;
    Ok(Some(stats))
}

// ---------------------------------------------------------------------------
// Shared utilities
// ---------------------------------------------------------------------------

/// Normalize a model name for display and merging.
/// Strips provider prefixes, normalizes version separators, lowercases.
pub fn normalize_model_for_display(model: &str) -> String {
    let base = if let Some(pos) = model.rfind('/') {
        &model[pos + 1..]
    } else {
        model
    };
    let normalized = normalize_claude_version(base);
    normalized.to_lowercase()
}

/// Display name for each tool.
pub fn tool_display_name(tool: &str) -> &str {
    match tool {
        "claude" => "Claude Code",
        "codex" => "Codex",
        "gemini" => "Gemini",
        "opencode" => "OpenCode",
        "pi" => "Pi",
        "chat" => "Chat",
        _ => tool,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_claude_line(line: &str) -> (u64, u64, u64, u64, Option<String>) {
        let v: Value = serde_json::from_str(line).unwrap();
        if v.get("type").and_then(|t| t.as_str()) != Some("assistant") {
            return (0, 0, 0, 0, None);
        }
        let usage = match v.get("message").and_then(|m| m.get("usage")) {
            Some(u) => u,
            None => return (0, 0, 0, 0, None),
        };
        let input = usage
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let output = usage
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let cr = usage
            .get("cache_read_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let cw = usage
            .get("cache_creation_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let model = v
            .get("message")
            .and_then(|m| m.get("model"))
            .and_then(|m| m.as_str())
            .map(String::from);
        (input, output, cr, cw, model)
    }

    #[test]
    fn claude_line_with_usage() {
        let line = r#"{"type":"assistant","message":{"model":"claude-sonnet-4-20250514","usage":{"input_tokens":100,"output_tokens":50,"cache_read_input_tokens":20,"cache_creation_input_tokens":10}},"sessionId":"abc"}"#;
        let (i, o, cr, cw, model) = parse_claude_line(line);
        assert_eq!(i, 100);
        assert_eq!(o, 50);
        assert_eq!(cr, 20);
        assert_eq!(cw, 10);
        assert_eq!(model.as_deref(), Some("claude-sonnet-4-20250514"));
    }

    #[test]
    fn claude_line_without_usage() {
        let line = r#"{"type":"assistant","message":{"model":"claude-sonnet-4-20250514"},"sessionId":"abc"}"#;
        let (i, o, cr, cw, _) = parse_claude_line(line);
        assert_eq!((i, o, cr, cw), (0, 0, 0, 0));
    }

    #[test]
    fn claude_line_non_assistant() {
        let line = r#"{"type":"progress","data":{"type":"hook_progress"}}"#;
        let (i, o, cr, cw, model) = parse_claude_line(line);
        assert_eq!((i, o, cr, cw), (0, 0, 0, 0));
        assert!(model.is_none());
    }

    #[test]
    fn gemini_message_parsing() {
        let json = r#"{"sessionId":"s1","messages":[
            {"type":"user","content":"hi"},
            {"type":"gemini","content":"hello","tokens":{"input":100,"output":50,"cached":20,"thoughts":10,"tool":0}},
            {"type":"gemini","content":"bye","tokens":{"input":200,"output":100,"cached":0,"thoughts":5,"tool":0}}
        ]}"#;
        let v: Value = serde_json::from_str(json).unwrap();
        let messages = v.get("messages").unwrap().as_array().unwrap();
        let mut total_input = 0u64;
        let mut total_output = 0u64;
        let mut total_cached = 0u64;
        for msg in messages {
            if msg.get("type").and_then(|t| t.as_str()) != Some("gemini") {
                continue;
            }
            if let Some(tokens) = msg.get("tokens") {
                total_input += tokens.get("input").and_then(|v| v.as_u64()).unwrap_or(0);
                total_output += tokens.get("output").and_then(|v| v.as_u64()).unwrap_or(0);
                total_cached += tokens.get("cached").and_then(|v| v.as_u64()).unwrap_or(0);
            }
        }
        assert_eq!(total_input, 300);
        assert_eq!(total_output, 150);
        assert_eq!(total_cached, 20);
    }

    #[test]
    fn pi_message_parsing() {
        let line = r#"{"type":"message","id":"x","message":{"role":"assistant","model":"gpt-5.2","usage":{"input":500,"output":200,"cacheRead":100,"cacheWrite":50,"totalTokens":700}}}"#;
        let v: Value = serde_json::from_str(line).unwrap();
        let usage = v.get("message").unwrap().get("usage").unwrap();
        assert_eq!(usage.get("input").unwrap().as_u64(), Some(500));
        assert_eq!(usage.get("output").unwrap().as_u64(), Some(200));
        assert_eq!(usage.get("cacheRead").unwrap().as_u64(), Some(100));
        assert_eq!(usage.get("cacheWrite").unwrap().as_u64(), Some(50));
    }

    #[test]
    fn codex_token_count_parsing() {
        let line = r#"{"type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":1000,"cached_input_tokens":500,"output_tokens":300,"reasoning_output_tokens":100,"total_tokens":1300},"model_context_window":258400},"rate_limits":null}}"#;
        let v: Value = serde_json::from_str(line).unwrap();
        let usage = v
            .get("payload")
            .unwrap()
            .get("info")
            .unwrap()
            .get("total_token_usage")
            .unwrap();
        assert_eq!(usage.get("input_tokens").unwrap().as_u64(), Some(1000));
        assert_eq!(usage.get("output_tokens").unwrap().as_u64(), Some(300));
        assert_eq!(
            usage.get("cached_input_tokens").unwrap().as_u64(),
            Some(500)
        );
    }

    #[test]
    fn codex_null_info_skipped() {
        let line = r#"{"type":"event_msg","payload":{"type":"token_count","info":null,"rate_limits":null}}"#;
        let v: Value = serde_json::from_str(line).unwrap();
        let info = v.get("payload").unwrap().get("info").unwrap();
        assert!(info.is_null());
    }

    #[test]
    fn normalize_model_strips_prefix_and_version() {
        assert_eq!(
            normalize_model_for_display("anthropic/claude-sonnet-4.6"),
            "claude-sonnet-4.6"
        );
        assert_eq!(
            normalize_model_for_display("claude-sonnet-4-6"),
            "claude-sonnet-4.6"
        );
        assert_eq!(
            normalize_model_for_display("anthropic/claude-sonnet-4-6"),
            "claude-sonnet-4.6"
        );
        assert_eq!(normalize_model_for_display("openai/gpt-5.4"), "gpt-5.4");
        assert_eq!(
            normalize_model_for_display("accounts/fireworks/models/kimi-k2-instruct-0905"),
            "kimi-k2-instruct-0905"
        );
        assert_eq!(normalize_model_for_display("MiniMax-M2.5"), "minimax-m2.5");
        assert_eq!(
            normalize_model_for_display("minimax/minimax-m2.5"),
            "minimax-m2.5"
        );
        assert_eq!(
            normalize_model_for_display("deepseek-chat"),
            "deepseek-chat"
        );
        assert_eq!(
            normalize_model_for_display("claude-haiku-4-5-20251001"),
            "claude-haiku-4-5-20251001"
        );
    }

    #[test]
    fn tool_display_names() {
        assert_eq!(tool_display_name("claude"), "Claude Code");
        assert_eq!(tool_display_name("codex"), "Codex");
        assert_eq!(tool_display_name("gemini"), "Gemini");
        assert_eq!(tool_display_name("pi"), "Pi");
        assert_eq!(tool_display_name("chat"), "Chat");
        assert_eq!(tool_display_name("unknown"), "unknown");
    }
}
