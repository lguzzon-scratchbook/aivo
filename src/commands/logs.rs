use anyhow::Result;
use serde_json::json;
use std::collections::HashSet;
use std::io::{self, Write};
use std::time::Duration;

use crate::cli::LogsArgs;
use crate::commands::chat::format_time_ago_short;
use crate::errors::ExitCode;
use crate::services::SessionStore;
use crate::services::log_store::{LogEntry, LogQuery};
use crate::style;

pub struct LogsCommand {
    session_store: SessionStore,
}

impl LogsCommand {
    pub fn new(session_store: SessionStore) -> Self {
        Self { session_store }
    }

    pub async fn execute(&self, args: LogsArgs) -> ExitCode {
        match self.execute_internal(args).await {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                ExitCode::UserError
            }
        }
    }

    async fn execute_internal(&self, mut args: LogsArgs) -> Result<ExitCode> {
        validate_args(&args)?;
        match args.action.as_deref() {
            Some("path") => self.show_path(&args).await,
            Some("status") => self.show_status(&args).await,
            Some("show") => self.show_entry(&args).await,
            Some(query) => {
                // Treat unknown action as a search query: `aivo logs claude`
                if args.search.is_none() {
                    args.search = Some(query.to_string());
                }
                self.list_entries(&args).await
            }
            None => self.list_entries(&args).await,
        }
    }

    async fn show_path(&self, args: &LogsArgs) -> Result<ExitCode> {
        ensure_no_target(args, "path")?;
        let path = self.session_store.logs().path().display().to_string();
        if args.json {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({ "path": path }))?
            );
        } else {
            println!("{}", path);
        }
        Ok(ExitCode::Success)
    }

    async fn show_status(&self, args: &LogsArgs) -> Result<ExitCode> {
        ensure_no_target(args, "status")?;
        let status = self.session_store.logs().status().await?;
        if args.json {
            println!("{}", serde_json::to_string_pretty(&status)?);
            return Ok(ExitCode::Success);
        }

        println!("{} {}", style::bold("Path:"), style::dim(&status.path));
        println!(
            "{} {}",
            style::bold("Entries:"),
            style::cyan(status.total_entries.to_string())
        );
        println!(
            "{} {} bytes",
            style::bold("Size:"),
            style::cyan(status.file_size_bytes.to_string())
        );
        if status.counts_by_source.is_empty() {
            println!("{}", style::dim("No log entries recorded yet."));
        } else {
            println!();
            println!("{}", style::bold("By Source:"));
            for row in status.counts_by_source {
                println!(
                    "  {} {}",
                    style::cyan(format!("{:<8}", row.source)),
                    style::cyan(row.count.to_string())
                );
            }
        }
        Ok(ExitCode::Success)
    }

    async fn show_entry(&self, args: &LogsArgs) -> Result<ExitCode> {
        let id = args
            .target
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("Usage: aivo logs show <id>"))?;
        let entry = self
            .session_store
            .logs()
            .get_by_reference(id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("No log entry with id '{}'", id))?;

        if args.json {
            println!("{}", serde_json::to_string_pretty(&entry)?);
            return Ok(ExitCode::Success);
        }

        print_entry(&entry);
        Ok(ExitCode::Success)
    }

    async fn list_entries(&self, args: &LogsArgs) -> Result<ExitCode> {
        if args.target.is_some() {
            anyhow::bail!("Unexpected target without an action. Use `aivo logs show <id>`");
        }
        if args.watch {
            return self.watch_entries(args).await;
        }
        let entries = self.fetch_entries(args).await?;

        if args.json {
            println!(
                "{}",
                serde_json::to_string_pretty(&order_entries(entries, args.latest_first))?
            );
            return Ok(ExitCode::Success);
        }

        render_text_entries(entries, args.limit, args.latest_first);
        Ok(ExitCode::Success)
    }

    async fn watch_entries(&self, args: &LogsArgs) -> Result<ExitCode> {
        let interval = Duration::from_secs_f32(args.interval.max(0.1));
        let mut seen_ids = HashSet::new();

        loop {
            let entries = self.fetch_entries(args).await?;

            if args.jsonl {
                let mut ordered = entries;
                ordered.reverse();
                for entry in ordered {
                    if seen_ids.insert(entry.id.clone()) {
                        println!("{}", serde_json::to_string(&entry)?);
                    }
                }
                io::stdout().flush()?;
            } else {
                print!("\x1b[2J\x1b[H");
                println!(
                    "{} {}",
                    style::bold("Watching logs"),
                    style::dim(format!(
                        "(refresh {}s, Ctrl+C to stop)",
                        args.interval.max(0.1)
                    ))
                );
                println!();
                render_text_entries(entries, args.limit, args.latest_first);
                io::stdout().flush()?;
            }

            tokio::time::sleep(interval).await;
        }
    }

    async fn fetch_entries(&self, args: &LogsArgs) -> Result<Vec<LogEntry>> {
        // Over-fetch to compensate for run event collapsing (start+finish pairs)
        let query_limit = if args.watch {
            args.limit.saturating_mul(5)
        } else if args.json {
            args.limit
        } else {
            args.limit.saturating_mul(3)
        };
        let entries = self
            .session_store
            .logs()
            .list(LogQuery {
                limit: query_limit,
                search: args.search.clone(),
                source: args.source.clone(),
                tool: args.tool.clone(),
                model: args.model.clone(),
                key_query: args.key.clone(),
                cwd: args.cwd.clone(),
                since: args.since.clone(),
                until: args.until.clone(),
                errors_only: args.errors,
            })
            .await?;
        Ok(entries)
    }

    pub fn print_help() {
        println!(
            "{} aivo logs [show <id>|path|status]",
            style::bold("Usage:")
        );
        println!();
        println!(
            "{}",
            style::dim("Query local SQLite logs for chat, run, and serve activity.")
        );
        println!();
        println!("{}", style::bold("Filters:"));
        let print_opt = |flag: &str, desc: &str| {
            println!(
                "  {}{}",
                style::cyan(format!("{:<26}", flag)),
                style::dim(desc)
            );
        };
        print_opt(
            "-n, --limit <N>",
            "Maximum number of rows to show (default: 20)",
        );
        print_opt("--json", "Output JSON");
        print_opt("--watch", "Continuously refresh matching logs");
        print_opt(
            "--interval <secs>",
            "Refresh interval for --watch (default: 1)",
        );
        print_opt("--jsonl", "Emit newly seen entries as JSONL while watching");
        print_opt("--latest-first", "Show newest entries first");
        print_opt("-s, --search <query>", "Search title/body text");
        print_opt("--source <source>", "Filter by source: chat, run, serve");
        print_opt("--tool <tool>", "Filter by tool name");
        print_opt("--model <model>", "Filter by model substring");
        print_opt("-k, --key <id|name>", "Filter by saved key ID or name");
        print_opt("--cwd <path>", "Filter by working directory substring");
        print_opt("--since <time>", "Only show entries on or after this time");
        print_opt("--until <time>", "Only show entries on or before this time");
        print_opt("--errors", "Only show HTTP >= 400 or non-zero exit code");
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo logs"));
        println!("  {}", style::dim("aivo logs --source chat -n 5"));
        println!("  {}", style::dim("aivo logs --tool claude --errors"));
        println!("  {}", style::dim("aivo logs --source run --watch"));
        println!("  {}", style::dim("aivo logs --watch --jsonl"));
        println!("  {}", style::dim("aivo logs show 7m2q8k4v9cpr"));
        println!("  {}", style::dim("aivo logs status"));
        println!("  {}", style::dim("aivo logs path"));
    }
}

fn ensure_no_target(args: &LogsArgs, action: &str) -> Result<()> {
    if args.target.is_some() {
        anyhow::bail!("`aivo logs {}` does not take a target", action);
    }
    Ok(())
}

fn validate_args(args: &LogsArgs) -> Result<()> {
    if args.interval <= 0.0 {
        anyhow::bail!("--interval must be greater than 0");
    }
    if args.jsonl && !args.watch {
        anyhow::bail!("--jsonl requires --watch");
    }
    if args.json && args.watch {
        anyhow::bail!("--json cannot be combined with --watch; use --jsonl for watch mode");
    }
    if args.json && args.jsonl {
        anyhow::bail!("--json and --jsonl cannot be combined");
    }
    if args.watch && args.action.is_some() {
        anyhow::bail!("--watch is only supported for `aivo logs` list output");
    }
    Ok(())
}

fn render_text_entries(entries: Vec<LogEntry>, limit: usize, latest_first: bool) {
    if entries.is_empty() {
        println!("{}", style::dim("No log entries found."));
        return;
    }

    let entries = order_entries(collapse_run_events(entries, limit), latest_first);
    for entry in entries {
        print_summary(&entry);
    }
}

fn order_entries(mut entries: Vec<LogEntry>, latest_first: bool) -> Vec<LogEntry> {
    if !latest_first {
        entries.reverse();
    }
    entries
}

fn print_summary(entry: &LogEntry) {
    let display_id = display_id(entry);
    let time_ago = format_time_ago_short(&entry.ts_utc);
    let detail = match entry.source.as_str() {
        "chat" => {
            let title = entry.title.clone().unwrap_or_else(|| "(chat)".to_string());
            let tokens = format_token_summary(entry);
            if tokens.is_empty() {
                title
            } else {
                format!("{title}  {tokens}")
            }
        }
        "run" => {
            let tool = entry.tool.as_deref().unwrap_or("run");
            let model = entry
                .model
                .clone()
                .unwrap_or_else(|| "(tool default)".to_string());
            let state = match entry.phase.as_deref() {
                Some("started") => "running".to_string(),
                _ => entry
                    .exit_code
                    .map(|code| format!("exit={code}"))
                    .unwrap_or_else(|| "exit=?".to_string()),
            };
            let duration = entry
                .duration_ms
                .map(|ms| format!(" ({})", format_duration_ms(ms)))
                .unwrap_or_default();
            format!("{tool} {model} {state}{duration}")
        }
        "serve" => {
            let title = entry.title.clone().unwrap_or_else(|| "request".to_string());
            let status = entry
                .status_code
                .map(|code| code.to_string())
                .unwrap_or_else(|| "?".to_string());
            let duration = entry
                .duration_ms
                .map(|ms| format!(" ({})", format_duration_ms(ms)))
                .unwrap_or_default();
            format!("{title} status={status}{duration}")
        }
        _ => entry.title.clone().unwrap_or_else(|| entry.kind.clone()),
    };
    println!(
        "{} {} {} {}",
        style::cyan(display_id),
        style::dim(format!("{:>5}", time_ago)),
        style::yellow(format!("[{}]", entry.source)),
        detail
    );
}

fn format_token_summary(entry: &LogEntry) -> String {
    match (entry.input_tokens, entry.output_tokens) {
        (Some(input), Some(output)) if input > 0 || output > 0 => {
            style::dim(format!("({input}\u{2192}{output} tokens)"))
        }
        _ => String::new(),
    }
}

fn format_duration_ms(ms: i64) -> String {
    let ms = ms.unsigned_abs();
    match ms {
        0..=999 => format!("{ms}ms"),
        1000..=59_999 => format!("{:.1}s", ms as f64 / 1000.0),
        60_000..=3_599_999 => {
            let minutes = ms / 60_000;
            let seconds = (ms % 60_000) / 1000;
            if seconds == 0 {
                format!("{minutes}m")
            } else {
                format!("{minutes}m {seconds}s")
            }
        }
        _ => {
            let hours = ms / 3_600_000;
            let minutes = (ms % 3_600_000) / 60_000;
            if minutes == 0 {
                format!("{hours}h")
            } else {
                format!("{hours}h {minutes}m")
            }
        }
    }
}

fn display_id(entry: &LogEntry) -> &str {
    if entry.source == "run"
        && let Some(group_id) = entry.event_group_id.as_deref()
    {
        return group_id;
    }
    &entry.id
}

fn print_entry(entry: &LogEntry) {
    println!("{} {}", style::bold("id:"), entry.id);
    println!("{} {}", style::bold("time:"), entry.ts_utc);
    println!("{} {}", style::bold("source:"), entry.source);
    println!("{} {}", style::bold("kind:"), entry.kind);
    if let Some(value) = &entry.event_group_id {
        println!("{} {}", style::bold("group:"), value);
    }
    if let Some(value) = &entry.phase {
        println!("{} {}", style::bold("phase:"), value);
    }
    if let Some(value) = &entry.key_name {
        println!("{} {}", style::bold("key:"), value);
    }
    if let Some(value) = &entry.key_id {
        println!("{} {}", style::bold("key id:"), value);
    }
    if let Some(value) = &entry.base_url {
        println!("{} {}", style::bold("base url:"), style::dim(value));
    }
    if let Some(value) = &entry.tool {
        println!("{} {}", style::bold("tool:"), value);
    }
    if let Some(value) = &entry.model {
        println!("{} {}", style::bold("model:"), value);
    }
    if let Some(value) = &entry.cwd {
        println!("{} {}", style::bold("cwd:"), style::dim(value));
    }
    if let Some(value) = &entry.session_id {
        println!("{} {}", style::bold("session:"), value);
    }
    if let Some(value) = entry.status_code {
        println!("{} {}", style::bold("status:"), value);
    }
    if let Some(value) = entry.exit_code {
        println!("{} {}", style::bold("exit code:"), value);
    }
    if let Some(value) = entry.duration_ms {
        println!("{} {}", style::bold("duration:"), format_duration_ms(value));
    }
    if entry.input_tokens.is_some() || entry.output_tokens.is_some() {
        println!(
            "{} input={} output={} cache_read={} cache_write={}",
            style::bold("tokens:"),
            entry.input_tokens.unwrap_or(0),
            entry.output_tokens.unwrap_or(0),
            entry.cache_read_input_tokens.unwrap_or(0),
            entry.cache_creation_input_tokens.unwrap_or(0)
        );
    }
    if let Some(value) = &entry.title {
        println!("{} {}", style::bold("title:"), value);
    }
    if let Some(value) = &entry.body_text {
        println!();
        println!("{}", style::bold("Body:"));
        println!("{}", value);
    }
    if let Some(value) = &entry.payload_json {
        println!();
        println!("{}", style::bold("Payload:"));
        println!(
            "{}",
            serde_json::to_string_pretty(value).unwrap_or_default()
        );
    }
}

fn collapse_run_events(entries: Vec<LogEntry>, limit: usize) -> Vec<LogEntry> {
    let mut seen_groups = HashSet::new();
    let mut collapsed = Vec::new();

    for entry in entries {
        if entry.source == "run"
            && let Some(group_id) = &entry.event_group_id
            && !seen_groups.insert(group_id.clone())
        {
            continue;
        }
        collapsed.push(entry);
        if collapsed.len() >= limit {
            break;
        }
    }

    collapsed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_args() -> LogsArgs {
        LogsArgs {
            action: None,
            target: None,
            limit: 20,
            json: false,
            watch: false,
            interval: 1.0,
            jsonl: false,
            latest_first: false,
            search: None,
            source: None,
            tool: None,
            model: None,
            key: None,
            cwd: None,
            since: None,
            until: None,
            errors: false,
        }
    }

    #[test]
    fn validate_args_rejects_jsonl_without_watch() {
        let mut args = base_args();
        args.jsonl = true;
        assert!(validate_args(&args).is_err());
    }

    #[test]
    fn validate_args_rejects_json_with_watch() {
        let mut args = base_args();
        args.json = true;
        args.watch = true;
        assert!(validate_args(&args).is_err());
    }

    fn test_entry(id: &str, ts: &str, source: &str) -> LogEntry {
        LogEntry {
            id: id.to_string(),
            ts_utc: ts.to_string(),
            source: source.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn collapse_run_events_prefers_latest_group_event() {
        let entries = vec![
            LogEntry {
                event_group_id: Some("run-1".to_string()),
                phase: Some("finished".to_string()),
                exit_code: Some(0),
                ..test_entry("2", "2026-03-27T12:00:01Z", "run")
            },
            LogEntry {
                event_group_id: Some("run-1".to_string()),
                phase: Some("started".to_string()),
                ..test_entry("1", "2026-03-27T12:00:00Z", "run")
            },
        ];

        let collapsed = collapse_run_events(entries, 20);
        assert_eq!(collapsed.len(), 1);
        assert_eq!(collapsed[0].id, "2");
    }

    #[test]
    fn order_entries_defaults_to_chronological() {
        let entries = vec![
            test_entry("2", "2026-03-27T12:00:01Z", "run"),
            test_entry("1", "2026-03-27T12:00:00Z", "run"),
        ];

        let ordered = order_entries(entries, false);
        assert_eq!(ordered[0].id, "1");
        assert_eq!(ordered[1].id, "2");
    }

    #[test]
    fn display_id_prefers_run_group_id() {
        let entry = LogEntry {
            event_group_id: Some("group123".to_string()),
            phase: Some("finished".to_string()),
            exit_code: Some(0),
            ..test_entry("event123", "2026-03-27T12:00:01Z", "run")
        };

        assert_eq!(display_id(&entry), "group123");
    }

    #[test]
    fn format_duration_ms_ranges() {
        assert_eq!(format_duration_ms(0), "0ms");
        assert_eq!(format_duration_ms(500), "500ms");
        assert_eq!(format_duration_ms(1234), "1.2s");
        assert_eq!(format_duration_ms(60_000), "1m");
        assert_eq!(format_duration_ms(90_000), "1m 30s");
        assert_eq!(format_duration_ms(3_600_000), "1h");
        assert_eq!(format_duration_ms(5_400_000), "1h 30m");
    }
}
