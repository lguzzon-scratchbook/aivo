/**
 * InfoCommand handler — unified system info and health check for aivo.
 *
 * `aivo info` shows config, keys, tools, directory state, and active defaults.
 * `aivo info --check` additionally pings all keys and shows a pass/fail summary.
 */
use anyhow::Result;

use crate::commands::keys::{PingResult, PingStatus, ping_keys_streaming};
use crate::commands::truncate_url_for_display;
use crate::errors::ExitCode;
use crate::services::models_cache::ModelsCache;
use crate::services::path_search::{collect_path_dirs, find_in_dirs};
use crate::services::session_store::{DirectoryStartRecord, SessionStore};
use crate::services::system_env;
use crate::style;
use crate::version;

pub struct InfoCommand {
    session_store: SessionStore,
    cache: ModelsCache,
}

impl InfoCommand {
    pub fn new(session_store: SessionStore, cache: ModelsCache) -> Self {
        Self {
            session_store,
            cache,
        }
    }

    pub async fn execute(&self, check: bool) -> ExitCode {
        match self.execute_internal(check).await {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                ExitCode::UserError
            }
        }
    }

    async fn execute_internal(&self, check: bool) -> Result<ExitCode> {
        // Header
        println!(
            "{} {}",
            style::cyan("aivo info"),
            style::dim(format!("v{}", version::VERSION)),
        );
        println!();

        let (keys, active_key_id) = self.session_store.get_keys_and_active_id_info().await?;
        let cwd = system_env::current_dir_string().unwrap_or_else(|| ".".to_string());
        let remembered = self.session_store.get_directory_start(&cwd).await?;
        let active_key = active_key_id
            .as_deref()
            .and_then(|active_id| keys.iter().find(|key| key.id == active_id));

        let mut has_problems = false;

        // 1. Config
        if check {
            has_problems |= self.check_config();
        }

        // 2. Keys
        println!("{}", style::bold("Keys:"));
        if keys.is_empty() {
            if check {
                println!(
                    "  {}",
                    style::dim("(none) — run `aivo keys add` to add a key")
                );
            } else {
                println!("  {}", style::dim("(none)"));
            }
        } else {
            let max_name_len = keys
                .iter()
                .map(|k| k.display_name().len())
                .max()
                .unwrap_or(0);

            if check {
                let active_id = active_key_id.as_deref();
                ping_keys_streaming(keys.clone(), |id, result| {
                    has_problems |= print_key_result(id, result, active_id, max_name_len);
                })
                .await;
            } else {
                for key in &keys {
                let is_active = active_key_id.as_deref() == Some(key.id.as_str());
                let marker = if is_active {
                    style::bullet_symbol()
                } else {
                    style::empty_bullet_symbol()
                };
                println!(
                    "  {} {}  {:width$}  {}",
                    marker,
                    style::cyan(key.short_id()),
                    key.display_name(),
                    style::dim(truncate_url_for_display(&key.base_url, 50)),
                    width = max_name_len
                );
                }
            }
        }

        // 3. Tools
        println!();
        println!("{}", style::bold("Tools:"));
        let path_dirs = collect_path_dirs();
        for tool in ["claude", "codex", "gemini", "opencode", "pi"] {
            match find_in_dirs(tool, &path_dirs) {
                Some(path) => println!(
                    "  {} {:8} {}",
                    style::success_symbol(),
                    style::cyan(tool),
                    style::dim(path.display().to_string())
                ),
                None => println!(
                    "  {} {:8} {}",
                    style::empty_bullet_symbol(),
                    style::cyan(tool),
                    style::dim("not found on PATH")
                ),
            }
        }

        // 4. Current directory + remembered start
        println!();
        println!("{}", style::bold("Current directory:"));
        println!("  {}", style::dim(&cwd));
        match remembered {
            Some(record) => Self::print_directory_start(&record, &keys),
            None => println!(
                "  {}",
                style::dim("No remembered start for this directory.")
            ),
        }

        // 5. Active defaults
        println!();
        println!("{}", style::bold("Active defaults:"));
        if let Some(key) = active_key {
            println!(
                "  {} {} {}",
                style::dim("key:"),
                style::cyan(key.display_name()),
                style::dim(format!("({})", key.base_url))
            );

            match self.session_store.get_chat_model(&key.id).await? {
                Some(model) => println!("  {} {}", style::dim("chat model:"), model),
                None => println!("  {} {}", style::dim("chat model:"), style::dim("(none)")),
            }

            match self.cache.get(&key.base_url).await {
                Some(models) => println!("  {} {}", style::dim("cached models:"), models.len()),
                None => println!(
                    "  {} {}",
                    style::dim("cached models:"),
                    style::dim("(none)")
                ),
            }
        } else {
            println!("  {}", style::dim("No active key."));
        }

        // 6. Summary (check mode only)
        if check {
            println!();
            if has_problems {
                println!(
                    "{}",
                    style::yellow("Some checks failed. See details above.")
                );
                return Ok(ExitCode::UserError);
            } else {
                println!("{}", style::green("All checks passed."));
            }
        }

        Ok(ExitCode::Success)
    }

    fn check_config(&self) -> bool {
        println!("{}", style::bold("Config:"));

        let config_path = self.session_store.get_config_path();
        let exists = config_path.exists();
        if exists {
            println!(
                "  {} config file  {}",
                style::green("✓"),
                style::dim(config_path.display().to_string())
            );
        } else {
            println!(
                "  {} config file  {}",
                style::red("✗"),
                style::dim("not found — run `aivo keys add` to create")
            );
            println!();
            return true;
        }

        println!();
        false
    }

    pub fn print_help() {
        println!("{} aivo ls [--ping]", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim("Show system info, keys, tools, and directory state.")
        );
        println!(
            "{}",
            style::dim("With --ping, also pings all keys and shows a pass/fail summary.")
        );
        println!();
        println!("{}", style::bold("Options:"));
        let print_opt = |flag: &str, desc: &str| {
            println!("  {:<18} {}", flag, style::dim(desc));
        };
        print_opt("--ping", "Ping all keys and show pass/fail summary");
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo ls"));
        println!("  {}", style::dim("aivo ls --ping"));
    }

    fn print_directory_start(
        record: &DirectoryStartRecord,
        keys: &[crate::services::session_store::ApiKey],
    ) {
        let (tool, key_name, model) = format_directory_start_line(record, keys);

        println!(
            "  {} {} · {} · {}",
            style::dim("start:"),
            style::cyan(&tool),
            key_name,
            model
        );
    }
}

/// Prints a single key ping result. Returns true if the result indicates a problem.
fn print_key_result(
    id: &str,
    result: &PingResult,
    active_key_id: Option<&str>,
    max_name_len: usize,
) -> bool {
    let is_active = active_key_id == Some(id);
    let active_marker = if is_active { " (active)" } else { "" };
    let message = result.status.message();
    let has_problem = !matches!(result.status, PingStatus::Ok);
    let (icon, status_styled) = if has_problem {
        (style::red("✗"), style::red(&message))
    } else {
        (style::green("✓"), style::green(&message))
    };
    let latency = result
        .latency
        .map(|d: std::time::Duration| format!(" {}ms", d.as_millis()))
        .unwrap_or_default();
    let name_padded = format!("{:<width$}", result.name, width = max_name_len);
    println!(
        "  {} {}{}  {}  {}{}",
        icon,
        name_padded,
        style::dim(active_marker),
        style::dim(truncate_url_for_display(&result.url, 40)),
        status_styled,
        style::dim(&latency),
    );
    has_problem
}

fn format_directory_start_line(
    record: &DirectoryStartRecord,
    keys: &[crate::services::session_store::ApiKey],
) -> (String, String, String) {
    let key_name = keys
        .iter()
        .find(|key| key.id == record.key_id)
        .map(|key| key.display_name().to_string())
        .unwrap_or_else(|| record.key_id.clone());
    let model = record
        .model
        .as_deref()
        .unwrap_or("(tool default)")
        .to_string();

    (record.tool.clone(), key_name, model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::session_store::{ApiKey, DirectoryStartRecord};

    fn make_key(id: &str, name: &str) -> ApiKey {
        ApiKey::new_with_protocol(
            id.to_string(),
            name.to_string(),
            "https://api.example.com/v1".to_string(),
            None,
            "sk-test".to_string(),
        )
    }

    fn make_record(key_id: &str, tool: &str, model: Option<&str>) -> DirectoryStartRecord {
        DirectoryStartRecord {
            key_id: key_id.to_string(),
            base_url: "https://api.example.com/v1".to_string(),
            tool: tool.to_string(),
            model: model.map(ToString::to_string),
            updated_at: "2026-01-01T00:00:00Z".to_string(),
        }
    }

    #[test]
    fn format_directory_start_uses_key_name_when_found() {
        let keys = vec![make_key("abc", "My OpenAI Key")];
        let record = make_record("abc", "claude", Some("gpt-4o"));

        let (_tool, key_name, _model) = format_directory_start_line(&record, &keys);
        assert_eq!(key_name, "My OpenAI Key");
    }

    #[test]
    fn format_directory_start_falls_back_to_key_id() {
        let keys = vec![make_key("xyz", "Other Key")];
        let record = make_record("abc", "claude", Some("gpt-4o"));

        let (_tool, key_name, _model) = format_directory_start_line(&record, &keys);
        assert_eq!(key_name, "abc");
    }

    #[test]
    fn format_directory_start_shows_tool_default_when_no_model() {
        let keys = vec![make_key("abc", "My Key")];
        let record = make_record("abc", "codex", None);

        let (_tool, _key_name, model) = format_directory_start_line(&record, &keys);
        assert_eq!(model, "(tool default)");
    }

    #[test]
    fn format_directory_start_shows_model_when_present() {
        let keys = vec![make_key("abc", "My Key")];
        let record = make_record("abc", "gemini", Some("gpt-4o"));

        let (_tool, _key_name, model) = format_directory_start_line(&record, &keys);
        assert_eq!(model, "gpt-4o");
    }
}
