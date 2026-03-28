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
use crate::services::path_search::{collect_path_dirs, find_in_dirs};
use crate::services::session_store::SessionStore;
use crate::services::system_env;
use crate::style;
use crate::version;

pub struct InfoCommand {
    session_store: SessionStore,
}

impl InfoCommand {
    pub fn new(session_store: SessionStore) -> Self {
        Self { session_store }
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

        let keys = self.session_store.get_keys().await?;
        let cwd = system_env::current_dir_string().unwrap_or_else(|| ".".to_string());
        let last_sel = self.session_store.get_last_selection(&cwd).await?;
        let selected_key_id = last_sel.as_ref().map(|s| s.key_id.as_str());
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
                ping_keys_streaming(keys.clone(), |id, result| {
                    has_problems |= print_key_result(id, result, selected_key_id, max_name_len);
                })
                .await;
            } else {
                for key in &keys {
                    let is_selected = selected_key_id == Some(key.id.as_str());
                    let marker = if is_selected {
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

        // 4. Current directory + last used selection
        println!();
        println!("{}", style::bold("Current directory:"));
        println!("  {}", style::dim(&cwd));
        match last_sel {
            Some(ref sel) => {
                let key_label = keys
                    .iter()
                    .find(|k| k.id == sel.key_id)
                    .map(|k| k.display_name().to_string())
                    .unwrap_or(sel.key_id.clone());
                let model_display =
                    crate::commands::models::model_display_label(sel.model.as_deref());
                println!(
                    "  {} {} · {} · {}",
                    style::bullet_symbol(),
                    style::cyan(&sel.tool),
                    key_label,
                    model_display,
                );
            }
            None => {
                println!("  {}", style::dim("No saved selection for this directory."));
            }
        }

        // 5. Summary (check mode only)
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
        println!("{} aivo info [--ping]", style::bold("Usage:"));
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
            println!(
                "  {}{}",
                style::cyan(format!("{:<26}", flag)),
                style::dim(desc)
            );
        };
        print_opt("--ping", "Ping all keys and show pass/fail summary");
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo info"));
        println!("  {}", style::dim("aivo info --ping"));
    }
}

/// Prints a single key ping result. Returns true if the result indicates a problem.
fn print_key_result(
    id: &str,
    result: &PingResult,
    selected_key_id: Option<&str>,
    max_name_len: usize,
) -> bool {
    let is_selected = selected_key_id == Some(id);
    let active_marker = if is_selected { " (selected)" } else { "" };
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
