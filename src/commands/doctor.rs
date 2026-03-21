/**
 * DoctorCommand handler — comprehensive health check for aivo.
 *
 * Checks: config file, encryption, all stored keys (ping), and tool binaries.
 */
use anyhow::Result;

use crate::commands::keys::{PingResult, PingStatus, ping_keys_streaming};
use crate::commands::truncate_url_for_display;
use crate::errors::ExitCode;
use crate::services::path_search::{collect_path_dirs, find_in_dirs};
use crate::services::session_store::SessionStore;
use crate::style;
use crate::version;

pub struct DoctorCommand {
    session_store: SessionStore,
}

impl DoctorCommand {
    pub fn new(session_store: SessionStore) -> Self {
        Self { session_store }
    }

    pub async fn execute(&self) -> ExitCode {
        match self.execute_internal().await {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                ExitCode::UserError
            }
        }
    }

    async fn execute_internal(&self) -> Result<ExitCode> {
        println!(
            "{} {} {}",
            style::cyan("aivo doctor"),
            style::dim(format!("v{}", version::VERSION)),
            style::dim("— system health check")
        );
        println!();

        let mut has_problems = false;

        // 1. Config file
        has_problems |= self.check_config().await;

        // 2. API keys (ping all)
        has_problems |= self.check_keys().await;

        // 3. Tool binaries
        self.check_tools();

        // 4. Summary
        println!();
        if has_problems {
            println!(
                "{}",
                style::yellow("Some checks failed. See details above.")
            );
            Ok(ExitCode::UserError)
        } else {
            println!("{}", style::green("All checks passed."));
            Ok(ExitCode::Success)
        }
    }

    async fn check_config(&self) -> bool {
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

    /// Pings all stored keys and prints results as they arrive.
    /// Returns true if any key has problems.
    async fn check_keys(&self) -> bool {
        println!("{}", style::bold("Keys:"));

        let keys = match self.session_store.get_keys().await {
            Ok(k) => k,
            Err(e) => {
                println!("  {} Failed to load keys: {}", style::red("✗"), e);
                println!();
                return true;
            }
        };

        if keys.is_empty() {
            println!(
                "  {}",
                style::dim("(none) — run `aivo keys add` to add a key")
            );
            println!();
            return false;
        }

        let active_key_id = self
            .session_store
            .get_active_key_info()
            .await
            .ok()
            .flatten()
            .map(|k| k.id.clone());

        let mut has_problems = false;
        let max_name_len = keys
            .iter()
            .map(|k| k.display_name().len())
            .max()
            .unwrap_or(0);

        ping_keys_streaming(keys, |id, result| {
            has_problems |=
                print_key_result(id, result, active_key_id.as_deref(), max_name_len);
        })
        .await;

        println!();
        has_problems
    }

    fn check_tools(&self) {
        println!("{}", style::bold("Tools:"));
        let path_dirs = collect_path_dirs();
        for tool in ["claude", "codex", "gemini", "opencode", "pi"] {
            match find_in_dirs(tool, &path_dirs) {
                Some(path) => println!(
                    "  {} {:8} {}",
                    style::green("✓"),
                    tool,
                    style::dim(path.display().to_string())
                ),
                None => println!(
                    "  {} {:8} {}",
                    style::yellow("—"),
                    tool,
                    style::dim("not found on PATH")
                ),
            }
        }
    }

    pub fn print_help() {
        println!("{} aivo doctor", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim("Run a comprehensive health check: config, keys, and tool binaries.")
        );
        println!();
        println!("{}", style::bold("Checks:"));
        let print_row = |label: &str, description: &str| {
            println!("  {:<18} {}", label, style::dim(description));
        };
        print_row("config", "- Config file exists");
        print_row("keys", "- Ping all stored API keys");
        print_row("tools", "- Detect AI tool binaries on PATH");
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo doctor"));
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
        (style::red("✗"), style::red(message.clone()))
    } else {
        (style::green("✓"), style::green(message.clone()))
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

// Tests for path search utilities are in services::path_search
