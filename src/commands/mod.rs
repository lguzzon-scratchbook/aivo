//! Command handlers module for the aivo CLI.
//! Provides implementations for all CLI commands.

use crate::services::ai_launcher::PreparedLaunch;
use crate::services::environment_injector::redact_env_value;
use crate::style;

/// Strips trailing slashes and a bare `/v1` suffix from a provider base URL.
pub(crate) fn normalize_base_url(url: &str) -> &str {
    let url = url.trim_end_matches('/');
    url.strip_suffix("/v1").unwrap_or(url)
}

/// Truncates a URL for display while preserving both the prefix and suffix.
pub(crate) fn truncate_url_for_display(url: &str, max_len: usize) -> String {
    if url.len() <= max_len {
        return url.to_string();
    }
    let keep_suffix = 15.min(max_len / 3);
    let keep_prefix = max_len.saturating_sub(keep_suffix + 1);
    format!(
        "{}…{}",
        &url[..keep_prefix],
        &url[url.len() - keep_suffix..]
    )
}

pub mod chat;
pub mod keys;
pub mod ls;
pub mod models;
pub mod run;
pub mod serve;
pub mod start;
pub mod update;

pub use chat::ChatCommand;
pub use keys::KeysCommand;
pub use ls::LsCommand;
pub use models::ModelsCommand;
pub use run::RunCommand;
pub use serve::ServeCommand;
pub use start::{StartCommand, StartFlowArgs};
pub use update::UpdateCommand;

pub(crate) fn print_launch_preview(plan: &PreparedLaunch) {
    println!(
        "{} {}",
        style::bold("Tool:"),
        style::cyan(plan.tool.as_str())
    );
    println!(
        "{} {} {}",
        style::bold("Key:"),
        style::cyan(plan.key.display_name()),
        style::dim(format!("({})", plan.key.base_url))
    );
    println!(
        "{} {}",
        style::bold("Model:"),
        plan.model.as_deref().unwrap_or("(tool default)")
    );
    println!(
        "{} {}",
        style::bold("Command:"),
        format_shell_command(&plan.command, &plan.args)
    );
    println!();
    println!("{}", style::bold("Environment:"));
    if plan.env_vars.is_empty() {
        println!("  {}", style::dim("(none)"));
    } else {
        let mut keys: Vec<_> = plan.env_vars.keys().collect();
        keys.sort();
        for key in keys {
            println!("  {}={}", key, redact_env_value(key, &plan.env_vars[key]));
        }
    }

    if !plan.notes.is_empty() {
        println!();
        println!("{}", style::bold("Notes:"));
        for note in &plan.notes {
            println!("  {} {}", style::arrow_symbol(), note);
        }
    }
}

fn format_shell_command(command: &str, args: &[String]) -> String {
    let mut parts = vec![shell_quote(command)];
    parts.extend(args.iter().map(|arg| shell_quote(arg)));
    parts.join(" ")
}

fn shell_quote(value: &str) -> String {
    if !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '/' | '.' | ':' | '='))
    {
        return value.to_string();
    }

    format!("'{}'", value.replace('\'', "'\\''"))
}

#[cfg(test)]
mod tests {
    use super::truncate_url_for_display;

    #[test]
    fn truncate_url_for_display_preserves_short_urls() {
        assert_eq!(
            truncate_url_for_display("https://api.example.com/v1", 50),
            "https://api.example.com/v1"
        );
    }

    #[test]
    fn truncate_url_for_display_shortens_long_urls() {
        let url = "https://very-long-provider-host.example.com/path/to/a/deeply/nested/resource/v1";
        let truncated = truncate_url_for_display(url, 32);

        assert_eq!(
            truncated,
            format!("{}…{}", &url[..21], &url[url.len() - 10..])
        );
    }
}
