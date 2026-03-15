//! Command handlers module for the aivo CLI.
//! Provides implementations for all CLI commands.

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
pub mod models;
pub mod run;
pub mod serve;
pub mod start;
pub mod update;

pub use chat::ChatCommand;
pub use keys::KeysCommand;
pub use models::ModelsCommand;
pub use run::RunCommand;
pub use serve::ServeCommand;
pub use start::{StartCommand, StartFlowArgs};
pub use update::UpdateCommand;

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
