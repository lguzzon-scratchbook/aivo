//! Command handlers module for the aivo CLI.
//! Provides implementations for all CLI commands.

/// Strips trailing slashes and a bare `/v1` suffix from a provider base URL.
pub(crate) fn normalize_base_url(url: &str) -> &str {
    let url = url.trim_end_matches('/');
    url.strip_suffix("/v1").unwrap_or(url)
}

pub mod chat;
pub mod keys;
pub mod models;
pub mod run;
pub mod serve;
pub mod update;

pub use chat::ChatCommand;
pub use keys::KeysCommand;
pub use models::ModelsCommand;
pub use run::RunCommand;
pub use serve::ServeCommand;
pub use update::UpdateCommand;
