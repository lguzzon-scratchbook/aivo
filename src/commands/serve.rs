//! ServeCommand — starts a local OpenAI-compatible HTTP server.

use anyhow::Result;

use crate::errors::ExitCode;
use crate::services::provider_protocol::{ProviderProtocol, detect_provider_protocol};
use crate::services::serve_router::{ServeRouter, ServeRouterConfig};
use crate::services::session_store::{ApiKey, SessionStore};
use crate::style;

pub struct ServeCommand {
    session_store: SessionStore,
}

impl ServeCommand {
    pub fn new(session_store: SessionStore) -> Self {
        Self { session_store }
    }

    pub async fn execute(&self, port: u16, key_override: Option<ApiKey>) -> ExitCode {
        match self.execute_internal(port, key_override).await {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                ExitCode::UserError
            }
        }
    }

    async fn execute_internal(&self, port: u16, key_override: Option<ApiKey>) -> Result<ExitCode> {
        let key = match key_override {
            Some(k) => k,
            None => match self.session_store.get_active_key().await? {
                Some(k) => k,
                None => {
                    eprintln!(
                        "{} No API key configured. Run 'aivo keys add' first.",
                        style::red("Error:")
                    );
                    return Ok(ExitCode::AuthError);
                }
            },
        };

        let is_copilot = key.base_url == "copilot";
        let is_openrouter = key.base_url.contains("openrouter");
        let upstream_protocol = if is_copilot {
            ProviderProtocol::Openai
        } else {
            detect_provider_protocol(&key.base_url)
        };

        // Capture display info before moving key into the router
        let display_name = key.display_name().to_string();
        let display_host = if is_copilot {
            "github.com/copilot".to_string()
        } else {
            key.base_url.clone()
        };

        let config = ServeRouterConfig {
            upstream_base_url: key.base_url.clone(),
            upstream_api_key: key.key.as_str().to_string(),
            upstream_protocol,
            is_copilot,
            is_openrouter,
        };

        let router = ServeRouter::new(config, key);

        // Bind eagerly — errors here (e.g. "address already in use") before printing startup
        router.start_background(port).await?;

        eprintln!(
            "{} Listening on http://127.0.0.1:{}",
            style::success_symbol(),
            port
        );
        eprintln!("  {} · {}", display_name, style::dim(&display_host));
        eprintln!("  {}", style::dim("Press Ctrl+C to stop"));

        tokio::signal::ctrl_c().await?;

        Ok(ExitCode::Success)
    }

    pub fn print_help() {
        println!("{} aivo serve", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim(
                "Start a local OpenAI-compatible server that proxies to the active provider."
            )
        );
        println!();
        println!("{}", style::bold("Options:"));
        println!(
            "  {}  {}",
            style::cyan("-p, --port <PORT>"),
            style::dim("Port to listen on (default: 24860)")
        );
        println!(
            "  {}   {}",
            style::cyan("-k, --key <id|name>"),
            style::dim("Select API key by ID or name")
        );
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo serve"));
        println!("  {}", style::dim("aivo serve -p 8080"));
        println!("  {}", style::dim("aivo serve -k openrouter"));
    }
}
