//! ServeCommand — starts a local OpenAI-compatible HTTP server.

use anyhow::Result;
use std::net::IpAddr;

use crate::errors::ExitCode;
use crate::services::provider_profile::provider_profile_for_key;
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

        let profile = provider_profile_for_key(&key);
        let is_copilot = profile.serve_flags.is_copilot;
        let is_openrouter = profile.serve_flags.is_openrouter;
        let upstream_protocol = profile.default_protocol;

        if is_self_proxy_target(&key.base_url, port) {
            anyhow::bail!(
                "Refusing to start `aivo serve`: active upstream {} points back to http://127.0.0.1:{} and would proxy into itself. Switch to a real provider key with `aivo use <name>` or pass `--key <name>`.",
                key.base_url,
                port
            );
        }

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
        let mut handle = router.start_background(port).await?;

        eprintln!(
            "{} Listening on http://127.0.0.1:{}",
            style::success_symbol(),
            port
        );
        eprintln!("  {} · {}", display_name, style::dim(&display_host));
        print_supported_paths();
        eprintln!("  {}", style::dim("Press Ctrl+C to stop"));

        tokio::select! {
            signal = tokio::signal::ctrl_c() => {
                signal?;
                handle.abort();
                let _ = handle.await;
            }
            result = &mut handle => match result {
                Ok(Ok(())) => {
                    anyhow::bail!("serve router stopped unexpectedly");
                }
                Ok(Err(err)) => {
                    return Err(err);
                }
                Err(err) if err.is_cancelled() => {}
                Err(err) => {
                    anyhow::bail!("serve router task failed: {}", err);
                }
            },
        }

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
            style::dim("Select API key by ID or name (-k opens key picker)")
        );
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo serve"));
        println!("  {}", style::dim("aivo serve -p 8080"));
        println!("  {}", style::dim("aivo serve -k openrouter"));
    }
}

fn print_supported_paths() {
    eprintln!();
    eprintln!("{}", style::bold("Supported paths"));
    eprintln!("  {}", style::blue("/v1/models"));
    eprintln!("  {}", style::blue("/v1/chat/completions"));
    eprintln!("  {}", style::blue("/v1/responses"));
    eprintln!();
}

fn is_self_proxy_target(base_url: &str, port: u16) -> bool {
    let Ok(url) = reqwest::Url::parse(base_url) else {
        return false;
    };

    let Some(host) = url.host_str() else {
        return false;
    };
    let Some(target_port) = url.port_or_known_default() else {
        return false;
    };

    if target_port != port {
        return false;
    }

    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }

    host.trim_matches(['[', ']'])
        .parse::<IpAddr>()
        .is_ok_and(|ip| ip.is_loopback())
}

#[cfg(test)]
mod tests {
    use super::is_self_proxy_target;

    #[test]
    fn detects_localhost_self_proxy() {
        assert!(is_self_proxy_target("http://127.0.0.1:24860", 24860));
        assert!(is_self_proxy_target("http://127.0.0.1:24860/v1", 24860));
        assert!(is_self_proxy_target("http://localhost:24860", 24860));
        assert!(is_self_proxy_target("http://[::1]:24860/v1", 24860));
    }

    #[test]
    fn ignores_other_ports_and_hosts() {
        assert!(!is_self_proxy_target("http://127.0.0.1:8080", 24860));
        assert!(!is_self_proxy_target("https://api.openai.com/v1", 24860));
        assert!(!is_self_proxy_target("not-a-url", 24860));
    }
}
