//! Ollama lifecycle management and model operations.
//!
//! Provides functions for detecting, starting, and querying a local Ollama instance.
//! Ollama exposes an OpenAI-compatible API at `{host}/v1`.

use std::io::Write;
use std::process::Command;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Deserialize;

/// Returns the Ollama host from `OLLAMA_HOST` or the default `http://localhost:11434`.
pub fn ollama_host() -> String {
    std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string())
}

/// Returns the OpenAI-compatible base URL for Ollama (`{host}/v1`).
pub fn ollama_openai_base_url() -> String {
    format!("{}/v1", ollama_host())
}

/// Returns `true` if the `ollama` binary is on `PATH`.
pub fn detect_binary() -> bool {
    Command::new("ollama")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok()
}

/// Builds a client that bypasses proxies (Ollama is always local).
fn local_client(timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .no_proxy()
        .timeout(timeout)
        .build()
        .unwrap_or_default()
}

fn health_client() -> reqwest::Client {
    local_client(Duration::from_secs(1))
}

async fn check_health(client: &reqwest::Client) -> bool {
    let url = format!("{}/api/tags", ollama_host());
    client.get(&url).send().await.is_ok()
}

/// Returns `true` if Ollama is responding to API requests.
pub async fn is_running() -> bool {
    check_health(&health_client()).await
}

/// Spawns `ollama serve` as a detached process and polls for readiness (up to 5s).
pub async fn auto_start() -> Result<()> {
    eprintln!("  {} Starting Ollama...", crate::style::dim("⟳"));

    Command::new("ollama")
        .arg("serve")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .context("Failed to spawn 'ollama serve'")?;

    let client = health_client();
    for _ in 0..10 {
        tokio::time::sleep(Duration::from_millis(500)).await;
        if check_health(&client).await {
            return Ok(());
        }
    }

    anyhow::bail!(
        "Ollama did not become ready within 5 seconds. Try running 'ollama serve' manually."
    )
}

/// Ensures Ollama is installed and running: checks binary → checks running → auto-starts.
pub async fn ensure_ready() -> Result<()> {
    if !detect_binary() {
        anyhow::bail!("Ollama is not installed. Install it from https://ollama.com and try again.");
    }
    if !is_running().await {
        auto_start().await?;
    }
    Ok(())
}

#[derive(Deserialize)]
struct TagsResponse {
    #[serde(default)]
    models: Vec<TagModel>,
}

#[derive(Deserialize)]
struct TagModel {
    name: String,
}

/// Lists locally available Ollama models via `GET /api/tags`.
pub async fn list_models() -> Result<Vec<String>> {
    let url = format!("{}/api/tags", ollama_host());
    let response = local_client(Duration::from_secs(10))
        .get(&url)
        .send()
        .await
        .context("Failed to connect to Ollama")?;
    let text = response
        .text()
        .await
        .context("Failed to read Ollama /api/tags response")?;
    let resp: TagsResponse = serde_json::from_str(&text)
        .with_context(|| format!("Failed to parse Ollama /api/tags: {}", truncate(&text, 200)))?;
    Ok(resp.models.into_iter().map(|m| m.name).collect())
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max { s } else { &s[..max] }
}

/// Returns `true` if the given model name is already pulled locally.
pub async fn is_model_available(name: &str) -> Result<bool> {
    let models = list_models().await?;
    Ok(models.iter().any(|m| {
        m == name
            || m.strip_suffix(":latest")
                .is_some_and(|stripped| stripped == name)
    }))
}

/// Pulls a model from the Ollama registry with streaming progress output.
pub async fn pull_model(name: &str) -> Result<()> {
    let url = format!("{}/api/pull", ollama_host());
    let client = local_client(Duration::from_secs(600));
    let mut resp = client
        .post(&url)
        .json(&serde_json::json!({ "name": name, "stream": true }))
        .send()
        .await
        .context("Failed to connect to Ollama for model pull")?;

    if !resp.status().is_success() {
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Ollama pull failed: {}", body);
    }

    let mut last_status = String::new();
    while let Some(chunk) = resp
        .chunk()
        .await
        .context("Stream error during model pull")?
    {
        // Each line is a JSON object with status and optional progress fields
        for line in chunk.split(|&b| b == b'\n') {
            if line.is_empty() {
                continue;
            }
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(line) {
                let status = v["status"].as_str().unwrap_or("");
                if !status.is_empty() && status != last_status {
                    eprint!("\r\x1b[2K  {} {}", crate::style::dim("⟳"), status);
                    let _ = std::io::stderr().flush();
                    last_status = status.to_string();
                }
                // Show download progress percentage
                if let (Some(completed), Some(total)) =
                    (v["completed"].as_u64(), v["total"].as_u64())
                    && total > 0
                {
                    let pct = (completed as f64 / total as f64 * 100.0) as u64;
                    eprint!(
                        "\r\x1b[2K  {} {} ({}%)",
                        crate::style::dim("⟳"),
                        status,
                        pct
                    );
                    let _ = std::io::stderr().flush();
                }
            }
        }
    }
    eprintln!(
        "\r\x1b[2K  {} Pull complete: {}",
        crate::style::success_symbol(),
        name
    );
    Ok(())
}

/// Checks if a model is locally available; if not, prompts the user and pulls it.
pub async fn ensure_model(name: &str) -> Result<()> {
    if is_model_available(name).await? {
        return Ok(());
    }

    eprint!(
        "  {} Model '{}' not found locally. Pull it? [Y/n] ",
        crate::style::yellow("?"),
        name
    );
    let _ = std::io::stderr().flush();

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    if matches!(input.trim().to_ascii_lowercase().as_str(), "" | "y" | "yes") {
        pull_model(name).await?;
    } else {
        anyhow::bail!(
            "Model '{}' is not available. Pull it with 'ollama pull {}'.",
            name,
            name
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_host_default() {
        // When OLLAMA_HOST is not set, returns default
        if std::env::var("OLLAMA_HOST").is_err() {
            assert_eq!(ollama_host(), "http://localhost:11434");
        }
    }

    #[test]
    fn test_ollama_openai_base_url() {
        if std::env::var("OLLAMA_HOST").is_err() {
            assert_eq!(ollama_openai_base_url(), "http://localhost:11434/v1");
        }
    }

    #[test]
    fn test_is_ollama_base() {
        use crate::services::provider_profile::is_ollama_base;
        assert!(is_ollama_base("ollama"));
        assert!(!is_ollama_base("copilot"));
        assert!(!is_ollama_base("http://localhost:11434"));
        assert!(!is_ollama_base(""));
    }
}
