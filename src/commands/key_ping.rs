/**
 * Ping subsystem for API key health-checking.
 */
use crate::services::api_key_store::ApiKeyStore;
use anyhow::Result;
use serde_json::{Value, json};

use std::time::{Duration, Instant};

use crate::commands::truncate_url_for_display;
use crate::services::session_store::{ApiKey, SessionStore};
use crate::style;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PingStatus {
    Ok,
    AuthError,
    Unreachable,
    Timeout,
    Error(String),
}

impl PingStatus {
    pub fn icon(&self) -> &'static str {
        match self {
            PingStatus::Ok => "\u{2713}",
            PingStatus::AuthError => "\u{2717}",
            PingStatus::Unreachable => "\u{2717}",
            PingStatus::Timeout => "\u{2717}",
            PingStatus::Error(_) => "\u{2717}",
        }
    }

    pub fn message(&self) -> String {
        match self {
            PingStatus::Ok => "ok".to_string(),
            PingStatus::AuthError => "auth failed".to_string(),
            PingStatus::Unreachable => "unreachable".to_string(),
            PingStatus::Timeout => "timeout".to_string(),
            PingStatus::Error(msg) => msg.clone(),
        }
    }

    /// Machine-readable identifier used in structured output.
    pub fn json_key(&self) -> &'static str {
        match self {
            PingStatus::Ok => "ok",
            PingStatus::AuthError => "auth_error",
            PingStatus::Unreachable => "unreachable",
            PingStatus::Timeout => "timeout",
            PingStatus::Error(_) => "error",
        }
    }

    pub fn from_http_status(status: u16) -> Self {
        match status {
            200..=299 => PingStatus::Ok,
            401 | 403 => PingStatus::AuthError,
            // 404/405 on probe endpoints means reachable but wrong path — still ok for ping
            404 | 405 => PingStatus::Ok,
            _ => PingStatus::Error(format!("HTTP {}", status)),
        }
    }
}

#[derive(Debug)]
pub struct PingResult {
    pub name: String,
    pub url: String,
    pub status: PingStatus,
    pub latency: Option<Duration>,
}

/// Builds a JSON metadata object for a key. Never includes the secret.
pub(crate) fn key_metadata_json(key: &ApiKey, selected_id: Option<&str>) -> Value {
    json!({
        "id": key.id,
        "name": key.name,
        "base_url": key.base_url,
        "active": selected_id == Some(key.id.as_str()),
        "created_at": key.created_at,
    })
}

/// Converts a PingResult into a JSON object for structured output.
pub(crate) fn ping_result_json(result: &PingResult) -> Value {
    json!({
        "ok": matches!(result.status, PingStatus::Ok),
        "status": result.status.json_key(),
        "message": result.status.message(),
        "latency_ms": result.latency.map(|d| d.as_millis() as u64),
    })
}

const PING_TIMEOUT: Duration = Duration::from_secs(5);

const PING_MAX_RETRIES: u32 = 3;

pub async fn ping_key(key: ApiKey) -> PingResult {
    let name = if key.name.is_empty() {
        key.short_id().to_string()
    } else {
        key.name.clone()
    };
    let url = key.base_url.clone();

    let start = Instant::now();
    let status = ping_with_retries(&key).await;
    let latency = Some(start.elapsed());

    PingResult {
        name,
        url,
        status,
        latency,
    }
}

/// Pings keys concurrently and calls `on_result(key_id, result)` for each as it resolves.
/// Decrypt failures are reported immediately; successful decrypts are spawned and awaited in order.
pub async fn ping_keys_streaming(
    keys: Vec<ApiKey>,
    mut on_result: impl FnMut(&str, &PingResult),
) {
    let mut handles = Vec::new();
    for mut key in keys {
        let id = key.id.clone();
        if ApiKeyStore::decrypt_key_secret(&mut key).is_err() {
            on_result(
                &id,
                &PingResult {
                    name: key.display_name().to_string(),
                    url: key.base_url.clone(),
                    status: PingStatus::Error("decrypt failed".to_string()),
                    latency: None,
                },
            );
            continue;
        }
        handles.push(tokio::spawn(async move {
            let result = ping_key(key).await;
            (id, result)
        }));
    }
    for handle in handles {
        if let Ok((id, result)) = handle.await {
            on_result(&id, &result);
        }
    }
}

pub(crate) async fn ping_with_retries(key: &ApiKey) -> PingStatus {
    let mut last_status = PingStatus::Unreachable;
    for _ in 0..PING_MAX_RETRIES {
        last_status = match tokio::time::timeout(PING_TIMEOUT, probe_key(key)).await {
            Ok(Ok(s)) => s,
            Ok(Err(_)) => PingStatus::Unreachable,
            Err(_) => PingStatus::Timeout,
        };
        if matches!(last_status, PingStatus::Ok) {
            return last_status;
        }
    }
    last_status
}

async fn probe_key(key: &ApiKey) -> Result<PingStatus> {
    use crate::services::provider_profile::{ModelListingStrategy, provider_profile_for_key};

    // OAuth keys have no reachable REST endpoint; tokens are consumed by the
    // native CLI against the provider's subscription backend. Report "Ok" so
    // the list doesn't look scary; the real health check is `aivo run <tool>`.
    if key.is_any_oauth() {
        return Ok(PingStatus::Ok);
    }

    let profile = provider_profile_for_key(key);
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(PING_TIMEOUT)
        .build()?;

    match profile.model_listing_strategy {
        ModelListingStrategy::Ollama => {
            match client.get("http://localhost:11434/").send().await {
                Ok(_) => Ok(PingStatus::Ok),
                Err(_) => Ok(PingStatus::Unreachable),
            }
        }
        ModelListingStrategy::Copilot => {
            use crate::services::copilot_auth::CopilotTokenManager;
            let tm = CopilotTokenManager::new(key.key.as_str().to_string());
            match tm.get_token().await {
                Ok(_) => Ok(PingStatus::Ok),
                Err(_) => Ok(PingStatus::AuthError),
            }
        }
        ModelListingStrategy::Google => {
            let url = format!(
                "https://generativelanguage.googleapis.com/v1beta/models?key={}",
                key.key.as_str()
            );
            match client.get(&url).send().await {
                Ok(r) => Ok(PingStatus::from_http_status(r.status().as_u16())),
                Err(_) => Ok(PingStatus::Unreachable),
            }
        }
        ModelListingStrategy::Anthropic | ModelListingStrategy::Static(_) => {
            let base = key.base_url.trim_end_matches('/');
            let url = if base.ends_with("/v1") {
                format!("{}/models", base)
            } else {
                format!("{}/v1/models", base)
            };
            match client
                .get(&url)
                .header("x-api-key", key.key.as_str())
                .header("anthropic-version", "2023-06-01")
                .send()
                .await
            {
                Ok(r) => Ok(PingStatus::from_http_status(r.status().as_u16())),
                Err(_) => Ok(PingStatus::Unreachable),
            }
        }
        ModelListingStrategy::AivoStarter => {
            let url = format!(
                "{}/v1/models",
                crate::constants::AIVO_STARTER_REAL_URL.trim_end_matches('/')
            );
            let req = crate::services::device_fingerprint::with_starter_headers(
                client
                    .get(&url)
                    .header("Authorization", format!("Bearer {}", key.key.as_str())),
            );
            match req.send().await {
                Ok(r) => Ok(PingStatus::from_http_status(r.status().as_u16())),
                Err(_) => Ok(PingStatus::Unreachable),
            }
        }
        ModelListingStrategy::CloudflareSearch | ModelListingStrategy::OpenAiCompatible => {
            let base = key.base_url.trim_end_matches('/');
            let url = if base.ends_with("/v1") {
                format!("{}/models", base)
            } else {
                format!("{}/v1/models", base)
            };
            match client
                .get(&url)
                .header("Authorization", format!("Bearer {}", key.key.as_str()))
                .send()
                .await
            {
                Ok(r) => Ok(PingStatus::from_http_status(r.status().as_u16())),
                Err(_) => Ok(PingStatus::Unreachable),
            }
        }
    }
}

/// Prints a single ping result in a formatted table row.
pub(crate) fn print_ping_result(result: &PingResult, max_name_len: usize) {
    let icon = match &result.status {
        PingStatus::Ok => style::green(result.status.icon()),
        _ => style::red(result.status.icon()),
    };
    let latency = result
        .latency
        .map(|d| format!("{}ms", d.as_millis()))
        .unwrap_or_default();
    let name_padded = format!("{:<width$}", result.name, width = max_name_len);
    let message = result.status.message();
    println!(
        " {} {}  {}  {:>6}  {}",
        icon,
        name_padded,
        style::dim(truncate_url_for_display(&result.url, 40)),
        style::dim(latency),
        match &result.status {
            PingStatus::Ok => style::green(&message),
            _ => style::red(&message),
        }
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::session_store::ApiKey;

    #[test]
    fn test_ping_status_from_http_status_ok() {
        assert_eq!(PingStatus::from_http_status(200), PingStatus::Ok);
        assert_eq!(PingStatus::from_http_status(204), PingStatus::Ok);
    }

    #[test]
    fn test_ping_status_from_http_status_auth_error() {
        assert_eq!(PingStatus::from_http_status(401), PingStatus::AuthError);
        assert_eq!(PingStatus::from_http_status(403), PingStatus::AuthError);
    }

    #[test]
    fn test_ping_status_from_http_status_other() {
        assert_eq!(
            PingStatus::from_http_status(500),
            PingStatus::Error("HTTP 500".to_string())
        );
        assert_eq!(
            PingStatus::from_http_status(429),
            PingStatus::Error("HTTP 429".to_string())
        );
    }

    #[test]
    fn test_ping_status_from_http_status_reachable_wrong_path() {
        assert_eq!(PingStatus::from_http_status(404), PingStatus::Ok);
        assert_eq!(PingStatus::from_http_status(405), PingStatus::Ok);
    }

    #[test]
    fn test_ping_status_icons_and_messages() {
        assert_eq!(PingStatus::Ok.icon(), "\u{2713}");
        assert_eq!(PingStatus::Ok.message(), "ok");
        assert_eq!(PingStatus::AuthError.icon(), "\u{2717}");
        assert_eq!(PingStatus::AuthError.message(), "auth failed");
        assert_eq!(PingStatus::Unreachable.icon(), "\u{2717}");
        assert_eq!(PingStatus::Unreachable.message(), "unreachable");
        assert_eq!(PingStatus::Timeout.icon(), "\u{2717}");
        assert_eq!(PingStatus::Timeout.message(), "timeout");
    }

    #[test]
    fn test_ping_result_empty_name_uses_short_id() {
        let result = PingResult {
            name: "abc".to_string(),
            url: "https://api.openai.com".to_string(),
            status: PingStatus::Ok,
            latency: Some(std::time::Duration::from_millis(42)),
        };
        assert_eq!(result.name, "abc");
        assert_eq!(result.status, PingStatus::Ok);
    }

    #[test]
    fn test_ping_result_json_ok_includes_latency() {
        let result = PingResult {
            name: "test".to_string(),
            url: "https://api.example.com".to_string(),
            status: PingStatus::Ok,
            latency: Some(std::time::Duration::from_millis(42)),
        };
        let payload = ping_result_json(&result);
        assert_eq!(payload["ok"], true);
        assert_eq!(payload["status"], "ok");
        assert_eq!(payload["latency_ms"], 42);
    }

    #[test]
    fn test_ping_result_json_error_status_keys() {
        for (status, expected_key) in [
            (PingStatus::AuthError, "auth_error"),
            (PingStatus::Unreachable, "unreachable"),
            (PingStatus::Timeout, "timeout"),
            (PingStatus::Error("boom".into()), "error"),
        ] {
            let result = PingResult {
                name: "n".to_string(),
                url: "u".to_string(),
                status,
                latency: None,
            };
            let payload = ping_result_json(&result);
            assert_eq!(payload["ok"], false);
            assert_eq!(payload["status"], expected_key);
            assert!(payload["latency_ms"].is_null());
        }
    }

    #[test]
    fn test_key_metadata_json_excludes_secret() {
        let key = ApiKey::new_with_protocol(
            "abc".to_string(),
            "test".to_string(),
            "https://api.example.com".to_string(),
            None,
            "sk-must-not-leak".to_string(),
        );
        let payload = key_metadata_json(&key, Some("abc"));
        let s = serde_json::to_string(&payload).unwrap();
        assert!(!s.contains("sk-must-not-leak"));
        assert!(s.contains("\"active\":true"));
        assert_eq!(payload["id"], "abc");
        assert_eq!(payload["name"], "test");
        assert_eq!(payload["base_url"], "https://api.example.com");
    }

    #[test]
    fn test_key_metadata_json_inactive() {
        let key = ApiKey::new_with_protocol(
            "abc".to_string(),
            "test".to_string(),
            "https://api.example.com".to_string(),
            None,
            "sk".to_string(),
        );
        let payload = key_metadata_json(&key, Some("xyz"));
        assert_eq!(payload["active"], false);
        let payload = key_metadata_json(&key, None);
        assert_eq!(payload["active"], false);
    }
}
