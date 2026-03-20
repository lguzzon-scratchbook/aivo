use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::constants::PLACEHOLDER_LOOPBACK_URL;
use crate::services::ai_launcher::AIToolType;
use crate::services::provider_protocol::ProviderProtocol;
use crate::services::session_store::{
    ApiKey, ClaudeProviderProtocol, GeminiProviderProtocol, SessionStore,
};

pub(crate) struct LaunchRuntimeState {
    pub(crate) env: HashMap<String, String>,
    pub(crate) router_protocol: Option<Arc<AtomicU8>>,
    pub(crate) responses_api_support: Option<Arc<AtomicU8>>,
    pub(crate) pi_agent_dir: Option<String>,
}

pub(crate) async fn prepare_runtime_env(
    tool: AIToolType,
    mut env: HashMap<String, String>,
) -> Result<LaunchRuntimeState> {
    let mut router_protocol = None;
    let mut responses_api_support = None;

    if tool == AIToolType::Claude && env.contains_key("AIVO_USE_ROUTER") {
        let port = start_anthropic_router(&env).await?;
        set_local_base_url(&mut env, "ANTHROPIC_BASE_URL", port);
    }

    if tool == AIToolType::Claude && env.contains_key("AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER") {
        let (port, active) = start_anthropic_to_openai_router(&env).await?;
        router_protocol = Some(active);
        set_local_base_url(&mut env, "ANTHROPIC_BASE_URL", port);
    }

    if tool == AIToolType::Claude && env.contains_key("AIVO_USE_COPILOT_ROUTER") {
        let port = start_copilot_router(&env).await?;
        set_local_base_url(&mut env, "ANTHROPIC_BASE_URL", port);
    }

    if tool == AIToolType::Codex && env.contains_key("AIVO_USE_RESPONSES_TO_CHAT_ROUTER") {
        let (port, _active, responses_api) = start_responses_to_chat_router(&env).await?;
        responses_api_support = Some(responses_api);
        set_local_base_url(&mut env, "OPENAI_BASE_URL", port);
    }

    if tool == AIToolType::Codex && env.contains_key("AIVO_USE_RESPONSES_TO_CHAT_COPILOT_ROUTER") {
        let port = start_responses_to_chat_copilot_router(&env).await?;
        set_local_base_url(&mut env, "OPENAI_BASE_URL", port);
    }

    if tool == AIToolType::Gemini && env.contains_key("AIVO_USE_GEMINI_ROUTER") {
        let (port, active) = start_gemini_router(&env).await?;
        router_protocol = Some(active);
        set_local_base_url(&mut env, "GOOGLE_GEMINI_BASE_URL", port);
    }

    if tool == AIToolType::Gemini && env.contains_key("AIVO_USE_GEMINI_COPILOT_ROUTER") {
        let port = start_gemini_copilot_router(&env).await?;
        set_local_base_url(&mut env, "GOOGLE_GEMINI_BASE_URL", port);
    }

    if tool == AIToolType::Opencode && env.contains_key("AIVO_USE_OPENCODE_COPILOT_ROUTER") {
        let port = start_responses_to_chat_copilot_router(&env).await?;
        patch_opencode_config_content(&mut env, port);
    }

    if tool == AIToolType::Opencode && env.contains_key("AIVO_USE_OPENCODE_ROUTER") {
        let (port, _active, _responses_api) = start_responses_to_chat_router(&env).await?;
        patch_opencode_config_content(&mut env, port);
    }

    if tool == AIToolType::Pi && env.contains_key("AIVO_SETUP_PI_AGENT_DIR") {
        // Direct connection — no router needed, just write the temp agent dir.
        write_pi_agent_dir(&mut env, None).await?;
    }

    if tool == AIToolType::Pi && env.contains_key("AIVO_USE_PI_COPILOT_ROUTER") {
        let port = start_responses_to_chat_copilot_router(&env).await?;
        write_pi_agent_dir(&mut env, Some(port)).await?;
    }

    let pi_agent_dir = env.get("PI_CODING_AGENT_DIR").cloned();

    Ok(LaunchRuntimeState {
        env,
        router_protocol,
        responses_api_support,
        pi_agent_dir,
    })
}

pub(crate) async fn record_launch_state(
    session_store: &SessionStore,
    key: &ApiKey,
    tool: AIToolType,
    model: Option<&str>,
) {
    let _ = session_store
        .record_selection(&key.id, tool.as_str(), model)
        .await;
    if let Some(cwd) = crate::services::system_env::current_dir_string() {
        let _ = session_store
            .set_directory_start(&cwd, &key.id, &key.base_url, tool.as_str(), model)
            .await;
    }
}

pub(crate) async fn persist_runtime_discoveries(
    session_store: &SessionStore,
    tool: AIToolType,
    key: &ApiKey,
    key_override_used: bool,
    router_protocol: Option<Arc<AtomicU8>>,
    responses_api_support: Option<Arc<AtomicU8>>,
) {
    if key_override_used {
        return;
    }

    if let Some(active) = router_protocol {
        let final_protocol = ProviderProtocol::from_u8(active.load(Ordering::Relaxed));
        match tool {
            AIToolType::Claude => {
                let current = key
                    .claude_protocol
                    .map(|p| match p {
                        ClaudeProviderProtocol::Openai => ProviderProtocol::Openai,
                        ClaudeProviderProtocol::Anthropic => ProviderProtocol::Anthropic,
                        ClaudeProviderProtocol::Google => ProviderProtocol::Google,
                    })
                    .unwrap_or(ProviderProtocol::Openai);
                if final_protocol != current {
                    let protocol = match final_protocol {
                        ProviderProtocol::Openai | ProviderProtocol::ResponsesApi => {
                            ClaudeProviderProtocol::Openai
                        }
                        ProviderProtocol::Anthropic => ClaudeProviderProtocol::Anthropic,
                        ProviderProtocol::Google => ClaudeProviderProtocol::Google,
                    };
                    let _ = session_store
                        .set_key_claude_protocol(&key.id, Some(protocol))
                        .await;
                }
            }
            AIToolType::Gemini => {
                let current = key
                    .gemini_protocol
                    .map(|p| match p {
                        GeminiProviderProtocol::Google => ProviderProtocol::Google,
                        GeminiProviderProtocol::Openai => ProviderProtocol::Openai,
                        GeminiProviderProtocol::Anthropic => ProviderProtocol::Anthropic,
                    })
                    .unwrap_or(ProviderProtocol::Openai);
                if final_protocol != current {
                    let protocol = match final_protocol {
                        ProviderProtocol::Google => GeminiProviderProtocol::Google,
                        ProviderProtocol::Openai | ProviderProtocol::ResponsesApi => {
                            GeminiProviderProtocol::Openai
                        }
                        ProviderProtocol::Anthropic => GeminiProviderProtocol::Anthropic,
                    };
                    let _ = session_store
                        .set_key_gemini_protocol(&key.id, Some(protocol))
                        .await;
                }
            }
            _ => {}
        }
    }

    if let Some(active) = responses_api_support {
        let final_val = match active.load(Ordering::Relaxed) {
            1 => Some(true),
            2 => Some(false),
            _ => None,
        };
        if final_val.is_some() && final_val != key.responses_api_supported {
            let _ = session_store
                .set_key_responses_api_supported(&key.id, final_val)
                .await;
        }
    }
}

pub(crate) async fn cleanup_runtime_artifacts(
    codex_model_catalog_path: Option<&str>,
    pi_agent_dir: Option<&str>,
) {
    if let Some(path) = codex_model_catalog_path {
        let _ = tokio::fs::remove_file(path).await;
    }
    if let Some(dir) = pi_agent_dir {
        let _ = tokio::fs::remove_dir_all(dir).await;
    }
}

/// Writes a temporary `PI_CODING_AGENT_DIR` with `models.json`, `auth.json`,
/// and `settings.json` so Pi discovers the aivo custom provider.
///
/// When `port` is `Some`, the placeholder `PLACEHOLDER_LOOPBACK_URL` in
/// `AIVO_PI_MODELS_JSON` is patched with the real router port.
/// When `port` is `None`, the JSON already contains the real upstream URL.
async fn write_pi_agent_dir(env: &mut HashMap<String, String>, port: Option<u16>) -> Result<()> {
    let raw = env
        .get("AIVO_PI_MODELS_JSON")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_PI_MODELS_JSON"))?
        .clone();

    let models_json = match port {
        Some(p) => raw.replace(PLACEHOLDER_LOOPBACK_URL, &format!("http://127.0.0.1:{p}")),
        None => raw,
    };

    let dir = tempfile::Builder::new()
        .prefix("aivo-pi-")
        .tempdir()?
        .keep();

    tokio::try_join!(
        tokio::fs::write(dir.join("models.json"), &models_json),
        tokio::fs::write(dir.join("auth.json"), "{}"),
        tokio::fs::write(dir.join("settings.json"), "{}"),
    )?;

    // Symlink the real pi agent's bin/ directory (contains managed fd, rg binaries)
    // so pi doesn't re-download them into the temp dir.
    #[cfg(unix)]
    if let Some(home) = crate::services::system_env::home_dir() {
        let real_bin = home.join(".pi").join("agent").join("bin");
        let _ = tokio::fs::symlink(&real_bin, dir.join("bin")).await;
    }

    env.insert(
        "PI_CODING_AGENT_DIR".to_string(),
        dir.to_string_lossy().to_string(),
    );
    Ok(())
}

fn set_local_base_url(env: &mut HashMap<String, String>, key: &str, port: u16) {
    env.insert(key.to_string(), format!("http://127.0.0.1:{port}"));
}

fn patch_opencode_config_content(env: &mut HashMap<String, String>, port: u16) {
    let real_url = format!("http://127.0.0.1:{port}");
    if let Some(content) = env.get("OPENCODE_CONFIG_CONTENT").cloned() {
        let patched = content.replace(PLACEHOLDER_LOOPBACK_URL, &real_url);
        env.insert("OPENCODE_CONFIG_CONTENT".to_string(), patched);
    }
}

/// Starts the built-in AnthropicRouter and returns the port it bound to
async fn start_anthropic_router(env: &HashMap<String, String>) -> Result<u16> {
    use crate::services::{AnthropicRouter, AnthropicRouterConfig};

    let api_key = env
        .get("AIVO_ROUTER_API_KEY")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_ROUTER_API_KEY"))?
        .clone();

    let base_url = env
        .get("AIVO_ROUTER_BASE_URL")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_ROUTER_BASE_URL"))?
        .clone();

    let config = AnthropicRouterConfig {
        upstream_base_url: base_url,
        upstream_api_key: api_key,
    };

    let router = AnthropicRouter::new(config);
    let (port, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: anthropic router exited unexpectedly: {e}");
        }
    });
    Ok(port)
}

async fn start_anthropic_to_openai_router(
    env: &HashMap<String, String>,
) -> Result<(u16, Arc<AtomicU8>)> {
    use crate::services::provider_protocol::detect_provider_protocol;
    use crate::services::{AnthropicToOpenAIRouter, AnthropicToOpenAIRouterConfig};

    let api_key = env
        .get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_API_KEY")
        .ok_or_else(|| anyhow::anyhow!("Missing anthropic-to-openai router API key"))?
        .clone();

    let base_url = env
        .get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_BASE_URL")
        .ok_or_else(|| anyhow::anyhow!("Missing anthropic-to-openai router base URL"))?
        .clone();

    let model_prefix = env
        .get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_MODEL_PREFIX")
        .cloned();
    let requires_reasoning_content = env
        .get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_REQUIRE_REASONING")
        .map(|v| v == "1")
        .unwrap_or(false);
    let max_tokens_cap = env
        .get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_MAX_TOKENS_CAP")
        .and_then(|v| v.parse::<u64>().ok());
    let target_protocol = env
        .get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_UPSTREAM_PROTOCOL")
        .and_then(|value| ProviderProtocol::parse(value))
        .unwrap_or_else(|| detect_provider_protocol(&base_url));
    let config = AnthropicToOpenAIRouterConfig {
        target_base_url: base_url,
        target_api_key: api_key,
        target_protocol,
        model_prefix,
        requires_reasoning_content,
        max_tokens_cap,
    };

    let router = AnthropicToOpenAIRouter::new(config);
    let (port, active_protocol, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: anthropic-to-openai router exited unexpectedly: {e}");
        }
    });
    Ok((port, active_protocol))
}

async fn start_responses_to_chat_router(
    env: &HashMap<String, String>,
) -> Result<(u16, Arc<AtomicU8>, Arc<AtomicU8>)> {
    use crate::services::provider_protocol::detect_provider_protocol;
    use crate::services::{ResponsesToChatRouter, ResponsesToChatRouterConfig};

    let api_key = env
        .get("AIVO_RESPONSES_TO_CHAT_ROUTER_API_KEY")
        .ok_or_else(|| anyhow::anyhow!("Missing responses-to-chat router API key"))?
        .clone();

    let base_url = env
        .get("AIVO_RESPONSES_TO_CHAT_ROUTER_BASE_URL")
        .ok_or_else(|| anyhow::anyhow!("Missing responses-to-chat router base URL"))?
        .clone();

    let model_prefix = env
        .get("AIVO_RESPONSES_TO_CHAT_ROUTER_MODEL_PREFIX")
        .cloned();
    let requires_reasoning_content = env
        .get("AIVO_RESPONSES_TO_CHAT_ROUTER_REQUIRE_REASONING")
        .map(|v| v == "1")
        .unwrap_or(false);
    let actual_model = env
        .get("AIVO_RESPONSES_TO_CHAT_ROUTER_ACTUAL_MODEL")
        .cloned();
    let max_tokens_cap = env
        .get("AIVO_RESPONSES_TO_CHAT_ROUTER_MAX_TOKENS_CAP")
        .and_then(|v| v.parse::<u64>().ok());
    let target_protocol = env
        .get("AIVO_RESPONSES_TO_CHAT_ROUTER_UPSTREAM_PROTOCOL")
        .and_then(|value| ProviderProtocol::parse(value))
        .unwrap_or_else(|| detect_provider_protocol(&base_url));
    let responses_api_supported = match env
        .get("AIVO_RESPONSES_TO_CHAT_ROUTER_RESPONSES_API")
        .map(|v| v.as_str())
    {
        Some("1") => Some(true),
        Some("0") => Some(false),
        _ => None,
    };

    let router = ResponsesToChatRouter::new(ResponsesToChatRouterConfig {
        target_base_url: base_url,
        api_key,
        target_protocol,
        copilot_token_manager: None,
        model_prefix,
        requires_reasoning_content,
        actual_model,
        max_tokens_cap,
        responses_api_supported,
    });
    let (port, active_protocol, responses_api, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: responses-to-chat router exited unexpectedly: {e}");
        }
    });
    Ok((port, active_protocol, responses_api))
}

async fn start_gemini_router(env: &HashMap<String, String>) -> Result<(u16, Arc<AtomicU8>)> {
    use crate::services::provider_protocol::detect_provider_protocol;
    use crate::services::{GeminiRouter, GeminiRouterConfig};

    let api_key = env
        .get("AIVO_GEMINI_ROUTER_API_KEY")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_GEMINI_ROUTER_API_KEY"))?
        .clone();

    let base_url = env
        .get("AIVO_GEMINI_ROUTER_BASE_URL")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_GEMINI_ROUTER_BASE_URL"))?
        .clone();

    let requires_reasoning_content = env
        .get("AIVO_GEMINI_ROUTER_REQUIRE_REASONING")
        .map(|v| v == "1")
        .unwrap_or(false);
    let max_tokens_cap = env
        .get("AIVO_GEMINI_ROUTER_MAX_TOKENS_CAP")
        .and_then(|v| v.parse::<u64>().ok());
    let upstream_protocol = env
        .get("AIVO_GEMINI_ROUTER_UPSTREAM_PROTOCOL")
        .and_then(|value| ProviderProtocol::parse(value))
        .unwrap_or_else(|| detect_provider_protocol(&base_url));
    let router = GeminiRouter::new(GeminiRouterConfig {
        target_base_url: base_url,
        api_key,
        upstream_protocol,
        forced_model: None,
        copilot_token_manager: None,
        requires_reasoning_content,
        max_tokens_cap,
    });
    let (port, active_protocol, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: gemini router exited unexpectedly: {e}");
        }
    });
    Ok((port, active_protocol))
}

async fn start_gemini_copilot_router(env: &HashMap<String, String>) -> Result<u16> {
    use crate::services::copilot_auth::CopilotTokenManager;
    use crate::services::{GeminiRouter, GeminiRouterConfig};

    let github_token = env
        .get("AIVO_COPILOT_GITHUB_TOKEN")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_COPILOT_GITHUB_TOKEN"))?
        .clone();

    let forced_model = env.get("AIVO_GEMINI_COPILOT_FORCED_MODEL").cloned();

    if forced_model.is_none() {
        eprintln!(
            "  {} Gemini + Copilot: no model specified. Gemini models are not available on \
             Copilot. Pass --model <model> (e.g., --model gpt-4o).",
            crate::style::yellow("Warning:")
        );
    }

    let router = GeminiRouter::new(GeminiRouterConfig {
        target_base_url: String::new(),
        api_key: String::new(),
        upstream_protocol: ProviderProtocol::Openai,
        forced_model,
        copilot_token_manager: Some(Arc::new(CopilotTokenManager::new(github_token))),
        requires_reasoning_content: false,
        max_tokens_cap: None,
    });
    let (port, _active_protocol, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: gemini copilot router exited unexpectedly: {e}");
        }
    });
    Ok(port)
}

async fn start_copilot_router(env: &HashMap<String, String>) -> Result<u16> {
    use crate::services::{CopilotRouter, CopilotRouterConfig};

    let github_token = env
        .get("AIVO_COPILOT_GITHUB_TOKEN")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_COPILOT_GITHUB_TOKEN"))?
        .clone();

    let router = CopilotRouter::new(CopilotRouterConfig { github_token });
    let (port, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: copilot router exited unexpectedly: {e}");
        }
    });
    Ok(port)
}

async fn start_responses_to_chat_copilot_router(env: &HashMap<String, String>) -> Result<u16> {
    use crate::services::copilot_auth::CopilotTokenManager;
    use crate::services::{ResponsesToChatRouter, ResponsesToChatRouterConfig};

    let github_token = env
        .get("AIVO_COPILOT_GITHUB_TOKEN")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_COPILOT_GITHUB_TOKEN"))?
        .clone();

    let router = ResponsesToChatRouter::new(ResponsesToChatRouterConfig {
        target_base_url: String::new(),
        api_key: String::new(),
        target_protocol: ProviderProtocol::Openai,
        copilot_token_manager: Some(Arc::new(CopilotTokenManager::new(github_token))),
        model_prefix: None,
        requires_reasoning_content: false,
        actual_model: None,
        max_tokens_cap: None,
        responses_api_supported: None,
    });
    let (port, _active_protocol, _responses_api, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: responses-to-chat copilot router exited unexpectedly: {e}");
        }
    });
    Ok(port)
}

#[cfg(test)]
mod tests {
    use super::patch_opencode_config_content;
    use std::collections::HashMap;

    #[test]
    fn patch_opencode_config_content_rewrites_placeholder_url() {
        let mut env = HashMap::from([(
            "OPENCODE_CONFIG_CONTENT".to_string(),
            "{\"baseUrl\":\"http://127.0.0.1:0\"}".to_string(),
        )]);

        patch_opencode_config_content(&mut env, 24860);

        assert_eq!(
            env.get("OPENCODE_CONFIG_CONTENT").unwrap(),
            "{\"baseUrl\":\"http://127.0.0.1:24860\"}"
        );
    }

    #[test]
    fn patch_opencode_config_content_ignores_missing_payload() {
        let mut env = HashMap::new();
        patch_opencode_config_content(&mut env, 24860);
        assert!(env.is_empty());
    }

    #[test]
    fn set_local_base_url_inserts_loopback_address() {
        use super::set_local_base_url;
        let mut env = HashMap::new();
        set_local_base_url(&mut env, "ANTHROPIC_BASE_URL", 9999);
        assert_eq!(
            env.get("ANTHROPIC_BASE_URL").unwrap(),
            "http://127.0.0.1:9999"
        );
    }
}
