//! AILauncher service for spawning AI tool processes.
//! Handles process spawning with environment injection and stdio passthrough.

use anyhow::{Context, Result};
use reqwest::Client;
use std::collections::HashMap;
use std::io::{self, Write};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::process::Command;
use tokio::signal;

use crate::errors::{CLIError, ErrorCategory};
use crate::services::environment_injector::EnvironmentInjector;
use crate::services::models_cache::ModelsCache;
use crate::services::session_store::{ApiKey, SessionStore};

/// Supported AI tool types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AIToolType {
    Claude,
    Codex,
    Gemini,
    Opencode,
}

impl AIToolType {
    /// Parses a string into an AIToolType
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "claude" => Some(Self::Claude),
            "codex" => Some(Self::Codex),
            "gemini" => Some(Self::Gemini),
            "opencode" => Some(Self::Opencode),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Claude => "claude",
            Self::Codex => "codex",
            Self::Gemini => "gemini",
            Self::Opencode => "opencode",
        }
    }
}

/// Launch options for AI tools
#[derive(Debug, Clone)]
pub struct LaunchOptions {
    pub tool: AIToolType,
    pub args: Vec<String>,
    pub debug: bool,
    pub model: Option<String>,
    pub env: Option<HashMap<String, String>>,
    /// Temporary key override for this launch (does not persist to config)
    pub key_override: Option<ApiKey>,
}

/// Tool configuration including command and environment variables
#[derive(Debug, Clone)]
pub struct ToolConfig {
    pub command: String,
    pub env_vars: HashMap<String, String>,
}

/// AILauncher spawns AI tool processes with configured environment and stdio passthrough
#[derive(Debug, Clone)]
pub struct AILauncher {
    session_store: SessionStore,
    env_injector: EnvironmentInjector,
    cache: ModelsCache,
}

impl AILauncher {
    /// Creates a new AILauncher
    pub fn new(
        session_store: SessionStore,
        env_injector: EnvironmentInjector,
        cache: ModelsCache,
    ) -> Self {
        Self {
            session_store,
            env_injector,
            cache,
        }
    }

    /// Spawns an AI tool with configured environment and stdio passthrough
    pub async fn launch(&self, options: &LaunchOptions) -> Result<i32> {
        let key = match &options.key_override {
            Some(k) => k.clone(),
            None => match self.session_store.get_active_key().await? {
                Some(k) => k,
                None => {
                    return Err(CLIError::new(
                        "No API key configured. Please add a key with 'aivo keys add'.",
                        ErrorCategory::Auth,
                        None::<String>,
                        Some("Run 'aivo keys add' to add an API key"),
                    )
                    .into());
                }
            },
        };

        self.output_key_info(&key);

        let (model, opencode_models) = if options.tool == AIToolType::Opencode {
            let (selected_model, discovered_models) = self
                .resolve_opencode_model_config(&key, options.model.as_deref())
                .await?;
            (selected_model, Some(discovered_models))
        } else {
            (options.model.clone(), None)
        };
        let tool_config = self.get_tool_config(
            options.tool,
            &key,
            model.as_deref(),
            opencode_models.as_deref(),
        );

        let mut env =
            self.env_injector
                .merge(&tool_config.env_vars, options.env.as_ref(), options.debug);

        // Start built-in router for OpenRouter + Claude, update ANTHROPIC_BASE_URL with actual port
        if options.tool == AIToolType::Claude && env.contains_key("AIVO_USE_ROUTER") {
            let port = start_router(&env).await?;
            env.insert(
                "ANTHROPIC_BASE_URL".to_string(),
                format!("http://127.0.0.1:{}", port),
            );
        }

        // Start CodexRouter for non-OpenAI providers, update OPENAI_BASE_URL with actual port
        if options.tool == AIToolType::Codex && env.contains_key("AIVO_USE_CODEX_ROUTER") {
            let port = start_codex_router(&env).await?;
            env.insert(
                "OPENAI_BASE_URL".to_string(),
                format!("http://127.0.0.1:{}", port),
            );
        }

        // Start GeminiRouter for non-Google providers, update GOOGLE_GEMINI_BASE_URL with actual port
        if options.tool == AIToolType::Gemini && env.contains_key("AIVO_USE_GEMINI_ROUTER") {
            let port = start_gemini_router(&env).await?;
            env.insert(
                "GOOGLE_GEMINI_BASE_URL".to_string(),
                format!("http://127.0.0.1:{}", port),
            );
        }

        // For Claude, inject --teammate-mode in-process to run in single window
        let args = inject_claude_teammate_mode(options.tool, &options.args);

        // Spawn the process with inherited stdio
        self.spawn_process(&tool_config.command, &args, env).await
    }

    /// Outputs information about which key is being used
    fn output_key_info(&self, key: &ApiKey) {
        use crate::style;

        eprintln!(
            "  {} Using key: {} {}",
            style::success_symbol(),
            style::cyan(&key.name),
            style::dim(format!("({})", key.base_url))
        );
    }

    async fn resolve_opencode_model_config(
        &self,
        key: &ApiKey,
        model: Option<&str>,
    ) -> Result<(Option<String>, Vec<String>)> {
        let requested_model = model.map(|m| m.strip_prefix("aivo/").unwrap_or(m).to_string());
        let client = Client::new();

        // Check cache first — skip the spinner if we get a hit
        let fetch_result = if let Some(cached) = self.cache.get(&key.base_url).await {
            Ok(cached)
        } else {
            // Cache miss: show spinner while fetching from network
            let spinning = Arc::new(AtomicBool::new(true));
            let spinning_clone = spinning.clone();
            let spinner_handle = tokio::task::spawn_blocking(move || {
                let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
                let mut i = 0;
                while spinning_clone.load(Ordering::Relaxed) {
                    eprint!("\r{} Fetching models...", frames[i % frames.len()]);
                    let _ = io::stderr().flush();
                    std::thread::sleep(std::time::Duration::from_millis(80));
                    i += 1;
                }
            });

            // bypass_cache=true: we know it's a miss; fetch_models_cached will still write result to cache
            let result =
                crate::commands::models::fetch_models_cached(&client, key, &self.cache, true).await;

            spinning.store(false, Ordering::Relaxed);
            std::thread::sleep(std::time::Duration::from_millis(100));
            eprint!("\r \r");
            let _ = io::stderr().flush();
            let _ = spinner_handle.await;

            result
        };

        let mut models = match fetch_result {
            Ok(models) => models,
            Err(e) => {
                if let Some(requested_model) = requested_model.clone() {
                    return Ok((Some(requested_model.clone()), vec![requested_model]));
                }
                return Err(e).with_context(|| {
                    "Unable to determine an OpenCode model from your provider. Pass --model <provider/model>."
                });
            }
        };
        if let Some(requested_model) = requested_model {
            if !models.contains(&requested_model) {
                models.push(requested_model.clone());
            }
            models.sort();
            models.dedup();
            return Ok((Some(requested_model), models));
        }

        models.sort();
        models.dedup();

        let selected_model = models
            .iter()
            .find(|m| m.contains("claude") && m.contains("sonnet"))
            .or_else(|| models.first())
            .cloned()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "No models returned by provider. Pass --model <provider/model> for opencode."
                )
            })?;
        Ok((Some(selected_model), models))
    }

    /// Gets tool-specific configuration including command and environment variables
    fn get_tool_config(
        &self,
        tool: AIToolType,
        key: &ApiKey,
        model: Option<&str>,
        opencode_models: Option<&[String]>,
    ) -> ToolConfig {
        let env_vars = match tool {
            AIToolType::Claude => self.env_injector.for_claude(key, model),
            AIToolType::Codex => self.env_injector.for_codex(key, model),
            AIToolType::Gemini => self.env_injector.for_gemini(key, model),
            AIToolType::Opencode => self.env_injector.for_opencode(key, model, opencode_models),
        };

        ToolConfig {
            command: tool.as_str().to_string(),
            env_vars,
        }
    }

    /// Spawns a child process with stdio inheritance and returns its exit code
    #[cfg(unix)]
    async fn spawn_process(
        &self,
        command: &str,
        args: &[String],
        env: HashMap<String, String>,
    ) -> Result<i32> {
        let mut cmd = Command::new(command);
        cmd.args(args)
            .envs(&env)
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());

        let mut child = cmd
            .spawn()
            .with_context(|| format!("Failed to spawn {}", command))?;

        // Get the child PID for signal forwarding
        let child_id = child.id();

        // Set up signal forwarding
        let mut sigint = signal::unix::signal(signal::unix::SignalKind::interrupt())?;
        let mut sigterm = signal::unix::signal(signal::unix::SignalKind::terminate())?;

        // Wait for the child to complete, while also listening for signals
        let result = tokio::select! {
            status = child.wait() => {
                status.map(|s| s.code().unwrap_or(1))
            }
            _ = sigint.recv() => {
                // Forward SIGINT to child
                if let Some(id) = child_id {
                    let _ = nix::sys::signal::kill(nix::unistd::Pid::from_raw(id as i32), nix::sys::signal::SIGINT);
                }
                child.wait().await.map(|s| s.code().unwrap_or(130)) // 128 + SIGINT (2)
            }
            _ = sigterm.recv() => {
                // Forward SIGTERM to child
                if let Some(id) = child_id {
                    let _ = nix::sys::signal::kill(nix::unistd::Pid::from_raw(id as i32), nix::sys::signal::SIGTERM);
                }
                child.wait().await.map(|s| s.code().unwrap_or(143)) // 128 + SIGTERM (15)
            }
        };

        result.map_err(|e| e.into())
    }

    /// Spawns a child process with stdio inheritance and returns its exit code (non-Unix)
    #[cfg(not(unix))]
    async fn spawn_process(
        &self,
        command: &str,
        args: &[String],
        env: HashMap<String, String>,
    ) -> Result<i32> {
        let mut cmd = Command::new(command);
        cmd.args(args)
            .envs(&env)
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());

        let mut child = cmd
            .spawn()
            .with_context(|| format!("Failed to spawn {}", command))?;

        // On non-Unix platforms, just wait for the child
        let status = child.wait().await?;
        Ok(status.code().unwrap_or(1))
    }
}

/// Starts the built-in Claude Code Router and returns the port it bound to
async fn start_router(env: &HashMap<String, String>) -> Result<u16> {
    use crate::services::{ClaudeCodeRouter, RouterConfig};

    let api_key = env
        .get("AIVO_ROUTER_API_KEY")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_ROUTER_API_KEY"))?
        .clone();

    let base_url = env
        .get("AIVO_ROUTER_BASE_URL")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_ROUTER_BASE_URL"))?
        .clone();

    let config = RouterConfig {
        openrouter_base_url: base_url,
        openrouter_api_key: api_key,
    };

    let router = ClaudeCodeRouter::new(config);
    let (port, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: claude code router exited unexpectedly: {e}");
        }
    });
    Ok(port)
}

/// Starts the built-in CodexRouter for non-OpenAI providers and returns the port it bound to
async fn start_codex_router(env: &HashMap<String, String>) -> Result<u16> {
    use crate::services::{CodexRouter, CodexRouterConfig};

    let api_key = env
        .get("AIVO_CODEX_ROUTER_API_KEY")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_CODEX_ROUTER_API_KEY"))?
        .clone();

    let base_url = env
        .get("AIVO_CODEX_ROUTER_BASE_URL")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_CODEX_ROUTER_BASE_URL"))?
        .clone();

    let router = CodexRouter::new(CodexRouterConfig {
        target_base_url: base_url,
        api_key,
    });
    let (port, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: codex router exited unexpectedly: {e}");
        }
    });
    Ok(port)
}

/// Starts the built-in GeminiRouter for non-Google providers and returns the port it bound to
async fn start_gemini_router(env: &HashMap<String, String>) -> Result<u16> {
    use crate::services::{GeminiRouter, GeminiRouterConfig};

    let api_key = env
        .get("AIVO_GEMINI_ROUTER_API_KEY")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_GEMINI_ROUTER_API_KEY"))?
        .clone();

    let base_url = env
        .get("AIVO_GEMINI_ROUTER_BASE_URL")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_GEMINI_ROUTER_BASE_URL"))?
        .clone();

    let router = GeminiRouter::new(GeminiRouterConfig {
        target_base_url: base_url,
        api_key,
    });
    let (port, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: gemini router exited unexpectedly: {e}");
        }
    });
    Ok(port)
}

/// Injects `--teammate-mode in-process` for Claude if not already specified by the user.
/// This ensures Claude runs in a single window instead of using split panels.
fn inject_claude_teammate_mode(tool: AIToolType, args: &[String]) -> Vec<String> {
    if tool != AIToolType::Claude {
        return args.to_vec();
    }

    // Check if the user already specified --teammate-mode
    let has_teammate_mode = args
        .iter()
        .any(|a| a == "--teammate-mode" || a.starts_with("--teammate-mode="));
    if has_teammate_mode {
        return args.to_vec();
    }

    let mut new_args = vec!["--teammate-mode".to_string(), "in-process".to_string()];
    new_args.extend_from_slice(args);
    new_args
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ai_tool_type_from_str() {
        assert_eq!(AIToolType::parse("claude"), Some(AIToolType::Claude));
        assert_eq!(AIToolType::parse("Claude"), Some(AIToolType::Claude));
        assert_eq!(AIToolType::parse("CLAUDE"), Some(AIToolType::Claude));
        assert_eq!(AIToolType::parse("codex"), Some(AIToolType::Codex));
        assert_eq!(AIToolType::parse("gemini"), Some(AIToolType::Gemini));
        assert_eq!(AIToolType::parse("opencode"), Some(AIToolType::Opencode));
        assert_eq!(AIToolType::parse("unknown"), None);
    }

    #[test]
    fn test_inject_claude_teammate_mode_for_claude() {
        let args = vec!["--verbose".to_string(), "prompt".to_string()];
        let result = inject_claude_teammate_mode(AIToolType::Claude, &args);
        assert_eq!(
            result,
            vec!["--teammate-mode", "in-process", "--verbose", "prompt"]
        );
    }

    #[test]
    fn test_inject_claude_teammate_mode_skips_non_claude() {
        let args = vec!["--verbose".to_string()];
        let result = inject_claude_teammate_mode(AIToolType::Codex, &args);
        assert_eq!(result, vec!["--verbose"]);

        let result = inject_claude_teammate_mode(AIToolType::Gemini, &args);
        assert_eq!(result, vec!["--verbose"]);

        let result = inject_claude_teammate_mode(AIToolType::Opencode, &args);
        assert_eq!(result, vec!["--verbose"]);
    }

    #[test]
    fn test_inject_claude_teammate_mode_respects_user_flag() {
        // User specified --teammate-mode explicitly
        let args = vec![
            "--teammate-mode".to_string(),
            "split".to_string(),
            "prompt".to_string(),
        ];
        let result = inject_claude_teammate_mode(AIToolType::Claude, &args);
        assert_eq!(result, vec!["--teammate-mode", "split", "prompt"]);

        // User specified --teammate-mode=value format
        let args = vec!["--teammate-mode=split".to_string(), "prompt".to_string()];
        let result = inject_claude_teammate_mode(AIToolType::Claude, &args);
        assert_eq!(result, vec!["--teammate-mode=split", "prompt"]);
    }

    #[test]
    fn test_inject_claude_teammate_mode_empty_args() {
        let args: Vec<String> = vec![];
        let result = inject_claude_teammate_mode(AIToolType::Claude, &args);
        assert_eq!(result, vec!["--teammate-mode", "in-process"]);
    }
}
