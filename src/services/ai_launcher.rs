//! AILauncher service for spawning AI tool processes.
//! Handles process spawning with environment injection and stdio passthrough.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::process::Stdio;
use tokio::process::Command;
use tokio::signal;

use crate::errors::{CLIError, ErrorCategory};
use crate::services::environment_injector::EnvironmentInjector;
use crate::services::session_store::{ApiKey, SessionStore};

/// Supported AI tool types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AIToolType {
    Claude,
    Codex,
    Gemini,
}

impl AIToolType {
    /// Parses a string into an AIToolType
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "claude" => Some(Self::Claude),
            "codex" => Some(Self::Codex),
            "gemini" => Some(Self::Gemini),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Claude => "claude",
            Self::Codex => "codex",
            Self::Gemini => "gemini",
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
}

impl AILauncher {
    /// Creates a new AILauncher
    pub fn new(session_store: SessionStore, env_injector: EnvironmentInjector) -> Self {
        Self {
            session_store,
            env_injector,
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

        let tool_config = self.get_tool_config(options.tool, &key, options.model.as_deref());

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

    /// Gets tool-specific configuration including command and environment variables
    fn get_tool_config(&self, tool: AIToolType, key: &ApiKey, model: Option<&str>) -> ToolConfig {
        let env_vars = match tool {
            AIToolType::Claude => self.env_injector.for_claude(key, model),
            AIToolType::Codex => self.env_injector.for_codex(key, model),
            AIToolType::Gemini => self.env_injector.for_gemini(key, model),
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
    let (port, _handle) = router.start_background().await?;
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
    let (port, _handle) = router.start_background().await?;
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
    let (port, _handle) = router.start_background().await?;
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
