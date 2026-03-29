/**
 * RunCommand handler for unified AI tool launching.
 */
use std::collections::HashMap;

use anyhow::Result;
use reqwest::Client;

use crate::commands::models::{
    fetch_models_for_select, prompt_model_picker, resolve_model_placeholder,
};
use crate::commands::print_launch_preview;
use crate::errors::ExitCode;
use crate::services::ai_launcher::{AILauncher, AIToolType, LaunchOptions};
use crate::services::http_utils;
use crate::services::models_cache::ModelsCache;
use crate::services::session_store::{ApiKey, SessionStore};
use crate::style;

/// RunCommand provides a unified interface to launch AI tools
pub struct RunCommand {
    session_store: SessionStore,
    ai_launcher: AILauncher,
    cache: ModelsCache,
}

impl RunCommand {
    pub fn new(session_store: SessionStore, ai_launcher: AILauncher, cache: ModelsCache) -> Self {
        Self {
            session_store,
            ai_launcher,
            cache,
        }
    }

    /// Resolves the model to use when --model flag is provided.
    /// --model <value> → use as-is. --model (no value) → show picker.
    /// No --model flag → returns None (let the tool use its own default).
    /// Returns None when the picker was cancelled or no flag was given.
    async fn resolve_model(
        &self,
        client: &Client,
        key: &ApiKey,
        flag_model: Option<String>,
        refresh: bool,
        tool: AIToolType,
    ) -> Result<Option<String>> {
        match flag_model {
            // No --model flag → don't override, let the tool use its default
            None => return Ok(None),
            // --model <value> → use it as-is
            Some(ref m) if !m.is_empty() => return Ok(Some(m.clone())),
            // --model with no value → show picker
            Some(_) => {}
        }

        let models_list = if refresh {
            crate::commands::models::fetch_models_cached(client, key, &self.cache, true)
                .await
                .unwrap_or_default()
        } else {
            fetch_models_for_select(client, key, &self.cache).await
        };

        if models_list.is_empty() {
            anyhow::bail!(
                "No model configured and could not fetch model list. Use --model <name> to specify one."
            );
        }

        Ok(prompt_model_picker(models_list, Some(tool)))
    }

    /// Executes the run command with the specified AI tool
    #[allow(clippy::too_many_arguments)]
    pub async fn execute(
        &self,
        tool: Option<&str>,
        args: Vec<String>,
        debug: bool,
        dry_run: bool,
        refresh: bool,
        model: Option<String>,
        env: Option<HashMap<String, String>>,
        key_override: Option<ApiKey>,
    ) -> ExitCode {
        match self
            .execute_internal(
                tool,
                args,
                debug,
                dry_run,
                refresh,
                model,
                env,
                key_override,
            )
            .await
        {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                ExitCode::UserError
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn execute_internal(
        &self,
        tool: Option<&str>,
        args: Vec<String>,
        debug: bool,
        dry_run: bool,
        refresh: bool,
        model: Option<String>,
        env: Option<HashMap<String, String>>,
        key_override: Option<ApiKey>,
    ) -> anyhow::Result<ExitCode> {
        let tool = match tool {
            Some(t) => t,
            None => {
                Self::print_help();
                return Ok(ExitCode::UserError);
            }
        };

        // Handle help flags
        if tool == "--help" || tool == "-h" {
            Self::print_help();
            return Ok(ExitCode::Success);
        }

        // Validate tool
        let ai_tool = match AIToolType::parse(tool) {
            Some(t) => t,
            None => {
                eprintln!("{} Unknown AI tool '{}'", style::red("Error:"), tool);
                eprintln!();
                eprintln!("Available tools:");
                eprintln!(
                    "  {}    {}",
                    style::cyan("claude"),
                    style::dim("Claude Code")
                );
                eprintln!("  {}     {}", style::cyan("codex"), style::dim("Codex"));
                eprintln!("  {}    {}", style::cyan("gemini"), style::dim("Gemini"));
                eprintln!("  {}  {}", style::cyan("opencode"), style::dim("OpenCode"));
                eprintln!("  {}        {}", style::cyan("pi"), style::dim("Pi"));
                eprintln!();
                eprintln!(
                    "{}",
                    style::dim("Usage: aivo run <tool> [options] [args...]")
                );
                return Ok(ExitCode::UserError);
            }
        };

        let picker_was_requested = model.as_ref().is_some_and(|m| m.is_empty());
        let client = http_utils::router_http_client();
        let resolved_model = if let Some(ref key) = key_override {
            let result = self
                .resolve_model(&client, key, model, refresh, ai_tool)
                .await?;
            if picker_was_requested && result.is_none() {
                return Ok(ExitCode::Success);
            }
            result
        } else {
            // key_override is always resolved in main.rs before reaching here; this
            // branch is unreachable in normal operation. Bail defensively rather than
            // silently discarding the picker trigger.
            anyhow::bail!("Internal error: no active key available for model resolution");
        };

        if let Some(ref key) = key_override {
            let _ = self
                .session_store
                .set_last_selection(key, tool, resolved_model.as_deref())
                .await;
        }

        let launch_model = resolve_model_placeholder(resolved_model);

        // Launch the AI tool
        let options = LaunchOptions {
            tool: ai_tool,
            args,
            debug,
            model: launch_model,
            env,
            key_override,
        };

        if dry_run {
            let plan = self.ai_launcher.prepare_launch(&options).await?;
            print_launch_preview(&plan);
            return Ok(ExitCode::Success);
        }

        let exit_code = self.ai_launcher.launch(&options).await?;
        Ok(match exit_code {
            0 => ExitCode::Success,
            n => ExitCode::ToolExit(n),
        })
    }

    /// Shows usage information
    pub fn print_help() {
        println!("{} aivo run [tool] [args...]", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim("Launch an AI coding assistant with local API keys.")
        );
        println!(
            "{}",
            style::dim(
                "When no tool is provided, `aivo run` falls back to the saved `start` flow."
            )
        );
        println!(
            "{}",
            style::dim("All arguments are passed through to the underlying tool.")
        );
        println!();
        println!("{}", style::bold("Options:"));
        let print_opt = |flag: &str, desc: &str| {
            println!(
                "  {}{}",
                style::cyan(format!("{:<26}", flag)),
                style::dim(desc)
            );
        };
        print_opt("-m, --model <model>", "Specify AI model to use");
        print_opt(
            "-k, --key <id|name>",
            "Select API key by ID or name (-k opens key picker)",
        );
        print_opt("-r, --refresh", "Bypass cache and fetch fresh model list");
        print_opt("--env <k=v>", "Inject environment variable");
        print_opt(
            "--dry-run",
            "Print resolved command and environment without launching",
        );
        println!();
        println!("{}", style::bold("Tools:"));
        let print_tool = |label: &str, desc: &str| {
            println!(
                "  {}{}",
                style::cyan(format!("{:<12}", label)),
                style::dim(desc)
            );
        };
        print_tool("claude", "Claude Code");
        print_tool("codex", "Codex");
        print_tool("gemini", "Gemini");
        print_tool("opencode", "OpenCode");
        print_tool("pi", "Pi");
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo run claude"));
        println!(
            "  {}",
            style::dim("aivo run claude --model claude-sonnet-4.5")
        );
        println!("  {}", style::dim("aivo claude \"fix the login bug\""));
        println!("  {}", style::dim("aivo codex \"refactor this function\""));
        println!("  {}", style::dim("aivo gemini \"explain this code\""));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_ai_tools() {
        assert!(AIToolType::parse("claude").is_some());
        assert!(AIToolType::parse("codex").is_some());
        assert!(AIToolType::parse("gemini").is_some());
        assert!(AIToolType::parse("opencode").is_some());
        assert!(AIToolType::parse("pi").is_some());
    }

    #[test]
    fn test_invalid_ai_tool() {
        assert!(AIToolType::parse("unknown").is_none());
        assert!(AIToolType::parse("").is_none());
        assert!(AIToolType::parse("chatgpt").is_none());
    }

    #[test]
    fn test_ai_tool_type_display_names() {
        // Ensure all tools have valid string representations
        let tools = ["claude", "codex", "gemini", "opencode", "pi"];
        for tool in &tools {
            let parsed = AIToolType::parse(tool).unwrap();
            // Roundtrip: parsing should give a valid tool type
            assert!(
                matches!(
                    parsed,
                    AIToolType::Claude
                        | AIToolType::Codex
                        | AIToolType::Gemini
                        | AIToolType::Opencode
                        | AIToolType::Pi
                ),
                "Tool {} should parse to a valid AIToolType",
                tool
            );
        }
    }
}
