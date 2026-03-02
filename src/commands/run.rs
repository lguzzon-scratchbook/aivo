/**
 * RunCommand handler for unified AI tool launching.
 */
use std::collections::HashMap;

use anyhow::Result;
use reqwest::Client;

use crate::commands::models::fetch_models_for_select;
use crate::errors::ExitCode;
use crate::services::ai_launcher::{AILauncher, AIToolType, LaunchOptions};
use crate::services::models_cache::ModelsCache;
use crate::services::session_store::{ApiKey, SessionStore};
use crate::style;
use crate::tui::FuzzySelect;

/// RunCommand provides a unified interface to launch AI tools
pub struct RunCommand {
    ai_launcher: AILauncher,
    session_store: SessionStore,
    cache: ModelsCache,
}

impl RunCommand {
    pub fn new(ai_launcher: AILauncher, session_store: SessionStore, cache: ModelsCache) -> Self {
        Self {
            ai_launcher,
            session_store,
            cache,
        }
    }

    /// Resolves the model to use: flag > persisted per-key > picker.
    /// Saves the resolved model to session_store so subsequent `aivo run` invocations
    /// remember the choice without needing the `--model` flag again.
    /// Returns None when the picker was cancelled.
    async fn resolve_model(
        &self,
        client: &Client,
        key: &ApiKey,
        flag_model: Option<String>,
    ) -> Result<Option<String>> {
        match flag_model {
            // --model with no value → force picker (bypass persisted model)
            Some(ref m) if m.is_empty() => {}
            // --model <value> → use it and save
            Some(model) => {
                let current = self.session_store.get_chat_model(&key.id).await?;
                if current.as_deref() != Some(&model) {
                    self.session_store.set_chat_model(&key.id, &model).await?;
                }
                return Ok(Some(model));
            }
            // No flag: check for persisted model
            None => {
                if let Some(persisted) = self.session_store.get_chat_model(&key.id).await? {
                    return Ok(Some(persisted));
                }
            }
        }

        // Show picker (either no model set, or --model with no value)
        let models_list = fetch_models_for_select(client, key, &self.cache).await;

        if models_list.is_empty() {
            anyhow::bail!(
                "No model configured and could not fetch model list. Use --model <name> to specify one."
            );
        }

        let selected = FuzzySelect::new()
            .with_prompt("Select model")
            .items(&models_list)
            .default(0)
            .interact_opt()
            .ok()
            .flatten()
            .map(|idx| models_list[idx].clone());

        if let Some(ref model) = selected {
            self.session_store.set_chat_model(&key.id, model).await?;
        }

        Ok(selected)
    }

    /// Executes the run command with the specified AI tool
    pub async fn execute(
        &self,
        tool: Option<&str>,
        args: Vec<String>,
        debug: bool,
        model: Option<String>,
        env: Option<HashMap<String, String>>,
        key_override: Option<ApiKey>,
    ) -> ExitCode {
        match self
            .execute_internal(tool, args, debug, model, env, key_override)
            .await
        {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                ExitCode::UserError
            }
        }
    }

    async fn execute_internal(
        &self,
        tool: Option<&str>,
        args: Vec<String>,
        debug: bool,
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
                eprintln!();
                eprintln!(
                    "{}",
                    style::dim("Usage: aivo run <tool> [options] [args...]")
                );
                return Ok(ExitCode::UserError);
            }
        };

        // Resolve model: picker triggered when model is None or Some("") and key is known
        let client = Client::new();
        let resolved_model = if let Some(ref key) = key_override {
            match self.resolve_model(&client, key, model).await? {
                Some(m) => Some(m),
                None => return Ok(ExitCode::Success), // user cancelled picker
            }
        } else {
            // key_override is always resolved in main.rs before reaching here; this
            // branch is unreachable in normal operation. Bail defensively rather than
            // silently discarding the picker trigger.
            anyhow::bail!("Internal error: no active key available for model resolution");
        };

        // Launch the AI tool
        let options = LaunchOptions {
            tool: ai_tool,
            args,
            debug,
            model: resolved_model,
            env,
            key_override,
        };

        let exit_code = self.ai_launcher.launch(&options).await?;
        Ok(match exit_code {
            0 => ExitCode::Success,
            n => ExitCode::ToolExit(n),
        })
    }

    /// Shows usage information
    pub fn print_help() {
        println!("{} aivo run <tool> [args...]", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim("Launch an AI coding assistant with local API keys.")
        );
        println!(
            "{}",
            style::dim("All arguments are passed through to the underlying tool.")
        );
        println!();
        println!("{}", style::bold("Options:"));
        println!(
            "  {}  {}",
            style::cyan("-m, --model <model>"),
            style::dim("Specify AI model to use")
        );
        println!(
            "  {}  {}",
            style::cyan("-k, --key <id|name>"),
            style::dim("Select API key by ID or name")
        );
        println!(
            "  {}          {}",
            style::cyan("--env <k=v>"),
            style::dim("Inject environment variable")
        );
        println!(
            "  {}              {}",
            style::cyan("--debug"),
            style::dim("Enable debug output")
        );
        println!();
        println!("{}", style::bold("Tools:"));
        println!(
            "  {}    {}",
            style::cyan("claude"),
            style::dim("Claude Code")
        );
        println!("  {}     {}", style::cyan("codex"), style::dim("Codex"));
        println!("  {}    {}", style::cyan("gemini"), style::dim("Gemini"));
        println!("  {}  {}", style::cyan("opencode"), style::dim("OpenCode"));
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo run claude"));
        println!(
            "  {}",
            style::dim("aivo run claude --model claude-sonnet-4.5")
        );
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
    }
}
