/**
 * RunCommand handler for unified AI tool launching.
 */
use std::collections::HashMap;

use crate::errors::ExitCode;
use crate::services::ai_launcher::{AILauncher, AIToolType, LaunchOptions};
use crate::style;

/// RunCommand provides a unified interface to launch AI tools
pub struct RunCommand {
    ai_launcher: AILauncher,
}

impl RunCommand {
    pub fn new(ai_launcher: AILauncher) -> Self {
        Self { ai_launcher }
    }

    /// Executes the run command with the specified AI tool
    pub async fn execute(
        &self,
        tool: Option<&str>,
        args: Vec<String>,
        debug: bool,
        model: Option<String>,
        env: Option<HashMap<String, String>>,
    ) -> ExitCode {
        match self.execute_internal(tool, args, debug, model, env).await {
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
    ) -> anyhow::Result<ExitCode> {
        let tool = match tool {
            Some(t) => t,
            None => {
                self.show_usage();
                return Ok(ExitCode::UserError);
            }
        };

        // Handle help flags
        if tool == "--help" || tool == "-h" {
            self.show_usage();
            return Ok(ExitCode::Success);
        }

        // Validate tool
        let ai_tool = match AIToolType::parse(tool) {
            Some(t) => t,
            None => {
                eprintln!("{} Unknown AI tool '{}'", style::red("Error:"), tool);
                eprintln!();
                eprintln!("Available tools:");
                eprintln!("  {}  {}", style::cyan("claude"), style::dim("- Claude AI"));
                eprintln!("  {}  {}", style::cyan("codex"), style::dim("- Codex AI"));
                eprintln!("  {}  {}", style::cyan("gemini"), style::dim("- Gemini AI"));
                eprintln!();
                eprintln!(
                    "{}",
                    style::dim("Usage: aivo run <tool> [options] [args...]")
                );
                return Ok(ExitCode::UserError);
            }
        };

        // Launch the AI tool
        let options = LaunchOptions {
            tool: ai_tool,
            args,
            debug,
            model,
            env,
        };

        let exit_code = self.ai_launcher.launch(&options).await?;
        Ok(match exit_code {
            0 => ExitCode::Success,
            n => ExitCode::ToolExit(n),
        })
    }

    /// Shows usage information
    fn show_usage(&self) {
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
            style::cyan("--env <k=v>"),
            style::dim("Inject environment variable")
        );
        println!(
            "  {}      {}",
            style::cyan("--debug"),
            style::dim("Enable debug output")
        );
        println!();
        println!("{}", style::bold("Tools:"));
        println!("  {}  {}", style::cyan("claude"), style::dim("- Claude AI"));
        println!("  {}  {}", style::cyan("codex"), style::dim("- Codex AI"));
        println!("  {}  {}", style::cyan("gemini"), style::dim("- Gemini AI"));
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
    }
}
