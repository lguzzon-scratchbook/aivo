/**
 * CLI argument parsing and command routing.
 * Uses clap for argument parsing.
 */
use clap::{Args, Parser, Subcommand};
use std::collections::HashMap;

/// The aivo CLI - unified access to AI coding assistants
#[derive(Parser, Debug)]
#[command(
    name = "aivo",
    about = "CLI tool for unified access to AI coding assistants (Claude, Codex, Gemini)",
    version = crate::version::VERSION,
    author = "yuanchuan",
    disable_help_flag = true,
    disable_version_flag = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Display help information
    #[arg(short, long, global = true, help = "Display help information")]
    pub help: bool,

    /// Display the current version
    #[arg(short, long, global = true, help = "Display the current version")]
    pub version: bool,
}

/// Available commands for the CLI
#[derive(Subcommand, Debug, Clone)]
pub enum Commands {
    /// Run AI tools (claude, codex, gemini) - all args passed through
    Run(RunArgs),

    /// Manage API keys (list, use <id|name>, rm <id|name>, add)
    Keys(KeysArgs),

    /// Update the CLI tool to the latest version
    Update,
}

/// Arguments for the keys command
#[derive(Args, Debug, Clone)]
pub struct KeysArgs {
    /// The action to perform (list, use, rm, add, cat)
    #[arg(
        value_name = "ACTION",
        help = "Action to perform: list, use, rm, add, cat"
    )]
    pub action: Option<String>,

    /// Additional arguments for the action (e.g., key ID or name)
    #[arg(value_name = "ARGS", help = "Additional arguments for the action")]
    pub args: Vec<String>,
}

/// Arguments for the run command
#[derive(Args, Debug, Clone)]
pub struct RunArgs {
    /// The AI tool to run (claude, codex, gemini)
    #[arg(value_name = "TOOL", help = "AI tool to run: claude, codex, or gemini")]
    pub tool: Option<String>,

    /// Specify AI model to use
    #[arg(short, long, value_name = "MODEL")]
    pub model: Option<String>,

    /// Enable debug output
    #[arg(long)]
    pub debug: bool,

    /// Inject environment variable (KEY=VALUE)
    #[arg(short, long = "env", value_name = "KEY=VALUE")]
    pub envs: Vec<String>,

    /// Additional arguments to pass through to the AI tool
    #[arg(
        value_name = "ARGS",
        help = "Arguments to pass through to the AI tool",
        trailing_var_arg = true,
        allow_hyphen_values = true
    )]
    pub args: Vec<String>,
}

/// Parse environment variable strings in the format KEY=VALUE
#[allow(dead_code)]
pub fn parse_env_vars(env_strings: &[String]) -> HashMap<String, String> {
    let mut env_map = HashMap::new();

    for env_str in env_strings {
        if let Some((key, value)) = env_str.split_once('=') {
            env_map.insert(key.to_string(), value.to_string());
        }
    }

    env_map
}

/// Get the list of valid commands
#[allow(dead_code)]
pub fn get_valid_commands() -> Vec<&'static str> {
    vec!["update", "keys", "run"]
}

/// Check if a command is a passthrough command (passes all args to underlying tool)
#[allow(dead_code)]
pub fn is_passthrough_command(command: &str) -> bool {
    command == "run"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_env_vars() {
        let env_strings = vec![
            "KEY1=value1".to_string(),
            "KEY2=value2".to_string(),
            "KEY3=nested=value".to_string(),
        ];

        let env_map = parse_env_vars(&env_strings);
        assert_eq!(env_map.get("KEY1"), Some(&"value1".to_string()));
        assert_eq!(env_map.get("KEY2"), Some(&"value2".to_string()));
        assert_eq!(env_map.get("KEY3"), Some(&"nested=value".to_string()));
    }

    #[test]
    fn test_parse_env_vars_invalid() {
        let env_strings = vec!["NO_EQUALS".to_string(), "VALID=key".to_string()];

        let env_map = parse_env_vars(&env_strings);
        assert_eq!(env_map.len(), 1);
        assert_eq!(env_map.get("VALID"), Some(&"key".to_string()));
    }

    #[test]
    fn test_get_valid_commands() {
        let commands = get_valid_commands();
        assert!(commands.contains(&"update"));
        assert!(commands.contains(&"keys"));
        assert!(commands.contains(&"run"));
    }

    #[test]
    fn test_is_passthrough_command() {
        assert!(is_passthrough_command("run"));
        assert!(!is_passthrough_command("keys"));
    }

    #[test]
    fn test_run_args_debug_flag() {
        let cli = Cli::try_parse_from(["aivo", "run", "claude", "--debug"]).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.tool, Some("claude".to_string()));
            assert!(run_args.debug);
            assert!(!run_args.args.contains(&"--debug".to_string()));
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_run_args_model_flag() {
        // --model value
        let cli = Cli::try_parse_from(["aivo", "run", "claude", "--model", "gpt-5"]).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.model, Some("gpt-5".to_string()));
        } else {
            panic!("Expected Run command");
        }

        // --model=value
        let cli =
            Cli::try_parse_from(["aivo", "run", "claude", "--model=gpt-5"]).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.model, Some("gpt-5".to_string()));
        } else {
            panic!("Expected Run command");
        }

        // -m value
        let cli = Cli::try_parse_from(["aivo", "run", "claude", "-m", "gpt-5"]).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.model, Some("gpt-5".to_string()));
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_run_args_env_flag() {
        let cli =
            Cli::try_parse_from(["aivo", "run", "claude", "--env", "FOO=bar", "-e", "BAZ=qux"])
                .unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.envs, vec!["FOO=bar", "BAZ=qux"]);
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_run_args_passthrough() {
        // Arguments not matching aivo flags should be passed through
        let cli = Cli::try_parse_from([
            "aivo", "run", "claude", "--debug", "--", "--some-tool-flag", "value",
        ])
        .unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert!(run_args.debug);
            assert!(run_args.args.contains(&"--some-tool-flag".to_string()));
            assert!(run_args.args.contains(&"value".to_string()));
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_run_args_passthrough_claude_teammate_flags() {
        // Real-world usage: claude flags mixed with aivo --model flag.
        // When unknown flags appear before --model, clap's trailing_var_arg swallows
        // --model into args. main.rs re-extracts aivo flags from args at runtime.
        let cli = Cli::try_parse_from([
            "aivo",
            "run",
            "claude",
            "--agent-name",
            "senior-engineer",
            "--team-name",
            "ai-gateway-team",
            "--agent-color",
            "blue",
            "--parent-session-id",
            "df205d21-e955-421c-b2b9-5ff42c900cb6",
            "--agent-type",
            "general-purpose",
            "--dangerously-skip-permissions",
            "--model",
            "claude-opus-4-6",
        ])
        .unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.tool, Some("claude".to_string()));
            // --model gets swallowed into args by trailing_var_arg; main.rs re-extracts it
            assert!(run_args.args.contains(&"--model".to_string()));
            assert!(run_args.args.contains(&"claude-opus-4-6".to_string()));
            // All unknown flags pass through
            assert!(run_args.args.contains(&"--agent-name".to_string()));
            assert!(run_args.args.contains(&"senior-engineer".to_string()));
            assert!(run_args.args.contains(&"--team-name".to_string()));
            assert!(run_args.args.contains(&"ai-gateway-team".to_string()));
            assert!(run_args.args.contains(&"--dangerously-skip-permissions".to_string()));
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_run_args_model_before_unknown_flags() {
        // When --model comes before unknown flags, clap parses it directly
        let cli = Cli::try_parse_from([
            "aivo",
            "run",
            "claude",
            "--model",
            "claude-opus-4-6",
            "--agent-name",
            "senior-engineer",
            "--dangerously-skip-permissions",
        ])
        .unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.model, Some("claude-opus-4-6".to_string()));
            assert!(run_args.args.contains(&"--agent-name".to_string()));
            assert!(run_args.args.contains(&"--dangerously-skip-permissions".to_string()));
        } else {
            panic!("Expected Run command");
        }
    }
}
