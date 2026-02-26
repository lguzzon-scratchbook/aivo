/**
 * Main entry point for the aivo CLI.
 * Initializes services with dependency injection and routes commands to handlers.
 */
use std::process;

use clap::Parser;

mod cli;
mod commands;
mod errors;
mod services;
mod style;
mod version;

use cli::{Cli, Commands};
use commands::{KeysCommand, RunCommand, UpdateCommand};
use errors::ExitCode;
use services::{AILauncher, EnvironmentInjector, SessionStore};

/// Known AI tool names that can be used as shortcut aliases for `run`.
const TOOL_ALIASES: &[&str] = &["claude", "codex", "gemini"];

/// Main entry point for the CLI
#[tokio::main(flavor = "current_thread")]
async fn main() {
    // Rewrite `aivo <tool> ...` to `aivo run <tool> ...` for known tool aliases
    let raw_args: Vec<String> = std::env::args().collect();
    let args = if raw_args.len() > 1 && TOOL_ALIASES.contains(&raw_args[1].as_str()) {
        let mut rewritten = vec![raw_args[0].clone(), "run".to_string()];
        rewritten.extend_from_slice(&raw_args[1..]);
        Cli::parse_from(rewritten)
    } else {
        Cli::parse()
    };

    // Initialize services early so we can show active key in help
    let session_store = SessionStore::new();

    // Handle help and version flags at the top level
    if args.help {
        print_help();
        print_active_key(&session_store).await;
        process::exit(0);
    }

    if args.version {
        print_version();
        process::exit(0);
    }

    // Get the command or show help if none provided
    let command = match args.command {
        Some(cmd) => cmd,
        None => {
            print_help();
            print_active_key(&session_store).await;
            process::exit(0);
        }
    };

    // Route to command handler
    let exit_code = match command {
        Commands::Keys(keys_args) => {
            let command = KeysCommand::new(session_store);
            let action = keys_args.action.as_deref();
            let args: Vec<_> = keys_args.args.iter().map(|s| s.as_str()).collect();
            command.execute(action, Some(&args)).await
        }

        Commands::Run(run_args) => {
            let env_injector = EnvironmentInjector::new();
            let ai_launcher = AILauncher::new(session_store.clone(), env_injector);
            let command = RunCommand::new(ai_launcher);

            // Re-extract aivo flags from passthrough args that clap's trailing_var_arg
            // may have swallowed (e.g. `aivo run claude --agent-name foo --model opus`
            // puts --model into args instead of parsing it as an aivo flag).
            let mut model = run_args.model;
            let mut debug = run_args.debug;
            let mut env_strings = run_args.envs;
            let mut remaining_args = Vec::new();
            let mut i = 0;
            let args = &run_args.args;
            while i < args.len() {
                let arg = &args[i];
                if let Some(value) = arg.strip_prefix("--model=") {
                    if !value.is_empty() && model.is_none() {
                        model = Some(value.to_string());
                    } else {
                        remaining_args.push(arg.clone());
                    }
                } else if (arg == "--model" || arg == "-m") && model.is_none() {
                    if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                        model = Some(args[i + 1].clone());
                        i += 1;
                    } else {
                        remaining_args.push(arg.clone());
                    }
                } else if arg == "--debug" {
                    debug = true;
                } else if let Some(value) = arg
                    .strip_prefix("--env=")
                    .or_else(|| arg.strip_prefix("-e="))
                {
                    if !value.is_empty() {
                        env_strings.push(value.to_string());
                    }
                } else if (arg == "--env" || arg == "-e") && i + 1 < args.len() {
                    env_strings.push(args[i + 1].clone());
                    i += 1;
                } else {
                    remaining_args.push(arg.clone());
                }
                i += 1;
            }

            let env = if !env_strings.is_empty() {
                let mut map = std::collections::HashMap::new();
                for env_str in &env_strings {
                    if let Some((key, value)) = env_str.split_once('=') {
                        map.insert(key.to_string(), value.to_string());
                    } else {
                        eprintln!(
                            "{} Ignoring malformed env value '{}' (expected KEY=VALUE format)",
                            style::yellow("Warning:"),
                            env_str
                        );
                    }
                }
                Some(map)
            } else {
                None
            };

            command
                .execute(run_args.tool.as_deref(), remaining_args, debug, model, env)
                .await
        }

        Commands::Update => match UpdateCommand::new() {
            Ok(command) => command.execute().await,
            Err(e) => {
                eprintln!(
                    "{} Failed to initialize update command: {}",
                    style::red("Error:"),
                    e
                );
                ExitCode::UserError
            }
        },
    };

    process::exit(exit_code.code());
}

/// Prints the active key info in the same format as `aivo keys`.
async fn print_active_key(session_store: &SessionStore) {
    let keys = session_store.get_keys().await.unwrap_or_default();
    let active_key = session_store.get_active_key().await.ok().flatten();

    println!("  {}", style::bold("Active key:"));
    if keys.is_empty() {
        println!(
            "    {} {}",
            style::dim("No keys found. Add one with"),
            style::bold("aivo keys add")
        );
    } else {
        for key in &keys {
            let is_active = active_key.as_ref().map(|k| k.id == key.id).unwrap_or(false);
            let indicator = if is_active {
                style::bullet_symbol()
            } else {
                style::empty_bullet_symbol()
            };
            let id_padded = format!("{:<4}", key.id);
            println!(
                "    {} {}  {}  {}",
                indicator,
                style::cyan(&id_padded),
                key.name,
                style::dim(&key.base_url)
            );
        }
    }
    println!();
}

/// Prints help information
fn print_help() {
    println!();
    println!(
        "  {} {}",
        style::cyan("aivo"),
        style::dim(format!("v{}", version::VERSION))
    );
    println!("  {}", style::dim("CLI for AI coding assistants"));
    println!();
    println!("  {} aivo [options] [command]", style::bold("Usage:"));
    println!();
    println!("  {}", style::bold("Commands:"));
    println!(
        "    {}  {}",
        style::cyan("run    "),
        style::dim("Run AI tools (claude, codex, gemini) - all args passed through")
    );
    println!(
        "    {}  {}",
        style::cyan("keys   "),
        style::dim("Manage API keys (list, use <id|name>, rm <id|name>, add)")
    );
    println!(
        "    {}  {}",
        style::cyan("update "),
        style::dim("Update the CLI tool to the latest version")
    );
    println!();
    println!("  {}", style::bold("Aliases:"));
    println!(
        "    {}  {}",
        style::cyan("claude "),
        style::dim("Shortcut for 'aivo run claude'")
    );
    println!(
        "    {}  {}",
        style::cyan("codex  "),
        style::dim("Shortcut for 'aivo run codex'")
    );
    println!(
        "    {}  {}",
        style::cyan("gemini "),
        style::dim("Shortcut for 'aivo run gemini'")
    );
    println!();
    println!("  {}", style::bold("Options:"));
    println!(
        "    {}  Display help information",
        style::dim("-h, --help    ")
    );
    println!(
        "    {}  Display the current version",
        style::dim("-v, --version ")
    );
    println!();
}

/// Prints version information
fn print_version() {
    println!(
        "{} {}",
        style::cyan("aivo"),
        style::dim(format!("v{}", version::VERSION))
    );
}
