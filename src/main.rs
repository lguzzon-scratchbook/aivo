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

/// Main entry point for the CLI
#[tokio::main(flavor = "current_thread")]
async fn main() {
    let args = Cli::parse();

    // Handle help and version flags at the top level
    if args.help {
        print_help();
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
            process::exit(0);
        }
    };

    // Initialize services
    let session_store = SessionStore::new();

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
                } else if let Some(value) = arg.strip_prefix("--env=").or_else(|| arg.strip_prefix("-e=")) {
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
                .execute(
                    run_args.tool.as_deref(),
                    remaining_args,
                    debug,
                    model,
                    env,
                )
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

/// Prints help information
fn print_help() {
    println!();
    println!(
        "  {} {}",
        style::cyan("aivo"),
        style::dim(format!("v{}", version::VERSION))
    );
    println!(
        "  {}",
        style::dim("CLI for AI coding assistants")
    );
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
