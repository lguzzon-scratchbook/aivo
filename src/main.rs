/**
 * Main entry point for the aivo CLI.
 * Initializes services with dependency injection and routes commands to handlers.
 */
use std::process;

use clap::Parser;

mod cli;
mod commands;
mod constants;
mod errors;
mod key_resolution;
mod services;
mod style;
mod tui;
mod version;

use cli::{Cli, Commands};
use commands::{
    AliasCommand, ChatCommand, InfoCommand, KeysCommand, ModelsCommand, RunCommand, ServeCommand,
    StartCommand, StartFlowArgs, StatsCommand, UpdateCommand, truncate_url_for_display,
};
use errors::ExitCode;
use key_resolution::{KeyLookupMode, KeyResolution, key_or_exit, resolve_key_override};
use services::{AILauncher, EnvironmentInjector, SessionStore};

/// Known AI tool names that can be used as shortcut aliases for `run`.
const TOOL_ALIASES: &[&str] = &["claude", "codex", "gemini", "opencode", "pi"];

/// Main entry point for the CLI
#[tokio::main(flavor = "current_thread")]
async fn main() {
    let raw_args: Vec<String> = std::env::args().collect();
    let args = Cli::parse_from(rewrite_cli_args(raw_args));

    // Initialize services early so we can show active key in help
    let session_store = SessionStore::new();
    let models_cache = services::ModelsCache::new();

    // Handle help and version flags at the top level
    if args.help {
        match &args.command {
            Some(Commands::Run(_)) => {
                RunCommand::print_help();
            }
            Some(Commands::Keys(_)) => {
                KeysCommand::print_help();
            }
            Some(Commands::Chat(_)) => {
                ChatCommand::print_help();
            }
            Some(Commands::Models(_)) => {
                ModelsCommand::print_help();
            }
            Some(Commands::Serve(_)) => {
                ServeCommand::print_help();
            }
            Some(Commands::Alias(_)) => {
                AliasCommand::print_help();
            }
            Some(Commands::Ls(_)) => {
                InfoCommand::print_help();
            }
            Some(Commands::Stats(_)) => {
                StatsCommand::print_help();
            }
            Some(Commands::Update(_)) => {
                UpdateCommand::print_help();
            }
            None => {
                print_help();
                print_active_key(&session_store).await;
            }
        }
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
        Commands::Alias(alias_args) => {
            let command = AliasCommand::new(session_store);
            command.execute(alias_args).await
        }

        Commands::Keys(keys_args) => {
            let command = KeysCommand::new(session_store);
            command.execute(keys_args).await
        }

        Commands::Chat(chat_args) => {
            let key_override = key_or_exit(
                resolve_key_override(
                    &session_store,
                    chat_args.key.as_deref(),
                    KeyLookupMode::RequireActiveOrPrompt,
                )
                .await,
            );
            let model = resolve_model_alias(&session_store, chat_args.model).await;
            let command = ChatCommand::new(session_store, models_cache.clone());
            command
                .execute(
                    model,
                    chat_args.execute,
                    chat_args.attachments,
                    chat_args.refresh,
                    key_override,
                )
                .await
        }

        Commands::Run(run_args) => {
            let env_injector = EnvironmentInjector::new();
            let ai_launcher =
                AILauncher::new(session_store.clone(), env_injector, models_cache.clone());

            // Re-extract aivo flags from passthrough args that clap's trailing_var_arg
            // may have swallowed (e.g. `aivo run claude --agent-name foo --model opus`
            // puts --model into args instead of parsing it as an aivo flag).
            let extracted = extract_aivo_flags(
                run_args.model,
                run_args.key,
                run_args.debug,
                run_args.dry_run,
                run_args.envs,
                &run_args.args,
            );
            let model = resolve_model_alias(&session_store, extracted.model).await;
            let key_flag = extracted.key_flag;
            let debug = extracted.debug;
            let dry_run = extracted.dry_run;
            let env_strings = extracted.env_strings;
            let remaining_args = extracted.remaining_args;

            if run_args.tool.is_none() {
                if !remaining_args.is_empty() {
                    eprintln!(
                        "{} `aivo run` without a tool does not accept passthrough args",
                        style::red("Error:")
                    );
                    eprintln!(
                        "  {}",
                        style::dim("Use `aivo run <tool> ...` for passthrough flags.")
                    );
                    process::exit(ExitCode::UserError.code());
                }

                let command = StartCommand::new(session_store, ai_launcher, models_cache);
                command
                    .execute(StartFlowArgs {
                        model,
                        key: key_flag,
                        tool: None,
                        debug,
                        dry_run,
                        yes: false,
                        envs: env_strings,
                    })
                    .await
            } else {
                let command = RunCommand::new(ai_launcher, models_cache);

                let key_override = key_or_exit(
                    resolve_key_override(
                        &session_store,
                        key_flag.as_deref(),
                        KeyLookupMode::RequireActiveOrPrompt,
                    )
                    .await,
                );

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
                        dry_run,
                        model,
                        env,
                        key_override,
                    )
                    .await
            }
        }

        Commands::Models(models_args) => {
            let key_override = key_or_exit(
                resolve_key_override(
                    &session_store,
                    models_args.key.as_deref(),
                    KeyLookupMode::RequireActiveOrPrompt,
                )
                .await,
            );
            let command = ModelsCommand::new(session_store, models_cache);
            command
                .execute(key_override, models_args.refresh, models_args.search)
                .await
        }

        Commands::Serve(serve_args) => {
            let key_override = match resolve_key_override(
                &session_store,
                serve_args.key.as_deref(),
                KeyLookupMode::PreferActiveAllowNone,
            )
            .await
            {
                Ok(KeyResolution::Selected(key)) => Some(key),
                Ok(KeyResolution::Cancelled) => process::exit(ExitCode::Success.code()),
                Ok(KeyResolution::MissingAuth) => None,
                Err(e) => {
                    eprintln!("{} {}", style::red("Error:"), e);
                    process::exit(ExitCode::UserError.code());
                }
            };
            // Build failover key list when --failover is enabled
            let failover_keys = if serve_args.failover {
                let mut all_keys = session_store.get_keys().await.unwrap_or_default();
                // Decrypt and exclude the primary key
                let primary_id = key_override.as_ref().map(|k| k.id.clone());
                all_keys.retain(|k| primary_id.as_deref() != Some(&k.id));
                all_keys.iter_mut().for_each(|k| {
                    let _ = SessionStore::decrypt_key_secret(k);
                });
                all_keys
            } else {
                Vec::new()
            };
            let command = ServeCommand::new();
            command
                .execute(serve_args.port, key_override, serve_args.log, failover_keys)
                .await
        }

        Commands::Ls(ls_args) => {
            let command = InfoCommand::new(session_store, models_cache);
            command.execute(ls_args.ping).await
        }

        Commands::Stats(stats_args) => {
            let command = StatsCommand::new(session_store);
            command.execute(stats_args).await
        }

        Commands::Update(update_args) => match UpdateCommand::new() {
            Ok(command) => command.execute(update_args.force).await,
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
fn rewrite_cli_args(raw_args: Vec<String>) -> Vec<String> {
    if raw_args.len() <= 1 {
        return raw_args;
    }

    if TOOL_ALIASES.contains(&raw_args[1].as_str()) {
        let mut rewritten = vec![raw_args[0].clone(), "run".to_string()];
        rewritten.extend_from_slice(&raw_args[1..]);
        return rewritten;
    }

    if raw_args[1] == "use" {
        let mut rewritten = vec![raw_args[0].clone(), "keys".to_string(), "use".to_string()];
        rewritten.extend_from_slice(&raw_args[2..]);
        return rewritten;
    }

    if raw_args[1] == "ping" {
        let mut rewritten = vec![raw_args[0].clone(), "keys".to_string(), "ping".to_string()];
        rewritten.extend_from_slice(&raw_args[2..]);
        return rewritten;
    }

    if raw_args[1] == "-x" || raw_args[1] == "--execute" {
        let mut rewritten = vec![raw_args[0].clone(), "chat".to_string()];
        rewritten.extend_from_slice(&raw_args[1..]);
        return rewritten;
    }

    raw_args
}
/// Prints the active key info.
/// Only prints if there is an active key configured.
async fn print_active_key(session_store: &SessionStore) {
    let active_key = match session_store.get_active_key_info().await.ok().flatten() {
        Some(key) => key,
        None => return,
    };

    println!();
    println!("{}", style::bold("Active key:"));
    let id_padded = format!("{:<3}", active_key.short_id());
    println!(
        "  {} {}  {}  {}",
        style::bullet_symbol(),
        style::cyan(&id_padded),
        active_key.display_name(),
        style::dim(truncate_url_for_display(&active_key.base_url, 50))
    );
}

/// Prints help information
fn print_help() {
    println!(
        "{} {} {}",
        style::cyan("aivo"),
        style::dim(format!("v{}", version::VERSION)),
        style::dim("— CLI for AI coding assistants")
    );
    println!();
    println!("{} aivo <command> [options]", style::bold("Usage:"));
    println!();
    println!("{}", style::bold("Commands:"));
    let print_cmd = |name: &str, desc: &str| {
        let padded = format!("{:<10}", name);
        println!("  {}{}", style::cyan(&padded), style::dim(desc));
    };
    print_cmd("run", "Launch AI tool, or use the saved start flow");
    print_cmd("chat", "Start the interactive chat TUI");
    print_cmd("serve", "Start a local OpenAI-compatible API server");
    print_cmd("keys", "Manage API keys (use, rm, add, cat, edit)");
    print_cmd("models", "List available models from the active provider");
    print_cmd("alias", "Create, list, or remove model aliases");
    print_cmd("ls", "Show system info, keys, tools, and directory state");
    print_cmd("stats", "Show usage statistics");
    print_cmd("update", "Update to the latest version");
    println!();
    println!("{}", style::bold("Shortcuts:"));
    let print_shortcut = |alias: &str, expansion: &str| {
        println!(
            "  {} {} {}",
            style::cyan(alias),
            style::arrow_symbol(),
            style::dim(expansion)
        );
    };
    print_shortcut("use", "keys use");
    print_shortcut("ping", "keys ping");
    print_shortcut("claude|codex|gemini|opencode|pi", "run <tool>");
    println!();
    println!("{}", style::bold("Examples:"));
    println!("  {}", style::dim("aivo claude -m kimi-k2.5"));
    println!("  {}", style::dim("aivo chat -x \"hello\""));
    println!("  {}", style::dim("aivo gemini -k mykey -m minimax-m2.7"));
    println!("  {}", style::dim("aivo ls --ping"));
    println!();
    println!("{}", style::bold("Options:"));
    let print_opt = |flag: &str, desc: &str| {
        println!(
            "  {}{}",
            style::cyan(format!("{:<16}", flag)),
            style::dim(desc)
        );
    };
    print_opt("-h, --help", "Display help information");
    print_opt("-v, --version", "Display the current version");
}

/// Prints version information
fn print_version() {
    println!(
        "{} {}",
        style::cyan("aivo"),
        style::dim(format!("v{}", version::VERSION))
    );
}

/// Resolves a model alias if the model is a non-empty Some value.
/// Returns the original value unchanged if resolution fails or if it's None/empty (picker).
async fn resolve_model_alias(
    session_store: &SessionStore,
    model: Option<String>,
) -> Option<String> {
    match model {
        Some(ref m) if !m.is_empty() => match session_store.resolve_alias(m).await {
            Ok(resolved) => Some(resolved),
            Err(_) => model,
        },
        other => other,
    }
}

/// Result of extracting aivo-specific flags from clap's trailing passthrough args.
struct ExtractedFlags {
    model: Option<String>,
    key_flag: Option<String>,
    debug: bool,
    dry_run: bool,
    env_strings: Vec<String>,
    remaining_args: Vec<String>,
}

/// Extracts aivo-owned flags (`--model`/`-m`, `--key`/`-k`, `--debug`, `--dry-run`, `--env`/`-e`) from
/// the passthrough `args` slice that clap's `trailing_var_arg` may have swallowed.
///
/// Flags already parsed by clap are supplied via `initial_*` parameters so that the
/// function produces a single consistent view regardless of where clap stopped.
fn extract_aivo_flags(
    initial_model: Option<String>,
    initial_key: Option<String>,
    initial_debug: bool,
    initial_dry_run: bool,
    initial_envs: Vec<String>,
    passthrough_args: &[String],
) -> ExtractedFlags {
    // Clap may have consumed a following flag as the value of -m/-k (e.g. `-m --resume`
    // gives model="--resume"). Detect and undo that by pushing the flag-like value back.
    let mut model = match initial_model {
        Some(m) if m.starts_with('-') => {
            // Will be pushed into remaining_args below via the passthrough loop seed
            // but we need it back in the stream — handled after the loop.
            Some((true, m)) // (is_flag_lookalike, value)
        }
        Some(m) => Some((false, m)),
        None => None,
    };
    let mut key_flag = match initial_key {
        Some(k) if k.starts_with('-') => Some((true, k)),
        Some(k) => Some((false, k)),
        None => None,
    };

    let mut debug = initial_debug;
    let mut dry_run = initial_dry_run;
    let mut env_strings = initial_envs;
    let mut remaining_args: Vec<String> = Vec::new();

    // Flush flag-lookalike values back into remaining_args before processing passthrough.
    if let Some((true, ref v)) = model {
        remaining_args.push(v.clone());
        model = Some((false, String::new())); // empty → picker
    }
    if let Some((true, ref v)) = key_flag {
        remaining_args.push(v.clone());
        key_flag = Some((false, String::new()));
    }

    let mut model: Option<String> = model.map(|(_, v)| v);
    let mut key_flag: Option<String> = key_flag.map(|(_, v)| v);

    let mut i = 0;
    while i < passthrough_args.len() {
        let arg = &passthrough_args[i];
        if let Some(value) = arg.strip_prefix("--model=") {
            if !value.is_empty() && model.is_none() {
                model = Some(value.to_string());
            } else {
                remaining_args.push(arg.clone());
            }
        } else if (arg == "--model" || arg == "-m") && model.is_none() {
            if i + 1 < passthrough_args.len() && !passthrough_args[i + 1].starts_with('-') {
                model = Some(passthrough_args[i + 1].clone());
                i += 1;
            } else {
                // --model with no value → trigger interactive picker
                model = Some(String::new());
            }
        } else if let Some(value) = arg.strip_prefix("--key=") {
            if !value.is_empty() && key_flag.is_none() {
                key_flag = Some(value.to_string());
            } else {
                remaining_args.push(arg.clone());
            }
        } else if (arg == "--key" || arg == "-k") && key_flag.is_none() {
            if i + 1 < passthrough_args.len() && !passthrough_args[i + 1].starts_with('-') {
                key_flag = Some(passthrough_args[i + 1].clone());
                i += 1;
            } else {
                key_flag = Some(String::new());
            }
        } else if arg == "--debug" {
            debug = true;
        } else if arg == "--dry-run" {
            dry_run = true;
        } else if let Some(value) = arg
            .strip_prefix("--env=")
            .or_else(|| arg.strip_prefix("-e="))
        {
            if !value.is_empty() {
                env_strings.push(value.to_string());
            }
        } else if (arg == "--env" || arg == "-e") && i + 1 < passthrough_args.len() {
            env_strings.push(passthrough_args[i + 1].clone());
            i += 1;
        } else {
            remaining_args.push(arg.clone());
        }
        i += 1;
    }

    ExtractedFlags {
        model,
        key_flag,
        debug,
        dry_run,
        env_strings,
        remaining_args,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn model_inline_form() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            vec![],
            &args(&["--model=gpt-4o", "file.ts"]),
        );
        assert_eq!(r.model, Some("gpt-4o".to_string()));
        assert_eq!(r.remaining_args, args(&["file.ts"]));
    }

    #[test]
    fn model_space_form() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            vec![],
            &args(&["--model", "gpt-4o", "file.ts"]),
        );
        assert_eq!(r.model, Some("gpt-4o".to_string()));
        assert_eq!(r.remaining_args, args(&["file.ts"]));
    }

    #[test]
    fn model_short_form() {
        let r = extract_aivo_flags(None, None, false, false, vec![], &args(&["-m", "gpt-4o"]));
        assert_eq!(r.model, Some("gpt-4o".to_string()));
        assert!(r.remaining_args.is_empty());
    }

    #[test]
    fn model_no_value_triggers_picker() {
        let r = extract_aivo_flags(None, None, false, false, vec![], &args(&["--model"]));
        assert_eq!(r.model, Some(String::new()));
    }

    #[test]
    fn model_flag_as_value_corrected() {
        // Clap swallowed `--resume` as the value of -m
        let r = extract_aivo_flags(
            Some("--resume".to_string()),
            None,
            false,
            false,
            vec![],
            &[],
        );
        assert_eq!(r.model, Some(String::new())); // picker triggered
        assert_eq!(r.remaining_args, args(&["--resume"]));
    }

    #[test]
    fn model_already_set_passthrough_not_overwritten() {
        // clap parsed --model correctly; a second --model in passthrough should pass through
        let r = extract_aivo_flags(
            Some("gpt-4o".to_string()),
            None,
            false,
            false,
            vec![],
            &args(&["--model", "other"]),
        );
        assert_eq!(r.model, Some("gpt-4o".to_string()));
        assert_eq!(r.remaining_args, args(&["--model", "other"]));
    }

    #[test]
    fn key_inline_form() {
        let r = extract_aivo_flags(None, None, false, false, vec![], &args(&["--key=mykey"]));
        assert_eq!(r.key_flag, Some("mykey".to_string()));
    }

    #[test]
    fn key_space_form() {
        let r = extract_aivo_flags(None, None, false, false, vec![], &args(&["--key", "mykey"]));
        assert_eq!(r.key_flag, Some("mykey".to_string()));
    }

    #[test]
    fn key_short_form() {
        let r = extract_aivo_flags(None, None, false, false, vec![], &args(&["-k", "mykey"]));
        assert_eq!(r.key_flag, Some("mykey".to_string()));
    }

    #[test]
    fn key_flag_as_value_corrected() {
        let r = extract_aivo_flags(
            None,
            Some("--something".to_string()),
            false,
            false,
            vec![],
            &[],
        );
        assert_eq!(r.key_flag, Some(String::new()));
        assert_eq!(r.remaining_args, args(&["--something"]));
    }

    #[test]
    fn key_no_value_triggers_picker() {
        let r = extract_aivo_flags(None, None, false, false, vec![], &args(&["-k"]));
        assert_eq!(r.key_flag, Some(String::new()));
    }

    #[test]
    fn debug_flag() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            vec![],
            &args(&["--debug", "file.ts"]),
        );
        assert!(r.debug);
        assert_eq!(r.remaining_args, args(&["file.ts"]));
    }

    #[test]
    fn debug_already_set_preserved() {
        let r = extract_aivo_flags(None, None, true, false, vec![], &[]);
        assert!(r.debug);
    }

    #[test]
    fn dry_run_flag() {
        let r = extract_aivo_flags(None, None, false, false, vec![], &args(&["--dry-run"]));
        assert!(r.dry_run);
    }

    #[test]
    fn env_inline_form() {
        let r = extract_aivo_flags(None, None, false, false, vec![], &args(&["--env=FOO=bar"]));
        assert_eq!(r.env_strings, vec!["FOO=bar"]);
    }

    #[test]
    fn env_short_inline_form() {
        let r = extract_aivo_flags(None, None, false, false, vec![], &args(&["-e=FOO=bar"]));
        assert_eq!(r.env_strings, vec!["FOO=bar"]);
    }

    #[test]
    fn env_space_form() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            vec![],
            &args(&["--env", "FOO=bar"]),
        );
        assert_eq!(r.env_strings, vec!["FOO=bar"]);
    }

    #[test]
    fn env_short_space_form() {
        let r = extract_aivo_flags(None, None, false, false, vec![], &args(&["-e", "FOO=bar"]));
        assert_eq!(r.env_strings, vec!["FOO=bar"]);
    }

    #[test]
    fn initial_envs_preserved() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            vec!["PRE=1".to_string()],
            &args(&["-e", "POST=2"]),
        );
        assert_eq!(r.env_strings, vec!["PRE=1", "POST=2"]);
    }

    #[test]
    fn unknown_args_pass_through() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            vec![],
            &args(&["--agent-name", "foo", "--resume"]),
        );
        assert_eq!(r.remaining_args, args(&["--agent-name", "foo", "--resume"]));
        assert_eq!(r.model, None);
    }

    #[test]
    fn mixed_flags() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            vec![],
            &args(&[
                "--agent-name",
                "foo",
                "--model",
                "gpt-4o",
                "--debug",
                "file.ts",
            ]),
        );
        assert_eq!(r.model, Some("gpt-4o".to_string()));
        assert!(r.debug);
        assert_eq!(r.remaining_args, args(&["--agent-name", "foo", "file.ts"]));
    }

    #[test]
    fn rewrite_injects_chat_for_top_level_execute() {
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "-x", "hello"])),
            args(&["aivo", "chat", "-x", "hello"])
        );
    }

    #[test]
    fn rewrite_injects_chat_for_long_execute() {
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "--execute", "hello"])),
            args(&["aivo", "chat", "--execute", "hello"])
        );
    }

    #[test]
    fn rewrite_keeps_explicit_chat() {
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "chat", "-x", "hello"])),
            args(&["aivo", "chat", "-x", "hello"])
        );
    }

    #[test]
    fn rewrite_keeps_tool_alias_precedence() {
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "claude", "--model", "gpt-5"])),
            args(&["aivo", "run", "claude", "--model", "gpt-5"])
        );
    }

    #[test]
    fn prompt_passes_through_extraction() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            vec![],
            &args(&["fix the login bug"]),
        );
        assert_eq!(r.remaining_args, args(&["fix the login bug"]));
        assert_eq!(r.model, None);
    }

    #[test]
    fn prompt_preserved_with_model_flag() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            vec![],
            &args(&["--model", "gpt-4o", "fix the login bug"]),
        );
        assert_eq!(r.model, Some("gpt-4o".to_string()));
        assert_eq!(r.remaining_args, args(&["fix the login bug"]));
    }

    #[test]
    fn multi_word_unquoted_args_pass_through() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            vec![],
            &args(&["fix", "the", "bug"]),
        );
        assert_eq!(r.remaining_args, args(&["fix", "the", "bug"]));
    }
}
