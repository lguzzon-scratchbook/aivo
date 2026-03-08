/**
 * KeysCommand handler for managing API keys.
 */
use anyhow::Result;

use crate::cli::KeysArgs;
use crate::tui::FuzzySelect;

use crate::errors::ExitCode;
use crate::services::session_store::{ApiKey, SessionStore};
use crate::style;

enum KeySelection {
    Key(ApiKey),
    Cancelled,
    Empty,
    NotFound,
}

// Reads a confirmation from stdin (y/yes for true, anything else for false).
fn confirm(prompt: &str) -> std::io::Result<bool> {
    print!("{} [y/N]: ", prompt);
    std::io::Write::flush(&mut std::io::stdout())?;
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    Ok(matches!(
        input.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

// Creates a safe preview of an API key, handling short keys without panicking.
fn key_preview(key: &str) -> String {
    if key.len() <= 10 {
        format!("{}...", &key[..3.min(key.len())])
    } else {
        format!("{}...{}", &key[..6], &key[key.len() - 4..])
    }
}

pub struct KeysCommand {
    session_store: SessionStore,
}

#[derive(Clone, Copy, Debug, Default)]
struct AddKeyOptions<'a> {
    name: Option<&'a str>,
    base_url: Option<&'a str>,
    key: Option<&'a str>,
}

fn detect_base_url(name: &str) -> Option<&'static str> {
    let lower = name.to_lowercase();
    let providers: &[(&str, &str)] = &[
        ("openrouter", "https://openrouter.ai/api/v1"),
        ("vercel",     "https://ai-gateway.vercel.sh/v1"),
        ("fireworks",  "https://api.fireworks.ai/inference/v1"),
        ("minimax",    "https://api.minimax.io/anthropic"),
        ("deepseek",   "https://api.deepseek.com/v1"),
        ("moonshot",   "https://api.moonshot.cn/v1"),
        ("groq",       "https://api.groq.com/openai/v1"),
        ("xai",        "https://api.x.ai/v1"),
        ("mistral",    "https://api.mistral.ai/v1"),
    ];
    providers.iter().find_map(|(kw, url)| lower.contains(kw).then_some(*url))
}

impl KeysCommand {
    /// Creates a new KeysCommand instance
    pub fn new(session_store: SessionStore) -> Self {
        Self { session_store }
    }

    /// Executes the keys command with the specified action
    pub async fn execute(&self, keys_args: KeysArgs) -> ExitCode {
        let action = keys_args.action.as_deref();
        let args: Vec<_> = keys_args.args.iter().map(|s| s.as_str()).collect();
        let add_options = AddKeyOptions {
            name: keys_args.name.as_deref(),
            base_url: keys_args.base_url.as_deref(),
            key: keys_args.key.as_deref(),
        };

        match self
            .execute_internal(action, Some(&args), add_options)
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
        action: Option<&str>,
        args: Option<&[&str]>,
        add_options: AddKeyOptions<'_>,
    ) -> Result<ExitCode> {
        match action {
            None => self.list_keys().await,
            Some("add") => {
                self.add_key(args.and_then(|a| a.first().copied()), add_options)
                    .await
            }
            Some("rm") => self.remove_key(args.and_then(|a| a.first().copied())).await,
            Some("use") => self.use_key(args.and_then(|a| a.first().copied())).await,
            Some("cat") => self.cat_key(args.and_then(|a| a.first().copied())).await,
            Some("edit") => self.edit_key(args.and_then(|a| a.first().copied())).await,
            Some(action) => {
                eprintln!("{} Unknown action '{}'", style::red("Error:"), action);
                Self::print_help();
                Ok(ExitCode::UserError)
            }
        }
    }

    /// Lists all API keys
    async fn list_keys(&self) -> Result<ExitCode> {
        let keys = self.session_store.get_keys().await?;
        let active_key = self.session_store.get_active_key().await?;

        if keys.is_empty() {
            println!("{}", style::dim("No API keys found."));
            return Ok(ExitCode::Success);
        }

        println!();
        for key in &keys {
            let is_active = active_key.as_ref().map(|k| k.id == key.id).unwrap_or(false);
            let active_indicator = if is_active {
                style::bullet_symbol()
            } else {
                style::empty_bullet_symbol()
            };
            let id_padded = format!("{:<4}", key.id);
            if key.name.is_empty() {
                println!(
                    "  {} {}  {}",
                    active_indicator,
                    style::cyan(&id_padded),
                    style::dim(&key.base_url)
                );
            } else {
                println!(
                    "  {} {}  {}  {}",
                    active_indicator,
                    style::cyan(&id_padded),
                    key.name,
                    style::dim(&key.base_url)
                );
            }
        }

        Ok(ExitCode::Success)
    }

    /// Activates a specific API key by ID or name
    async fn use_key(&self, key_id_or_name: Option<&str>) -> Result<ExitCode> {
        match self
            .resolve_key_selection(
                key_id_or_name,
                "Select a key to activate",
                "No API keys found.",
            )
            .await?
        {
            KeySelection::Key(key) => {
                self.activate_key(&key).await?;
                Ok(ExitCode::Success)
            }
            KeySelection::Cancelled => {
                println!("{}", style::dim("Cancelled."));
                Ok(ExitCode::Success)
            }
            KeySelection::Empty => Ok(ExitCode::Success),
            KeySelection::NotFound => Ok(ExitCode::UserError),
        }
    }

    /// Activates a key and prints confirmation
    async fn activate_key(&self, key: &ApiKey) -> Result<()> {
        self.session_store.set_active_key(&key.id).await?;
        let preview = key_preview(&key.key);
        println!(
            "{} Activated key: {} {}",
            style::success_symbol(),
            style::cyan(key.display_name()),
            style::dim(&preview)
        );
        Ok(())
    }

    /// Displays details for a specific API key
    async fn cat_key(&self, key_id_or_name: Option<&str>) -> Result<ExitCode> {
        match self
            .resolve_key_selection(
                key_id_or_name,
                "Select a key to inspect",
                "No API keys found.",
            )
            .await?
        {
            KeySelection::Key(key) => {
                self.display_key_details(&key);
                Ok(ExitCode::Success)
            }
            KeySelection::Cancelled => {
                println!("{}", style::dim("Cancelled."));
                Ok(ExitCode::Success)
            }
            KeySelection::Empty => Ok(ExitCode::Success),
            KeySelection::NotFound => Ok(ExitCode::UserError),
        }
    }

    /// Displays key details
    fn display_key_details(&self, key: &ApiKey) {
        println!();
        println!("Name:     {}", style::cyan(key.display_name()));
        println!("Base URL: {}", style::blue(&key.base_url));
        println!("API Key:  {}", style::yellow(&*key.key));
        println!();
    }

    /// Interactively edits an API key
    async fn edit_key(&self, key_id_or_name: Option<&str>) -> Result<ExitCode> {
        use std::io::{self, Write};

        let key = match self
            .resolve_key_selection(key_id_or_name, "Select a key to edit", "No API keys found.")
            .await?
        {
            KeySelection::Key(key) => key,
            KeySelection::Cancelled => {
                println!("{}", style::dim("Cancelled."));
                return Ok(ExitCode::Success);
            }
            KeySelection::Empty => return Ok(ExitCode::Success),
            KeySelection::NotFound => return Ok(ExitCode::UserError),
        };

        println!("{}", style::bold("Edit API Key"));
        println!();
        println!("Press Enter to keep the current value.");
        println!();

        fn read_line_with_default(prompt: &str) -> io::Result<String> {
            print!("{}", prompt);
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            Ok(input.trim().to_string())
        }

        // Name
        let current_name = if key.name.is_empty() {
            format!("unnamed; shown as {}", key.id)
        } else {
            key.name.clone()
        };
        let name = {
            let input = read_line_with_default(&format!("Name [{}]: ", current_name))?;
            if input.is_empty() {
                key.name.clone()
            } else {
                input
            }
        };

        // Base URL
        let base_url = loop {
            let input = read_line_with_default(&format!("Base URL [{}]: ", key.base_url))?;
            let value = if input.is_empty() {
                key.base_url.clone()
            } else {
                input
            };
            if value == "copilot" || value.starts_with("http://") || value.starts_with("https://") {
                break value;
            }
            eprintln!(
                "{} URL must start with http:// or https:// (or enter 'copilot' for GitHub Copilot)",
                style::red("Error:")
            );
        };

        // API Key
        let api_key = loop {
            let preview = key_preview(&key.key);
            let input = read_line_with_default(&format!("API Key [{}]: ", preview))?;
            let value = if input.is_empty() {
                key.key.as_str().to_string()
            } else {
                input
            };
            if value.is_empty() {
                eprintln!("{} API Key cannot be empty", style::red("Error:"));
            } else {
                break value;
            }
        };

        println!();

        let updated = self
            .session_store
            .update_key(
                &key.id,
                &name,
                &base_url,
                if base_url == key.base_url {
                    key.claude_protocol
                } else {
                    None
                },
                &api_key,
            )
            .await?;

        if updated && base_url != key.base_url {
            let _ = self
                .session_store
                .set_key_gemini_protocol(&key.id, None)
                .await?;
            let _ = self.session_store.set_key_codex_mode(&key.id, None).await?;
            let _ = self
                .session_store
                .set_key_opencode_mode(&key.id, None)
                .await?;
        }

        if !updated {
            eprintln!("{} Key no longer exists", style::red("Error:"));
            return Ok(ExitCode::UserError);
        }

        println!(
            "{} Updated key: {}",
            style::success_symbol(),
            style::cyan(if name.is_empty() { &key.id } else { &name })
        );

        Ok(ExitCode::Success)
    }

    /// Interactively adds an API key
    async fn add_key(
        &self,
        provided_name: Option<&str>,
        add_options: AddKeyOptions<'_>,
    ) -> Result<ExitCode> {
        use std::io::{self, Write};

        fn read_line(prompt: &str) -> io::Result<String> {
            print!("{}", prompt);
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            Ok(input.trim().to_string())
        }

        if provided_name.is_some() && add_options.name.is_some() {
            eprintln!(
                "{} Specify the key name either positionally or with --name",
                style::red("Error:")
            );
            return Ok(ExitCode::UserError);
        }

        let name = if let Some(n) = add_options.name.or(provided_name) {
            n.to_string()
        } else if add_options.base_url.is_some() && add_options.key.is_some() {
            String::new()
        } else {
            read_line("Name (optional): ")?
        };

        // Shortcut: `aivo keys add copilot` skips all prompts unless flags conflict.
        let base_url = if name == "copilot" {
            match add_options.base_url {
                Some("copilot") | None => "copilot".to_string(),
                Some(_) => {
                    eprintln!(
                        "{} Name 'copilot' is reserved for GitHub Copilot. Use a different name or omit --base-url.",
                        style::red("Error:")
                    );
                    return Ok(ExitCode::UserError);
                }
            }
        } else {
            let detected_url = detect_base_url(&name);
            let mut provided_base_url = add_options.base_url.map(str::to_string);
            loop {
                let value = if let Some(value) = provided_base_url.take() {
                    value
                } else {
                    let prompt = match detected_url {
                        Some(default) => format!("Base URL [{}]: ", default),
                        None => "Base URL (e.g., https://api.openai.com/v1): ".to_string(),
                    };
                    let input = read_line(&prompt)?;
                    if input.is_empty() {
                        detected_url.unwrap_or("").to_string()
                    } else {
                        input
                    }
                };
                if value == "copilot" {
                    eprintln!(
                        "{} GitHub Copilot login requires the explicit shortcut 'aivo keys add copilot'.",
                        style::red("Error:")
                    );
                    if add_options.base_url.is_some() {
                        return Ok(ExitCode::UserError);
                    }
                    continue;
                }
                if value.starts_with("http://") || value.starts_with("https://") {
                    break value;
                }
                eprintln!(
                    "{} URL must start with http:// or https:// (or enter 'copilot' for GitHub Copilot)",
                    style::red("Error:")
                );
                if add_options.base_url.is_some() {
                    return Ok(ExitCode::UserError);
                }
            }
        };

        // GitHub Copilot: use device flow instead of manual key entry
        if base_url == "copilot" {
            if add_options.key.is_some() {
                eprintln!(
                    "{} Do not pass --key for GitHub Copilot. Use 'aivo keys add copilot' to start device login.",
                    style::red("Error:")
                );
                return Ok(ExitCode::UserError);
            }

            // Check for an existing Copilot key and prompt to replace
            let existing_keys = self.session_store.get_keys().await?;
            let existing_copilot_id =
                if let Some(existing) = existing_keys.iter().find(|k| k.base_url == "copilot") {
                    eprint!(
                        "{} Copilot key '{}' (ID: {}) already exists. Replace it? [y/N] ",
                        style::yellow("Warning:"),
                        existing.name,
                        existing.id
                    );
                    use std::io::Write as _;
                    std::io::stderr().flush()?;
                    let answer = read_line("")?;
                    if !matches!(answer.to_lowercase().as_str(), "y" | "yes") {
                        println!("Aborted.");
                        return Ok(ExitCode::Success);
                    }
                    Some(existing.id.clone())
                } else {
                    None
                };

            let token = crate::services::copilot_auth::device_flow_login().await?;

            // Device flow succeeded — now safe to remove the old key
            if let Some(old_id) = existing_copilot_id {
                self.session_store.delete_key(&old_id).await?;
            }

            let id = self
                .session_store
                .add_key_with_protocol(&name, "copilot", None, &token)
                .await?;
            self.session_store.set_active_key(&id).await?;

            println!();
            println!(
                "{} Added and activated key: {}",
                style::success_symbol(),
                style::cyan(&name)
            );
            println!("  {}", style::dim(format!("ID: {}", id)));
            println!("  {}", style::dim("Provider: GitHub Copilot"));
            println!();
            println!(
                "{} {} {}",
                style::yellow("Next:"),
                style::bold("aivo run claude"),
                style::dim("(uses Copilot subscription)")
            );

            return Ok(ExitCode::Success);
        }

        let key = if let Some(key) = add_options.key {
            key.to_string()
        } else {
            read_line("API Key: ")?
        };
        if key.is_empty() {
            eprintln!("{} API Key cannot be empty", style::red("Error:"));
            return Ok(ExitCode::UserError);
        }

        println!();

        let id = self
            .session_store
            .add_key_with_protocol(&name, &base_url, None, &key)
            .await?;
        self.session_store.set_active_key(&id).await?;

        println!(
            "{} Added and activated key: {}",
            style::success_symbol(),
            style::cyan(if name.is_empty() { &id } else { &name })
        );
        println!("  {}", style::dim(format!("ID: {}", id)));
        println!("  {}", style::dim(format!("Base URL: {}", base_url)));
        println!();
        println!(
            "{} {} {}",
            style::yellow("Next:"),
            style::bold("aivo run <tool>"),
            style::dim("(uses this key)")
        );

        Ok(ExitCode::Success)
    }

    /// Removes an API key by ID or name
    async fn remove_key(&self, key_id_or_name: Option<&str>) -> Result<ExitCode> {
        let key_to_remove = match self
            .resolve_key_selection(
                key_id_or_name,
                "Select a key to remove",
                "No keys to remove.",
            )
            .await?
        {
            KeySelection::Key(key) => key,
            KeySelection::Cancelled => {
                println!("{}", style::dim("Cancelled."));
                return Ok(ExitCode::Success);
            }
            KeySelection::Empty => return Ok(ExitCode::Success),
            KeySelection::NotFound => return Ok(ExitCode::UserError),
        };

        // Show confirmation
        let preview = key_preview(&key_to_remove.key);
        println!("Key: {} {}", style::cyan(&key_to_remove.id), preview);
        println!("URL: {}", style::dim(&key_to_remove.base_url));
        println!();

        let confirmed = confirm(&format!("Remove \"{}\"?", key_to_remove.display_name()))?;

        if !confirmed {
            println!("{}", style::dim("Cancelled."));
            return Ok(ExitCode::Success);
        }

        if self.session_store.delete_key(&key_to_remove.id).await? {
            println!(
                "{} Removed key: {}",
                style::success_symbol(),
                style::cyan(key_to_remove.display_name())
            );
            Ok(ExitCode::Success)
        } else {
            eprintln!("{} Failed to remove key", style::red("Error:"));
            Ok(ExitCode::UserError)
        }
    }

    async fn resolve_key_selection(
        &self,
        key_id_or_name: Option<&str>,
        prompt: &str,
        empty_message: &str,
    ) -> Result<KeySelection> {
        let all_keys = self.session_store.get_keys().await?;

        if all_keys.is_empty() {
            println!("{}", style::dim(empty_message));
            return Ok(KeySelection::Empty);
        }

        let Some(key_id_or_name) = key_id_or_name else {
            let active_key = self.session_store.get_active_key().await?;
            let default_idx = active_key
                .and_then(|ak| all_keys.iter().position(|k| k.id == ak.id))
                .unwrap_or(0);
            return Ok(match prompt_pick_key(&all_keys, prompt, default_idx)? {
                Some(key) => KeySelection::Key(key),
                None => KeySelection::Cancelled,
            });
        };

        if let Some(key) = all_keys.iter().find(|k| k.id == key_id_or_name) {
            return Ok(KeySelection::Key(key.clone()));
        }

        let name_matches: Vec<ApiKey> = all_keys
            .iter()
            .filter(|k| k.name == key_id_or_name)
            .cloned()
            .collect();

        match name_matches.len() {
            0 => {
                eprintln!(
                    "{} API key \"{}\" not found",
                    style::red("Error:"),
                    key_id_or_name
                );
                eprintln!();
                eprintln!("{}", style::dim("Run 'aivo keys' to see available keys."));
                Ok(KeySelection::NotFound)
            }
            1 => Ok(KeySelection::Key(name_matches[0].clone())),
            _ => {
                println!(
                    "{} Multiple keys found with name \"{}\":",
                    style::yellow("Note:"),
                    key_id_or_name
                );
                Ok(match prompt_pick_key(&name_matches, prompt, 0)? {
                    Some(key) => KeySelection::Key(key),
                    None => KeySelection::Cancelled,
                })
            }
        }
    }

    // Shows usage information.
    pub fn print_help() {
        let print_row = |label: &str, description: &str| {
            println!("  {:<18} {}", label, style::dim(description));
        };

        println!("{}", style::bold("Usage: aivo keys [action]"));
        println!();
        println!("{}", style::bold("Actions:"));
        print_row("(no action)", "- List all API keys");
        print_row("use [id|name]", "- Activate a specific API key");
        print_row("cat [id|name]", "- Display details for a key");
        print_row("rm [id|name]", "- Remove an API key");
        print_row("add [name]", "- Add an API key");
        print_row("edit [id|name]", "- Edit an API key");
        println!();
        println!("{}", style::bold("Add Flags:"));
        print_row("--name <name>", "- Set key name");
        print_row("--base-url <url>", "- Set provider base URL");
        print_row("--key <api-key>", "- Set provider API key");
        println!(
            "  {}",
            style::dim(
                "Example: aivo keys add --name abc --base-url https://example.io --key sk-..."
            )
        );
    }
}

// Formats an API key as a choice string for interactive selectors.
pub(crate) fn format_key_choice(key: &ApiKey) -> String {
    format!(
        "{}  {}  {}",
        style::cyan(format!("{:<4}", key.id)),
        key.display_name(),
        style::dim(&key.base_url)
    )
}

// Prompts the user to select a key from the given list.
fn prompt_pick_key(keys: &[ApiKey], prompt: &str, default: usize) -> Result<Option<ApiKey>> {
    let choices: Vec<String> = keys.iter().map(format_key_choice).collect();
    let selection = FuzzySelect::new()
        .with_prompt(prompt)
        .items(&choices)
        .default(default)
        .interact_opt()?;
    Ok(selection.map(|idx| keys[idx].clone()))
}

// Prompts the user to select a key from the given list and activates it.
// Returns `Ok(Some(key))` if selected, `Ok(None)` if cancelled.
#[allow(dead_code)]
pub(crate) async fn prompt_select_key(
    session_store: &SessionStore,
    keys: &[ApiKey],
    prompt: &str,
    default: usize,
) -> Result<Option<ApiKey>> {
    match prompt_pick_key(keys, prompt, default)? {
        Some(key) => {
            session_store.set_active_key(&key.id).await?;
            let preview = key_preview(&key.key);
            eprintln!(
                "{} Activated key: {} {}",
                style::success_symbol(),
                style::cyan(key.display_name()),
                style::dim(&preview)
            );
            Ok(Some(key))
        }
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::KeysArgs;

    fn keys_args(action: Option<&str>, args: &[&str]) -> KeysArgs {
        KeysArgs {
            action: action.map(str::to_string),
            args: args.iter().map(|s| s.to_string()).collect(),
            name: None,
            base_url: None,
            key: None,
        }
    }

    #[test]
    fn test_keys_command_creation() {
        let session_store = SessionStore::new();
        let _command = KeysCommand::new(session_store);
    }

    #[tokio::test]
    async fn test_edit_key_missing_id() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store);
        let code = cmd.execute(keys_args(Some("edit"), &[])).await;
        assert_eq!(code, crate::errors::ExitCode::Success);
    }

    #[tokio::test]
    async fn test_edit_key_not_found() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        store
            .add_key_with_protocol(
                "openrouter",
                "https://openrouter.ai/api/v1",
                None,
                "sk-test",
            )
            .await
            .unwrap();
        let cmd = KeysCommand::new(store);
        let code = cmd.execute(keys_args(Some("edit"), &["nonexistent"])).await;
        assert_eq!(code, crate::errors::ExitCode::UserError);
    }

    #[tokio::test]
    async fn test_use_key_no_arg_no_keys() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store);
        // No keys stored — should succeed (prints "No API keys found.")
        let code = cmd.execute(keys_args(Some("use"), &[])).await;
        assert_eq!(code, crate::errors::ExitCode::Success);
    }

    #[tokio::test]
    async fn test_cat_key_no_arg_no_keys() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store);
        let code = cmd.execute(keys_args(Some("cat"), &[])).await;
        assert_eq!(code, crate::errors::ExitCode::Success);
    }

    #[tokio::test]
    async fn test_remove_key_no_arg_no_keys() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store);
        let code = cmd.execute(keys_args(Some("rm"), &[])).await;
        assert_eq!(code, crate::errors::ExitCode::Success);
    }

    #[tokio::test]
    async fn test_keys_list_action_is_rejected() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store);
        let code = cmd.execute(keys_args(Some("list"), &[])).await;
        assert_eq!(code, crate::errors::ExitCode::UserError);
    }

    #[tokio::test]
    async fn test_add_key_with_flags() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store.clone());

        let code = cmd
            .execute(KeysArgs {
                action: Some("add".to_string()),
                args: Vec::new(),
                name: Some("minimax".to_string()),
                base_url: Some("https://api.minimax.io/anthropic".to_string()),
                key: Some("sk-minimax-test".to_string()),
            })
            .await;

        assert_eq!(code, crate::errors::ExitCode::Success);

        let keys = store.get_keys().await.unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].name, "minimax");
        assert_eq!(keys[0].base_url, "https://api.minimax.io/anthropic");
        assert_eq!(keys[0].claude_protocol, None);
        assert_eq!(keys[0].key.as_str(), "sk-minimax-test");

        let active = store.get_active_key().await.unwrap().unwrap();
        assert_eq!(active.id, keys[0].id);
    }

    #[tokio::test]
    async fn test_add_key_rejects_conflicting_name_sources() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store);

        let code = cmd
            .execute(KeysArgs {
                action: Some("add".to_string()),
                args: vec!["positional-name".to_string()],
                name: Some("flag-name".to_string()),
                base_url: Some("https://openrouter.ai/api/v1".to_string()),
                key: Some("sk-or-v1-test".to_string()),
            })
            .await;

        assert_eq!(code, crate::errors::ExitCode::UserError);
    }

    #[tokio::test]
    async fn test_add_key_without_name_uses_empty_stored_name() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store.clone());

        let code = cmd
            .execute(KeysArgs {
                action: Some("add".to_string()),
                args: Vec::new(),
                name: None,
                base_url: Some("https://openrouter.ai/api/v1".to_string()),
                key: Some("sk-or-v1-test".to_string()),
            })
            .await;

        assert_eq!(code, crate::errors::ExitCode::Success);

        let keys = store.get_keys().await.unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].name, "");
        assert_eq!(keys[0].display_name(), keys[0].id);

        let active = store.get_active_key().await.unwrap().unwrap();
        assert_eq!(active.id, keys[0].id);
    }

    #[tokio::test]
    async fn test_add_key_rejects_copilot_base_url_without_copilot_name() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store);

        let code = cmd
            .execute(KeysArgs {
                action: Some("add".to_string()),
                args: Vec::new(),
                name: None,
                base_url: Some("copilot".to_string()),
                key: None,
            })
            .await;

        assert_eq!(code, crate::errors::ExitCode::UserError);
    }

    #[test]
    fn test_detect_base_url_exact_match() {
        assert_eq!(detect_base_url("openrouter"), Some("https://openrouter.ai/api/v1"));
        assert_eq!(detect_base_url("deepseek"), Some("https://api.deepseek.com/v1"));
        assert_eq!(detect_base_url("groq"), Some("https://api.groq.com/openai/v1"));
        assert_eq!(detect_base_url("mistral"), Some("https://api.mistral.ai/v1"));
        assert_eq!(detect_base_url("xai"), Some("https://api.x.ai/v1"));
        assert_eq!(detect_base_url("fireworks"), Some("https://api.fireworks.ai/inference/v1"));
        assert_eq!(detect_base_url("moonshot"), Some("https://api.moonshot.cn/v1"));
        assert_eq!(detect_base_url("minimax"), Some("https://api.minimax.io/anthropic"));
        assert_eq!(detect_base_url("vercel"), Some("https://ai-gateway.vercel.sh/v1"));
    }

    #[test]
    fn test_detect_base_url_case_insensitive() {
        assert_eq!(detect_base_url("OpenRouter"), Some("https://openrouter.ai/api/v1"));
        assert_eq!(detect_base_url("GROQ"), Some("https://api.groq.com/openai/v1"));
        assert_eq!(detect_base_url("DeepSeek"), Some("https://api.deepseek.com/v1"));
    }

    #[test]
    fn test_detect_base_url_substring() {
        assert_eq!(detect_base_url("my-openrouter-key"), Some("https://openrouter.ai/api/v1"));
        assert_eq!(detect_base_url("work_groq"), Some("https://api.groq.com/openai/v1"));
    }

    #[test]
    fn test_detect_base_url_no_match() {
        assert_eq!(detect_base_url("random"), None);
        assert_eq!(detect_base_url(""), None);
        assert_eq!(detect_base_url("anthropic"), None);
        assert_eq!(detect_base_url("openai"), None);
    }

    #[test]
    fn test_format_key_choice_uses_id_for_unnamed_keys() {
        let key = ApiKey::new_with_protocol(
            "1a2b".to_string(),
            String::new(),
            "https://openrouter.ai/api/v1".to_string(),
            None,
            "sk-test".to_string(),
        );

        let choice = format_key_choice(&key);

        assert!(choice.contains("1a2b"));
        assert!(choice.contains("https://openrouter.ai/api/v1"));
    }
}
