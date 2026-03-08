/**
 * KeysCommand handler for managing API keys.
 */
use anyhow::Result;

use crate::cli::KeysArgs;
use crate::tui::FuzzySelect;

use crate::errors::ExitCode;
use crate::services::session_store::{ApiKey, SessionStore};
use crate::style;

/// Reads a confirmation from stdin (y/yes for true, anything else for false).
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

/// Creates a safe preview of an API key, handling short keys without panicking.
fn key_preview(key: &str) -> String {
    if key.len() <= 10 {
        format!("{}...", &key[..3.min(key.len())])
    } else {
        format!("{}...{}", &key[..6], &key[key.len() - 4..])
    }
}

/// KeysCommand provides management of API keys
pub struct KeysCommand {
    session_store: SessionStore,
}

#[derive(Clone, Copy, Debug, Default)]
struct AddKeyOptions<'a> {
    name: Option<&'a str>,
    base_url: Option<&'a str>,
    key: Option<&'a str>,
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
        let action = action.unwrap_or("list");

        match action {
            "add" => {
                self.add_key(args.and_then(|a| a.first().copied()), add_options)
                    .await
            }
            "list" => self.list_keys().await,
            "rm" => self.remove_key(args.and_then(|a| a.first().copied())).await,
            "use" => self.use_key(args.and_then(|a| a.first().copied())).await,
            "cat" => self.cat_key(args.and_then(|a| a.first().copied())).await,
            "edit" => self.edit_key(args.and_then(|a| a.first().copied())).await,
            _ => {
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

        for key in &keys {
            let is_active = active_key.as_ref().map(|k| k.id == key.id).unwrap_or(false);
            let active_indicator = if is_active {
                style::bullet_symbol()
            } else {
                style::empty_bullet_symbol()
            };
            let id_padded = format!("{:<4}", key.id);
            println!(
                "  {} {}  {}  {}",
                active_indicator,
                style::cyan(&id_padded),
                key.name,
                style::dim(&key.base_url)
            );
        }

        Ok(ExitCode::Success)
    }

    /// Activates a specific API key by ID or name
    async fn use_key(&self, key_id_or_name: Option<&str>) -> Result<ExitCode> {
        // No argument — show interactive selector
        let Some(key_id_or_name) = key_id_or_name else {
            let all_keys = self.session_store.get_keys().await?;
            if all_keys.is_empty() {
                println!("{}", style::dim("No API keys found."));
                return Ok(ExitCode::Success);
            }
            let active_key = self.session_store.get_active_key().await?;
            let active_idx = active_key
                .and_then(|ak| all_keys.iter().position(|k| k.id == ak.id))
                .unwrap_or(0);
            let result = prompt_select_key(
                &self.session_store,
                &all_keys,
                "Select a key to activate",
                active_idx,
            )
            .await?;
            if result.is_none() {
                println!("{}", style::dim("Cancelled."));
            }
            return Ok(ExitCode::Success);
        };

        let all_keys = self.session_store.get_keys().await?;

        // Try exact ID match first
        if let Some(key) = all_keys.iter().find(|k| k.id == key_id_or_name) {
            self.activate_key(key).await?;
            return Ok(ExitCode::Success);
        }

        // Try name match
        let name_matches: Vec<_> = all_keys
            .iter()
            .filter(|k| k.name == key_id_or_name)
            .collect();

        if name_matches.is_empty() {
            eprintln!(
                "{} API key \"{}\" not found",
                style::red("Error:"),
                key_id_or_name
            );
            eprintln!();
            eprintln!(
                "{}",
                style::dim("Run 'aivo keys list' to see available keys.")
            );
            return Ok(ExitCode::UserError);
        }

        if name_matches.len() == 1 {
            self.activate_key(name_matches[0]).await?;
            return Ok(ExitCode::Success);
        }

        // Multiple matches - interactive selection
        println!(
            "{} Multiple keys found with name \"{}\":",
            style::yellow("Note:"),
            key_id_or_name
        );

        let choices: Vec<_> = name_matches
            .iter()
            .map(|k| format!("{} - {} - {}", k.id, k.base_url, key_preview(&k.key)))
            .collect();

        let selection = FuzzySelect::new()
            .with_prompt("Select a key")
            .items(&choices)
            .default(0)
            .interact_opt()
            .ok()
            .flatten();

        if let Some(idx) = selection {
            self.activate_key(name_matches[idx]).await?;
        } else {
            println!("{}", style::dim("Cancelled."));
        }
        Ok(ExitCode::Success)
    }

    /// Activates a key and prints confirmation
    async fn activate_key(&self, key: &ApiKey) -> Result<()> {
        self.session_store.set_active_key(&key.id).await?;
        let preview = key_preview(&key.key);
        println!(
            "{} Activated key: {} {}",
            style::success_symbol(),
            style::cyan(&key.name),
            style::dim(&preview)
        );
        Ok(())
    }

    /// Displays details for a specific API key
    async fn cat_key(&self, key_id_or_name: Option<&str>) -> Result<ExitCode> {
        let key_id_or_name = match key_id_or_name {
            Some(k) => k,
            None => {
                eprintln!("{} Missing key ID or name", style::red("Error:"));
                eprintln!();
                eprintln!("{}", style::dim("Usage: aivo keys cat <key-id-or-name>"));
                return Ok(ExitCode::UserError);
            }
        };

        let all_keys = self.session_store.get_keys().await?;

        if let Some(key) = all_keys.iter().find(|k| k.id == key_id_or_name) {
            self.display_key_details(key);
            return Ok(ExitCode::Success);
        }

        // Try name match
        let name_matches: Vec<_> = all_keys
            .iter()
            .filter(|k| k.name == key_id_or_name)
            .collect();
        if name_matches.len() == 1 {
            self.display_key_details(name_matches[0]);
            return Ok(ExitCode::Success);
        }

        eprintln!(
            "{} API key \"{}\" not found",
            style::red("Error:"),
            key_id_or_name
        );
        eprintln!();
        eprintln!(
            "{}",
            style::dim("Run 'aivo keys list' to see available keys.")
        );
        Ok(ExitCode::UserError)
    }

    /// Displays key details
    fn display_key_details(&self, key: &ApiKey) {
        println!();
        println!("Name:     {}", style::cyan(&key.name));
        println!("Base URL: {}", style::blue(&key.base_url));
        println!("API Key:  {}", style::yellow(&*key.key));
        println!();
    }

    /// Interactively edits an API key
    async fn edit_key(&self, key_id_or_name: Option<&str>) -> Result<ExitCode> {
        use std::io::{self, Write};

        let key_id_or_name = match key_id_or_name {
            Some(k) => k,
            None => {
                eprintln!("{} Missing key ID or name", style::red("Error:"));
                eprintln!();
                eprintln!("{}", style::dim("Usage: aivo keys edit <key-id-or-name>"));
                return Ok(ExitCode::UserError);
            }
        };

        let key = match self
            .session_store
            .resolve_key_by_id_or_name(key_id_or_name)
            .await
        {
            Ok(k) => k,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                eprintln!();
                eprintln!(
                    "{}",
                    style::dim("Run 'aivo keys list' to see available keys.")
                );
                return Ok(ExitCode::UserError);
            }
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
        let name = loop {
            let input = read_line_with_default(&format!("Name [{}]: ", key.name))?;
            let value = if input.is_empty() {
                key.name.clone()
            } else {
                input
            };
            if value.is_empty() {
                eprintln!("{} Name cannot be empty", style::red("Error:"));
            } else {
                break value;
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
            style::cyan(&name)
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
        } else {
            let input = read_line("Name (e.g., my-openai-proxy): ")?;
            if input.is_empty() {
                eprintln!("{} Name cannot be empty", style::red("Error:"));
                return Ok(ExitCode::UserError);
            }
            input
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
            let mut provided_base_url = add_options.base_url.map(str::to_string);
            loop {
                let value = if let Some(value) = provided_base_url.take() {
                    value
                } else {
                    read_line("Base URL (e.g., http://localhost:8080 or 'copilot'): ")?
                };
                if value == "copilot"
                    || value.starts_with("http://")
                    || value.starts_with("https://")
                {
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
            style::cyan(&name)
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
        let key_id_or_name = match key_id_or_name {
            Some(k) => k,
            None => {
                eprintln!("{} Missing key ID or name", style::red("Error:"));
                eprintln!();
                eprintln!("{}", style::dim("Usage: aivo keys rm <key-id-or-name>"));
                return Ok(ExitCode::UserError);
            }
        };

        let keys = self.session_store.get_keys().await?;

        if keys.is_empty() {
            println!("{}", style::dim("No keys to remove."));
            return Ok(ExitCode::Success);
        }

        // Try exact ID match
        let key_to_remove = if let Some(key) = keys.iter().find(|k| k.id == key_id_or_name) {
            key.clone()
        } else {
            // Try name match
            let name_matches: Vec<_> = keys.iter().filter(|k| k.name == key_id_or_name).collect();

            if name_matches.is_empty() {
                eprintln!(
                    "{} Key \"{}\" not found",
                    style::red("Error:"),
                    key_id_or_name
                );
                eprintln!();
                eprintln!(
                    "{}",
                    style::dim("Run 'aivo keys list' to see available keys.")
                );
                return Ok(ExitCode::UserError);
            }

            if name_matches.len() == 1 {
                name_matches[0].clone()
            } else {
                // Multiple matches - interactive selection
                println!(
                    "{} Multiple keys found with name \"{}\":",
                    style::yellow("Note:"),
                    key_id_or_name
                );

                let choices: Vec<_> = name_matches
                    .iter()
                    .map(|k| format!("{} - {} - {}", k.id, k.base_url, key_preview(&k.key)))
                    .collect();

                let selection = FuzzySelect::new()
                    .with_prompt("Select a key to remove")
                    .items(&choices)
                    .default(0)
                    .interact_opt()
                    .ok()
                    .flatten();

                if let Some(idx) = selection {
                    name_matches[idx].clone()
                } else {
                    eprintln!("{} Invalid selection", style::red("Error:"));
                    return Ok(ExitCode::UserError);
                }
            }
        };

        // Show confirmation
        let preview = key_preview(&key_to_remove.key);
        println!("Key: {} {}", style::cyan(&key_to_remove.id), preview);
        println!("URL: {}", style::dim(&key_to_remove.base_url));
        println!();

        let confirmed = confirm(&format!("Remove \"{}\"?", key_to_remove.name))?;

        if !confirmed {
            println!("{}", style::dim("Cancelled."));
            return Ok(ExitCode::Success);
        }

        if self.session_store.delete_key(&key_to_remove.id).await? {
            println!(
                "{} Removed key: {}",
                style::success_symbol(),
                style::cyan(&key_to_remove.name)
            );
            Ok(ExitCode::Success)
        } else {
            eprintln!("{} Failed to remove key", style::red("Error:"));
            Ok(ExitCode::UserError)
        }
    }

    /// Shows usage information
    pub fn print_help() {
        println!("{}", style::bold("Usage: aivo keys [action]"));
        println!();
        println!("{}", style::bold("Actions:"));
        println!(
            "  list            {}",
            style::dim("- List all API keys (default)")
        );
        println!(
            "  use <id|name>   {}",
            style::dim("- Activate a specific API key")
        );
        println!(
            "  cat <id|name>   {}",
            style::dim("- Display details for a key")
        );
        println!("  rm <id|name>    {}", style::dim("- Remove an API key"));
        println!("  add [name]      {}", style::dim("- Add an API key"));
        println!("  edit <id|name>  {}", style::dim("- Edit an API key"));
        println!();
        println!("{}", style::bold("Add Flags:"));
        println!("  --name <name>         {}", style::dim("- Set key name"));
        println!(
            "  --base-url <url>     {}",
            style::dim("- Set provider base URL")
        );
        println!(
            "  --key <api-key>       {}",
            style::dim("- Set provider API key")
        );
        println!(
            "  {}",
            style::dim(
                "Example: aivo keys add --name openrouter --base-url https://openrouter.ai/api/v1 --key sk-or-v1-..."
            )
        );
    }
}

/// Formats an API key as a choice string for interactive selectors.
pub(crate) fn format_key_choice(key: &ApiKey) -> String {
    format!(
        "{}  {}  {}",
        style::cyan(format!("{:<4}", key.id)),
        key.name,
        style::dim(&key.base_url)
    )
}

/// Prompts the user to select a key from the given list and activates it.
/// Returns `Ok(Some(key))` if selected, `Ok(None)` if cancelled.
pub(crate) async fn prompt_select_key(
    session_store: &SessionStore,
    keys: &[ApiKey],
    prompt: &str,
    default: usize,
) -> Result<Option<ApiKey>> {
    let choices: Vec<String> = keys.iter().map(format_key_choice).collect();
    let selection = FuzzySelect::new()
        .with_prompt(prompt)
        .items(&choices)
        .default(default)
        .interact_opt()?;
    match selection {
        Some(idx) => {
            let key = &keys[idx];
            session_store.set_active_key(&key.id).await?;
            let preview = key_preview(&key.key);
            eprintln!(
                "{} Activated key: {} {}",
                style::success_symbol(),
                style::cyan(&key.name),
                style::dim(&preview)
            );
            Ok(Some(key.clone()))
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
        assert_eq!(code, crate::errors::ExitCode::UserError);
    }

    #[tokio::test]
    async fn test_edit_key_not_found() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
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
}
