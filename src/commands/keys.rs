/**
 * KeysCommand handler for managing API keys.
 */
use anyhow::Result;
use dialoguer::{Confirm, Select};

use crate::errors::ExitCode;
use crate::services::session_store::{ApiKey, SessionStore};
use crate::style;

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

impl KeysCommand {
    /// Creates a new KeysCommand instance
    pub fn new(session_store: SessionStore) -> Self {
        Self { session_store }
    }

    /// Executes the keys command with the specified action
    pub async fn execute(&self, action: Option<&str>, args: Option<&[&str]>) -> ExitCode {
        match self.execute_internal(action, args).await {
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
    ) -> Result<ExitCode> {
        let action = action.unwrap_or("list");

        match action {
            "add" => self.add_key(args.and_then(|a| a.first().copied())).await,
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
            let choices: Vec<_> = all_keys
                .iter()
                .map(|k| {
                    format!(
                        "{}  {}  {}",
                        style::cyan(&format!("{:<4}", k.id)),
                        k.name,
                        style::dim(&k.base_url)
                    )
                })
                .collect();
            let selection = Select::new()
                .with_prompt("Select a key to activate")
                .items(&choices)
                .default(active_idx)
                .interact()
                .ok();
            if let Some(idx) = selection {
                self.activate_key(&all_keys[idx]).await?;
            } else {
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

        let selection = Select::new()
            .with_prompt("Select a key")
            .items(&choices)
            .interact()
            .ok();

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
            if value.starts_with("http://") || value.starts_with("https://") {
                break value;
            }
            eprintln!(
                "{} URL must start with http:// or https://",
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
            .update_key(&key.id, &name, &base_url, &api_key)
            .await?;

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
    async fn add_key(&self, provided_name: Option<&str>) -> Result<ExitCode> {
        use std::io::{self, Write};

        println!("{}", style::bold("Add API Key"));
        println!();

        fn read_line(prompt: &str) -> io::Result<String> {
            print!("{}", prompt);
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            Ok(input.trim().to_string())
        }

        let name = if let Some(n) = provided_name {
            if n.is_empty() {
                eprintln!("{} Name cannot be empty", style::red("Error:"));
                return Ok(ExitCode::UserError);
            }
            n.to_string()
        } else {
            let input = read_line("Name (e.g., my-openai-proxy): ")?;
            if input.is_empty() {
                eprintln!("{} Name cannot be empty", style::red("Error:"));
                return Ok(ExitCode::UserError);
            }
            input
        };

        let base_url = loop {
            let input = read_line("Base URL (e.g., http://localhost:8080): ")?;
            if input.starts_with("http://") || input.starts_with("https://") {
                break input;
            }
            eprintln!(
                "{} URL must start with http:// or https://",
                style::red("Error:")
            );
        };

        let key = read_line("API Key: ")?;
        if key.is_empty() {
            eprintln!("{} API Key cannot be empty", style::red("Error:"));
            return Ok(ExitCode::UserError);
        }

        println!();

        let id = self.session_store.add_key(&name, &base_url, &key).await?;
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

                let selection = Select::new()
                    .with_prompt("Select a key to remove")
                    .items(&choices)
                    .interact()
                    .ok();

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

        let confirmed = Confirm::new()
            .with_prompt(format!("Remove \"{}\"?", key_to_remove.name))
            .default(false)
            .interact()?;

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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let code = cmd.execute(Some("edit"), Some(&[])).await;
        assert_eq!(code, crate::errors::ExitCode::UserError);
    }

    #[tokio::test]
    async fn test_edit_key_not_found() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store);
        let code = cmd.execute(Some("edit"), Some(&["nonexistent"])).await;
        assert_eq!(code, crate::errors::ExitCode::UserError);
    }

    #[tokio::test]
    async fn test_use_key_no_arg_no_keys() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store);
        // No keys stored — should succeed (prints "No API keys found.")
        let code = cmd.execute(Some("use"), Some(&[])).await;
        assert_eq!(code, crate::errors::ExitCode::Success);
    }
}
