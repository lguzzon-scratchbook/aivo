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
            "add" => self.add_key().await,
            "list" => self.list_keys().await,
            "rm" => self.remove_key(args.and_then(|a| a.first().copied())).await,
            "use" => self.use_key(args.and_then(|a| a.first().copied())).await,
            "cat" => self.cat_key(args.and_then(|a| a.first().copied())).await,
            _ => {
                eprintln!("{} Unknown action '{}'", style::red("Error:"), action);
                self.show_usage();
                Ok(ExitCode::UserError)
            }
        }
    }

    /// Lists all API keys
    async fn list_keys(&self) -> Result<ExitCode> {
        let keys = self.session_store.get_keys().await?;
        let active_key = self.session_store.get_active_key().await?;

        if keys.is_empty() {
            println!();
            println!("  {}", style::dim("No API keys found."));
            println!();
            println!(
                "  {} {}",
                style::yellow("Add a key:"),
                style::bold("aivo keys add")
            );
            println!();
            return Ok(ExitCode::Success);
        }

        println!();

        println!("  {}", style::dim("Keys:"));
        for key in &keys {
            let is_active = active_key.as_ref().map(|k| k.id == key.id).unwrap_or(false);
            let active_indicator = if is_active {
                style::bullet_symbol()
            } else {
                style::empty_bullet_symbol()
            };
            let id_padded = format!("{:<4}", key.id);
            println!(
                "    {} {}  {}  {}",
                active_indicator,
                style::cyan(&id_padded),
                key.name,
                style::dim(&key.base_url)
            );
        }
        println!();

        println!("  {}", style::yellow("Commands:"));
        println!(
            "    aivo keys use <id|name>    {}",
            style::dim("- Activate a specific key by ID or name")
        );
        println!(
            "    aivo keys cat <id|name>    {}",
            style::dim("- Display details for a key")
        );
        println!(
            "    aivo keys rm <id|name>     {}",
            style::dim("- Remove an API key")
        );
        println!(
            "    aivo keys add              {}",
            style::dim("- Add an API key")
        );
        println!();

        Ok(ExitCode::Success)
    }

    /// Activates a specific API key by ID or name
    async fn use_key(&self, key_id_or_name: Option<&str>) -> Result<ExitCode> {
        let key_id_or_name = match key_id_or_name {
            Some(k) => k,
            None => {
                eprintln!("{} Missing key ID or name", style::red("Error:"));
                eprintln!();
                eprintln!("{}", style::dim("Usage: aivo keys use <key-id-or-name>"));
                return Ok(ExitCode::UserError);
            }
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
            Ok(ExitCode::Success)
        } else {
            eprintln!("{} Invalid selection", style::red("Error:"));
            Ok(ExitCode::UserError)
        }
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
        println!("  Name:     {}", style::cyan(&key.name));
        println!("  Base URL: {}", style::blue(&key.base_url));
        println!("  API Key:  {}", style::yellow(&*key.key));
        println!();
    }

    /// Interactively adds an API key
    async fn add_key(&self) -> Result<ExitCode> {
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

        let name = read_line("Name (e.g., my-openai-proxy): ")?;
        if name.is_empty() {
            eprintln!("{} Name cannot be empty", style::red("Error:"));
            return Ok(ExitCode::UserError);
        }

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
    fn show_usage(&self) {
        println!();
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
        println!("  add             {}", style::dim("- Add an API key"));
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
}
