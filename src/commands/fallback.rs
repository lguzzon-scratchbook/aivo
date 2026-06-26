//! Fallback command handler — manage virtual models backed by ordered
//! `<provider>:<model>` pairs that are tried in sequence.
//!
//! Commands: `create`, `list`, `get`, `update`, `delete`, `status`,
//! `clear-exclusions`, `reorder`.

use anyhow::Result;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::cli::FallbackArgs;
use crate::services::session_store::{FallbackConfig, FallbackEntry, SessionStore};
use crate::style;

pub struct FallbackCommand {
    session_store: SessionStore,
}

impl FallbackCommand {
    pub fn new(session_store: SessionStore) -> Self {
        Self { session_store }
    }

    pub async fn execute(&self, args: FallbackArgs) -> i32 {
        match self.execute_inner(args).await {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                1
            }
        }
    }

    async fn execute_inner(&self, args: FallbackArgs) -> Result<()> {
        let action = args.action.as_deref().unwrap_or("list");
        match action {
            "create" => self.cmd_create(args.name, args.entries).await,
            "list" | "ls" => self.cmd_list().await,
            "get" | "show" => self.cmd_get(args.name).await,
            "update" => self.cmd_update(args.name, args.entries).await,
            "delete" | "rm" | "remove" => self.cmd_delete(args.name).await,
            "status" => self.cmd_status(args.name).await,
            "clear-exclusions" | "clear" => self.cmd_clear_exclusions(args.name, args.entries).await,
            "reorder" => self.cmd_reorder(args.name, args.entries).await,
            other => {
                eprintln!("{} Unknown fallback action: {other}", style::red("Error:"));
                eprintln!("  Valid actions: create, list, get, update, delete, status, clear-exclusions, reorder");
                Ok(())
            }
        }
    }

    /// Validate that a string looks like a valid fallback name.
    fn validate_name(name: &str) -> Result<()> {
        if name.is_empty() {
            anyhow::bail!("Fallback name cannot be empty");
        }
        if name.contains(':') || name.contains(' ') || name.contains('/') {
            anyhow::bail!("Fallback name must not contain ':', spaces, or '/'");
        }
        Ok(())
    }

    /// `fallback create <name> <provider:model> [<provider:model> ...]`
    async fn cmd_create(&self, name: Option<String>, entries: Vec<String>) -> Result<()> {
        let name = name.ok_or_else(|| anyhow::anyhow!("Usage: fallback create <name> <provider:model> [...]"))?;
        Self::validate_name(&name)?;
        if entries.is_empty() {
            anyhow::bail!("At least one <provider:model> entry is required");
        }

        let parsed: Vec<FallbackEntry> = entries
            .iter()
            .map(|e| FallbackEntry::parse(e).ok_or_else(|| anyhow::anyhow!("Invalid entry {e:?}. Expected format: <provider>:<model>")))
            .collect::<Result<Vec<_>>>()?;

        // Check not already existing
        let existing = self.session_store.get_fallbacks().await?;
        if existing.contains_key(&name) {
            anyhow::bail!("Fallback '{name}' already exists. Use `fallback update` to modify, or `fallback delete` first.");
        }

        let config = FallbackConfig::new(parsed);
        self.session_store.set_fallback(name.clone(), config).await?;
        println!("  {} Created fallback '{name}'", style::green("✓"));
        Ok(())
    }

    /// `fallback list`
    async fn cmd_list(&self) -> Result<()> {
        let fallbacks = self.session_store.get_fallbacks().await?;
        if fallbacks.is_empty() {
            println!("  No fallbacks defined. Use `aivo fallback create <name> <provider:model> [...]`.");
            return Ok(());
        }

        println!("{}", style::bold("Fallbacks:"));
        for (name, config) in &fallbacks {
            let entries: Vec<String> = config.entries.iter().map(|e| e.to_provider_model()).collect();
            let active = active_entries(config, now());
            let excluded = config.entries.len() - active.len();
            let excl_str = if excluded > 0 {
                format!(" ({} excluded)", excluded)
            } else {
                String::new()
            };
            let last = config
                .last_used
                .as_ref()
                .map(|l| format!(", last used: {}", style::dim(l)))
                .unwrap_or_default();
            println!(
                "  {}  {}{}{}",
                style::cyan(name),
                entries.join(", "),
                excl_str,
                last,
            );
        }
        Ok(())
    }

    /// `fallback get <name>`
    async fn cmd_get(&self, name: Option<String>) -> Result<()> {
        let name = name.ok_or_else(|| anyhow::anyhow!("Usage: fallback get <name>"))?;
        let fallbacks = self.session_store.get_fallbacks().await?;
        let config = fallbacks
            .get(&name)
            .ok_or_else(|| anyhow::anyhow!("Fallback '{name}' not found"))?;

        println!("{} {}:", style::bold("Fallback"), style::cyan(&name));
        println!("  {}:", style::bold("Entries"));
        for (i, entry) in config.entries.iter().enumerate() {
            let excluded = config
                .exclusions
                .iter()
                .find(|e| e.provider == entry.provider && e.model == entry.model);
            let excl = match excluded {
                Some(ex) => format!("  {}", style::dim(format!("(excluded: {}, expires: {:?})", ex.reason, ex.expires_at))),
                None => String::new(),
            };
            println!("    {}. {}{}", i + 1, entry.to_provider_model(), excl);
        }
        if let Some(ref last) = config.last_used {
            println!("  {}: {}", style::bold("Last used"), last);
        }
        if config.exclusions.is_empty() {
            println!("  {}: none", style::bold("Exclusions"));
        } else {
            println!("  {}:", style::bold("Exclusions"));
            for ex in &config.exclusions {
                let expired = now() >= ex.expires_at.unwrap_or(i64::MAX);
                let exp_str = if expired {
                    " (expired)".to_string()
                } else {
                    format!(" (expires: {})", ex.expires_at.map_or("never".into(), |t| t.to_string()))
                };
                println!(
                    "    {}{}: {}",
                    ex.provider_model(),
                    exp_str,
                    style::dim(&ex.reason),
                );
            }
        }
        Ok(())
    }

    /// `fallback update <name> <provider:model> [<provider:model> ...]`
    async fn cmd_update(&self, name: Option<String>, entries: Vec<String>) -> Result<()> {
        let name = name.ok_or_else(|| anyhow::anyhow!("Usage: fallback update <name> <provider:model> [...]"))?;
        if entries.is_empty() {
            anyhow::bail!("At least one <provider:model> entry is required");
        }

        let parsed: Vec<FallbackEntry> = entries
            .iter()
            .map(|e| FallbackEntry::parse(e).ok_or_else(|| anyhow::anyhow!("Invalid entry {e:?}. Expected format: <provider>:<model>")))
            .collect::<Result<Vec<_>>>()?;

        let mut fallbacks = self.session_store.get_fallbacks().await?;
        let config = fallbacks
            .get_mut(&name)
            .ok_or_else(|| anyhow::anyhow!("Fallback '{name}' not found"))?;

        config.entries = parsed;
        // Reset last_used if entries changed (it no longer points to a valid entry)
        config.last_used = None;
        // Clear exclusions on update — they may not be relevant with new entries
        config.exclusions.clear();

        self.session_store.set_fallback(name.clone(), config.clone()).await?;
        println!("  {} Updated fallback '{name}'", style::green("✓"));
        Ok(())
    }

    /// `fallback delete <name>`
    async fn cmd_delete(&self, name: Option<String>) -> Result<()> {
        let name = name.ok_or_else(|| anyhow::anyhow!("Usage: fallback delete <name>"))?;
        let existed = self.session_store.remove_fallback(&name).await?;
        if existed {
            println!("  {} Deleted fallback '{name}'", style::green("✓"));
        } else {
            anyhow::bail!("Fallback '{name}' not found");
        }
        Ok(())
    }

    /// `fallback status <name>`
    async fn cmd_status(&self, name: Option<String>) -> Result<()> {
        let name = name.ok_or_else(|| anyhow::anyhow!("Usage: fallback status <name>"))?;
        let fallbacks = self.session_store.get_fallbacks().await?;
        let config = fallbacks
            .get(&name)
            .ok_or_else(|| anyhow::anyhow!("Fallback '{name}' not found"))?;

        let now = now();
        let active: Vec<&FallbackEntry> = config
            .entries
            .iter()
            .filter(|e| !config.exclusions.iter().any(|ex| {
                ex.provider == e.provider && ex.model == e.model && !ex.is_expired(now)
            }))
            .collect();
        let excluded_entries: Vec<&FallbackEntry> = config
            .entries
            .iter()
            .filter(|e| config.exclusions.iter().any(|ex| {
                ex.provider == e.provider && ex.model == e.model && !ex.is_expired(now)
            }))
            .collect();

        println!("{} {}:", style::bold("Fallback"), style::cyan(&name));
        println!("  {}: {} total", style::bold("Entries"), config.entries.len());
        println!("  {}: {}", style::bold("Active"), active.len());
        println!("  {}: {}", style::bold("Excluded"), excluded_entries.len());
        if let Some(ref last) = config.last_used {
            println!("  {}: {}", style::bold("Last used"), last);
        }

        if active.is_empty() && !config.entries.is_empty() {
            println!(
                "  {} All entries excluded — fallback will likely fail",
                style::yellow("⚠")
            );
        }
        Ok(())
    }

    /// `fallback clear-exclusions <name> [provider:model]`
    async fn cmd_clear_exclusions(&self, name: Option<String>, targets: Vec<String>) -> Result<()> {
        let name = name.ok_or_else(|| anyhow::anyhow!("Usage: fallback clear-exclusions <name> [provider:model]"))?;
        let mut fallbacks = self.session_store.get_fallbacks().await?;
        let config = fallbacks
            .get_mut(&name)
            .ok_or_else(|| anyhow::anyhow!("Fallback '{name}' not found"))?;

        if targets.is_empty() {
            // Clear all exclusions
            config.exclusions.clear();
            println!("  {} Cleared all exclusions for '{name}'", style::green("✓"));
        } else {
            // Clear specific entries
            for target in &targets {
                if let Some(entry) = FallbackEntry::parse(target) {
                    config.exclusions.retain(|e| !(e.provider == entry.provider && e.model == entry.model));
                }
            }
            println!("  {} Cleared exclusions for {targets:?} in '{name}'", style::green("✓"));
        }

        self.session_store
            .set_fallback(name.clone(), config.clone())
            .await?;
        Ok(())
    }

    /// `fallback reorder <name> <provider:model> [<provider:model> ...]`
    async fn cmd_reorder(&self, name: Option<String>, entries: Vec<String>) -> Result<()> {
        let name = name.ok_or_else(|| anyhow::anyhow!("Usage: fallback reorder <name> <provider:model> [...]"))?;
        if entries.is_empty() {
            anyhow::bail!("Must provide all entries in the desired order");
        }

        let reordered: Vec<FallbackEntry> = entries
            .iter()
            .map(|e| FallbackEntry::parse(e).ok_or_else(|| anyhow::anyhow!("Invalid entry {e:?}")))
            .collect::<Result<Vec<_>>>()?;

        let mut fallbacks = self.session_store.get_fallbacks().await?;
        let config = fallbacks
            .get_mut(&name)
            .ok_or_else(|| anyhow::anyhow!("Fallback '{name}' not found"))?;

        // Validate that reordered list contains exactly the same entries
        let original: std::collections::HashSet<(String, String)> = config
            .entries
            .iter()
            .map(|e| (e.provider.clone(), e.model.clone()))
            .collect();
        let reordered_set: std::collections::HashSet<(String, String)> = reordered
            .iter()
            .map(|e| (e.provider.clone(), e.model.clone()))
            .collect();

        if original != reordered_set {
            anyhow::bail!(
                "Reordered entries don't match existing entries. Existing: {}",
                config
                    .entries
                    .iter()
                    .map(|e| e.to_provider_model())
                    .collect::<Vec<_>>()
                    .join(", "),
            );
        }

        config.entries = reordered;
        self.session_store.set_fallback(name.clone(), config.clone()).await?;
        println!("  {} Reordered fallback '{name}'", style::green("✓"));
        Ok(())
    }
}

fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// Returns entries that are not currently excluded (or whose exclusion has expired).
fn active_entries(config: &FallbackConfig, now: i64) -> Vec<&FallbackEntry> {
    config
        .entries
        .iter()
        .filter(|e| {
            !config
                .exclusions
                .iter()
                .any(|ex| ex.provider == e.provider && ex.model == e.model && !ex.is_expired(now))
        })
        .collect()
}

impl FallbackCommand {
    pub fn print_help() {
        println!("{} aivo fallback <action> [name] [entries...]", crate::style::bold("Usage:"));
        println!();
        println!("{}", crate::style::bold("Manage fallback definitions — virtual models that try provider:model pairs in sequence."));
        println!();
        println!("{}", crate::style::bold("Actions:"));
        let print_opt = |act: &str, desc: &str| {
            println!("  {}  {}", crate::style::cyan(format!("{:<20}", act)), crate::style::dim(desc));
        };
        print_opt("create <name> <p:m>...", "Create a new fallback");
        print_opt("list", "List all fallbacks");
        print_opt("get <name>", "Show fallback details");
        print_opt("update <name> <p:m>...", "Replace entries in an existing fallback");
        print_opt("delete <name>", "Remove a fallback");
        print_opt("status <name>", "Show active/excluded entry counts");
        print_opt("clear-exclusions <name> [p:m]", "Clear exclusions (all or for one entry)");
        print_opt("reorder <name> <p:m>...", "Reorder entries (provide all in new order)");
        println!();
        println!("{}", crate::style::bold("A fallback name can be used anywhere a model is accepted (e.g. --model)."));
        println!("  Entries are tried in order until one succeeds.");
    }
}
