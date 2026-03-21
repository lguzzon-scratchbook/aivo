use anyhow::Result;

use crate::commands::truncate_url_for_display;
use crate::errors::ExitCode;
use crate::services::models_cache::ModelsCache;
use crate::services::path_search::{collect_path_dirs, find_in_dirs};
use crate::services::session_store::{DirectoryStartRecord, SessionStore};
use crate::services::system_env;
use crate::style;

pub struct LsCommand {
    session_store: SessionStore,
    cache: ModelsCache,
}

impl LsCommand {
    pub fn new(session_store: SessionStore, cache: ModelsCache) -> Self {
        Self {
            session_store,
            cache,
        }
    }

    pub async fn execute(&self) -> ExitCode {
        match self.execute_internal().await {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                ExitCode::UserError
            }
        }
    }

    async fn execute_internal(&self) -> Result<ExitCode> {
        let (keys, active_key_id) = self.session_store.get_keys_and_active_id_info().await?;
        let cwd = system_env::current_dir_string().unwrap_or_else(|| ".".to_string());
        let remembered = self.session_store.get_directory_start(&cwd).await?;
        let active_key = active_key_id
            .as_deref()
            .and_then(|active_id| keys.iter().find(|key| key.id == active_id));

        println!("{}", style::bold("Keys:"));
        if keys.is_empty() {
            println!("  {}", style::dim("(none)"));
        } else {
            let max_name_len = keys
                .iter()
                .map(|k| k.display_name().len())
                .max()
                .unwrap_or(0);
            for key in &keys {
                let is_active = active_key_id.as_deref() == Some(key.id.as_str());
                let marker = if is_active {
                    style::bullet_symbol()
                } else {
                    style::empty_bullet_symbol()
                };
                println!(
                    "  {} {}  {:width$}  {}",
                    marker,
                    style::cyan(key.short_id()),
                    key.display_name(),
                    style::dim(truncate_url_for_display(&key.base_url, 50)),
                    width = max_name_len
                );
            }
        }

        println!();
        println!("{}", style::bold("Tools:"));
        let path_dirs = collect_path_dirs();
        for tool in ["claude", "codex", "gemini", "opencode", "pi"] {
            match find_in_dirs(tool, &path_dirs) {
                Some(path) => println!(
                    "  {} {:8} {}",
                    style::success_symbol(),
                    style::cyan(tool),
                    style::dim(path.display().to_string())
                ),
                None => println!(
                    "  {} {:8} {}",
                    style::empty_bullet_symbol(),
                    style::cyan(tool),
                    style::dim("not found on PATH")
                ),
            }
        }

        println!();
        println!("{}", style::bold("Current directory:"));
        println!("  {}", style::dim(&cwd));
        match remembered {
            Some(record) => Self::print_directory_start(&record, &keys),
            None => println!(
                "  {}",
                style::dim("No remembered start for this directory.")
            ),
        }

        println!();
        println!("{}", style::bold("Active defaults:"));
        if let Some(key) = active_key {
            println!(
                "  {} {} {}",
                style::dim("key:"),
                style::cyan(key.display_name()),
                style::dim(format!("({})", key.base_url))
            );

            match self.session_store.get_chat_model(&key.id).await? {
                Some(model) => println!("  {} {}", style::dim("chat model:"), model),
                None => println!("  {} {}", style::dim("chat model:"), style::dim("(none)")),
            }

            match self.cache.get(&key.base_url).await {
                Some(models) => println!("  {} {}", style::dim("cached models:"), models.len()),
                None => println!(
                    "  {} {}",
                    style::dim("cached models:"),
                    style::dim("(none)")
                ),
            }
        } else {
            println!("  {}", style::dim("No active key."));
        }

        Ok(ExitCode::Success)
    }

    pub fn print_help() {
        println!("{} aivo ls", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim("Show saved keys, installed tool binaries, and current directory state.")
        );
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo ls"));
    }

    fn print_directory_start(
        record: &DirectoryStartRecord,
        keys: &[crate::services::session_store::ApiKey],
    ) {
        let key_name = keys
            .iter()
            .find(|key| key.id == record.key_id)
            .map(|key| key.display_name().to_string())
            .unwrap_or_else(|| record.key_id.clone());
        let model = record.model.as_deref().unwrap_or("(tool default)");

        println!(
            "  {} {} · {} · {}",
            style::dim("start:"),
            style::cyan(&record.tool),
            key_name,
            model
        );
    }
}

// Tests for path search utilities are in services::path_search
