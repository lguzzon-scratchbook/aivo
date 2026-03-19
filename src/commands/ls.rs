use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::commands::truncate_url_for_display;
use crate::errors::ExitCode;
use crate::services::models_cache::ModelsCache;
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

fn collect_path_dirs() -> Vec<PathBuf> {
    collect_path_dirs_from(std::env::var_os("PATH"))
}

fn collect_path_dirs_from(path_var: Option<std::ffi::OsString>) -> Vec<PathBuf> {
    let Some(path_var) = path_var else {
        return Vec::new();
    };
    std::env::split_paths(&path_var).collect()
}

fn find_in_dirs(program: &str, dirs: &[PathBuf]) -> Option<PathBuf> {
    #[cfg(windows)]
    let exts: Vec<String> = std::env::var_os("PATHEXT")
        .map(|value| {
            value
                .to_string_lossy()
                .split(';')
                .filter(|ext| !ext.is_empty())
                .map(|ext| ext.to_string())
                .collect()
        })
        .unwrap_or_else(|| vec![".EXE".to_string(), ".BAT".to_string(), ".CMD".to_string()]);

    for dir in dirs {
        let candidate = dir.join(program);
        if is_executable(&candidate) {
            return Some(candidate);
        }

        #[cfg(windows)]
        {
            for ext in &exts {
                let candidate = dir.join(format!("{}{}", program, ext));
                if is_executable(&candidate) {
                    return Some(candidate);
                }
            }
        }
    }

    None
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    match path.metadata() {
        Ok(meta) => meta.is_file() && (meta.permissions().mode() & 0o111 != 0),
        Err(_) => false,
    }
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn collect_path_dirs_from_none_returns_empty() {
        assert!(collect_path_dirs_from(None).is_empty());
    }

    #[test]
    fn collect_path_dirs_from_splits_multiple_entries() {
        let joined = std::env::join_paths(["/tmp/aivo-bin", "/usr/local/bin"]).unwrap();
        let dirs = collect_path_dirs_from(Some(joined));
        assert_eq!(
            dirs,
            vec![
                PathBuf::from("/tmp/aivo-bin"),
                PathBuf::from("/usr/local/bin")
            ]
        );
    }

    #[cfg(unix)]
    #[test]
    fn find_in_dirs_returns_only_executable_files() {
        use std::os::unix::fs::PermissionsExt;

        let dir = TempDir::new().unwrap();
        let script = dir.path().join("claude");
        let plain = dir.path().join("codex");

        std::fs::write(&script, "#!/bin/sh\n").unwrap();
        std::fs::write(&plain, "plain-text").unwrap();

        let mut perms = std::fs::metadata(&script).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&script, perms).unwrap();

        let mut plain_perms = std::fs::metadata(&plain).unwrap().permissions();
        plain_perms.set_mode(0o644);
        std::fs::set_permissions(&plain, plain_perms).unwrap();

        let dirs = vec![dir.path().to_path_buf()];

        assert_eq!(find_in_dirs("claude", &dirs), Some(script));
        assert_eq!(find_in_dirs("codex", &dirs), None);
    }

    #[cfg(not(unix))]
    #[test]
    fn find_in_dirs_matches_existing_file_on_non_unix() {
        let dir = TempDir::new().unwrap();
        let program = dir.path().join("claude");
        std::fs::write(&program, "binary").unwrap();

        let dirs = vec![dir.path().to_path_buf()];
        assert_eq!(find_in_dirs("claude", &dirs), Some(program));
    }
}
