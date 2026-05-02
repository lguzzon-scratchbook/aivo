//! Shared PATH-scanning utilities for finding executables.

use std::path::{Path, PathBuf};

pub fn collect_path_dirs() -> Vec<PathBuf> {
    collect_path_dirs_from(std::env::var_os("PATH"))
}

pub fn collect_path_dirs_from(path_var: Option<std::ffi::OsString>) -> Vec<PathBuf> {
    let Some(path_var) = path_var else {
        return Vec::new();
    };
    std::env::split_paths(&path_var).collect()
}

pub fn find_in_dirs(program: &str, dirs: &[PathBuf]) -> Option<PathBuf> {
    #[cfg(windows)]
    {
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

        // On Windows, CreateProcessW only spawns files with a recognized
        // executable extension. Skip the bare-name probe when the program has
        // no extension, otherwise we'd return e.g. npm's bash-style `claude`
        // shim (no extension) instead of the spawnable `claude.cmd` sibling.
        let has_explicit_ext = Path::new(program).extension().is_some();
        for dir in dirs {
            if has_explicit_ext {
                let candidate = dir.join(program);
                if is_executable(&candidate) {
                    return Some(candidate);
                }
                continue;
            }
            for ext in &exts {
                let candidate = dir.join(format!("{}{}", program, ext));
                if is_executable(&candidate) {
                    return Some(candidate);
                }
            }
        }
        return None;
    }

    #[cfg(not(windows))]
    {
        for dir in dirs {
            let candidate = dir.join(program);
            if is_executable(&candidate) {
                return Some(candidate);
            }
        }
        None
    }
}

#[cfg(unix)]
pub fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    match path.metadata() {
        Ok(meta) => meta.is_file() && (meta.permissions().mode() & 0o111 != 0),
        Err(_) => false,
    }
}

#[cfg(not(unix))]
pub fn is_executable(path: &Path) -> bool {
    path.is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn find_in_dirs_empty_returns_none() {
        assert_eq!(find_in_dirs("claude", &[]), None);
    }

    #[cfg(unix)]
    #[test]
    fn find_in_dirs_returns_only_executable_files() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::TempDir::new().unwrap();
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

    #[cfg(windows)]
    #[test]
    fn find_in_dirs_prefers_pathext_over_bare_name_on_windows() {
        // npm on Windows drops THREE shims for each global binary:
        //   `<name>`      — bash script for Cygwin/Git Bash
        //   `<name>.cmd`  — cmd.exe shim
        //   `<name>.ps1`  — PowerShell shim
        // CreateProcessW can only spawn the .cmd one. The bare-name file
        // exists too, so a naive lookup that returns the extensionless path
        // breaks `aivo claude` / `aivo codex`. This test pins that contract.
        let dir = tempfile::TempDir::new().unwrap();
        let bare = dir.path().join("claude");
        let cmd = dir.path().join("claude.cmd");
        std::fs::write(&bare, "bash shim").unwrap();
        std::fs::write(&cmd, "@echo off\r\n").unwrap();

        let dirs = vec![dir.path().to_path_buf()];
        assert_eq!(find_in_dirs("claude", &dirs), Some(cmd));
    }

    #[cfg(windows)]
    #[test]
    fn find_in_dirs_skips_extensionless_files_on_windows() {
        // Same logic but without a .cmd sibling — must NOT match the bare
        // name, since CreateProcessW would refuse to spawn it.
        let dir = tempfile::TempDir::new().unwrap();
        let bare = dir.path().join("claude");
        std::fs::write(&bare, "bash shim").unwrap();

        let dirs = vec![dir.path().to_path_buf()];
        assert_eq!(find_in_dirs("claude", &dirs), None);
    }

    #[cfg(windows)]
    #[test]
    fn find_in_dirs_uses_explicit_extension_when_program_has_one() {
        // If the caller passes `claude.cmd`, look up that exact filename.
        let dir = tempfile::TempDir::new().unwrap();
        let cmd = dir.path().join("claude.cmd");
        std::fs::write(&cmd, "@echo off\r\n").unwrap();

        let dirs = vec![dir.path().to_path_buf()];
        assert_eq!(find_in_dirs("claude.cmd", &dirs), Some(cmd));
    }
}
