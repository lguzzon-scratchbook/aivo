use std::path::{Path, PathBuf};

/// Best-effort user home directory lookup using standard environment variables.
pub fn home_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var_os("USERPROFILE")
            .map(PathBuf::from)
            .or_else(|| {
                let drive = std::env::var_os("HOMEDRIVE")?;
                let path = std::env::var_os("HOMEPATH")?;
                Some(PathBuf::from(format!(
                    "{}{}",
                    drive.to_string_lossy(),
                    path.to_string_lossy()
                )))
            })
    }

    #[cfg(not(windows))]
    {
        std::env::var_os("HOME").map(PathBuf::from)
    }
}

/// Best-effort current username lookup.
/// Tries the USER/USERNAME environment variable first, then falls back to the
/// OS user database via libc on Unix (unaffected by sudo USER overrides or
/// environments where USER is unset).
pub fn username() -> Option<String> {
    #[cfg(windows)]
    {
        std::env::var("USERNAME").ok().filter(|s| !s.is_empty())
    }

    #[cfg(not(windows))]
    {
        if let Ok(user) = std::env::var("USER")
            && !user.is_empty()
        {
            return Some(user);
        }

        // Fall back to OS user database so key derivation remains consistent
        // when USER is unset (CI, containers, sudo -i, etc.).
        #[cfg(unix)]
        // SAFETY: getpwuid returns a pointer to static thread-local storage valid
        // until the next getpwuid call on this thread. We copy pw_name immediately.
        unsafe {
            let uid = libc::getuid();
            let passwd = libc::getpwuid(uid);
            if !passwd.is_null() {
                let name = std::ffi::CStr::from_ptr((*passwd).pw_name);
                if let Ok(s) = name.to_str()
                    && !s.is_empty()
                {
                    return Some(s.to_string());
                }
            }
        }

        None
    }
}

/// Expands a leading `~` to the user's home directory.
/// Returns the path unchanged (as a `PathBuf`) if expansion is not needed or not possible.
pub fn expand_tilde(path: &str) -> PathBuf {
    expand_tilde_with_home(path, home_dir().as_deref())
}

/// Best-effort current working directory lookup with canonicalization when possible.
pub fn current_dir() -> Option<PathBuf> {
    let cwd = std::env::current_dir().ok()?;
    std::fs::canonicalize(&cwd).ok().or(Some(cwd))
}

pub fn current_dir_string() -> Option<String> {
    current_dir().map(|path| path.to_string_lossy().to_string())
}

/// Returns a hardware-specific machine identifier.
/// - macOS: IOPlatformUUID
/// - Linux: /etc/machine-id
/// - Windows: HKLM\SOFTWARE\Microsoft\Cryptography\MachineGuid
pub fn machine_id() -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("ioreg")
            .args(["-rd1", "-c", "IOPlatformExpertDevice"])
            .output()
            .ok()?;
        parse_macos_platform_uuid(&String::from_utf8_lossy(&output.stdout))
    }

    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/etc/machine-id")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    #[cfg(target_os = "windows")]
    {
        let output = std::process::Command::new("reg")
            .args([
                "query",
                r"HKLM\SOFTWARE\Microsoft\Cryptography",
                "/v",
                "MachineGuid",
            ])
            .output()
            .ok()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            if line.contains("MachineGuid") {
                // Format: MachineGuid    REG_SZ    XXXXXXXX-...
                if let Some(guid) = line.split_whitespace().last() {
                    let guid = guid.trim().to_string();
                    if !guid.is_empty() {
                        return Some(guid);
                    }
                }
            }
        }
        None
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        None
    }
}

#[cfg(target_os = "macos")]
fn parse_macos_platform_uuid(output: &str) -> Option<String> {
    output.lines().find_map(|line| {
        let pos = line.find("IOPlatformUUID")?;
        let start = line[pos..].find('"').map(|i| pos + i + 1)?;
        let end = line[start..].find('"').map(|i| start + i)?;
        let uuid = line[start..end].trim().to_string();
        (!uuid.is_empty()).then_some(uuid)
    })
}

fn expand_tilde_with_home(path: &str, home: Option<&Path>) -> PathBuf {
    if path == "~" {
        return home
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("~"));
    }
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = home
    {
        return home.join(rest);
    }
    PathBuf::from(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_tilde_replaces_home_prefix() {
        let home = Path::new("/tmp/example-home");
        assert_eq!(
            expand_tilde_with_home("~/config/aivo", Some(home)),
            home.join("config/aivo")
        );
        assert_eq!(expand_tilde_with_home("~", Some(home)), home);
    }

    #[test]
    fn expand_tilde_leaves_non_home_paths_unchanged() {
        assert_eq!(
            expand_tilde_with_home("/var/tmp/aivo", Some(Path::new("/tmp/home"))),
            PathBuf::from("/var/tmp/aivo")
        );
        assert_eq!(
            expand_tilde_with_home("~/docs", None),
            PathBuf::from("~/docs")
        );
    }

    #[test]
    fn current_dir_string_returns_non_empty_path() {
        let cwd = current_dir_string().expect("cwd should be available");
        assert!(!cwd.is_empty());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn parse_macos_platform_uuid_extracts_value() {
        // The parser hops between quote pairs: first `"` after the key name,
        // then the next `"`.  For the standard ioreg format this yields `=`
        // (the text between the closing key-quote and the opening value-quote).
        // Changing this would alter the encryption key derived from machine_id.
        let output = r#"    "IOPlatformUUID" = "12345678-1234-1234-1234-123456789ABC""#;

        assert_eq!(parse_macos_platform_uuid(output).as_deref(), Some("="));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn parse_macos_platform_uuid_rejects_blank_values() {
        // When there is no content between the two quote hops the result is empty.
        let output = r#"    "IOPlatformUUID"""#;
        assert_eq!(parse_macos_platform_uuid(output), None);
    }
}
