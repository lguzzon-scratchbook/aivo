/**
 * UpdateCommand handler for CLI self-update functionality.
 */
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use reqwest::{Client, RequestBuilder};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use crate::errors::ExitCode;
use crate::services::path_search::{collect_path_dirs, find_in_dirs};
use crate::style;

const GITHUB_API: &str = "https://api.github.com/repos/yuanchuan/aivo/releases/latest";
const GITHUB_RELEASES_LATEST: &str = "https://github.com/yuanchuan/aivo/releases/latest";
const GITHUB_LATEST_DOWNLOAD_BASE: &str =
    "https://github.com/yuanchuan/aivo/releases/latest/download";
const NPM_UPDATE_COMMAND: &str = "npm install -g @yuanchuan/aivo@latest";
const NPM_UPDATE_ARGS: [&str; 3] = ["install", "-g", "@yuanchuan/aivo@latest"];

/// UpdateCommand handles CLI self-update via GitHub Releases
pub struct UpdateCommand {
    client: Client,
}

/// GitHub Release asset from the API
#[derive(Debug, Clone, serde::Deserialize)]
struct GitHubAsset {
    name: String,
    #[serde(rename = "browser_download_url")]
    browser_download_url: String,
    #[serde(default)]
    digest: Option<String>,
}

/// GitHub Release response from the API
#[derive(Debug, Clone, serde::Deserialize)]
struct GitHubRelease {
    #[serde(rename = "tag_name")]
    tag_name: String,
    assets: Vec<GitHubAsset>,
}

impl GitHubRelease {
    fn version(&self) -> &str {
        self.tag_name.trim_start_matches('v')
    }
}

impl UpdateCommand {
    /// Shows usage information for the update command
    pub fn print_help() {
        println!("{} aivo update [OPTIONS]", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim("Update the CLI tool to the latest version.")
        );
        println!(
            "{}",
            style::dim("Delegates to Homebrew or npm when installed via those package managers.")
        );
        println!();
        println!("{}", style::bold("Options:"));
        let print_opt = |flag: &str, desc: &str| {
            println!(
                "  {}{}",
                style::cyan(format!("{:<26}", flag)),
                style::dim(desc)
            );
        };
        print_opt(
            "-f, --force",
            "Force update even if installed via a package manager",
        );
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo update"));
        println!("  {}", style::dim("aivo update --force"));
    }

    /// Creates a new UpdateCommand instance
    pub fn new() -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self { client })
    }

    /// Executes the update command
    pub async fn execute(&self, force: bool) -> ExitCode {
        match self.execute_internal(force).await {
            Ok(code) => code,
            Err(e) => {
                self.handle_error(e);
                ExitCode::UserError
            }
        }
    }

    async fn execute_internal(&self, force: bool) -> Result<ExitCode> {
        // Check for package-manager-managed installations
        if !force {
            let install_path = get_install_path()?;
            if let Some(manager) = detect_managed_install(&install_path) {
                match manager.kind {
                    PackageManager::Homebrew => {
                        return Ok(self.update_via_homebrew());
                    }
                    PackageManager::Npm => {
                        return self.update_via_npm(&manager);
                    }
                    PackageManager::Cargo => {
                        eprintln!(
                            "{} aivo was installed via {}.",
                            style::yellow("Warning:"),
                            manager.name
                        );
                        eprintln!(
                            "  Self-update would bypass {} and may cause issues.",
                            manager.name
                        );
                        eprintln!();
                        eprintln!(
                            "  {} {}",
                            style::dim("Update with:"),
                            style::green(manager.upgrade_command)
                        );
                        eprintln!(
                            "  {} {}",
                            style::dim("Force self-update:"),
                            style::green("aivo update --force")
                        );
                        return Ok(ExitCode::UserError);
                    }
                }
            }
        }

        println!("{} Checking for updates...", style::arrow_symbol());

        let current_version = crate::version::VERSION;
        let release = self.get_latest_release().await?;
        let latest_version = release.version();

        let binary_name = get_binary_name()?;
        let asset = release
            .assets
            .iter()
            .find(|a| a.name == binary_name)
            .ok_or_else(|| anyhow::anyhow!("No binary found for {}", binary_name))?;
        let expected_sha256 = self
            .resolve_expected_sha256(&release, &binary_name, asset)
            .await?;

        if !self.is_newer_version(latest_version, current_version) {
            println!(
                "{} Already up to date {}",
                style::success_symbol(),
                style::dim(format!("({})", current_version))
            );
            return Ok(ExitCode::Success);
        }

        println!("  Current: {}", style::dim(current_version));
        println!("  Latest:  {}", style::green(latest_version));
        println!("{} Downloading update...", style::arrow_symbol());

        self.install_update(&asset.browser_download_url, &expected_sha256)
            .await?;

        println!(
            "{} Updated to version {}",
            style::success_symbol(),
            latest_version
        );

        Ok(ExitCode::Success)
    }

    /// Fetches the latest release from GitHub API
    async fn get_latest_release(&self) -> Result<GitHubRelease> {
        let response = self
            .github_request(GITHUB_API)
            .header("Accept", "application/vnd.github.v3+json")
            .send()
            .await
            .context("Failed to fetch latest release")?;

        let status = response.status();
        let text = response.text().await.unwrap_or_default();

        if !status.is_success() {
            if status.as_u16() == 404 {
                return Err(anyhow::anyhow!("No releases found"));
            }
            if (status.as_u16() == 403 || status.as_u16() == 429)
                && let Ok(release) = self.get_latest_release_fallback().await
            {
                eprintln!(
                    "{} GitHub API returned {}. Falling back to GitHub Releases web endpoint.",
                    style::yellow("Warning:"),
                    status
                );
                return Ok(release);
            }
            if let Some(message) = parse_github_error_message(&text) {
                return Err(anyhow::anyhow!(
                    "GitHub API returned {}: {}",
                    status,
                    message
                ));
            }
            return Err(anyhow::anyhow!("GitHub API returned {}", status));
        }

        serde_json::from_str(&text).context("Failed to parse release response")
    }

    async fn get_latest_release_fallback(&self) -> Result<GitHubRelease> {
        let response = self
            .github_request(GITHUB_RELEASES_LATEST)
            .send()
            .await
            .context("Failed to resolve latest release via web endpoint")?;

        let status = response.status();
        if !status.is_success() {
            return Err(anyhow::anyhow!(
                "Fallback latest release lookup failed: HTTP {}",
                status
            ));
        }

        let tag_name = parse_release_tag_from_url(response.url()).ok_or_else(|| {
            anyhow::anyhow!("Could not determine latest release tag from redirect URL")
        })?;

        let mut assets = Vec::new();
        for binary_name in supported_binary_asset_names() {
            assets.push(GitHubAsset {
                name: binary_name.to_string(),
                browser_download_url: format!("{}/{}", GITHUB_LATEST_DOWNLOAD_BASE, binary_name),
                digest: None,
            });
            assets.push(GitHubAsset {
                name: format!("{}.sha256", binary_name),
                browser_download_url: format!(
                    "{}/{}.sha256",
                    GITHUB_LATEST_DOWNLOAD_BASE, binary_name
                ),
                digest: None,
            });
        }
        assets.push(GitHubAsset {
            name: "checksums.txt".to_string(),
            browser_download_url: format!("{}/checksums.txt", GITHUB_LATEST_DOWNLOAD_BASE),
            digest: None,
        });

        Ok(GitHubRelease { tag_name, assets })
    }

    /// Resolves the expected SHA-256 checksum for the binary.
    /// Refuses update when no checksum source is available.
    async fn resolve_expected_sha256(
        &self,
        release: &GitHubRelease,
        binary_name: &str,
        binary_asset: &GitHubAsset,
    ) -> Result<String> {
        if let Some(digest) = binary_asset.digest.as_deref()
            && let Some(sha256) = parse_digest_sha256(digest)
        {
            return Ok(sha256);
        }

        let sha256_name = format!("{}.sha256", binary_name);
        if let Some(asset) = release.assets.iter().find(|a| a.name == sha256_name) {
            let checksum_text = self.fetch_text(&asset.browser_download_url).await?;
            if let Some(sha256) = parse_checksum_text(&checksum_text, binary_name) {
                return Ok(sha256);
            }
            return Err(anyhow::anyhow!(
                "Checksum asset '{}' could not be parsed",
                sha256_name
            ));
        }

        if let Some(asset) = release.assets.iter().find(|a| a.name == "checksums.txt") {
            let checksum_text = self.fetch_text(&asset.browser_download_url).await?;
            if let Some(sha256) = parse_checksum_text(&checksum_text, binary_name) {
                return Ok(sha256);
            }
            return Err(anyhow::anyhow!(
                "checksums.txt does not contain an entry for {}",
                binary_name
            ));
        }

        Err(anyhow::anyhow!(
            "No checksum available for {}. Refusing insecure update.",
            binary_name
        ))
    }

    async fn fetch_text(&self, url: &str) -> Result<String> {
        let response = self
            .github_request(url)
            .send()
            .await
            .context("Failed to fetch checksum asset")?;

        let status = response.status();
        if !status.is_success() {
            return Err(anyhow::anyhow!("Checksum download failed: HTTP {}", status));
        }

        response
            .text()
            .await
            .context("Failed to read checksum asset response")
    }

    /// Downloads and installs the update
    async fn install_update(&self, download_url: &str, expected_sha256: &str) -> Result<()> {
        let mut response = self
            .github_request(download_url)
            .timeout(std::time::Duration::from_secs(600)) // 10 minutes
            .send()
            .await
            .context("Failed to download update")?;

        let status = response.status();
        if !status.is_success() {
            return Err(anyhow::anyhow!("Download failed: HTTP {}", status));
        }

        let total_size = response.content_length().unwrap_or(0);

        // Determine install path
        let exec_path = get_install_path()?;
        let tmp_path = exec_path.with_extension("tmp");

        // Stream the download directly to file with incremental hashing
        let mut hasher = Sha256::new();
        let mut downloaded: u64 = 0;
        let mut file = tokio::fs::File::create(&tmp_path)
            .await
            .with_context(|| format!("Failed to create temporary file at {:?}", tmp_path))?;

        while let Some(chunk) = response
            .chunk()
            .await
            .context("Error reading download stream")?
        {
            hasher.update(&chunk);
            file.write_all(&chunk)
                .await
                .with_context(|| format!("Failed to write to temporary file at {:?}", tmp_path))?;
            downloaded += chunk.len() as u64;

            if total_size > 0 {
                let mb = downloaded as f64 / 1024.0 / 1024.0;
                let total_mb = total_size as f64 / 1024.0 / 1024.0;
                let percent = (downloaded as f64 / total_size as f64) * 100.0;
                eprint!(
                    "\r  {} {:.1}/{:.1} MB ({:.0}%)",
                    style::dim("Downloading:"),
                    mb,
                    total_mb,
                    percent
                );
            }
        }
        if let Err(e) = file.flush().await {
            tokio::fs::remove_file(&tmp_path).await.ok();
            return Err(e.into());
        }
        if total_size > 0 {
            eprintln!(); // newline after progress
        }

        let actual_sha256 = format!("{:x}", hasher.finalize());
        if actual_sha256 != expected_sha256 {
            tokio::fs::remove_file(&tmp_path).await.ok();
            return Err(anyhow::anyhow!(
                "Checksum verification failed for downloaded update"
            ));
        }
        println!(
            "  {} {}",
            style::dim("Checksum (SHA-256):"),
            style::green("verified")
        );

        // Make executable (Unix only)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let permissions = std::fs::Permissions::from_mode(0o755);
            if let Err(e) = tokio::fs::set_permissions(&tmp_path, permissions).await {
                tokio::fs::remove_file(&tmp_path).await.ok();
                return Err(e.into());
            }
        }

        // Atomically replace the old binary
        if let Err(e) = tokio::fs::rename(&tmp_path, &exec_path).await {
            tokio::fs::remove_file(&tmp_path).await.ok();
            return Err(e).with_context(|| format!("Failed to replace binary at {:?}", exec_path));
        }

        println!("  {} {}", style::dim("Installed to:"), exec_path.display());

        Ok(())
    }

    /// Compares two semantic version strings.
    /// Strips pre-release suffixes (e.g. -rc1, -beta.1) before comparing.
    /// A pre-release version is considered older than its release counterpart.
    fn is_newer_version(&self, latest: &str, current: &str) -> bool {
        let parse_version = |version: &str| -> (Vec<u32>, bool) {
            let cleaned = version.trim_start_matches('v');
            // Split off pre-release suffix at the first hyphen
            let (version_str, has_prerelease) = match cleaned.split_once('-') {
                Some((v, _)) => (v, true),
                None => (cleaned, false),
            };
            let parts = version_str
                .split('.')
                .filter_map(|part| part.parse::<u32>().ok())
                .collect();
            (parts, has_prerelease)
        };

        let (latest_parts, latest_pre) = parse_version(latest);
        let (current_parts, current_pre) = parse_version(current);

        let max_len = latest_parts.len().max(current_parts.len());

        for i in 0..max_len {
            let latest_part = latest_parts.get(i).copied().unwrap_or(0);
            let current_part = current_parts.get(i).copied().unwrap_or(0);

            if latest_part > current_part {
                return true;
            }
            if latest_part < current_part {
                return false;
            }
        }

        // Same numeric version: release is newer than pre-release
        // e.g. "2.0.0" is newer than "2.0.0-rc1"
        if current_pre && !latest_pre {
            return true;
        }

        false
    }

    /// Handles errors
    fn handle_error(&self, error: anyhow::Error) {
        eprintln!("{} {}", style::red("Error:"), error);
        eprintln!();
        let msg = format!("{:#}", error);
        if msg.contains("GitHub API returned 403") || msg.contains("GitHub API returned 429") {
            eprintln!(
                "{} GitHub may be rate-limiting anonymous API requests from your IP.",
                style::yellow("Hint:")
            );
            eprintln!(
                "  {}",
                style::dim("Set GITHUB_TOKEN (or GH_TOKEN/AIVO_GITHUB_TOKEN) and retry.")
            );
            eprintln!();
        }
        eprintln!(
            "{} Check your internet connection and try again.",
            style::yellow("Suggestion:")
        );
    }

    fn github_request(&self, url: &str) -> RequestBuilder {
        let mut req = self.client.get(url).header("User-Agent", "aivo-cli");
        if let Some(token) = github_token_from_env() {
            req = req.bearer_auth(token);
        }
        req
    }

    /// Delegates update to Homebrew
    fn update_via_homebrew(&self) -> ExitCode {
        println!("{} Updating via Homebrew...", style::arrow_symbol());

        // Run brew update first to fetch latest formulas (ignore errors)
        let _ = Command::new("brew").args(["update", "--quiet"]).status();

        // Then upgrade aivo (--overwrite to handle symlink conflicts)
        println!("{} Upgrading aivo...", style::arrow_symbol());
        match Command::new("brew")
            .args(["upgrade", "--overwrite", "aivo"])
            .status()
        {
            Ok(status) if status.success() => ExitCode::Success,
            Ok(_) => ExitCode::Success,
            Err(e) => {
                eprintln!("{} brew upgrade failed: {}", style::red("Error:"), e);
                ExitCode::UserError
            }
        }
    }

    fn update_via_npm(&self, manager: &ManagedInstall) -> Result<ExitCode> {
        #[cfg(windows)]
        {
            eprintln!(
                "{} Windows npm installs are updated by the npm shim, not by aivo.exe directly.",
                style::yellow("Warning:")
            );
            eprintln!("  {} {}", style::dim("Run:"), style::green("aivo update"));
            eprintln!(
                "  {} {}",
                style::dim("Or repair with:"),
                style::green(manager.upgrade_command)
            );
            return Ok(ExitCode::UserError);
        }

        #[cfg(not(windows))]
        let npm_path = resolve_command_path("npm").ok_or_else(|| {
            anyhow::anyhow!(
                "Could not find npm on PATH. Run this command manually: {}",
                manager.upgrade_command
            )
        })?;

        #[cfg(not(windows))]
        {
            println!("{} Updating via npm...", style::arrow_symbol());
            println!(
                "  {} {}",
                style::dim("Running:"),
                style::green(manager.upgrade_command)
            );

            let status = Command::new(&npm_path)
                .args(NPM_UPDATE_ARGS)
                .status()
                .with_context(|| format!("Failed to launch npm at {}", npm_path.display()))?;

            return Ok(if status.success() {
                ExitCode::Success
            } else {
                ExitCode::UserError
            });
        }
    }
}

fn parse_digest_sha256(digest: &str) -> Option<String> {
    let value = digest.trim();
    let raw = value.strip_prefix("sha256:").unwrap_or(value);
    normalize_sha256(raw)
}

fn parse_github_error_message(text: &str) -> Option<String> {
    #[derive(serde::Deserialize)]
    struct GitHubErrorBody {
        message: Option<String>,
    }

    serde_json::from_str::<GitHubErrorBody>(text)
        .ok()
        .and_then(|body| body.message)
        .filter(|message| !message.trim().is_empty())
}

fn parse_release_tag_from_url(url: &reqwest::Url) -> Option<String> {
    let path = url.path();
    path.strip_prefix("/yuanchuan/aivo/releases/tag/")
        .filter(|tag| !tag.is_empty())
        .map(ToString::to_string)
}

fn github_token_from_env() -> Option<String> {
    ["AIVO_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"]
        .iter()
        .find_map(|key| env::var(key).ok())
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
}

fn supported_binary_asset_names() -> &'static [&'static str] {
    &[
        "aivo-darwin-arm64",
        "aivo-darwin-x64",
        "aivo-linux-arm64",
        "aivo-linux-x64",
        "aivo-windows-x64.exe",
    ]
}

fn normalize_sha256(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.len() != 64 || !trimmed.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    Some(trimmed.to_ascii_lowercase())
}

fn parse_checksum_text(text: &str, binary_name: &str) -> Option<String> {
    let mut fallback_hash: Option<String> = None;

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if let Some((left, right)) = line.split_once(" = ")
            && left.starts_with("SHA256 (")
            && left.ends_with(')')
            && (left.contains(binary_name) || binary_name.is_empty())
            && let Some(hash) = normalize_sha256(right)
        {
            return Some(hash);
        }

        let mut parts = line.split_whitespace();
        if let Some(first) = parts.next()
            && let Some(hash) = normalize_sha256(first)
        {
            let remainder = line[first.len()..].trim_start();
            let cleaned_remainder = remainder.trim_start_matches('*').trim_start();
            if cleaned_remainder.is_empty() {
                fallback_hash = Some(hash);
            } else if cleaned_remainder.ends_with(binary_name) || cleaned_remainder == binary_name {
                return Some(hash);
            }
        }
    }

    fallback_hash
}

/// Gets the expected binary asset name for the current platform/arch
fn get_binary_name() -> Result<String> {
    let platform = env::consts::OS;
    let arch = env::consts::ARCH;

    let name = match (platform, arch) {
        ("macos", "aarch64") => "aivo-darwin-arm64",
        ("macos", "x86_64") => "aivo-darwin-x64",
        ("linux", "aarch64") => "aivo-linux-arm64",
        ("linux", "x86_64") => "aivo-linux-x64",
        ("windows", "x86_64") => "aivo-windows-x64.exe",
        _ => {
            return Err(anyhow::anyhow!(
                "Unsupported platform: {}-{}",
                platform,
                arch
            ));
        }
    };

    Ok(name.to_string())
}

fn get_install_path() -> Result<PathBuf> {
    if let Ok(path) = env::var("AIVO_PATH") {
        return Ok(PathBuf::from(path));
    }
    let current_exe = env::current_exe()?;
    Ok(current_exe)
}

fn resolve_command_path(program: &str) -> Option<PathBuf> {
    let dirs = collect_path_dirs();
    find_in_dirs(program, &dirs)
}

fn normalize_install_path(path: &Path) -> String {
    path.to_string_lossy()
        .replace('\\', "/")
        .trim_start_matches("//?/")
        .to_ascii_lowercase()
}

/// Detected package manager type
enum PackageManager {
    Homebrew,
    Cargo,
    Npm,
}

/// Information about a detected package manager
struct ManagedInstall {
    kind: PackageManager,
    name: &'static str,
    upgrade_command: &'static str,
}

/// Detects whether the binary is managed by a package manager based on its path.
/// Returns None if the binary appears to be a direct download or AIVO_PATH is set.
fn detect_managed_install(install_path: &Path) -> Option<ManagedInstall> {
    // If AIVO_PATH is explicitly set, user opted into this path — skip detection
    if env::var("AIVO_PATH").is_ok() {
        return None;
    }

    let path_str = normalize_install_path(install_path);

    // npm: .../node_modules/@yuanchuan/aivo/...
    if path_str.contains("/node_modules/") {
        return Some(ManagedInstall {
            kind: PackageManager::Npm,
            name: "npm",
            upgrade_command: NPM_UPDATE_COMMAND,
        });
    }

    // Homebrew: /opt/homebrew/Cellar/..., /usr/local/Cellar/..., /home/linuxbrew/.linuxbrew/Cellar/...
    if path_str.contains("/cellar/") || path_str.contains("/homebrew/") {
        return Some(ManagedInstall {
            kind: PackageManager::Homebrew,
            name: "Homebrew",
            upgrade_command: "brew upgrade aivo",
        });
    }

    // Cargo: ~/.cargo/bin/aivo
    if path_str.contains("/.cargo/bin/") {
        return Some(ManagedInstall {
            kind: PackageManager::Cargo,
            name: "Cargo",
            upgrade_command: "cargo install aivo",
        });
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_newer_version() {
        let cmd = UpdateCommand::new().unwrap();

        assert!(cmd.is_newer_version("1.1.0", "1.0.0"));
        assert!(cmd.is_newer_version("2.0.0", "1.0.0"));
        assert!(cmd.is_newer_version("1.0.1", "1.0.0"));
        assert!(!cmd.is_newer_version("1.0.0", "1.0.0"));
        assert!(!cmd.is_newer_version("0.9.0", "1.0.0"));
        assert!(!cmd.is_newer_version("1.0.0", "1.0.1"));
    }

    #[test]
    fn test_parse_version() {
        let cmd = UpdateCommand::new().unwrap();

        assert!(cmd.is_newer_version("v1.1.0", "v1.0.0"));
        assert!(cmd.is_newer_version("1.1.0", "v1.0.0"));
    }

    #[test]
    fn test_prerelease_version() {
        let cmd = UpdateCommand::new().unwrap();

        // Release is newer than same-version pre-release
        assert!(cmd.is_newer_version("2.0.0", "2.0.0-rc1"));
        assert!(cmd.is_newer_version("2.0.0", "2.0.0-beta.1"));

        // Pre-release is not newer than its release
        assert!(!cmd.is_newer_version("2.0.0-rc1", "2.0.0"));

        // Same pre-release versions are not newer
        assert!(!cmd.is_newer_version("2.0.0-rc1", "2.0.0-rc1"));

        // Higher version is still newer regardless of pre-release
        assert!(cmd.is_newer_version("2.1.0-rc1", "2.0.0"));
        assert!(cmd.is_newer_version("2.1.0", "2.0.0-rc1"));
    }

    #[test]
    fn test_parse_digest_sha256() {
        let digest = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        assert_eq!(
            parse_digest_sha256(digest),
            Some("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string())
        );
        assert_eq!(parse_digest_sha256("invalid"), None);
    }

    #[test]
    fn test_parse_checksum_text_variants() {
        let artifact = "aivo-darwin-arm64";
        let plain = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n";
        assert_eq!(
            parse_checksum_text(plain, artifact),
            Some("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string())
        );

        let with_name =
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef  aivo-darwin-arm64";
        assert_eq!(
            parse_checksum_text(with_name, artifact),
            Some("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string())
        );

        let bsd = "SHA256 (aivo-darwin-arm64) = 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        assert_eq!(
            parse_checksum_text(bsd, artifact),
            Some("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string())
        );
    }

    #[test]
    fn test_parse_github_error_message() {
        let json = r#"{"message":"API rate limit exceeded for 1.2.3.4"}"#;
        assert_eq!(
            parse_github_error_message(json),
            Some("API rate limit exceeded for 1.2.3.4".to_string())
        );
        assert_eq!(parse_github_error_message("{}"), None);
        assert_eq!(parse_github_error_message("not-json"), None);
    }

    #[test]
    fn test_parse_release_tag_from_url() {
        let release_url =
            reqwest::Url::parse("https://github.com/yuanchuan/aivo/releases/tag/v0.5.0").unwrap();
        assert_eq!(
            parse_release_tag_from_url(&release_url),
            Some("v0.5.0".to_string())
        );

        let latest_url =
            reqwest::Url::parse("https://github.com/yuanchuan/aivo/releases/latest").unwrap();
        assert_eq!(parse_release_tag_from_url(&latest_url), None);
    }

    #[test]
    fn test_supported_binary_asset_names() {
        let assets = supported_binary_asset_names();
        assert!(assets.contains(&"aivo-darwin-arm64"));
        assert!(assets.contains(&"aivo-darwin-x64"));
        assert!(assets.contains(&"aivo-linux-arm64"));
        assert!(assets.contains(&"aivo-linux-x64"));
        assert!(assets.contains(&"aivo-windows-x64.exe"));
    }

    #[test]
    fn test_detect_npm_global() {
        let path = Path::new("/opt/homebrew/lib/node_modules/@yuanchuan/aivo/native/aivo");
        let result = detect_managed_install(path);
        assert!(result.is_some());
        let m = result.unwrap();
        assert_eq!(m.name, "npm");
        assert_eq!(m.upgrade_command, NPM_UPDATE_COMMAND);
    }

    #[test]
    fn test_detect_npm_nvm() {
        let path = Path::new(
            "/Users/user/.nvm/versions/node/v22.0.0/lib/node_modules/@yuanchuan/aivo/native/aivo",
        );
        let result = detect_managed_install(path);
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "npm");
    }

    #[test]
    fn test_detect_npm_windows_path() {
        let path = Path::new(
            r"C:\Users\user\AppData\Roaming\npm\node_modules\@yuanchuan\aivo\native\aivo.exe",
        );
        let result = detect_managed_install(path);
        assert!(result.is_some());
        let m = result.unwrap();
        assert_eq!(m.name, "npm");
        assert_eq!(m.upgrade_command, NPM_UPDATE_COMMAND);
    }

    #[test]
    fn test_detect_homebrew_cellar_arm() {
        let path = Path::new("/opt/homebrew/Cellar/aivo/0.4.3/bin/aivo");
        let result = detect_managed_install(path);
        assert!(result.is_some());
        let m = result.unwrap();
        assert_eq!(m.name, "Homebrew");
        assert_eq!(m.upgrade_command, "brew upgrade aivo");
    }

    #[test]
    fn test_detect_homebrew_cellar_intel() {
        let path = Path::new("/usr/local/Cellar/aivo/0.4.3/bin/aivo");
        let result = detect_managed_install(path);
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "Homebrew");
    }

    #[test]
    fn test_detect_homebrew_linuxbrew() {
        let path = Path::new("/home/linuxbrew/.linuxbrew/Cellar/aivo/0.4.3/bin/aivo");
        let result = detect_managed_install(path);
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "Homebrew");
    }

    #[test]
    fn test_detect_homebrew_opt_path() {
        let path = Path::new("/opt/homebrew/bin/aivo");
        let result = detect_managed_install(path);
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "Homebrew");
    }

    #[test]
    fn test_detect_cargo_install() {
        let path = Path::new("/Users/user/.cargo/bin/aivo");
        let result = detect_managed_install(path);
        assert!(result.is_some());
        let m = result.unwrap();
        assert_eq!(m.name, "Cargo");
        assert_eq!(m.upgrade_command, "cargo install aivo");
    }

    #[test]
    fn test_detect_cargo_windows_path() {
        let path = Path::new(r"C:\Users\user\.cargo\bin\aivo.exe");
        let result = detect_managed_install(path);
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "Cargo");
    }

    #[test]
    fn test_detect_direct_download() {
        let path = Path::new("/usr/local/bin/aivo");
        let result = detect_managed_install(path);
        assert!(result.is_none());
    }

    #[test]
    fn test_detect_custom_path() {
        let path = Path::new("/home/user/bin/aivo");
        let result = detect_managed_install(path);
        assert!(result.is_none());
    }

    #[test]
    fn test_normalize_install_path_strips_verbatim_prefix() {
        let path = Path::new(
            r"\\?\C:\Users\User\AppData\Roaming\npm\node_modules\@yuanchuan\aivo\aivo.exe",
        );
        assert_eq!(
            normalize_install_path(path),
            "c:/users/user/appdata/roaming/npm/node_modules/@yuanchuan/aivo/aivo.exe"
        );
    }
}
