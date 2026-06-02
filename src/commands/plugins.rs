//! `aivo plugins` — manage sibling-binary plugins (install/update/list/remove).
//! Plugins are `aivo-<name>` executables in `~/.config/aivo/plugins/`; dispatch
//! lives in `crate::plugin`.

use std::collections::BTreeMap;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::cli::{
    PluginInstallArgs, PluginRemoveArgs, PluginUpdateArgs, PluginsArgs, PluginsSubcommand,
};
use crate::errors::ExitCode;
use crate::plugin::{
    PLUGIN_PREFIX, discover, infer_plugin_name, installed_plugins, is_reserved_plugin_name,
    plugins_dir,
};
use crate::style;

const INSTALL_HINT: &str = "Install one with `aivo plugins install <path|url>`.";

#[derive(Default)]
pub struct PluginsCommand;

impl PluginsCommand {
    pub fn new() -> Self {
        Self
    }

    pub async fn execute(&self, args: PluginsArgs) -> ExitCode {
        let cmd = args.command.unwrap_or(PluginsSubcommand::List);
        let result = match cmd {
            PluginsSubcommand::List => list_action(),
            PluginsSubcommand::Install(a) => install_action(a).await,
            PluginsSubcommand::Update(a) => update_action(a).await,
            PluginsSubcommand::Remove(a) => remove_action(a),
        };
        match result {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{} {:#}", style::red("Error:"), e);
                ExitCode::UserError
            }
        }
    }

    pub fn print_help() {
        println!("{} aivo plugins [SUBCOMMAND]", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim(
                "Manage plugins — sibling `aivo-<name>` binaries under ~/.config/aivo/plugins.\n\
                 Once installed, `aivo <name> …` (or `aivo run <name> …`) runs the plugin."
            )
        );
        println!();
        println!("{}", style::bold("Subcommands:"));
        let row = |a: &str, b: &str| {
            println!("  {}  {}", style::cyan(format!("{:<26}", a)), style::dim(b));
        };
        row(
            "list",
            "Show installed plugins and where each resolves (default)",
        );
        row(
            "install <path|url> [--name N]",
            "Install from a local file or http(s) URL (--force to overwrite)",
        );
        row(
            "update [name]",
            "Re-install from the recorded source (all plugins if no name)",
        );
        row("remove <name> [-y]", "Remove an installed plugin");
        println!();
        println!("{}", style::bold("Examples:"));
        for ex in [
            "aivo plugins",
            "aivo plugins install ./target/release/aivo-amp",
            "aivo plugins install https://example.com/dl/aivo-amp --name amp",
            "aivo plugins update amp",
            "aivo plugins remove amp",
            "aivo amp --help        # run an installed plugin",
        ] {
            println!("  {}", style::dim(ex));
        }
    }
}

fn list_action() -> Result<ExitCode> {
    let plugins = installed_plugins();
    if plugins.is_empty() {
        eprintln!("  {} No plugins installed.", style::dim("·"));
        eprintln!("  {} {}", style::dim("·"), style::dim(INSTALL_HINT));
        if let Some(dir) = plugins_dir() {
            eprintln!(
                "  {} {}",
                style::dim("·"),
                style::dim(format!("Plugins live in {}", dir.display()))
            );
        }
        return Ok(ExitCode::Success);
    }

    let managed_dir = plugins_dir();
    let width = plugins
        .iter()
        .map(|(n, _)| n.len())
        .max()
        .unwrap_or(0)
        .min(24);
    for (name, path) in &plugins {
        let is_managed = managed_dir
            .as_deref()
            .is_some_and(|d| path.parent() == Some(d));
        let tag = if is_managed { "" } else { " (external)" };
        println!(
            "  {}  {}{}",
            style::cyan(format!("{name:<width$}")),
            style::dim(path.display().to_string()),
            style::dim(tag),
        );
    }
    Ok(ExitCode::Success)
}

async fn install_action(args: PluginInstallArgs) -> Result<ExitCode> {
    let dir =
        plugins_dir().context("could not resolve the home directory for ~/.config/aivo/plugins")?;

    let name = match args.name {
        Some(n) => n,
        None => infer_plugin_name(&args.source)
            .context("could not infer a plugin name from the source — pass --name <name>")?,
    };
    validate_name(&name)?;
    if is_reserved_plugin_name(&name) {
        anyhow::bail!(
            "`{name}` collides with a built-in command or tool, so it would never run as a plugin. Choose a different --name."
        );
    }

    let target = dir.join(plugin_filename(&name));
    if target.exists() && !args.force {
        anyhow::bail!(
            "plugin `{name}` is already installed at {}. Pass --force to overwrite.",
            target.display()
        );
    }

    // Stable, re-fetchable source (absolute path for local files) for `update`.
    let source = canonical_source(&args.source);
    reinstall(&name, &source, &dir).await?;
    record_source(&name, &source);

    eprintln!(
        "  {} Installed plugin `{}` — run it with {}",
        style::success_symbol(),
        name,
        style::cyan(format!("aivo {name}")),
    );
    eprintln!(
        "  {} {}",
        style::dim("·"),
        style::dim(target.display().to_string())
    );
    Ok(ExitCode::Success)
}

async fn update_action(args: PluginUpdateArgs) -> Result<ExitCode> {
    let dir = plugins_dir().context("could not resolve ~/.config/aivo/plugins")?;
    let sources = load_sources();

    let targets: Vec<String> = match args.name {
        Some(n) => vec![n.strip_prefix(PLUGIN_PREFIX).unwrap_or(&n).to_string()],
        None => sources.keys().cloned().collect(),
    };
    if targets.is_empty() {
        eprintln!(
            "  {} No plugins with a recorded source to update.",
            style::dim("·")
        );
        eprintln!("  {} {}", style::dim("·"), style::dim(INSTALL_HINT));
        return Ok(ExitCode::Success);
    }

    let mut any_failed = false;
    for name in &targets {
        let Some(source) = sources.get(name) else {
            any_failed = true;
            if discover(name).is_some() {
                eprintln!(
                    "  {} `{name}`: no recorded source (installed manually or externally) — reinstall with `aivo plugins install <source>`.",
                    style::yellow("!")
                );
            } else {
                eprintln!("  {} `{name}` is not installed.", style::yellow("!"));
            }
            continue;
        };
        match reinstall(name, source, &dir).await {
            Ok(()) => eprintln!(
                "  {} Updated `{name}` from {}",
                style::success_symbol(),
                style::dim(source)
            ),
            Err(e) => {
                any_failed = true;
                eprintln!("  {} `{name}`: {e:#}", style::red("✗"));
            }
        }
    }

    if any_failed {
        Ok(ExitCode::UserError)
    } else {
        Ok(ExitCode::Success)
    }
}

fn remove_action(args: PluginRemoveArgs) -> Result<ExitCode> {
    let dir = plugins_dir().context("could not resolve ~/.config/aivo/plugins")?;
    let name = args
        .name
        .strip_prefix(PLUGIN_PREFIX)
        .unwrap_or(&args.name)
        .to_string();
    let target = dir.join(plugin_filename(&name));

    if !target.exists() {
        if let Some(found) = discover(&name) {
            anyhow::bail!(
                "`{name}` isn't managed by aivo — it's at {}. Remove it there (e.g. `cargo uninstall`).",
                found.display()
            );
        }
        anyhow::bail!("plugin `{name}` is not installed. See `aivo plugins list`.");
    }

    if !args.yes && !confirm(&format!("Remove plugin `{name}`?"))? {
        return Ok(ExitCode::Success);
    }

    std::fs::remove_file(&target).with_context(|| format!("removing {}", target.display()))?;
    forget_source(&name);
    eprintln!("  {} Removed plugin `{name}`", style::success_symbol());
    Ok(ExitCode::Success)
}

/// Interactive y/N prompt; bails non-interactively (pass `--yes`).
fn confirm(prompt: &str) -> Result<bool> {
    if !std::io::stdin().is_terminal() {
        anyhow::bail!("{prompt} (non-interactive; pass --yes to confirm)");
    }
    eprint!("  {} {prompt} [y/N] ", style::yellow("?"));
    let _ = std::io::stderr().flush();
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    Ok(matches!(
        input.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

/// Reject names that can't dispatch: empty, flag-shaped, or containing a path
/// separator (which would escape the plugins dir / break the `aivo-<name>` map).
fn validate_name(name: &str) -> Result<()> {
    if name.starts_with('-') {
        anyhow::bail!("plugin name `{name}` must not start with `-`");
    }
    if name.contains('/') || name.contains('\\') {
        anyhow::bail!("plugin name `{name}` must not contain a path separator");
    }
    Ok(())
}

/// On-disk filename for a plugin: `aivo-<name>` (`.exe` on Windows).
fn plugin_filename(name: &str) -> String {
    if cfg!(windows) {
        format!("{PLUGIN_PREFIX}{name}.exe")
    } else {
        format!("{PLUGIN_PREFIX}{name}")
    }
}

/// Fetch `source` and write `aivo-<name>` into `dir`, overwriting any existing
/// binary. Shared by install and update.
async fn reinstall(name: &str, source: &str, dir: &Path) -> Result<()> {
    let bytes = fetch_source(source).await?;
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    write_executable(&dir.join(plugin_filename(name)), &bytes)
}

/// A stable, re-fetchable form of the install source: URLs verbatim, local
/// paths made absolute so `update` works regardless of the current directory.
fn canonical_source(source: &str) -> String {
    if is_url(source) {
        source.to_string()
    } else {
        std::fs::canonicalize(source)
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| source.to_string())
    }
}

fn is_url(source: &str) -> bool {
    source.starts_with("http://") || source.starts_with("https://")
}

// ── Source index (.sources.json) ──────────────────────────────────────────
// Where each plugin came from, so `update` can re-fetch. Dotfile → not discovered.

fn sources_path() -> Option<PathBuf> {
    plugins_dir().map(|d| d.join(".sources.json"))
}

fn load_sources() -> BTreeMap<String, String> {
    sources_path()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

/// Load the index, apply `mutate`, save if it changed. Best-effort — a write
/// failure only costs a future `update` (which warns).
fn update_sources(mutate: impl FnOnce(&mut BTreeMap<String, String>) -> bool) {
    let mut map = load_sources();
    if mutate(&mut map) {
        let _ = save_sources(&map);
    }
}

fn record_source(name: &str, source: &str) {
    update_sources(|m| {
        m.insert(name.to_string(), source.to_string());
        true
    });
}

fn forget_source(name: &str) {
    update_sources(|m| m.remove(name).is_some());
}

fn save_sources(map: &BTreeMap<String, String>) -> Result<()> {
    let path = sources_path().context("could not resolve the plugin source index path")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(map).context("serializing plugin sources")?;
    std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))
}

async fn fetch_source(source: &str) -> Result<Vec<u8>> {
    if is_url(source) {
        download(source).await
    } else {
        let path = Path::new(source);
        let meta =
            std::fs::metadata(path).with_context(|| format!("reading local source `{source}`"))?;
        if !meta.is_file() {
            anyhow::bail!("`{source}` is not a file");
        }
        std::fs::read(path).with_context(|| format!("reading `{source}`"))
    }
}

async fn download(url: &str) -> Result<Vec<u8>> {
    eprintln!("  {} Downloading {url}", style::dim("·"));
    // Reuse aivo's shared client so proxy / IPv4-only / Termux-DNS handling apply.
    let client = crate::services::http_utils::aivo_http_client_builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .context("failed to build HTTP client")?;
    let resp = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("requesting {url}"))?
        .error_for_status()
        .with_context(|| format!("downloading {url}"))?;
    let bytes = resp.bytes().await.context("reading download body")?;
    Ok(bytes.to_vec())
}

/// Write via a temp dotfile + rename, so a partial write never leaves a
/// half-written (or discoverable `aivo-*`) binary. Sets the exec bit on Unix.
fn write_executable(target: &Path, bytes: &[u8]) -> Result<()> {
    let file_name = target
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let tmp = target.with_file_name(format!(".{file_name}.tmp"));

    std::fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    set_executable(&tmp)?;
    std::fs::rename(&tmp, target).with_context(|| format!("installing {}", target.display()))?;
    Ok(())
}

#[cfg(unix)]
fn set_executable(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(path)?.permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(path, perms)?;
    Ok(())
}

#[cfg(not(unix))]
fn set_executable(_path: &Path) -> Result<()> {
    Ok(())
}
