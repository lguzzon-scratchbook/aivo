//! Cross-platform audio playback for `aivo speak` / `aivo audio --play`.
//!
//! We hand an audio file to the OS audio stack via a per-platform CLI:
//!
//! - macOS: `afplay` (built-in; handles MP3, WAV, AAC, M4A, …)
//! - Linux: try `paplay` / `pw-play` / `aplay` / `ffplay` / `mpv` / `mpg123`
//!   in order, falling through only when a binary isn't installed
//! - Windows: PowerShell `System.Media.SoundPlayer.PlaySync()` (built-in;
//!   WAV only)
//!
//! For "play-only, no save" mode (no `-o`) the audio command requests WAV
//! from the provider so the path works on every platform. With `-o` set the
//! caller's chosen format is honored and playback becomes best-effort —
//! Windows can fail on MP3, and Linux falls through `paplay`/`aplay` (which
//! reject MP3) to one of the codec-aware players if installed.

use std::path::Path;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
use std::process::Command;

use anyhow::{Result, bail};

/// Linux CLI players in priority order. Each entry is
/// `(binary, args_before_path)`.
#[cfg(target_os = "linux")]
const LINUX_PLAYERS: &[(&str, &[&str])] = &[
    ("paplay", &[]),
    ("pw-play", &[]),
    ("aplay", &["-q"]),
    ("ffplay", &["-autoexit", "-nodisp", "-loglevel", "quiet"]),
    ("mpv", &["--no-video", "--really-quiet"]),
    ("mpg123", &["-q"]),
];

/// Plays an audio file synchronously, returning when playback finishes (or
/// the player exits with a non-success status). The path must already exist.
/// On Linux/Windows the file should be WAV for guaranteed playback; on macOS
/// `afplay` handles most common formats.
pub fn play_audio_blocking(path: &Path) -> Result<()> {
    if !path.exists() {
        bail!("audio file '{}' does not exist", path.display());
    }
    play_impl(path)
}

#[cfg(target_os = "macos")]
fn play_impl(path: &Path) -> Result<()> {
    let status = Command::new("afplay").arg(path).status().map_err(|e| {
        anyhow::anyhow!(
            "failed to invoke `afplay`: {e} \
             (afplay is preinstalled on macOS — is your $PATH unusual?)"
        )
    })?;
    if !status.success() {
        bail!("afplay exited with status {}", status);
    }
    Ok(())
}

#[cfg(target_os = "windows")]
fn play_impl(path: &Path) -> Result<()> {
    // PowerShell single-quoted strings need internal `'` doubled.
    let path_str = path.to_string_lossy().replace('\'', "''");
    let script = format!("(New-Object System.Media.SoundPlayer '{path_str}').PlaySync()");
    let status = Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", &script])
        .status()
        .map_err(|e| anyhow::anyhow!("failed to invoke `powershell`: {e}"))?;
    if !status.success() {
        bail!(
            "powershell SoundPlayer exited with status {} (is the WAV file valid?)",
            status
        );
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn play_impl(path: &Path) -> Result<()> {
    let mut not_found = Vec::new();
    for (binary, args) in LINUX_PLAYERS {
        let mut cmd = Command::new(binary);
        cmd.args(*args).arg(path);
        match cmd.status() {
            Ok(status) if status.success() => return Ok(()),
            Ok(status) => bail!("{binary} exited with status {status}"),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                not_found.push(*binary);
                continue;
            }
            Err(e) => bail!("failed to invoke `{binary}`: {e}"),
        }
    }
    bail!(
        "no audio player found on PATH (tried: {}). \
         Install one (`apt install alsa-utils`, `brew install mpg123`, …) \
         or use `-o <path>` to save the audio to a file instead.",
        not_found.join(", ")
    );
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
fn play_impl(_path: &Path) -> Result<()> {
    bail!(
        "audio playback isn't supported on this platform; use `-o <path>` to save to a file instead"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_missing_file() {
        let err =
            play_audio_blocking(Path::new("/definitely/does/not/exist/aivo-test.wav")).unwrap_err();
        assert!(err.to_string().contains("does not exist"), "got: {err}");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_player_list_is_non_empty_and_distinct() {
        assert!(!LINUX_PLAYERS.is_empty());
        let names: Vec<_> = LINUX_PLAYERS.iter().map(|(n, _)| *n).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            names.len(),
            "duplicate binary in LINUX_PLAYERS"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_player_list_prefers_pulse_over_alsa() {
        // PulseAudio handles long-running streams more gracefully than raw
        // ALSA on most desktop distros; bias the probe toward it.
        let pulse_idx = LINUX_PLAYERS
            .iter()
            .position(|(n, _)| *n == "paplay")
            .expect("paplay should be in LINUX_PLAYERS");
        let alsa_idx = LINUX_PLAYERS
            .iter()
            .position(|(n, _)| *n == "aplay")
            .expect("aplay should be in LINUX_PLAYERS");
        assert!(pulse_idx < alsa_idx);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_path_with_apostrophe_is_doubled_in_script() {
        // Verifies the escape rule we apply before passing `path` into a
        // PowerShell single-quoted string literal: each `'` must become `''`.
        let raw = r"C:\Users\bob's stuff\hi.wav";
        let escaped = raw.replace('\'', "''");
        assert_eq!(escaped, r"C:\Users\bob''s stuff\hi.wav");
    }
}
