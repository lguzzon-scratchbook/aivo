//! Inline image rendering for modern terminal emulators.
//!
//! Two protocols are wired up. The Kitty graphics protocol (`\x1b_G…`)
//! covers Kitty and Warp but only accepts PNG bytes. The iTerm2 inline
//! image protocol (`\x1b]1337;File=…`) covers iTerm2, WezTerm, and
//! Ghostty and accepts any image format the terminal can decode (PNG /
//! JPEG / WebP / GIF). Several terminals speak both — we prefer iTerm2's
//! protocol when it's available because providers (xAI, Google) often
//! return non-PNG bytes that Kitty graphics would silently reject.
//!
//! ### tmux
//!
//! Inside tmux, `TERM_PROGRAM` reports `tmux`, masking the host emulator.
//! We detect the host through env vars tmux preserves (`WEZTERM_PANE`,
//! `KITTY_WINDOW_ID`, `ITERM_SESSION_ID`, `GHOSTTY_RESOURCES_DIR`) and
//! wrap escape sequences in tmux's DCS passthrough envelope. The user
//! still needs tmux 3.3+ with `set -g allow-passthrough on` for the
//! wrapped sequences to reach the host terminal — without that, preview
//! is silently dropped by tmux.
//!
//! ### Knobs
//!
//! - Stderr is the output stream so `aivo image` piping stays clean.
//! - Failures are silent — a broken preview must never fail the parent.
//! - `AIVO_PREVIEW=0` force-disables; `AIVO_PREVIEW=1` force-enables in
//!   a terminal auto-detect doesn't recognize.

use std::env;
use std::io::{self, IsTerminal, Write};
use std::process::Command;

use base64::Engine;

use crate::style;

const KITTY_CHUNK_SIZE: usize = 4096;

#[derive(Debug, PartialEq, Eq)]
enum Protocol {
    /// Kitty graphics protocol. PNG-only payload.
    Kitty,
    /// iTerm2 inline image protocol. Accepts any format the terminal
    /// itself can decode (PNG / JPEG / WebP / GIF / …).
    Iterm2,
}

/// Public entry point. Renders `bytes` inline if stdout is a TTY and the
/// active terminal speaks one of the supported protocols. Silent no-op
/// otherwise.
pub fn display_image(bytes: &[u8]) {
    if !io::stdout().is_terminal() {
        return;
    }
    let Some(proto) = detect_protocol() else {
        return;
    };

    // Build the protocol payload into a buffer first so the tmux
    // passthrough wrapper can rewrite escape bytes before flushing.
    let mut payload = Vec::with_capacity(bytes.len() * 4 / 3 + 256);
    let built = match proto {
        Protocol::Iterm2 => emit_iterm2(bytes, &mut payload),
        Protocol::Kitty if is_png(bytes) => emit_kitty_png(bytes, &mut payload),
        // Kitty graphics is PNG-only. Skip non-PNG silently rather than
        // emit invalid bytes. Future work: add an image-decode dep and
        // re-encode JPEG/WebP into PNG for Kitty/Warp users.
        Protocol::Kitty => return,
    };
    if built.is_err() {
        return;
    }

    let in_tmux = is_tmux();
    let mut stderr = io::stderr().lock();

    // tmux drops DCS passthrough sequences unless `allow-passthrough` is
    // on. Detect explicit-off and surface a tip instead of emitting
    // bytes that would visibly garble the next prompt. `None` means we
    // couldn't probe (no `tmux` binary on PATH despite TMUX env, etc.)
    // — fall through to best-effort emission.
    if in_tmux && tmux_passthrough_enabled() == Some(false) {
        print_tmux_tip(&mut stderr);
        return;
    }

    // Blank line above the image so it doesn't butt up against the
    // "saved …" status line. Written only when we're actually about to
    // emit a payload — non-rendering paths (unsupported terminal,
    // non-PNG on Kitty, tmux-without-passthrough) returned earlier.
    let _ = writeln!(stderr);
    let _ = if in_tmux {
        write_tmux_wrapped(&payload, &mut stderr)
    } else {
        stderr.write_all(&payload)
    };
    let _ = writeln!(stderr);
}

/// Probe `tmux show-options -g allow-passthrough` for the global
/// passthrough setting. Returns:
/// - `Some(true)` when set to `on` or `all`,
/// - `Some(false)` when set to `off` (or unset, since `off` is tmux's
///   default for this option in 3.3+),
/// - `None` when we couldn't run `tmux` or it failed — caller falls
///   back to best-effort emission.
fn tmux_passthrough_enabled() -> Option<bool> {
    let out = Command::new("tmux")
        .args(["show-options", "-g", "allow-passthrough"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_passthrough(&String::from_utf8_lossy(&out.stdout))
}

/// Parse the value token from `tmux show-options` output. Format is
/// either `allow-passthrough on|off|all` (option set) or empty (option
/// at its default, which is `off` in tmux 3.3+).
fn parse_passthrough(output: &str) -> Option<bool> {
    let last = output.split_whitespace().last()?.to_ascii_lowercase();
    Some(matches!(last.as_str(), "on" | "all"))
}

/// One-time tip shown when tmux is clearly going to swallow the
/// preview. Emitted to stderr like the image itself, so it doesn't
/// pollute stdout pipes.
fn print_tmux_tip<W: Write>(w: &mut W) {
    let _ = writeln!(w);
    let _ = writeln!(
        w,
        "  {} inline preview needs tmux passthrough — run:",
        style::dim("tip:")
    );
    let _ = writeln!(
        w,
        "    {}",
        style::cyan("tmux set-option -g allow-passthrough on")
    );
}

/// Pick the best protocol for the active terminal. Returns `None` when
/// preview is disabled or the terminal is unrecognized. Order matters
/// in two ways:
///
/// 1. Host-via-tmux env vars (`WEZTERM_PANE`, etc.) are checked **first**
///    because tmux overwrites `TERM_PROGRAM=tmux`, hiding the real host.
///    These vars survive tmux because they're plain env vars, not
///    `TERM_PROGRAM`.
/// 2. Among terminals that support both protocols (WezTerm, Ghostty),
///    iTerm2 wins because it accepts non-PNG bytes without re-encoding.
fn detect_protocol() -> Option<Protocol> {
    if env::var("AIVO_PREVIEW").as_deref() == Ok("0") {
        return None;
    }
    let force_on = env::var("AIVO_PREVIEW").as_deref() == Ok("1");

    // Host detection that survives tmux. These env vars are set by the
    // host terminal at shell-start and propagate into tmux sessions.
    if env::var("WEZTERM_PANE").is_ok() || env::var("WEZTERM_EXECUTABLE").is_ok() {
        return Some(Protocol::Iterm2);
    }
    if env::var("GHOSTTY_RESOURCES_DIR").is_ok() {
        return Some(Protocol::Iterm2);
    }
    if env::var("ITERM_SESSION_ID").is_ok() {
        return Some(Protocol::Iterm2);
    }
    if env::var("KITTY_WINDOW_ID").is_ok() {
        return Some(Protocol::Kitty);
    }

    // Standard outside-tmux detection. Inside tmux these are usually
    // overwritten (`TERM_PROGRAM=tmux`, `TERM=tmux-256color`) so the
    // host-specific env vars above do the work.
    let term = env::var("TERM").unwrap_or_default();
    let term_program = env::var("TERM_PROGRAM")
        .unwrap_or_default()
        .to_ascii_lowercase();
    let lc_terminal = env::var("LC_TERMINAL")
        .unwrap_or_default()
        .to_ascii_lowercase();

    if lc_terminal == "iterm2" {
        return Some(Protocol::Iterm2);
    }
    if matches!(term_program.as_str(), "wezterm" | "ghostty") {
        return Some(Protocol::Iterm2);
    }
    if term == "xterm-kitty" {
        return Some(Protocol::Kitty);
    }
    if matches!(term_program.as_str(), "kitty" | "warpterminal") {
        return Some(Protocol::Kitty);
    }

    if force_on {
        return Some(Protocol::Iterm2);
    }
    None
}

fn is_png(bytes: &[u8]) -> bool {
    bytes.starts_with(b"\x89PNG\r\n\x1a\n")
}

fn is_tmux() -> bool {
    env::var("TMUX").is_ok()
        || env::var("TERM_PROGRAM").as_deref() == Ok("tmux")
        || env::var("TERM")
            .map(|t| t.starts_with("tmux") || t.starts_with("screen"))
            .unwrap_or(false)
}

/// Wrap a graphics payload in tmux's DCS passthrough envelope. Each
/// `ESC` byte inside the payload must be doubled so tmux's DCS parser
/// doesn't terminate early; the surrounding `\x1bPtmux;…\x1b\\` opens
/// and closes the DCS.
///
/// Requires tmux 3.3+ with `set -g allow-passthrough on`. Without that
/// option the wrapped sequence is silently dropped — there's no way to
/// detect that from inside tmux, so we trust the user has configured
/// it (or accept that preview won't render).
fn write_tmux_wrapped<W: Write>(payload: &[u8], w: &mut W) -> io::Result<()> {
    w.write_all(b"\x1bPtmux;")?;
    // Stream rather than allocate. Writing one byte at a time would be
    // wasteful; instead, find ESC runs and split the payload into slices
    // we can write whole.
    let mut i = 0;
    while i < payload.len() {
        let next_esc = payload[i..]
            .iter()
            .position(|&b| b == 0x1b)
            .map(|p| i + p)
            .unwrap_or(payload.len());
        if next_esc > i {
            w.write_all(&payload[i..next_esc])?;
        }
        if next_esc < payload.len() {
            w.write_all(b"\x1b\x1b")?;
            i = next_esc + 1;
        } else {
            break;
        }
    }
    w.write_all(b"\x1b\\")?;
    Ok(())
}

/// Kitty graphics protocol emission. Format flag `f=100` is PNG; `a=T`
/// transmits-and-displays in one step. Only the first chunk carries the
/// format header — subsequent chunks need just the `m=` continuation
/// flag. 4096 bytes is the chunk size guaranteed by the protocol.
fn emit_kitty_png<W: Write>(png: &[u8], w: &mut W) -> io::Result<()> {
    let encoded = base64::engine::general_purpose::STANDARD.encode(png);
    let bytes = encoded.as_bytes();
    let total = bytes.len();
    let mut i = 0;
    while i < total {
        let end = (i + KITTY_CHUNK_SIZE).min(total);
        let is_last = end == total;
        let chunk = std::str::from_utf8(&bytes[i..end]).expect("base64 is ASCII");
        if i == 0 {
            write!(
                w,
                "\x1b_Ga=T,f=100,m={};{}\x1b\\",
                if is_last { 0 } else { 1 },
                chunk
            )?;
        } else {
            write!(w, "\x1b_Gm={};{}\x1b\\", if is_last { 0 } else { 1 }, chunk)?;
        }
        i = end;
    }
    Ok(())
}

/// iTerm2 inline image protocol emission. Single envelope (no chunking).
/// `inline=1` requests inline rendering; `height=20` caps the preview at
/// 20 rows so a 1024×1024 image doesn't swallow the screen;
/// `preserveAspectRatio=1` keeps width sensible. `name=` is base64 of
/// `"image"` to satisfy parsers that require the field.
fn emit_iterm2<W: Write>(bytes: &[u8], w: &mut W) -> io::Result<()> {
    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    let name_b64 = base64::engine::general_purpose::STANDARD.encode("image");
    write!(
        w,
        "\x1b]1337;File=name={};inline=1;size={};height=20;preserveAspectRatio=1:{}\x07",
        name_b64,
        bytes.len(),
        b64
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Saves and restores a set of env vars so detection tests don't
    /// leak state into one another. Tests grab `ENV_LOCK` so concurrent
    /// threads don't fight over process-global env state.
    struct EnvGuard {
        keys: Vec<(&'static str, Option<String>)>,
    }

    impl EnvGuard {
        fn new(keys: &[&'static str]) -> Self {
            let saved = keys
                .iter()
                .map(|k| (*k, env::var(k).ok()))
                .collect::<Vec<_>>();
            for k in keys {
                unsafe { env::remove_var(k) };
            }
            Self { keys: saved }
        }

        fn set(&self, k: &str, v: &str) {
            unsafe { env::set_var(k, v) };
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (k, v) in &self.keys {
                match v {
                    Some(val) => unsafe { env::set_var(k, val) },
                    None => unsafe { env::remove_var(k) },
                }
            }
        }
    }

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    const ENV_KEYS: &[&str] = &[
        "AIVO_PREVIEW",
        "TERM",
        "TERM_PROGRAM",
        "LC_TERMINAL",
        "TMUX",
        "WEZTERM_PANE",
        "WEZTERM_EXECUTABLE",
        "GHOSTTY_RESOURCES_DIR",
        "ITERM_SESSION_ID",
        "KITTY_WINDOW_ID",
    ];

    #[test]
    fn override_off_beats_everything() {
        let _g = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(ENV_KEYS);
        guard.set("AIVO_PREVIEW", "0");
        guard.set("TERM", "xterm-kitty");
        guard.set("TERM_PROGRAM", "ghostty");
        guard.set("WEZTERM_PANE", "0");
        assert_eq!(detect_protocol(), None);
    }

    #[test]
    fn override_on_falls_back_to_iterm2() {
        let _g = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(ENV_KEYS);
        guard.set("AIVO_PREVIEW", "1");
        assert_eq!(detect_protocol(), Some(Protocol::Iterm2));
    }

    #[test]
    fn wezterm_inside_tmux_picks_iterm2() {
        // Reproduces the user's bug: shell is in tmux, host is WezTerm.
        // TERM_PROGRAM=tmux masks the host; WEZTERM_PANE survives.
        let _g = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(ENV_KEYS);
        guard.set("TERM_PROGRAM", "tmux");
        guard.set("TERM", "tmux-256color");
        guard.set("TMUX", "/tmp/tmux-501/default,1234,0");
        guard.set("WEZTERM_PANE", "0");
        assert_eq!(detect_protocol(), Some(Protocol::Iterm2));
    }

    #[test]
    fn kitty_inside_tmux_picks_kitty() {
        let _g = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(ENV_KEYS);
        guard.set("TERM_PROGRAM", "tmux");
        guard.set("TERM", "tmux-256color");
        guard.set("TMUX", "/tmp/tmux-501/default,1234,0");
        guard.set("KITTY_WINDOW_ID", "1");
        assert_eq!(detect_protocol(), Some(Protocol::Kitty));
    }

    #[test]
    fn iterm2_inside_tmux_picks_iterm2() {
        let _g = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(ENV_KEYS);
        guard.set("TERM_PROGRAM", "tmux");
        guard.set("TMUX", "/tmp/tmux-501/default,1234,0");
        guard.set("ITERM_SESSION_ID", "abc");
        assert_eq!(detect_protocol(), Some(Protocol::Iterm2));
    }

    #[test]
    fn ghostty_inside_tmux_picks_iterm2() {
        let _g = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(ENV_KEYS);
        guard.set("TERM_PROGRAM", "tmux");
        guard.set("TMUX", "/tmp/tmux-501/default,1234,0");
        guard.set("GHOSTTY_RESOURCES_DIR", "/Applications/Ghostty.app/...");
        assert_eq!(detect_protocol(), Some(Protocol::Iterm2));
    }

    #[test]
    fn wezterm_outside_tmux_picks_iterm2() {
        let _g = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(ENV_KEYS);
        guard.set("TERM_PROGRAM", "WezTerm");
        assert_eq!(detect_protocol(), Some(Protocol::Iterm2));
    }

    #[test]
    fn kitty_outside_tmux_picks_kitty() {
        let _g = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(ENV_KEYS);
        guard.set("TERM", "xterm-kitty");
        assert_eq!(detect_protocol(), Some(Protocol::Kitty));
    }

    #[test]
    fn warp_picks_kitty() {
        let _g = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(ENV_KEYS);
        guard.set("TERM_PROGRAM", "WarpTerminal");
        assert_eq!(detect_protocol(), Some(Protocol::Kitty));
    }

    #[test]
    fn rejects_plain_xterm() {
        let _g = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(ENV_KEYS);
        guard.set("TERM", "xterm-256color");
        guard.set("TERM_PROGRAM", "Apple_Terminal");
        assert_eq!(detect_protocol(), None);
    }

    #[test]
    fn is_png_recognizes_signature() {
        assert!(is_png(b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR"));
        assert!(!is_png(b"\xff\xd8\xff\xe0\0\x10JFIF"));
        assert!(!is_png(b"RIFF\x00\x00\x00\x00WEBP"));
        assert!(!is_png(&[]));
        assert!(!is_png(b"\x89PN"));
    }

    #[test]
    fn is_tmux_detects_canonical_signals() {
        let _g = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["TMUX", "TERM_PROGRAM", "TERM"]);
        guard.set("TMUX", "/tmp/tmux-501/default,1,0");
        assert!(is_tmux());
    }

    #[test]
    fn parse_passthrough_handles_known_values() {
        assert_eq!(parse_passthrough("allow-passthrough on\n"), Some(true));
        assert_eq!(parse_passthrough("allow-passthrough off\n"), Some(false));
        assert_eq!(parse_passthrough("allow-passthrough all\n"), Some(true));
        assert_eq!(parse_passthrough(""), None);
        // Trailing whitespace shouldn't fool the parser.
        assert_eq!(parse_passthrough("  allow-passthrough  off  "), Some(false));
    }

    #[test]
    fn emit_kitty_png_single_chunk_has_format_and_terminator() {
        let png = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR";
        let mut buf = Vec::new();
        emit_kitty_png(png, &mut buf).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert!(out.starts_with("\x1b_Ga=T,f=100,m=0;"));
        assert!(out.ends_with("\x1b\\"));
        assert_eq!(out.matches("\x1b_G").count(), 1);
    }

    #[test]
    fn emit_kitty_png_chunks_at_4096() {
        let png = vec![0xABu8; 7000];
        let mut buf = Vec::new();
        emit_kitty_png(&png, &mut buf).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert_eq!(out.matches("\x1b_G").count(), 3);
        assert!(out.starts_with("\x1b_Ga=T,f=100,m=1;"));
        assert!(out.contains("\x1b_Gm=1;"));
        assert!(out.contains("\x1b_Gm=0;"));
    }

    #[test]
    fn emit_iterm2_envelopes_with_size_and_terminator() {
        let bytes = b"\xff\xd8\xff\xe0\0\x10JFIF\x00\x01";
        let mut buf = Vec::new();
        emit_iterm2(bytes, &mut buf).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert!(out.starts_with("\x1b]1337;File=name="));
        assert!(out.contains("inline=1"));
        assert!(out.contains("size=12"));
        assert!(out.contains("preserveAspectRatio=1"));
        assert!(out.ends_with("\x07"));
        assert!(out.contains("name=aW1hZ2U="));
    }

    #[test]
    fn tmux_passthrough_doubles_escape_bytes() {
        // Iterm2 payload: \x1b]1337;...:base64\x07
        let payload = b"\x1b]1337;File=inline=1:Zm9v\x07";
        let mut buf = Vec::new();
        write_tmux_wrapped(payload, &mut buf).unwrap();
        let out = buf;
        // DCS opener.
        assert!(out.starts_with(b"\x1bPtmux;"));
        // The single ESC inside the payload must be doubled.
        assert!(
            out.windows(2).filter(|w| *w == b"\x1b\x1b").count() >= 1,
            "expected at least one doubled ESC inside the wrapper"
        );
        // DCS closer.
        assert!(out.ends_with(b"\x1b\\"));
    }

    #[test]
    fn tmux_passthrough_preserves_kitty_payload() {
        // Kitty payload ends in \x1b\\ — verify both ESCs are doubled.
        let payload = b"\x1b_Ga=T,f=100,m=0;Zm9v\x1b\\";
        let mut buf = Vec::new();
        write_tmux_wrapped(payload, &mut buf).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert!(out.starts_with("\x1bPtmux;"));
        assert!(out.ends_with("\x1b\\"));
        // Two ESCs in the payload → both doubled. After the opener
        // consumes its own ESC, the body should contain the doubled
        // pairs and nothing else with bare ESC except the closer.
        assert!(out.contains("\x1b\x1b_G"));
        assert!(out.contains("\x1b\x1b\\"));
    }
}
