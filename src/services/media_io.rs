//! Output handling shared by media-generating commands (`image`, `video`,
//! `audio`).
//!
//! Each modality has its own provider dispatch and MIME → extension table,
//! but they share the user-facing output UX: parsing `-o`, default
//! timestamped filenames, directory targets, template tokens, the overwrite
//! prompt, atomic writes, and JSON-error extraction. Keeping that logic in
//! one place keeps the per-modality modules tiny.

use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use chrono::Utc;
use serde_json::Value;

/// How `-o`/`--output` was specified.
#[derive(Debug, Clone)]
pub enum OutputTarget {
    /// No `-o` given — use default timestamped filename in CWD.
    Default,
    /// `-o path` where the path ends with `/` or is an existing directory.
    Directory(PathBuf),
    /// `-o path` pointing at a specific file.
    File(PathBuf),
    /// `-o "tmpl.png"` — a template with `{ts}`/`{model}` tokens.
    Template(String),
}

impl OutputTarget {
    /// Parse the raw `-o` argument. Returns `Default` when `arg` is `None`.
    pub fn parse(arg: Option<&str>) -> Self {
        let raw = match arg {
            None => return Self::Default,
            Some(r) => r,
        };

        if raw.contains('{') && raw.contains('}') {
            return Self::Template(raw.to_string());
        }

        let path = PathBuf::from(raw);
        let looks_like_dir =
            raw.ends_with('/') || raw.ends_with(std::path::MAIN_SEPARATOR) || path.is_dir();
        if looks_like_dir {
            Self::Directory(path)
        } else {
            Self::File(path)
        }
    }

    /// True when the user's `-o` value pins a file extension. For `Default`
    /// and `Directory` the extension is chosen by the caller (e.g. `png` /
    /// `mp4` / `mp3`), so the caller may swap in the server's actual
    /// content-type-derived extension instead.
    pub fn pins_extension(&self) -> bool {
        match self {
            Self::Default | Self::Directory(_) => false,
            Self::File(p) => p.extension().is_some(),
            Self::Template(s) => Path::new(s).extension().is_some(),
        }
    }
}

/// Resolves the concrete output path before any API call is made. Collision
/// resolution happens after the response, once the extension is known, via
/// [`apply_overwrite_policy`].
pub fn resolve_output_path(target: &OutputTarget, model: &str, ext: &str) -> Result<PathBuf> {
    let ts = Utc::now().format("%Y%m%d-%H%M%S").to_string();

    match target {
        OutputTarget::Default => Ok(PathBuf::from(format!("./aivo-{ts}.{ext}"))),
        OutputTarget::Directory(dir) => {
            verify_writable_dir(dir)?;
            Ok(dir.join(format!("aivo-{ts}.{ext}")))
        }
        OutputTarget::File(path) => {
            if let Some(parent) = parent_if_nonempty(path) {
                verify_writable_dir(parent)?;
            }
            let (stem, real_ext) = split_stem_ext(path, ext);
            let dir = path.parent().unwrap_or(Path::new("."));
            Ok(dir.join(format!("{stem}.{real_ext}")))
        }
        OutputTarget::Template(tmpl) => {
            let expanded = expand_template(tmpl, &ts, model);
            let path = PathBuf::from(&expanded);
            if let Some(parent) = parent_if_nonempty(&path) {
                verify_writable_dir(parent)?;
            }
            Ok(path)
        }
    }
}

/// Returns `path.parent()` only when it's a non-empty path. `Path::parent()`
/// returns `Some("")` for bare filenames like `cat.png`, which isn't useful
/// for directory checks.
fn parent_if_nonempty(path: &Path) -> Option<&Path> {
    path.parent().filter(|p| !p.as_os_str().is_empty())
}

fn expand_template(tmpl: &str, ts: &str, model: &str) -> String {
    tmpl.replace("{ts}", ts)
        .replace("{model}", &sanitize_model(model))
}

/// Replaces filesystem-hostile characters in a model id with underscores.
pub fn sanitize_model(model: &str) -> String {
    model
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            c => c,
        })
        .collect()
}

fn split_stem_ext(path: &Path, default_ext: &str) -> (String, String) {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("aivo")
        .to_string();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| default_ext.to_string());
    (stem, ext)
}

fn verify_writable_dir(dir: &Path) -> Result<()> {
    if !dir.exists() {
        bail!(
            "directory '{}' does not exist (create it first, or omit -o)",
            dir.display()
        );
    }
    if !dir.is_dir() {
        bail!("'{}' is not a directory", dir.display());
    }
    let meta = fs::metadata(dir)
        .with_context(|| format!("cannot access directory '{}'", dir.display()))?;
    if meta.permissions().readonly() {
        bail!("cannot write to '{}': permission denied", dir.display());
    }
    Ok(())
}

/// Decision made by [`apply_overwrite_policy`] for a single file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OverwriteDecision {
    /// Write at this exact path (replacing any existing file, or writing a
    /// fresh file, or a `-1`/`-2`/… auto-suffix when the user chose skip).
    Write(PathBuf),
    /// Abort the whole run (non-TTY / JSON / explicit "no").
    Abort,
}

/// Controls how [`apply_overwrite_policy`] resolves existing-file collisions.
#[derive(Debug, Clone, Copy)]
pub struct OverwritePolicy {
    pub force: bool,
    pub interactive: bool,
}

impl OverwritePolicy {
    pub fn from_flags(force: bool, json_mode: bool) -> Self {
        Self {
            force,
            interactive: !json_mode && io::stdin().is_terminal() && io::stderr().is_terminal(),
        }
    }
}

/// Decides what to do with a single intended target path.
///
/// When `force` is set, always returns `Write`. When the path doesn't exist,
/// also `Write`. Otherwise prompts (if `interactive`) or aborts.
pub fn apply_overwrite_policy(
    path: &Path,
    policy: OverwritePolicy,
    prompt_answer: Option<char>,
) -> OverwriteDecision {
    if !path.exists() {
        return OverwriteDecision::Write(path.to_path_buf());
    }
    if policy.force {
        return OverwriteDecision::Write(path.to_path_buf());
    }
    if !policy.interactive {
        return OverwriteDecision::Abort;
    }
    let answer = prompt_answer.unwrap_or('n');
    match answer {
        'y' | 'Y' | 'a' | 'A' => OverwriteDecision::Write(path.to_path_buf()),
        's' | 'S' => OverwriteDecision::Write(next_free_path(path)),
        _ => OverwriteDecision::Abort,
    }
}

/// Finds the next free path by appending `-1`, `-2`, … before the extension.
/// Preserves the original extension (or lack thereof).
pub fn next_free_path(path: &Path) -> PathBuf {
    let parent = path.parent().unwrap_or(Path::new("."));
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("aivo");
    let ext = path.extension().and_then(|e| e.to_str());
    for i in 1..=9999 {
        let candidate = match ext {
            Some(e) => parent.join(format!("{stem}-{i}.{e}")),
            None => parent.join(format!("{stem}-{i}")),
        };
        if !candidate.exists() {
            return candidate;
        }
    }
    match ext {
        Some(e) => parent.join(format!("{stem}-overflow.{e}")),
        None => parent.join(format!("{stem}-overflow")),
    }
}

/// Prompts the user once about overwriting `path`. Returns the raw char the
/// user entered (lowercased) or `'n'` on empty/EOF.
pub fn prompt_overwrite(path: &Path) -> char {
    eprint!(
        "File '{}' exists. Overwrite? [y/N/s(kip-and-suffix)]: ",
        path.display()
    );
    let _ = io::stderr().flush();
    let mut buf = String::new();
    if io::stdin().read_line(&mut buf).is_err() {
        return 'n';
    }
    buf.trim()
        .chars()
        .next()
        .map(|c| c.to_ascii_lowercase())
        .unwrap_or('n')
}

/// When the user didn't pin an extension and the server reported one, swap
/// the path's extension to match. When the user pinned, or the server didn't
/// say, the path is returned unchanged.
///
/// `server_ext` is the resolved extension string (e.g. `"jpg"`, `"mp4"`,
/// `"wav"`), already mapped from a Content-Type header by per-modality
/// logic. Pass `None` when the server didn't report a content-type.
pub fn align_extension(path: &Path, server_ext: Option<&str>, pinned: bool) -> PathBuf {
    if pinned {
        return path.to_path_buf();
    }
    let Some(server_ext) = server_ext else {
        return path.to_path_buf();
    };
    let server_ext = server_ext.to_ascii_lowercase();
    let current_ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    if current_ext == server_ext {
        return path.to_path_buf();
    }
    path.with_extension(server_ext)
}

/// Writes bytes atomically via a `.part` sibling + rename. Cleans up the
/// partial file on failure so a failed run never leaves a half-written file
/// at the final name.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<u64> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let stem = path.file_name().and_then(|s| s.to_str()).unwrap_or("aivo");
    let suffix: u32 = rand::random();
    let tmp = parent.join(format!(".{stem}.aivo-tmp-{suffix:x}.part"));

    let write_result = (|| -> Result<u64> {
        let mut f = fs::File::create(&tmp)
            .with_context(|| format!("cannot create temp file '{}'", tmp.display()))?;
        f.write_all(bytes)
            .with_context(|| format!("write failed for '{}'", tmp.display()))?;
        f.sync_all().ok();
        drop(f);
        replace_with_temp_file(&tmp, path)?;
        Ok(bytes.len() as u64)
    })();

    if write_result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    write_result
}

fn replace_with_temp_file(tmp: &Path, path: &Path) -> Result<()> {
    #[cfg(windows)]
    if path.exists() {
        fs::remove_file(path)
            .with_context(|| format!("cannot replace existing file '{}'", path.display()))?;
    }

    fs::rename(tmp, path)
        .with_context(|| format!("rename '{}' -> '{}' failed", tmp.display(), path.display()))
}

/// Pulls a human-readable error message out of a JSON error body.
///
/// Handles three shapes:
/// - OpenAI: `{"error":{"message":"...","type":"..."}}`
/// - flat: `{"message":"..."}`
/// - Google AIP: `{"error":{"message":"Invalid argument","status":"...",
///   "details":[{"fieldViolations":[{"field":"...","description":"..."}]}]}}`
///
/// For Google's shape we append the field-violation descriptions to the
/// base message — without that, all the user sees is `"Invalid argument"`
/// and the actual reason (which API field was wrong) gets dropped on the
/// floor.
pub fn extract_error_message(body: &str) -> Option<String> {
    let v: Value = serde_json::from_str(body).ok()?;
    let err_node = v.get("error");

    let base = err_node
        .and_then(|e| e.get("message"))
        .and_then(|m| m.as_str())
        .or_else(|| v.get("message").and_then(|m| m.as_str()))
        .map(str::to_string);

    let details = err_node
        .and_then(|e| e.get("details"))
        .and_then(|d| d.as_array())
        .map(|details_arr| {
            let mut parts: Vec<String> = Vec::new();
            for entry in details_arr {
                // BadRequest details: fieldViolations[].{field, description}.
                if let Some(violations) = entry.get("fieldViolations").and_then(|f| f.as_array()) {
                    for fv in violations {
                        let field = fv
                            .get("field")
                            .and_then(|f| f.as_str())
                            .unwrap_or("")
                            .trim();
                        let desc = fv
                            .get("description")
                            .and_then(|d| d.as_str())
                            .unwrap_or("")
                            .trim();
                        if !desc.is_empty() {
                            parts.push(if field.is_empty() {
                                desc.to_string()
                            } else {
                                format!("{field}: {desc}")
                            });
                        }
                    }
                }
                // ErrorInfo details: {reason, domain}. Useful when present
                // (rate-limit / auth surfaces include it).
                if let Some(reason) = entry.get("reason").and_then(|r| r.as_str()) {
                    parts.push(format!("reason: {reason}"));
                }
            }
            parts.join("; ")
        })
        .filter(|s| !s.is_empty());

    match (base, details) {
        (Some(b), Some(d)) => Some(format!("{b} — {d}")),
        (Some(b), None) => Some(b),
        (None, Some(d)) => Some(d),
        (None, None) => None,
    }
}

/// Formats a byte count as `<1KB`/`<1MB`/`<1GB` for end-user display.
pub fn human_bytes(b: u64) -> String {
    const K: u64 = 1024;
    if b < K {
        format!("{b}B")
    } else if b < K * K {
        format!("{:.1}KB", b as f64 / K as f64)
    } else {
        format!("{:.1}MB", b as f64 / (K * K) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn parse_none_returns_default() {
        assert!(matches!(OutputTarget::parse(None), OutputTarget::Default));
    }

    #[test]
    fn parse_file_path() {
        let t = OutputTarget::parse(Some("cat.png"));
        assert!(matches!(t, OutputTarget::File(_)));
    }

    #[test]
    fn parse_trailing_slash_is_directory() {
        let t = OutputTarget::parse(Some("out/"));
        assert!(matches!(t, OutputTarget::Directory(_)));
    }

    #[test]
    fn parse_template_with_braces() {
        let t = OutputTarget::parse(Some("cat-{n}.png"));
        assert!(matches!(t, OutputTarget::Template(_)));
    }

    #[test]
    fn resolve_default() {
        let path = resolve_output_path(&OutputTarget::Default, "gpt-image-1", "png").unwrap();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.starts_with("aivo-"));
        assert!(name.ends_with(".png"));
    }

    #[test]
    fn resolve_default_supports_arbitrary_ext() {
        let path = resolve_output_path(&OutputTarget::Default, "sora-2", "mp4").unwrap();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.ends_with(".mp4"));
    }

    #[test]
    fn resolve_exact_file() {
        let t = OutputTarget::File(PathBuf::from("cat.png"));
        let path = resolve_output_path(&t, "gpt-image-1", "png").unwrap();
        assert_eq!(path.to_string_lossy(), "cat.png");
    }

    #[test]
    fn resolve_directory_existing() {
        let tmp = TempDir::new().unwrap();
        let t = OutputTarget::Directory(tmp.path().to_path_buf());
        let path = resolve_output_path(&t, "gpt-image-1", "png").unwrap();
        assert!(path.starts_with(tmp.path()));
    }

    #[test]
    fn resolve_directory_missing_errors() {
        let t = OutputTarget::Directory(PathBuf::from("/definitely/does/not/exist/aivo-test"));
        let err = resolve_output_path(&t, "gpt-image-1", "png").unwrap_err();
        assert!(err.to_string().contains("does not exist"));
    }

    #[test]
    fn resolve_template_with_model_token() {
        let t = OutputTarget::Template("{model}.png".into());
        let path = resolve_output_path(&t, "gpt-image-1", "png").unwrap();
        assert_eq!(path.to_string_lossy(), "gpt-image-1.png");
    }

    #[test]
    fn resolve_template_with_ts_token() {
        let t = OutputTarget::Template("shot-{ts}.png".into());
        let path = resolve_output_path(&t, "gpt-image-1", "png").unwrap();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.starts_with("shot-"));
        assert!(name.ends_with(".png"));
    }

    #[test]
    fn resolve_template_checks_parent_directory() {
        let t = OutputTarget::Template("/definitely/does/not/exist/cat.png".into());
        let err = resolve_output_path(&t, "gpt-image-1", "png").unwrap_err();
        assert!(err.to_string().contains("does not exist"));
    }

    #[test]
    fn parent_if_nonempty_skips_bare_filename() {
        assert!(parent_if_nonempty(Path::new("cat.png")).is_none());
        assert_eq!(
            parent_if_nonempty(Path::new("out/cat.png")),
            Some(Path::new("out"))
        );
        assert_eq!(
            parent_if_nonempty(Path::new("/abs/dir/cat.png")),
            Some(Path::new("/abs/dir"))
        );
    }

    #[test]
    fn sanitize_model_replaces_slashes_and_colons() {
        assert_eq!(sanitize_model("org/model"), "org_model");
        assert_eq!(sanitize_model("name:tag"), "name_tag");
        assert_eq!(sanitize_model("normal-name_1"), "normal-name_1");
    }

    #[test]
    fn pins_extension_distinguishes_user_vs_auto() {
        assert!(!OutputTarget::Default.pins_extension());
        assert!(!OutputTarget::Directory(PathBuf::from("out")).pins_extension());
        assert!(OutputTarget::File(PathBuf::from("cat.png")).pins_extension());
        assert!(!OutputTarget::File(PathBuf::from("cat")).pins_extension());
        assert!(OutputTarget::Template("{model}-{ts}.png".into()).pins_extension());
        assert!(!OutputTarget::Template("{model}-{ts}".into()).pins_extension());
    }

    #[test]
    fn align_extension_swaps_when_unpinned_and_mismatched() {
        assert_eq!(
            align_extension(Path::new("aivo.png"), Some("jpg"), false),
            PathBuf::from("aivo.jpg")
        );
        assert_eq!(
            align_extension(Path::new("out/aivo.png"), Some("webp"), false),
            PathBuf::from("out/aivo.webp")
        );
    }

    #[test]
    fn align_extension_keeps_user_pinned_path() {
        assert_eq!(
            align_extension(Path::new("cat.png"), Some("jpg"), true),
            PathBuf::from("cat.png")
        );
    }

    #[test]
    fn align_extension_noop_when_matched_or_no_server_ext() {
        assert_eq!(
            align_extension(Path::new("aivo.png"), Some("png"), false),
            PathBuf::from("aivo.png")
        );
        assert_eq!(
            align_extension(Path::new("aivo.png"), None, false),
            PathBuf::from("aivo.png")
        );
    }

    #[test]
    fn align_extension_handles_arbitrary_modality_extensions() {
        assert_eq!(
            align_extension(Path::new("clip.mp4"), Some("webm"), false),
            PathBuf::from("clip.webm")
        );
        assert_eq!(
            align_extension(Path::new("speech.mp3"), Some("wav"), false),
            PathBuf::from("speech.wav")
        );
    }

    #[test]
    fn next_free_path_finds_suffix() {
        let tmp = TempDir::new().unwrap();
        let existing = tmp.path().join("cat.png");
        fs::write(&existing, b"x").unwrap();
        let free = next_free_path(&existing);
        assert_eq!(free, tmp.path().join("cat-1.png"));

        fs::write(tmp.path().join("cat-1.png"), b"x").unwrap();
        assert_eq!(next_free_path(&existing), tmp.path().join("cat-2.png"));
    }

    #[test]
    fn next_free_path_preserves_extensionless_paths() {
        let tmp = TempDir::new().unwrap();
        let existing = tmp.path().join("notes");
        fs::write(&existing, b"x").unwrap();
        let free = next_free_path(&existing);
        assert_eq!(free, tmp.path().join("notes-1"));
    }

    #[test]
    fn overwrite_policy_force_writes() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("a.png");
        fs::write(&path, b"old").unwrap();
        let d = apply_overwrite_policy(
            &path,
            OverwritePolicy {
                force: true,
                interactive: false,
            },
            None,
        );
        assert_eq!(d, OverwriteDecision::Write(path));
    }

    #[test]
    fn overwrite_policy_nontty_aborts() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("a.png");
        fs::write(&path, b"old").unwrap();
        let d = apply_overwrite_policy(
            &path,
            OverwritePolicy {
                force: false,
                interactive: false,
            },
            None,
        );
        assert_eq!(d, OverwriteDecision::Abort);
    }

    #[test]
    fn overwrite_policy_missing_file_writes() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("nope.png");
        let d = apply_overwrite_policy(
            &path,
            OverwritePolicy {
                force: false,
                interactive: false,
            },
            None,
        );
        assert_eq!(d, OverwriteDecision::Write(path));
    }

    #[test]
    fn overwrite_policy_yes_answer_writes() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("a.png");
        fs::write(&path, b"old").unwrap();
        let d = apply_overwrite_policy(
            &path,
            OverwritePolicy {
                force: false,
                interactive: true,
            },
            Some('y'),
        );
        assert_eq!(d, OverwriteDecision::Write(path));
    }

    #[test]
    fn overwrite_policy_no_answer_aborts() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("a.png");
        fs::write(&path, b"old").unwrap();
        let d = apply_overwrite_policy(
            &path,
            OverwritePolicy {
                force: false,
                interactive: true,
            },
            Some('n'),
        );
        assert_eq!(d, OverwriteDecision::Abort);
    }

    #[test]
    fn overwrite_policy_skip_finds_free_path() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("a.png");
        fs::write(&path, b"old").unwrap();
        let d = apply_overwrite_policy(
            &path,
            OverwritePolicy {
                force: false,
                interactive: true,
            },
            Some('s'),
        );
        match d {
            OverwriteDecision::Write(p) => {
                assert_eq!(p, tmp.path().join("a-1.png"));
            }
            _ => panic!("expected Write with suffixed path"),
        }
    }

    #[test]
    fn atomic_write_produces_final_file_only() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("out.png");
        let n = atomic_write(&path, b"hello").unwrap();
        assert_eq!(n, 5);
        assert_eq!(fs::read(&path).unwrap(), b"hello");
        let dir_entries: Vec<_> = fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(dir_entries.len(), 1);
        assert_eq!(dir_entries[0], "out.png");
    }

    #[test]
    fn atomic_write_replaces_existing_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("out.png");
        fs::write(&path, b"old").unwrap();
        let n = atomic_write(&path, b"new bytes").unwrap();
        assert_eq!(n, 9);
        assert_eq!(fs::read(&path).unwrap(), b"new bytes");
    }

    #[test]
    fn extract_error_message_reads_openai_shape() {
        let body = r#"{"error":{"message":"bad prompt","type":"invalid"}}"#;
        assert_eq!(extract_error_message(body).as_deref(), Some("bad prompt"));
    }

    #[test]
    fn extract_error_message_reads_flat_shape() {
        let body = r#"{"message":"rate limited"}"#;
        assert_eq!(extract_error_message(body).as_deref(), Some("rate limited"));
    }

    #[test]
    fn extract_error_message_returns_none_for_plain_text() {
        assert!(extract_error_message("not json").is_none());
    }

    #[test]
    fn extract_error_message_appends_google_field_violations() {
        // Real Gemini TTS error shape — the useful part lives in
        // error.details[].fieldViolations[], not in error.message.
        let body = r#"{
            "error": {
                "code": 400,
                "message": "Invalid argument",
                "status": "INVALID_ARGUMENT",
                "details": [{
                    "@type": "type.googleapis.com/google.rpc.BadRequest",
                    "fieldViolations": [{
                        "field": "session_context.tts_voice_info.selected_voice_name",
                        "description": "must not be empty for non-voice-replication requests."
                    }]
                }]
            }
        }"#;
        let msg = extract_error_message(body).expect("should extract");
        assert!(msg.contains("Invalid argument"), "got: {msg}");
        assert!(
            msg.contains("session_context.tts_voice_info.selected_voice_name"),
            "field name should appear: {msg}"
        );
        assert!(
            msg.contains("must not be empty"),
            "description should appear: {msg}"
        );
    }

    #[test]
    fn extract_error_message_handles_multiple_field_violations() {
        let body = r#"{
            "error": {
                "message": "Invalid argument",
                "details": [{
                    "fieldViolations": [
                        {"field": "voice", "description": "is required"},
                        {"field": "model", "description": "must be a TTS model"}
                    ]
                }]
            }
        }"#;
        let msg = extract_error_message(body).expect("should extract");
        assert!(msg.contains("voice: is required"), "got: {msg}");
        assert!(msg.contains("model: must be a TTS model"), "got: {msg}");
    }

    #[test]
    fn extract_error_message_appends_error_info_reason() {
        // ErrorInfo details (rate-limit, auth) come back with `reason` only.
        let body = r#"{
            "error": {
                "message": "Quota exceeded",
                "details": [{
                    "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                    "reason": "RATE_LIMIT_EXCEEDED",
                    "domain": "googleapis.com"
                }]
            }
        }"#;
        let msg = extract_error_message(body).expect("should extract");
        assert!(msg.contains("Quota exceeded"), "got: {msg}");
        assert!(msg.contains("reason: RATE_LIMIT_EXCEEDED"), "got: {msg}");
    }

    #[test]
    fn extract_error_message_still_works_for_simple_openai_shape() {
        // Backward-compat: details-less OpenAI-style errors must keep
        // returning just the message, no trailing separator.
        let body = r#"{"error":{"message":"bad prompt","type":"invalid"}}"#;
        assert_eq!(extract_error_message(body).as_deref(), Some("bad prompt"));
    }

    #[test]
    fn human_bytes_formats_ranges() {
        assert_eq!(human_bytes(500), "500B");
        assert_eq!(human_bytes(1024), "1.0KB");
        assert_eq!(human_bytes(1024 * 1024), "1.0MB");
    }
}
