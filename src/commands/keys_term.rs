/**
 * Terminal I/O and interactive picker utilities for key management.
 */
use crate::services::api_key_store::ApiKeyStore;
use anyhow::Result;

use crate::services::provider_profile::is_aivo_starter_base;
use crate::services::session_store::{ApiKey, SessionStore};
use crate::style;
use crate::tui::FuzzySelect;

// Force cooked (canonical + echo) mode on stdin. The `console` crate's FuzzySelect
// can leave termios flags off on macOS, and a previously-crashed `aivo` invocation
// may have left the terminal corrupted. `crossterm::disable_raw_mode` is a no-op
// when crossterm didn't enable raw mode itself, so we shell out to `stty sane`.
// Call this at entry of an interactive flow and after any FuzzySelect exits.
#[cfg(unix)]
pub(crate) fn restore_cooked_mode() {
    let _ = std::process::Command::new("stty")
        .arg("sane")
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
}

#[cfg(not(unix))]
pub(crate) fn restore_cooked_mode() {
    let _ = crossterm::terminal::disable_raw_mode();
}

// Reads a line from stdin, flushing the prompt first so it appears before blocking.
pub(crate) fn term_read_line(prompt: &str) -> std::io::Result<String> {
    use std::io::{BufRead, Write};
    print!("{}", prompt);
    std::io::stdout().flush()?;
    let mut input = String::new();
    std::io::stdin().lock().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

// Reads a line from stdin with masked echo (prints '*' per character) for secrets.
pub(crate) fn term_read_secret(prompt: &str) -> std::io::Result<String> {
    use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
    use crossterm::terminal;
    use std::io::Write;

    print!("{}", prompt);
    std::io::stdout().flush()?;

    terminal::enable_raw_mode()?;
    let mut input = String::new();
    let mut stdout = std::io::stdout();
    let result = loop {
        match event::read() {
            Ok(Event::Key(KeyEvent {
                code, modifiers, ..
            })) => match code {
                KeyCode::Enter => {
                    let _ = write!(stdout, "\r\n");
                    let _ = stdout.flush();
                    break Ok(input);
                }
                KeyCode::Backspace if !input.is_empty() => {
                    input.pop();
                    let _ = write!(stdout, "\x08 \x08");
                    let _ = stdout.flush();
                }
                KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
                    let _ = write!(stdout, "\r\n");
                    let _ = stdout.flush();
                    break Err(std::io::Error::new(
                        std::io::ErrorKind::Interrupted,
                        "interrupted",
                    ));
                }
                KeyCode::Char(c) => {
                    input.push(c);
                    let _ = write!(stdout, "*");
                    let _ = stdout.flush();
                }
                _ => {}
            },
            Ok(_) => {}
            Err(e) => break Err(e),
        }
    };
    let _ = terminal::disable_raw_mode();
    result
}

// Reads a confirmation from stdin (y/yes for true, anything else for false).
pub(crate) fn confirm(prompt: &str) -> std::io::Result<bool> {
    let input = term_read_line(&format!("{} [y/N]: ", prompt))?;
    Ok(matches!(input.to_ascii_lowercase().as_str(), "y" | "yes"))
}

// Creates a safe preview of an API key, handling short keys without panicking.
pub(crate) fn key_preview(key: &str) -> String {
    let chars: Vec<char> = key.chars().collect();
    if chars.len() <= 10 {
        let prefix: String = chars.iter().take(3).collect();
        format!("{prefix}...")
    } else {
        let prefix: String = chars.iter().take(6).collect();
        let suffix: String = chars.iter().skip(chars.len() - 4).collect();
        format!("{prefix}...{suffix}")
    }
}

pub(crate) fn display_secret(key: &ApiKey) -> String {
    key.credential_label()
        .map(str::to_string)
        .unwrap_or_else(|| key_preview(&key.key))
}

/// Prompts for a secret. If the user enters nothing, asks whether to save
/// without one; loops until a value is provided or confirmation is given.
pub(crate) fn prompt_secret(prompt: &str, secret_noun: &str) -> std::io::Result<String> {
    loop {
        let input = term_read_secret(&style::dim(prompt))?;
        if !input.is_empty() {
            return Ok(input);
        }
        let confirm_prompt = style::yellow(format!("Save without {}?", secret_noun));
        if confirm(&confirm_prompt)? {
            return Ok(String::new());
        }
    }
}

// Formats an API key as a choice string for interactive selectors.
pub(crate) fn format_key_choice(key: &ApiKey) -> String {
    format!(
        "{}  {}  {}",
        style::cyan(format!("{:<3}", key.short_id())),
        key.display_name(),
        style::dim(&key.base_url)
    )
}

/// Formats the common key-entry prefix (active indicator, short ID, name) for
/// listing displays. Used by both `list_keys` and `list_keys_with_ping`.
pub(crate) fn format_key_entry(
    key: &ApiKey,
    selected_key_id: Option<&str>,
    max_name_len: usize,
) -> String {
    let is_selected = selected_key_id == Some(key.id.as_str());
    let active_indicator = if is_selected {
        style::bullet_symbol()
    } else {
        style::empty_bullet_symbol()
    };
    let id_padded = format!("{:<3}", key.short_id());
    let name_padded = format!("{:<width$}", key.name, width = max_name_len);
    let is_starter = is_aivo_starter_base(&key.base_url);
    let name_col = if is_starter {
        style::magenta(&name_padded)
    } else {
        name_padded
    };
    format!(
        "{} {}  {}",
        active_indicator,
        style::cyan(&id_padded),
        name_col,
    )
}

// Prompts the user to select a key from the given list. `annotations`
// parallels `keys`: `Some(reason)` disables that row and shows the reason
// as a dim suffix. Pass `&[]` for no annotations.
pub(crate) fn prompt_pick_key(
    keys: &[ApiKey],
    annotations: &[Option<String>],
    prompt: &str,
    default: usize,
) -> Result<Option<ApiKey>> {
    let choices: Vec<String> = keys.iter().map(format_key_choice).collect();
    let mut picker = FuzzySelect::new()
        .with_prompt(prompt)
        .items(&choices)
        .default(default);
    if !annotations.is_empty() {
        picker = picker.annotations(annotations.to_vec());
    }
    let selection = picker.interact_opt()?;
    Ok(selection.map(|idx| keys[idx].clone()))
}

pub(crate) fn prompt_pick_key_without_activation(
    keys: &[ApiKey],
    annotations: &[Option<String>],
    prompt: &str,
    default: usize,
) -> Result<Option<ApiKey>> {
    match prompt_pick_key(keys, annotations, prompt, default)? {
        Some(mut key) => {
            ApiKeyStore::decrypt_key_secret(&mut key)?;
            Ok(Some(key))
        }
        None => Ok(None),
    }
}

// Picks a key from `keys` and activates it. Returns `Ok(None)` if cancelled.
#[allow(dead_code)] // used by binary crate (key_resolution.rs)
pub(crate) async fn prompt_select_key(
    session_store: &SessionStore,
    keys: &[ApiKey],
    annotations: &[Option<String>],
    prompt: &str,
    default: usize,
) -> Result<Option<ApiKey>> {
    match prompt_pick_key(keys, annotations, prompt, default)? {
        Some(mut key) => {
            ApiKeyStore::decrypt_key_secret(&mut key)?;
            session_store.api_keys().api_keys().set_active_key(&key.id).await?;
            let preview = display_secret(&key);
            eprintln!(
                "{} Activated key: {} {}",
                style::success_symbol(),
                style::cyan(key.display_name()),
                style::dim(&preview)
            );
            Ok(Some(key))
        }
        None => Ok(None),
    }
}

/// Offers a picker of compatible keys when `bad_key` is an OAuth credential
/// the current command can't use. OAuth keys stay visible but are disabled
/// with an inline reason. `context_phrase` is the user-visible command name
/// inserted into messages (e.g. `"aivo chat"` or `"aivo run codex"`).
///
/// Returns `Ok(Some(new_key))` when the user picks a replacement; `Ok(None)`
/// when there's no TTY, no eligible key, or the user cancelled — callers
/// should exit with `ExitCode::UserError`.
pub(crate) async fn swap_incompatible_key(
    session_store: &SessionStore,
    bad_key: &ApiKey,
    compat: crate::services::key_compat::KeyCompatContext,
    context_phrase: &str,
) -> Result<Option<ApiKey>> {
    use std::io::IsTerminal;

    let all_keys = session_store.api_keys().api_keys().get_keys().await?;
    let annotations = compat.annotations_for(&all_keys);
    let has_eligible = annotations.iter().any(Option::is_none);

    if !has_eligible || !std::io::stderr().is_terminal() {
        eprintln!(
            "{} Key '{}' is a {} OAuth account — `{}` can't use it.",
            style::red("Error:"),
            bad_key.display_name(),
            bad_key.oauth_kind_label(),
            context_phrase,
        );
        eprintln!(
            "  {} Use `{}` or select a regular API key.",
            style::dim("hint:"),
            bad_key.oauth_tool_hint(),
        );
        return Ok(None);
    }

    eprintln!(
        "{} Key '{}' is a {} OAuth account — pick a regular API key for `{}`.",
        style::yellow("Note:"),
        bad_key.display_name(),
        bad_key.oauth_kind_label(),
        context_phrase,
    );

    prompt_pick_key_without_activation(&all_keys, &annotations, "Select a key", 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::session_store::ApiKey;

    #[test]
    fn test_format_key_choice_uses_id_for_unnamed_keys() {
        let key = ApiKey::new_with_protocol(
            "a2b".to_string(),
            String::new(),
            "https://openrouter.ai/api/v1".to_string(),
            None,
            "sk-test".to_string(),
        );

        let choice = format_key_choice(&key);

        assert!(choice.contains("a2b"));
        assert!(choice.contains("https://openrouter.ai/api/v1"));
    }

    #[test]
    fn key_preview_redacts_by_length() {
        for (input, expected) in [
            ("sk-abc", "sk-..."),
            ("1234567890", "123..."),
            ("sk-abcdefghijklmnop", "sk-abc...mnop"),
            ("", "..."),
        ] {
            assert_eq!(key_preview(input), expected, "input: {input:?}");
        }
    }

    #[test]
    fn key_preview_unicode_safe() {
        // Multi-byte chars: prove we slice on char boundaries, not bytes.
        let key = "\u{1f511}\u{1f511}\u{1f511}\u{1f511}\u{1f511}\u{1f511}\u{1f511}\u{1f511}\u{1f511}\u{1f511}\u{1f511}\u{1f511}";
        let out = key_preview(key);
        assert!(out.contains("..."));
        assert!(out.starts_with("\u{1f511}"));
    }

    #[test]
    fn display_secret_labels_oauth_and_copilot() {
        use crate::services::claude_oauth::CLAUDE_OAUTH_SENTINEL;
        use crate::services::codex_oauth::CODEX_OAUTH_SENTINEL;
        use crate::services::gemini_oauth::GEMINI_OAUTH_SENTINEL;

        let cases = [
            (CLAUDE_OAUTH_SENTINEL, "<Claude OAuth>"),
            (CODEX_OAUTH_SENTINEL, "<Codex OAuth>"),
            (GEMINI_OAUTH_SENTINEL, "<Gemini OAuth>"),
            ("copilot", "<Copilot>"),
        ];
        for (base_url, expected) in cases {
            let key = ApiKey::new_with_protocol(
                "id".to_string(),
                "name".to_string(),
                base_url.to_string(),
                None,
                "must-not-leak-this-credential-blob".to_string(),
            );
            let out = display_secret(&key);
            assert_eq!(out, expected, "base_url: {base_url}");
            assert!(!out.contains("must-not-leak"), "base_url: {base_url}");
        }
    }

    #[test]
    fn display_secret_falls_back_to_preview_for_api_keys() {
        let key = ApiKey::new_with_protocol(
            "id".to_string(),
            "name".to_string(),
            "https://api.example.com".to_string(),
            None,
            "sk-abcdefghijklmnop".to_string(),
        );
        assert_eq!(display_secret(&key), "sk-abc...mnop");
    }
}
