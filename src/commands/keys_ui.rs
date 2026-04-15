//! Lightweight step chips and provider info lines for the `aivo keys add`
//! flow.
//!
//! The visual vocabulary:
//!
//! ```text
//!  1/3  Name — a short label for this key
//! Name (optional): █
//!
//!  2/3  Provider — preset or custom URL
//! [fuzzy picker]
//!
//! ● OpenRouter  https://openrouter.ai/keys
//!
//!  3/3  API Key
//! API Key: █
//! ```
//!
//! Each helper emits a *leading* blank line so the caller never has to
//! scatter `println!()` calls between steps — the separator travels with
//! the next step.
//!
//! Output goes to stderr (matching `FuzzySelect` and the spinner) so
//! piping `aivo keys add` stdout still captures only actual data.

use crate::style;
use std::io::{self, Write};

/// Render a one-line step chip like:
/// ```text
///  2/3  Provider — preset or custom URL
/// ```
/// The chip ` N/M ` is rendered with a cyan background + black foreground.
/// `subtitle` is separated by an em-dash and rendered dim; pass `""` to
/// omit the trailing clause when the prompt is self-explanatory.
pub fn step_header(step: usize, total: usize, title: &str, subtitle: &str) {
    let chip = format!(" {step}/{total} ");
    let styled_chip = console::style(chip).black().on_cyan().to_string();
    let styled_title = style::bold(title);

    let mut out = io::stderr().lock();
    let _ = writeln!(out);
    if subtitle.is_empty() {
        let _ = writeln!(out, "{} {}", styled_chip, styled_title);
    } else {
        let _ = writeln!(
            out,
            "{} {} {} {}",
            styled_chip,
            styled_title,
            style::dim("—"),
            style::dim(subtitle),
        );
    }
    let _ = out.flush();
}

/// Render a one-line provider info bullet like:
/// ```text
/// ● OpenRouter  https://openrouter.ai/keys
/// ```
/// Pass an empty `note` to skip rendering entirely (e.g. when a known
/// provider has no signup URL seeded — the base URL was already visible
/// in the picker, so there's nothing useful to add).
pub fn provider_info(name: &str, note: &str) {
    if note.is_empty() {
        return;
    }
    let mut out = io::stderr().lock();
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "{} {}  {}",
        style::bullet_symbol(),
        style::cyan(style::bold(name)),
        style::dim(note),
    );
    let _ = out.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_header_without_subtitle() {
        step_header(1, 3, "Name", "");
    }

    #[test]
    fn step_header_with_subtitle() {
        step_header(2, 3, "Provider", "preset or custom URL");
    }

    #[test]
    fn provider_info_with_url() {
        provider_info("OpenRouter", "https://openrouter.ai/keys");
    }

    #[test]
    fn provider_info_empty_note_is_silent() {
        // No panic, no output. Caller supplies empty string when there's
        // nothing useful to surface.
        provider_info("Some Provider", "");
    }
}
