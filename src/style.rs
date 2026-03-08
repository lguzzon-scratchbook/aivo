/**
 * Terminal styling utility using the console crate.
 * Provides cross-platform styling with ANSI fallback support.
 */
use console::style;
use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::task::JoinHandle;

/// Supported style names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StyleName {
    Bold,
    Dim,
    Red,
    Green,
    Yellow,
    Blue,
    Cyan,
}

/// Styles text using the console crate.
/// On Windows without ANSI support, console handles the fallback automatically.
pub fn style_text(style_name: StyleName, text: impl AsRef<str>) -> String {
    let text = text.as_ref();

    match style_name {
        StyleName::Bold => style(text).bold().to_string(),
        StyleName::Dim => style(text).dim().to_string(),
        StyleName::Red => style(text).red().to_string(),
        StyleName::Green => style(text).green().to_string(),
        StyleName::Yellow => style(text).yellow().to_string(),
        StyleName::Blue => style(text).blue().to_string(),
        StyleName::Cyan => style(text).cyan().to_string(),
    }
}

/// Convenience function to style text as cyan (commonly used in the CLI).
pub fn cyan(text: impl AsRef<str>) -> String {
    style_text(StyleName::Cyan, text)
}

/// Convenience function to style text as green (for success).
pub fn green(text: impl AsRef<str>) -> String {
    style_text(StyleName::Green, text)
}

/// Convenience function to style text as red (for errors).
pub fn red(text: impl AsRef<str>) -> String {
    style_text(StyleName::Red, text)
}

/// Convenience function to style text as yellow (for warnings/notes).
pub fn yellow(text: impl AsRef<str>) -> String {
    style_text(StyleName::Yellow, text)
}

/// Convenience function to style text as dim (for secondary information).
pub fn dim(text: impl AsRef<str>) -> String {
    style_text(StyleName::Dim, text)
}

/// Convenience function to style text as bold.
pub fn bold(text: impl AsRef<str>) -> String {
    style_text(StyleName::Bold, text)
}

/// Convenience function to style text as blue.
pub fn blue(text: impl AsRef<str>) -> String {
    style_text(StyleName::Blue, text)
}

/// Convenience function for the "✓" success symbol.
pub fn success_symbol() -> String {
    green("✓")
}

/// Convenience function for the "→" arrow symbol.
pub fn arrow_symbol() -> String {
    cyan("→")
}

/// Convenience function for the "●" bullet symbol.
pub fn bullet_symbol() -> String {
    green("●")
}

/// Convenience function for the "○" empty bullet symbol.
pub fn empty_bullet_symbol() -> String {
    dim("○")
}

/// Starts a braille spinner on stderr. Returns the flag and join handle.
/// Pass an optional label to display after the spinner character.
pub fn start_spinner(label: Option<&str>) -> (Arc<AtomicBool>, JoinHandle<()>) {
    let spinning = Arc::new(AtomicBool::new(true));
    let spinning_clone = spinning.clone();
    let label = label.map(str::to_owned).unwrap_or_default();
    let first_frame = "\u{280b}";

    // Paint the first frame synchronously so short operations still show feedback.
    eprint!("\r{}{}", dim(first_frame), label);
    let _ = io::stderr().flush();

    let handle = tokio::task::spawn_blocking(move || {
        let frames = [
            "\u{280b}", "\u{2819}", "\u{2839}", "\u{2838}", "\u{283c}", "\u{2834}", "\u{2826}",
            "\u{2827}", "\u{2807}", "\u{280f}",
        ];
        let mut i = 1;
        while spinning_clone.load(Ordering::Relaxed) {
            eprint!("\r{}{}", dim(frames[i % frames.len()]), label);
            let _ = io::stderr().flush();
            std::thread::sleep(std::time::Duration::from_millis(80));
            i += 1;
        }
    });
    (spinning, handle)
}

/// Stops the spinner and clears its character from the line.
pub fn stop_spinner(spinning: &Arc<AtomicBool>) {
    if spinning.swap(false, Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(100));
        eprint!("\r \r");
        let _ = io::stderr().flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_style_text() {
        let styled = style_text(StyleName::Cyan, "test");
        assert!(!styled.is_empty());
        assert!(styled.contains("test"));
    }

    #[test]
    fn test_convenience_functions() {
        assert!(!cyan("test").is_empty());
        assert!(!green("test").is_empty());
        assert!(!red("test").is_empty());
        assert!(!yellow("test").is_empty());
        assert!(!dim("test").is_empty());
        assert!(!bold("test").is_empty());
        assert!(!blue("test").is_empty());
    }

    #[test]
    fn test_symbols() {
        assert!(!success_symbol().is_empty());
        assert!(!arrow_symbol().is_empty());
        assert!(!bullet_symbol().is_empty());
        assert!(!empty_bullet_symbol().is_empty());
    }
}
