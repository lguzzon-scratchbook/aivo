use console::{Key, Term};

impl Default for FuzzySelect {
    fn default() -> Self {
        Self::new()
    }
}

pub struct FuzzySelect {
    prompt: String,
    items: Vec<String>,
    default: usize,
}

impl FuzzySelect {
    pub fn new() -> Self {
        Self {
            prompt: "Select".to_string(),
            items: Vec::new(),
            default: 0,
        }
    }

    pub fn with_prompt(mut self, prompt: &str) -> Self {
        self.prompt = prompt.to_string();
        self
    }

    pub fn items(mut self, items: &[String]) -> Self {
        self.items = items.to_vec();
        self
    }

    pub fn default(mut self, default: usize) -> Self {
        self.default = default;
        self
    }

    pub fn interact_opt(self) -> std::io::Result<Option<usize>> {
        let term = Term::stderr();
        term.hide_cursor()?;

        let mut query = String::new();
        let mut selection = self.default.min(self.items.len().saturating_sub(1));
        let mut page_start = 0;
        let page_size = 10;

        loop {
            let filtered: Vec<(usize, &String)> = self
                .items
                .iter()
                .enumerate()
                .filter(|(_, item)| matches_fuzzy(&query, item))
                .collect();

            let count = filtered.len();

            if selection >= count {
                selection = count.saturating_sub(1);
            }

            if selection < page_start {
                page_start = selection;
            } else if selection >= page_start + page_size {
                page_start = selection.saturating_sub(page_size).saturating_add(1);
            }

            if page_start > count.saturating_sub(1) {
                page_start = count.saturating_sub(1);
            }

            let end_idx = (page_start + page_size).min(count);

            let term_width = term.size().1 as usize;

            let hint = if query.is_empty() && count > page_size {
                format!(" {}", crate::style::dim("(type to filter)"))
            } else {
                String::new()
            };
            let prompt_line = format!("{}: {}{}", crate::style::bold(&self.prompt), query, hint);
            term.write_line(&truncate_to_width(&prompt_line, term_width))?;

            let items_drawn = if count == 0 {
                term.write_line(&format!("  {}", crate::style::dim("(no matches)")))?;
                1
            } else {
                let mut lines = 0;
                if page_start > 0 {
                    let above = page_start;
                    term.write_line(&format!(
                        "  {}",
                        crate::style::dim(&format!("↑ {} more above", above))
                    ))?;
                    lines += 1;
                }
                for (i, (_, item)) in filtered.iter().enumerate().take(end_idx).skip(page_start) {
                    let is_selected = i == selection;
                    let symbol = if is_selected {
                        crate::style::cyan(">")
                    } else {
                        " ".to_string()
                    };
                    let styled_item = if is_selected {
                        crate::style::cyan(item)
                    } else {
                        crate::style::dim(item)
                    };
                    let line = format!("{} {}", symbol, styled_item);
                    term.write_line(&truncate_to_width(&line, term_width))?;
                    lines += 1;
                }
                if end_idx < count {
                    let below = count - end_idx;
                    term.write_line(&format!(
                        "  {}",
                        crate::style::dim(&format!("↓ {} more below", below))
                    ))?;
                    lines += 1;
                }
                lines
            };

            let key = match term.read_key_raw() {
                Ok(key) => key,
                Err(e) => {
                    let _ = term.clear_last_lines(1 + items_drawn);
                    let _ = term.show_cursor();
                    return Err(e);
                }
            };

            // Clear drawn lines before next iteration or exit
            term.clear_last_lines(1 + items_drawn)?;

            match key {
                key if is_previous_key(&key) => {
                    if selection > 0 {
                        selection -= 1;
                    } else if count > 0 {
                        selection = count - 1;
                    }
                }
                key if is_next_key(&key) => {
                    if count > 0 {
                        if selection < count - 1 {
                            selection += 1;
                        } else {
                            selection = 0;
                        }
                    }
                }
                Key::Enter => {
                    term.show_cursor()?;
                    if count > 0 {
                        return Ok(Some(filtered[selection].0));
                    }
                    return Ok(None);
                }
                Key::Escape | Key::CtrlC => {
                    term.show_cursor()?;
                    return Ok(None);
                }
                Key::Backspace => {
                    if !query.is_empty() {
                        query.pop();
                        selection = 0;
                        page_start = 0;
                    }
                }
                Key::Char(c) => {
                    if !c.is_control() {
                        query.push(c);
                        selection = 0;
                        page_start = 0;
                    }
                }
                _ => {}
            }
        }
    }
}

fn is_previous_key(key: &Key) -> bool {
    matches!(key, Key::ArrowUp | Key::Char('\x10')) || matches_application_arrow(key, 'A')
}

fn is_next_key(key: &Key) -> bool {
    matches!(key, Key::ArrowDown | Key::Char('\x0e')) || matches_application_arrow(key, 'B')
}

fn matches_application_arrow(key: &Key, direction: char) -> bool {
    matches!(key, Key::UnknownEscSeq(seq) if seq.as_slice() == ['O', direction])
}

/// Truncate a string to fit within terminal width, accounting for ANSI escape codes.
/// ANSI sequences are not visible characters, so we track visible width separately.
fn truncate_to_width(s: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let mut visible = 0;
    let mut result = String::with_capacity(s.len());
    let mut in_escape = false;
    for c in s.chars() {
        if c == '\x1b' {
            in_escape = true;
            result.push(c);
        } else if in_escape {
            result.push(c);
            if c.is_ascii_alphabetic() {
                in_escape = false;
            }
        } else {
            if visible >= width {
                break;
            }
            result.push(c);
            visible += 1;
        }
    }
    result
}

pub(crate) fn matches_fuzzy(query: &str, target: &str) -> bool {
    let mut q_chars = query.chars();
    let mut current_q_char = match q_chars.next() {
        Some(c) => c,
        None => return true,
    };

    for c in target.chars() {
        if c.eq_ignore_ascii_case(&current_q_char) {
            current_q_char = match q_chars.next() {
                Some(next) => next,
                None => return true, // All query chars found
            };
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::{is_next_key, is_previous_key};
    use console::Key;

    #[test]
    fn recognizes_application_cursor_mode_arrows() {
        assert!(is_previous_key(&Key::UnknownEscSeq(vec!['O', 'A'])));
        assert!(is_next_key(&Key::UnknownEscSeq(vec!['O', 'B'])));
    }

    #[test]
    fn recognizes_standard_navigation_shortcuts() {
        assert!(is_previous_key(&Key::ArrowUp));
        assert!(is_previous_key(&Key::Char('\x10')));
        assert!(is_next_key(&Key::ArrowDown));
        assert!(is_next_key(&Key::Char('\x0e')));
    }
}
