use console::{Key, Term};
use std::io::Write as _;

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
        let mut term = Term::stderr();

        // Setup initial state
        let mut query = String::new();
        let mut selection = self.default.min(self.items.len().saturating_sub(1));
        let mut page_start = 0;
        let page_size = 10;

        loop {
            // Filter items based on query
            // Store (original_index, item_string)
            let filtered: Vec<(usize, &String)> = self
                .items
                .iter()
                .enumerate()
                .filter(|(_, item)| matches_fuzzy(&query, item))
                .collect();

            let count = filtered.len();

            // Adjust selection if out of bounds (e.g. after filtering)
            if selection >= count {
                selection = count.saturating_sub(1);
            }

            // Calculate visible range for pagination
            // Ensure selection is visible
            if selection < page_start {
                page_start = selection;
            } else if selection >= page_start + page_size {
                page_start = selection.saturating_sub(page_size).saturating_add(1);
            }

            // Ensure page_start is valid
            if page_start > count.saturating_sub(1) {
                page_start = count.saturating_sub(1);
            }

            let end_idx = (page_start + page_size).min(count);

            // Save cursor position before drawing
            term.write_all(b"\x1b7")?;

            // Draw prompt and query
            term.write_line(&format!("{}: {}", crate::style::bold(&self.prompt), query))?;

            if count == 0 {
                term.write_line(&format!("  {}", crate::style::dim("(no matches)")))?;
            } else {
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

                    term.write_line(&format!("{} {}", symbol, styled_item))?;
                }
            }

            // Wait for input
            let key = match term.read_key() {
                Ok(key) => key,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {
                    // Ctrl-C pressed - restore cursor and clear before exiting
                    let _ = term.write_all(b"\x1b8\x1b[0J");
                    let _ = term.show_cursor();
                    return Ok(None);
                }
                Err(e) => {
                    let _ = term.write_all(b"\x1b8\x1b[0J");
                    let _ = term.show_cursor();
                    return Err(e);
                }
            };

            // Restore cursor and clear to end of screen for next redraw or exit
            term.write_all(b"\x1b8\x1b[0J")?;

            match key {
                Key::ArrowUp | Key::Char('\x10') => {
                    // Ctrl-P
                    if selection > 0 {
                        selection -= 1;
                    } else if count > 0 {
                        selection = count - 1; // Wrap around
                    }
                }
                Key::ArrowDown | Key::Char('\x0e') => {
                    // Ctrl-N
                    if count > 0 {
                        if selection < count - 1 {
                            selection += 1;
                        } else {
                            selection = 0; // Wrap around
                        }
                    }
                }
                Key::Enter => {
                    let _ = term.show_cursor();
                    if count > 0 {
                        return Ok(Some(filtered[selection].0));
                    }
                    return Ok(None);
                }
                Key::Escape => {
                    let _ = term.show_cursor();
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

fn matches_fuzzy(query: &str, target: &str) -> bool {
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
