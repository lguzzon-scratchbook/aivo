use super::*;

const SELECT_BG: Color = Color::Rgb(78, 108, 136);
const SELECT_TEXT: Color = Color::Rgb(242, 245, 247);
const SELECT_ACCENT: Color = SELECT_WARM;

fn picker_content_width(width: u16) -> usize {
    usize::from(width.max(1))
        .saturating_sub(PICKER_ROW_PREFIX_WIDTH)
        .max(1)
}

pub(super) fn filter_slash_commands(query: &str) -> Vec<&'static SlashCommandSpec> {
    if query.is_empty() {
        return SLASH_COMMANDS.iter().collect();
    }

    let mut prefix_matches = Vec::new();
    let mut fuzzy_matches = Vec::new();
    for command in SLASH_COMMANDS {
        if command.name.starts_with(query) {
            prefix_matches.push(command);
        } else if matches_fuzzy(query, command.name) {
            fuzzy_matches.push(command);
        }
    }
    prefix_matches.extend(fuzzy_matches);
    prefix_matches
}

pub(super) fn collect_attach_path_suggestions(cwd: &str, query: &str) -> Vec<PathMenuEntry> {
    let trimmed = query.trim_start();
    let (dir_part, prefix) = match trimmed.rfind('/') {
        Some(index) => (&trimmed[..=index], &trimmed[index + 1..]),
        None => ("", trimmed),
    };

    let dir_path = {
        let expanded = crate::services::system_env::expand_tilde(dir_part);
        if expanded.is_absolute() {
            expanded
        } else {
            Path::new(cwd).join(dir_part)
        }
    };

    let Ok(read_dir) = std::fs::read_dir(&dir_path) else {
        return Vec::new();
    };

    let mut entries = read_dir
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().to_string_lossy().to_string();
            if !prefix.is_empty() && !name.starts_with(prefix) && !matches_fuzzy(prefix, &name) {
                return None;
            }
            let file_type = entry.file_type().ok()?;
            let is_dir = file_type.is_dir();
            let suffix = if is_dir { "/" } else { "" };
            let display_name = format!("{name}{suffix}");
            Some(PathMenuEntry {
                label: display_name.clone(),
                is_dir,
                description: if is_dir { "directory" } else { "file" }.to_string(),
                insertion_text: format!("/attach {dir_part}{display_name}"),
            })
        })
        .collect::<Vec<_>>();

    entries.sort_by(|a, b| {
        // Prefix matches rank above fuzzy-only matches, then dirs before files, then alphabetical.
        let a_prefix = a.label.starts_with(prefix);
        let b_prefix = b.label.starts_with(prefix);
        b_prefix
            .cmp(&a_prefix)
            .then_with(|| b.is_dir.cmp(&a.is_dir))
            .then_with(|| a.label.to_lowercase().cmp(&b.label.to_lowercase()))
    });
    entries.truncate(64);
    entries
}

pub(super) fn command_menu_area(
    composer_area: Rect,
    frame_area: Rect,
    item_count: usize,
    preferred_placement: Option<CommandMenuPlacement>,
) -> (Rect, CommandMenuPlacement) {
    let left_offset = composer_area.x.saturating_sub(frame_area.x);
    let max_width = frame_area.width.saturating_sub(left_offset).max(1);
    let min_width = max_width.min(24);
    let width = composer_area.width.min(max_width).max(min_width);
    let row_count = item_count.clamp(1, COMMAND_MENU_MAX_ROWS) as u16;
    let desired_height = row_count.saturating_add(3);
    let above_space = composer_area.y.saturating_sub(frame_area.y);
    let below_space = frame_area
        .y
        .saturating_add(frame_area.height)
        .saturating_sub(composer_area.y.saturating_add(composer_area.height));
    let placement = preferred_placement.unwrap_or({
        if above_space >= desired_height || above_space >= below_space {
            CommandMenuPlacement::Above
        } else {
            CommandMenuPlacement::Below
        }
    });
    // Cap height to the available space in the chosen direction so the menu never
    // overlaps the composer when space is tight.
    let available = match placement {
        CommandMenuPlacement::Above => above_space,
        CommandMenuPlacement::Below => below_space,
    };
    let height = desired_height
        .min(available.max(4))
        .min(frame_area.height.max(4));
    let y = match placement {
        CommandMenuPlacement::Above => composer_area.y.saturating_sub(height).max(frame_area.y),
        CommandMenuPlacement::Below => composer_area.y.saturating_add(composer_area.height),
    };
    (Rect::new(composer_area.x, y, width, height), placement)
}

pub(super) fn command_menu_item_line(
    label: &str,
    description: &str,
    selected: bool,
    width: u16,
    label_column_width: usize,
) -> Line<'static> {
    const SELECT_TEXT: Color = SELECT_WARM;
    const COLUMN_GAP: usize = 2;

    let content_width = picker_content_width(width);
    let description_width = content_width
        .saturating_sub(label_column_width)
        .saturating_sub(COLUMN_GAP);
    let rendered_label = truncate_for_display_width(label, label_column_width.max(1));
    let rendered_description = if description_width >= 8 {
        truncate_for_display_width(description, description_width)
    } else {
        String::new()
    };
    let label_padding = label_column_width.saturating_sub(display_width(&rendered_label));
    let description_gap = if rendered_description.is_empty() {
        0
    } else {
        COLUMN_GAP
    };
    let plain = format!(
        "{}{}{}{}",
        rendered_label,
        " ".repeat(label_padding),
        " ".repeat(description_gap),
        rendered_description
    );
    let fill_width = content_width.saturating_sub(display_width(&plain));

    let prefix_style = if selected {
        Style::default()
            .fg(SELECT_TEXT)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(FAINT)
    };
    let label_style = if selected {
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(ASSISTANT).add_modifier(Modifier::BOLD)
    };
    let description_style = if selected {
        Style::default().fg(TEXT)
    } else {
        Style::default().fg(MUTED)
    };

    let mut spans = vec![Span::styled(
        if selected { "> " } else { "  " },
        prefix_style,
    )];
    spans.push(Span::styled(rendered_label, label_style));
    if label_padding > 0 {
        spans.push(Span::styled(" ".repeat(label_padding), description_style));
    }
    if description_gap > 0 {
        spans.push(Span::styled(" ".repeat(description_gap), description_style));
    }
    if !rendered_description.is_empty() {
        spans.push(Span::styled(rendered_description, description_style));
    }
    if fill_width > 0 {
        spans.push(Span::styled(" ".repeat(fill_width), description_style));
    }
    Line::from(spans)
}

pub(super) fn render_command_menu_rows(
    menu: &VisibleCommandMenu,
    width: u16,
) -> Vec<Line<'static>> {
    if menu.entries.is_empty() {
        return vec![Line::from(Span::styled(
            match menu.kind {
                MenuKind::Commands => "No matching command",
                MenuKind::AttachPath => "No matching path",
            },
            Style::default().fg(MUTED),
        ))];
    }

    let selected = menu.selected.unwrap_or(0);
    let start = if menu.entries.len() <= COMMAND_MENU_MAX_ROWS {
        0
    } else {
        selected
            .saturating_sub(COMMAND_MENU_MAX_ROWS / 2)
            .min(menu.entries.len().saturating_sub(COMMAND_MENU_MAX_ROWS))
    };
    let end = (start + COMMAND_MENU_MAX_ROWS).min(menu.entries.len());
    let content_width = picker_content_width(width);
    let labels: Vec<String> = menu.entries[start..end]
        .iter()
        .map(ComposerMenuEntry::label)
        .collect();
    let label_column_width = labels
        .iter()
        .map(|label| display_width(label))
        .max()
        .unwrap_or(0)
        .min(content_width.saturating_sub(8))
        .max(4);
    menu.entries[start..end]
        .iter()
        .zip(labels.iter())
        .enumerate()
        .map(|(index, (entry, label))| {
            command_menu_item_line(
                label,
                entry.description(),
                start + index == selected,
                width,
                label_column_width,
            )
        })
        .collect()
}

pub(super) fn picker_kind_noun(kind: &PickerKind) -> &'static str {
    match kind {
        PickerKind::Key => "keys",
        PickerKind::Model { .. } => "models",
        PickerKind::Session => "chats",
    }
}

pub(super) fn picker_search_placeholder(kind: &PickerKind) -> &'static str {
    match kind {
        PickerKind::Key => "filter key name or endpoint",
        PickerKind::Model { .. } => "filter model names",
        PickerKind::Session => "filter saved chats",
    }
}

pub(super) fn key_search_text(key: &ApiKey) -> String {
    format!(
        "{} {} {}",
        key.id,
        key.display_name(),
        footer_host_label(&key.base_url)
    )
}

pub(super) fn key_picker_item_line(key: &ApiKey, selected: bool, width: u16) -> Line<'static> {
    const SEPARATOR: &str = " · ";

    let name = key.display_name().to_string();
    let endpoint = key.base_url.clone();
    let content_width = picker_content_width(width);
    let separator_width = display_width(SEPARATOR);
    let name_width = display_width(&name);
    let max_endpoint_width = content_width.saturating_sub(name_width + separator_width);

    let (rendered_name, rendered_endpoint) = if max_endpoint_width >= 12 {
        (
            name,
            truncate_for_display_width(&endpoint, max_endpoint_width.max(1)),
        )
    } else {
        let combined = format!("{}{}{}", key.display_name(), SEPARATOR, key.base_url);
        (
            truncate_for_display_width(&combined, content_width),
            String::new(),
        )
    };

    let plain = if rendered_endpoint.is_empty() {
        rendered_name.clone()
    } else {
        format!("{rendered_name}{SEPARATOR}{rendered_endpoint}")
    };
    let fill_width = content_width.saturating_sub(display_width(&plain));

    let fill_style = if selected {
        Style::default().bg(SELECT_BG)
    } else {
        Style::default()
    };
    let prefix_style = if selected {
        fill_style.fg(SELECT_ACCENT).add_modifier(Modifier::BOLD)
    } else {
        fill_style
    };
    let name_style = if selected {
        Style::default()
            .fg(SELECT_TEXT)
            .bg(SELECT_BG)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
    };
    let endpoint_style = if selected {
        Style::default().fg(SELECT_ACCENT).bg(SELECT_BG)
    } else {
        Style::default().fg(MUTED)
    };

    let mut spans = vec![Span::styled(
        if selected { "> " } else { "  " },
        prefix_style,
    )];
    spans.push(Span::styled(rendered_name, name_style));
    if !rendered_endpoint.is_empty() {
        spans.push(Span::styled(SEPARATOR, endpoint_style));
        spans.push(Span::styled(rendered_endpoint, endpoint_style));
    }
    spans.push(Span::styled(" ".repeat(fill_width), fill_style));
    Line::from(spans)
}

pub(super) fn session_picker_item_lines(
    preview: &SessionPreview,
    selected: bool,
    armed_delete: bool,
    width: u16,
) -> Vec<Line<'static>> {
    const SELECT_TIME: Color = SELECT_WARM;
    const DELETE_BG: Color = Color::Rgb(104, 63, 63);
    const DELETE_TEXT: Color = Color::Rgb(255, 241, 233);
    const DELETE_TIME: Color = Color::Rgb(255, 198, 176);

    let time = format_session_time(&preview.updated_at);
    let content_width = picker_content_width(width);
    let time_width = display_width(&time);
    let preview_width = content_width
        .saturating_sub(time_width.saturating_add(2))
        .max(1);
    let summary = truncate_for_display_width(&preview.preview_text, preview_width);
    let summary_width = display_width(&summary);
    let gap_width = content_width
        .saturating_sub(summary_width + time_width)
        .max(1);

    let (active_bg, active_text, active_time) = if armed_delete {
        (DELETE_BG, DELETE_TEXT, DELETE_TIME)
    } else {
        (SELECT_BG, SELECT_TEXT, SELECT_TIME)
    };

    let line_style = if selected {
        Style::default()
            .fg(active_text)
            .bg(active_bg)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(TEXT)
    };
    let time_style = if selected {
        Style::default()
            .fg(active_time)
            .bg(active_bg)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(ACCENT)
    };
    let fill_style = if selected {
        Style::default().bg(active_bg)
    } else {
        Style::default()
    };

    vec![Line::from(vec![
        Span::styled(
            if armed_delete {
                "! "
            } else if selected {
                "> "
            } else {
                "  "
            },
            if selected {
                fill_style.fg(active_time).add_modifier(Modifier::BOLD)
            } else {
                fill_style
            },
        ),
        Span::styled(summary, line_style),
        Span::styled(" ".repeat(gap_width), fill_style),
        Span::styled(time, time_style),
    ])]
}

pub(super) fn render_session_picker_rows(
    picker: &PickerState,
    max_rows: usize,
    width: u16,
) -> (Vec<Line<'static>>, Vec<Option<usize>>) {
    let filtered = picker.filtered_items();
    if filtered.is_empty() || max_rows == 0 {
        let msg = if picker.items.is_empty() {
            "No saved chats yet"
        } else {
            "No matches"
        };
        return (
            vec![Line::from(Span::styled(msg, Style::default().fg(MUTED)))],
            Vec::new(),
        );
    }

    let mut all_rows: Vec<(Line<'static>, Option<usize>)> = Vec::new();
    let mut previous_group = String::new();
    for (filtered_index, (_, item)) in filtered.iter().enumerate() {
        let PickerValue::Session(preview) = &item.value else {
            continue;
        };
        let group = format_session_group_label(&preview.updated_at);
        if group != previous_group {
            if !all_rows.is_empty() {
                all_rows.push((Line::from(""), None));
            }
            all_rows.push((
                Line::from(Span::styled(
                    group.clone(),
                    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                )),
                None,
            ));
            previous_group = group;
        }
        let selected = filtered_index == picker.selected;
        let armed_delete = selected && picker.delete_is_armed_for_session(preview);
        for line in session_picker_item_lines(preview, selected, armed_delete, width) {
            all_rows.push((line, Some(filtered_index)));
        }
    }

    if all_rows.len() <= max_rows {
        let (lines, row_map): (Vec<_>, Vec<_>) = all_rows.into_iter().unzip();
        return (lines, row_map);
    }

    let selected_row = all_rows
        .iter()
        .position(|(_, index)| *index == Some(picker.selected))
        .unwrap_or(0);
    let mut start = selected_row.saturating_sub(max_rows / 2);
    let mut end = (start + max_rows).min(all_rows.len());
    if end - start < max_rows {
        start = end.saturating_sub(max_rows);
    }
    while start > 0 && all_rows[start].1 == all_rows[start - 1].1 {
        start -= 1;
        end = (start + max_rows).min(all_rows.len());
    }
    if start > 0 && all_rows[start].1.is_some() && all_rows[start - 1].1.is_none() {
        start -= 1;
        end = (start + max_rows).min(all_rows.len());
    }

    let (lines, row_map): (Vec<_>, Vec<_>) = all_rows[start..end].iter().cloned().unzip();
    (lines, row_map)
}

pub(super) fn picker_entry_lines(
    item: &PickerEntry,
    selected: bool,
    width: u16,
) -> Vec<Line<'static>> {
    match &item.value {
        PickerValue::Session(preview) => session_picker_item_lines(preview, selected, false, width),
        PickerValue::Key(key) => vec![key_picker_item_line(key, selected, width)],
        _ => {
            let content_width = usize::from(width.max(1))
                .saturating_sub(PICKER_ROW_PREFIX_WIDTH)
                .max(1);
            let label = truncate_for_display_width(&item.label, content_width);
            let fill_width = content_width.saturating_sub(display_width(&label));
            let fill_style = if selected {
                Style::default().bg(SELECT_BG)
            } else {
                Style::default()
            };
            let prefix_style = if selected {
                fill_style.fg(SELECT_ACCENT).add_modifier(Modifier::BOLD)
            } else {
                fill_style
            };
            let label_style = if selected {
                Style::default()
                    .fg(SELECT_TEXT)
                    .bg(SELECT_BG)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(TEXT)
            };
            vec![Line::from(vec![
                Span::styled(if selected { "> " } else { "  " }, prefix_style),
                Span::styled(label, label_style),
                Span::styled(" ".repeat(fill_width), fill_style),
            ])]
        }
    }
}

pub(super) fn centered_rect(width_pct: u16, height_pct: u16, area: Rect) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - height_pct) / 2),
            Constraint::Percentage(height_pct),
            Constraint::Percentage((100 - height_pct) / 2),
        ])
        .split(area);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - width_pct) / 2),
            Constraint::Percentage(width_pct),
            Constraint::Percentage((100 - width_pct) / 2),
        ])
        .split(vertical[1])[1]
}

pub(super) fn cursor_position(
    text: &str,
    cursor: usize,
    width: u16,
    line_prefix_width: u16,
) -> (u16, u16) {
    let width = usize::from(width.max(1));
    let text_before = &text[..cursor.min(text.len())];
    let line_prefix_width = usize::from(line_prefix_width);
    let mut x = line_prefix_width.min(width.saturating_sub(1));
    let mut y = 0usize;

    for ch in text_before.chars() {
        if ch == '\n' {
            y += 1;
            x = line_prefix_width.min(width.saturating_sub(1));
            continue;
        }

        let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
        if ch_width == 0 {
            continue;
        }
        if x + ch_width > width {
            y += 1;
            x = 0;
        }
        x += ch_width;
    }

    (x as u16, y as u16)
}
