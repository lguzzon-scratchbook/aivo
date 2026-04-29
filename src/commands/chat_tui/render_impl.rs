use super::*;

impl ChatTuiApp {
    pub(super) fn is_transcript_empty(&self) -> bool {
        self.history.is_empty()
            && self.pending_response.is_empty()
            && self.pending_reasoning.is_empty()
            && !self.sending
    }

    pub(super) fn build_transcript(&self) -> RenderedTranscript {
        let mut lines = Vec::new();
        let mut previous_role: Option<&str> = None;

        if self.is_transcript_empty() {
            push_styled_line(&mut lines, "", Style::default());
            return lines.into();
        }

        push_transcript_intro(&mut lines, &self.raw_model, &self.key.base_url, &self.cwd);
        push_message_spacing(&mut lines);

        for message in &self.history {
            if should_add_message_spacing(previous_role, message.role.as_str()) {
                push_message_spacing(&mut lines);
            }
            match message.role.as_str() {
                "user" => render_user_message(&mut lines, &message.content, &message.attachments),
                "assistant" => render_assistant_message(
                    &mut lines,
                    message.reasoning_content.as_deref(),
                    &message.content,
                ),
                other => render_system_message(&mut lines, other, &message.content),
            }
            previous_role = Some(message.role.as_str());
        }

        let has_visible_streaming =
            !self.pending_response.is_empty() || !self.pending_reasoning.is_empty();
        if self.sending && !has_visible_streaming {
            if should_add_message_spacing(previous_role, "assistant") {
                push_message_spacing(&mut lines);
            }
            render_pending_status(
                &mut lines,
                self.frame_tick,
                self.reduce_motion,
                self.request_started_at
                    .map(|started_at| started_at.elapsed())
                    .unwrap_or_default(),
            );
        } else if has_visible_streaming {
            if should_add_message_spacing(previous_role, "assistant") {
                push_message_spacing(&mut lines);
            }
            render_assistant_message(
                &mut lines,
                (!self.pending_reasoning.is_empty()).then_some(self.pending_reasoning.as_str()),
                &self.pending_response,
            );
        }

        if let Some((color, text)) = notice_display(self.notice.as_ref()) {
            push_message_spacing(&mut lines);
            render_notice_line(&mut lines, color, text.as_ref());
        }

        compact_styled_lines(&mut lines);
        lines.into()
    }

    pub(super) fn transcript_intro_lines(&self) -> Vec<String> {
        vec![
            "AIVO Chat".to_string(),
            format!("{} · {}", self.raw_model, self.key.base_url),
            self.cwd.clone(),
        ]
    }

    pub(super) fn render(&mut self, frame: &mut Frame<'_>) {
        let outer = frame.area();
        self.picker_hitbox = None;
        self.transcript_hitbox = None;
        let composer_area = self.render_main(frame, outer);
        if let Some(menu) = self.visible_command_menu() {
            let (area, placement) = command_menu_area(
                composer_area,
                outer,
                menu.entries.len(),
                self.command_menu.placement,
            );
            self.command_menu.placement = Some(placement);
            self.render_command_menu(frame, area, &menu);
        }
        let body = outer;

        match self.overlay.clone() {
            Overlay::Picker(picker) => {
                self.render_picker(frame, centered_rect(68, 72, body), &picker);
            }
            Overlay::Help => {
                self.render_help_overlay(frame, centered_rect(64, 88, body));
            }
            Overlay::None => {}
        }

        self.render_copy_toast(frame, outer);
    }

    pub(super) fn render_main(&mut self, frame: &mut Frame<'_>, area: Rect) -> Rect {
        let composer_height = self.composer_height();
        let footer_height = 1u16;
        let max_transcript_height = area
            .height
            .saturating_sub(composer_height + footer_height)
            .max(1);
        let is_empty = self.is_transcript_empty();
        let transcript = self.build_transcript();
        let visual_rows = wrap_plain_lines(&transcript.plain_lines, area.width.max(1));
        let min_transcript_height = self
            .empty_state_height(area.width.max(1))
            .clamp(1, max_transcript_height);
        let transcript_height = if is_empty {
            min_transcript_height
        } else {
            (visual_rows.len() as u16).clamp(min_transcript_height, max_transcript_height)
        };
        let stack_height = transcript_height
            .saturating_add(composer_height)
            .saturating_add(footer_height)
            .min(area.height.max(1));

        let stack = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(stack_height), Constraint::Min(0)])
            .split(area);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(transcript_height),
                Constraint::Length(composer_height),
                Constraint::Length(footer_height),
            ])
            .split(stack[0]);

        let transcript_area = chunks[0];
        let transcript_total_lines = visual_rows.len();
        let transcript_line_height = transcript_total_lines as u16;
        let transcript_padding =
            if transcript_area.height > 2 && transcript_line_height > transcript_area.height {
                TRANSCRIPT_BOTTOM_PADDING
            } else {
                0
            };
        let transcript_view_area = Rect {
            x: transcript_area.x,
            y: transcript_area.y,
            width: transcript_area.width,
            height: transcript_area
                .height
                .saturating_sub(transcript_padding)
                .max(1),
        };
        let view_height = transcript_view_area.height.max(1);
        let needs_scrollbar = transcript_total_lines > usize::from(view_height);
        let transcript_content_area = Rect {
            x: transcript_view_area.x,
            y: transcript_view_area.y,
            width: if needs_scrollbar {
                transcript_view_area.width.saturating_sub(1).max(1)
            } else {
                transcript_view_area.width
            },
            height: transcript_view_area.height,
        };
        let visual_rows = if needs_scrollbar {
            wrap_plain_lines(
                &transcript.plain_lines,
                transcript_content_area.width.max(1),
            )
        } else {
            visual_rows
        };
        let transcript_total_lines = visual_rows.len();
        let width = transcript_content_area.width.max(1);
        self.transcript_width = width;
        self.transcript_view_height = view_height;
        let max_scroll = transcript_total_lines.saturating_sub(usize::from(view_height));
        if self.follow_output {
            self.transcript_scroll = max_scroll;
        } else {
            self.transcript_scroll = self.transcript_scroll.min(max_scroll);
        }
        self.transcript_hitbox = Some(TranscriptHitbox {
            area: transcript_view_area,
            first_row: self.transcript_scroll,
            rows: visual_rows.clone(),
        });

        frame.render_widget(Clear, chunks[0]);

        if is_empty {
            self.render_empty_state(frame, transcript_area);
        } else {
            let transcript_widget = Paragraph::new(transcript.text)
                .style(Style::default().fg(TEXT))
                .scroll(((self.transcript_scroll.min(u16::MAX as usize)) as u16, 0))
                .wrap(Wrap { trim: false });
            frame.render_widget(transcript_widget, transcript_content_area);
            self.render_transcript_selection_highlight(frame, transcript_content_area);
            let total_lines = transcript_total_lines;
            if total_lines > usize::from(view_height) {
                let mut scrollbar_state =
                    ScrollbarState::new(total_lines.saturating_sub(usize::from(view_height)))
                        .position(self.transcript_scroll);
                let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
                    .thumb_style(Style::default().fg(FAINT))
                    .track_style(Style::default().fg(Color::Rgb(50, 54, 56)))
                    .begin_symbol(None)
                    .end_symbol(None);
                frame.render_stateful_widget(scrollbar, transcript_view_area, &mut scrollbar_state);
            }
        }

        if transcript_padding > 0 {
            frame.render_widget(
                Clear,
                Rect {
                    x: transcript_area.x,
                    y: transcript_view_area
                        .y
                        .saturating_add(transcript_view_area.height),
                    width: transcript_area.width,
                    height: transcript_padding,
                },
            );
        }

        let composer_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(1)])
            .split(chunks[1]);

        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "─".repeat(usize::from(composer_chunks[0].width.max(1))),
                Style::default().fg(FAINT),
            ))),
            composer_chunks[0],
        );

        let composer_area = composer_chunks[1];
        let composer = Paragraph::new(self.render_composer_text()).wrap(Wrap { trim: false });
        frame.render_widget(composer, composer_area);

        if self.should_show_input_cursor() {
            let (cursor_x, cursor_y) = {
                let (x, y) = cursor_position(
                    &self.draft,
                    self.cursor,
                    composer_area.width.max(1),
                    COMPOSER_PREFIX_WIDTH,
                );
                (x, y.saturating_add(self.draft_attachments.len() as u16))
            };
            frame.set_cursor_position((
                composer_area.x + cursor_x,
                composer_area.y + cursor_y.min(composer_area.height.saturating_sub(1)),
            ));
        }

        self.render_footer(frame, chunks[2]);
        composer_area
    }

    pub(super) fn empty_state_height(&self, width: u16) -> u16 {
        let content_width = width.saturating_sub(1).max(1);
        let mut height = if let Some(loading) = &self.loading_resume {
            let mut rows = vec![
                "Loading saved chat…".to_string(),
                loading.preview.title.clone(),
                plain_text_from_spans(&resume_metadata_spans(
                    &loading.preview,
                    content_width.saturating_sub(1).max(1),
                )),
                self.cwd.clone(),
            ];
            rows.extend(self.notice_plain_lines(content_width));
            wrap_plain_lines(&rows, content_width).len() as u16
        } else {
            let mut rows = self.transcript_intro_lines();
            rows.extend(self.notice_plain_lines(content_width));
            wrap_plain_lines(&rows, content_width).len() as u16
        };
        height = height.saturating_add(EMPTY_STATE_BOTTOM_GAP);
        height.max(1)
    }

    fn notice_plain_lines(&self, width: u16) -> Vec<String> {
        notice_display(self.notice.as_ref())
            .map(|(_, text)| {
                let mut lines = vec![String::new()];
                lines.extend(wrap_plain_lines(&[text.into_owned()], width));
                lines
            })
            .unwrap_or_default()
    }

    fn render_transcript_selection_highlight(&self, frame: &mut Frame<'_>, area: Rect) {
        let Some(selection) = self
            .transcript_selection
            .filter(|selection| !selection.is_empty())
        else {
            return;
        };
        let Some(hitbox) = &self.transcript_hitbox else {
            return;
        };

        let (start, end) = normalized_selection(selection);
        let visible_start = hitbox.first_row;
        let visible_end = visible_start.saturating_add(usize::from(area.height));
        let row_start = start.row.max(visible_start);
        let row_end = end.row.min(visible_end.saturating_sub(1));
        if row_start > row_end {
            return;
        }

        let buffer = frame.buffer_mut();
        for row in row_start..=row_end {
            let local_y = row.saturating_sub(visible_start) as u16;
            let start_col = if row == start.row { start.column } else { 0 };
            let end_col = if row == end.row {
                end.column
            } else {
                area.width
            };
            let start_col = start_col.min(area.width);
            let end_col = end_col.min(area.width);
            if start_col >= end_col {
                continue;
            }

            for column in start_col..end_col {
                if let Some(cell) = buffer.cell_mut((area.x + column, area.y + local_y)) {
                    cell.set_bg(SELECT_WARM);
                }
            }
        }
    }

    pub(super) fn render_copy_toast(&mut self, frame: &mut Frame<'_>, area: Rect) {
        let Some(toast) = self.copy_toast.clone() else {
            return;
        };
        let now = Instant::now();
        if now >= toast.expires_at {
            self.copy_toast = None;
            return;
        }

        let text_width = display_width(&toast.text).min(usize::from(area.width));
        let toast_width = (text_width as u16).saturating_add(4).min(area.width.max(1));
        let toast_area = Rect {
            x: area
                .x
                .saturating_add(area.width.saturating_sub(toast_width)),
            y: area.y,
            width: toast_width,
            height: 1,
        };
        let color = if now.duration_since(toast.created_at) >= COPY_TOAST_FADE_AFTER {
            FAINT
        } else {
            ACCENT
        };
        frame.render_widget(Clear, toast_area);
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::raw(" "),
                Span::styled(&toast.text, Style::default().fg(color)),
                Span::raw(" "),
            ]))
            .style(Style::default().bg(Color::Rgb(24, 26, 27))),
            toast_area,
        );
    }

    pub(super) fn render_empty_state(&self, frame: &mut Frame<'_>, area: Rect) {
        let content_area = Rect {
            x: area.x,
            y: area.y,
            width: area.width,
            height: area.height.saturating_sub(EMPTY_STATE_BOTTOM_GAP),
        };

        let lines = if let Some(loading) = &self.loading_resume {
            vec![
                Line::from(vec![
                    Span::styled(
                        "Loading",
                        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        " saved chat…",
                        Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::from(Span::styled(
                    loading.preview.title.clone(),
                    Style::default().fg(TEXT),
                )),
                Line::from(resume_metadata_spans(
                    &loading.preview,
                    area.width.max(1).saturating_sub(2),
                )),
                Line::from(Span::styled(self.cwd.as_str(), Style::default().fg(FAINT))),
            ]
        } else {
            vec![
                Line::from(vec![
                    Span::styled(
                        "AIVO",
                        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        " Chat",
                        Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::from(Span::styled(
                    format!("{} · {}", self.raw_model, self.key.base_url),
                    Style::default().fg(MUTED),
                )),
                Line::from(Span::styled(self.cwd.as_str(), Style::default().fg(FAINT))),
            ]
        };

        let mut lines = lines;
        if let Some((color, text)) = notice_display(self.notice.as_ref()) {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                text.into_owned(),
                Style::default().fg(color),
            )));
        }

        frame.render_widget(
            Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }),
            content_area,
        );
    }

    pub(super) fn render_composer_text(&self) -> Text<'static> {
        let prompt = if self.draft_history_index.is_some() {
            Span::styled(
                "^ ",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            )
        } else {
            Span::styled("> ", Style::default().fg(USER).add_modifier(Modifier::BOLD))
        };
        let mut lines = composer_attachment_lines(&self.draft_attachments);
        if self.draft.is_empty() {
            let placeholder = if self.loading_resume.is_some() {
                Span::styled("Resume loading…", Style::default().fg(FAINT))
            } else if self.sending {
                Span::styled("", Style::default())
            } else {
                Span::styled(" Ask anything · / for commands", Style::default().fg(FAINT))
            };
            lines.push(Line::from(vec![prompt, placeholder]));
            return Text::from(lines);
        }

        for (index, line) in self.draft.split('\n').enumerate() {
            if line.is_empty() && index > 0 {
                lines.push(Line::from(""));
            } else {
                let prefix = if index == 0 {
                    prompt.clone()
                } else {
                    Span::raw("  ")
                };
                lines.push(Line::from(vec![
                    prefix,
                    Span::styled(line.to_string(), Style::default().fg(TEXT)),
                ]));
            }
        }

        Text::from(lines)
    }

    pub(super) fn render_footer(&self, frame: &mut Frame<'_>, area: Rect) {
        let (right_label, right_color) = self.footer_status_label();
        let right_label_width = display_width(&right_label) as u16;
        let left_width = if right_label_width == 0 {
            area.width
        } else {
            area.width.saturating_sub(right_label_width + 1)
        };
        let left_text =
            build_footer_text(&self.raw_model, &self.key.base_url, &self.cwd, left_width);
        let left_len = display_width(&left_text) as u16;
        let pad = left_width.saturating_sub(left_len);
        let mut spans = vec![Span::styled(left_text, Style::default().fg(MUTED))];
        if right_label_width > 0 {
            spans.push(Span::raw(" ".repeat(usize::from(pad) + 1)));
            spans.push(Span::styled(right_label, Style::default().fg(right_color)));
        }
        frame.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    pub(super) fn footer_status_label(&self) -> (String, Color) {
        (
            format_token_count(self.context_tokens, self.last_usage),
            MUTED,
        )
    }

    pub(super) fn composer_height(&self) -> u16 {
        let lines = (self.draft.split('\n').count().max(1) + self.draft_attachments.len()) as u16;
        (lines + 2).clamp(3, 9)
    }
}
