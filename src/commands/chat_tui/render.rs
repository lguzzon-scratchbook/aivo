use super::*;

pub(super) struct StyledLine {
    pub(super) line: Line<'static>,
    pub(super) plain: String,
}

pub(super) struct RenderedTranscript {
    pub(super) text: Text<'static>,
}

impl From<Vec<StyledLine>> for RenderedTranscript {
    fn from(lines: Vec<StyledLine>) -> Self {
        let text = Text::from(lines.into_iter().map(|line| line.line).collect::<Vec<_>>());
        Self { text }
    }
}

pub(super) fn push_message_spacing(lines: &mut Vec<StyledLine>) {
    if !lines.is_empty() {
        lines.push(blank_line());
    }
}

pub(super) fn push_transcript_intro(
    lines: &mut Vec<StyledLine>,
    raw_model: &str,
    base_url: &str,
    cwd: &str,
) {
    lines.push(line_with_plain(vec![
        Span::styled(
            "AIVO".to_string(),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            " Chat".to_string(),
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        ),
    ]));
    push_styled_line(
        lines,
        format!("{raw_model} · {base_url}"),
        Style::default().fg(MUTED),
    );
    push_styled_line(lines, cwd.to_string(), Style::default().fg(FAINT));
}

pub(super) fn should_add_message_spacing(previous_role: Option<&str>, next_role: &str) -> bool {
    previous_role.is_some() && !next_role.is_empty()
}

pub(super) fn attachment_kind_label(attachment: &MessageAttachment) -> &'static str {
    if attachment.mime_type.starts_with("image/") {
        "image"
    } else {
        "file"
    }
}

pub(super) fn render_user_attachment_lines(
    lines: &mut Vec<StyledLine>,
    attachments: &[MessageAttachment],
) {
    for attachment in attachments {
        push_styled_line(
            lines,
            format!(
                "  [{}] {}",
                attachment_kind_label(attachment),
                attachment.name
            ),
            Style::default().fg(MUTED),
        );
    }
}

pub(super) fn composer_attachment_lines(attachments: &[MessageAttachment]) -> Vec<Line<'static>> {
    attachments
        .iter()
        .enumerate()
        .map(|(index, attachment)| {
            Line::from(vec![
                Span::styled("· ", Style::default().fg(ACCENT)),
                Span::styled(
                    format!(
                        "{}. [{}] {}",
                        index + 1,
                        attachment_kind_label(attachment),
                        attachment.name
                    ),
                    Style::default().fg(MUTED),
                ),
            ])
        })
        .collect()
}

pub(super) fn render_user_message(
    lines: &mut Vec<StyledLine>,
    content: &str,
    attachments: &[MessageAttachment],
) {
    let mut had_line = false;
    for (idx, raw_line) in content.lines().enumerate() {
        let prefix = if idx == 0 { "> " } else { "  " };
        push_styled_line(
            lines,
            format!("{prefix}{raw_line}"),
            Style::default().fg(USER),
        );
        had_line = true;
    }
    if !had_line {
        push_styled_line(lines, "> ", Style::default().fg(USER));
    }
    render_user_attachment_lines(lines, attachments);
}

pub(super) fn render_reasoning_block(
    lines: &mut Vec<StyledLine>,
    reasoning: &str,
    show_reasoning: bool,
) {
    if !show_reasoning {
        return;
    }

    lines.push(line_with_plain(vec![Span::styled(
        "Thinking".to_string(),
        Style::default().fg(MUTED).add_modifier(Modifier::BOLD),
    )]));

    let reasoning_lines = normalized_reasoning_lines(reasoning);
    let mut had_line = false;
    for raw_line in reasoning_lines {
        lines.push(line_with_plain(vec![
            Span::styled("  ".to_string(), Style::default()),
            Span::styled(
                raw_line,
                Style::default().fg(MUTED).add_modifier(Modifier::ITALIC),
            ),
        ]));
        had_line = true;
    }

    if !had_line {
        push_styled_line(
            lines,
            "  ".to_string(),
            Style::default().fg(MUTED).add_modifier(Modifier::ITALIC),
        );
    }
}

pub(super) fn normalized_reasoning_lines(reasoning: &str) -> Vec<String> {
    let mut lines = Vec::new();

    for raw_line in reasoning.lines() {
        let trimmed = raw_line.trim();
        if !trimmed.is_empty() {
            lines.push(trimmed.to_string());
        }
    }

    lines
}

pub(super) fn extend_without_leading_blank(
    lines: &mut Vec<StyledLine>,
    mut rendered: Vec<StyledLine>,
) {
    while rendered
        .first()
        .is_some_and(|line| line.plain.trim().is_empty())
    {
        rendered.remove(0);
    }
    lines.extend(rendered);
}

pub(super) fn render_assistant_message(
    lines: &mut Vec<StyledLine>,
    show_reasoning: bool,
    reasoning: Option<&str>,
    content: &str,
) {
    if let Some(reasoning) = reasoning.filter(|text| !text.trim().is_empty()) {
        render_reasoning_block(lines, reasoning, show_reasoning);
        if show_reasoning && !content.is_empty() {
            push_styled_line(lines, "", Style::default());
        }
    }

    if !content.is_empty() {
        extend_without_leading_blank(lines, render_markdown_lines(content));
    }
}

pub(super) fn render_pending_status(
    lines: &mut Vec<StyledLine>,
    frame_tick: usize,
    reduce_motion: bool,
    elapsed: Duration,
) {
    let spinner = spinner_frame_indexed(frame_tick, reduce_motion);
    let text = format!(
        "{spinner} Thinking ({} • esc to interrupt)",
        format_request_elapsed(elapsed)
    );
    push_styled_line(
        lines,
        text,
        Style::default().fg(MUTED).add_modifier(Modifier::ITALIC),
    );
}

pub(super) fn spinner_frame_indexed(frame_tick: usize, reduce_motion: bool) -> &'static str {
    if reduce_motion {
        return spinner_frame(0);
    }
    spinner_frame(frame_tick / 5)
}

pub(super) fn error_notice(notice: Option<&(Color, String)>) -> Option<&str> {
    notice
        .filter(|(color, _)| *color == ERROR)
        .map(|(_, text)| text.as_str())
}

pub(super) fn rect_contains(area: Rect, point: (u16, u16)) -> bool {
    let (x, y) = point;
    x >= area.x
        && x < area.x.saturating_add(area.width)
        && y >= area.y
        && y < area.y.saturating_add(area.height)
}

pub(super) fn render_error_notice(lines: &mut Vec<StyledLine>, error: &str) {
    push_styled_line(lines, format!("Error: {error}"), Style::default().fg(ERROR));
}

pub(super) fn render_system_message(lines: &mut Vec<StyledLine>, role: &str, content: &str) {
    push_styled_line(
        lines,
        role.to_string(),
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
    );
    if !content.is_empty() {
        extend_without_leading_blank(lines, render_markdown_lines(content));
    }
}

pub(super) fn render_markdown_lines(content: &str) -> Vec<StyledLine> {
    let mut options = MdOptions::empty();
    options.insert(MdOptions::ENABLE_STRIKETHROUGH);
    options.insert(MdOptions::ENABLE_TABLES);
    options.insert(MdOptions::ENABLE_TASKLISTS);
    let parser = Parser::new_ext(content, options);
    let mut renderer = MarkdownRenderer::new();

    for event in parser {
        renderer.push_event(event);
    }

    renderer.finish()
}

pub(super) struct MarkdownRenderer {
    lines: Vec<StyledLine>,
    current_spans: Vec<Span<'static>>,
    current_plain: String,
    inline_style: InlineStyle,
    heading: Option<HeadingLevel>,
    quote_depth: usize,
    list_stack: Vec<ListState>,
    item_prefix: Option<String>,
    code_block: Option<CodeFence>,
}

impl MarkdownRenderer {
    fn new() -> Self {
        Self {
            lines: Vec::new(),
            current_spans: Vec::new(),
            current_plain: String::new(),
            inline_style: InlineStyle::default(),
            heading: None,
            quote_depth: 0,
            list_stack: Vec::new(),
            item_prefix: None,
            code_block: None,
        }
    }

    fn finish(mut self) -> Vec<StyledLine> {
        self.flush_line();
        self.lines
    }

    fn push_event(&mut self, event: MdEvent<'_>) {
        match event {
            MdEvent::Start(tag) => self.start_tag(tag),
            MdEvent::End(tag) => self.end_tag(tag),
            MdEvent::Text(text) => self.push_text(text.as_ref()),
            MdEvent::Code(text) => self.push_inline_code(text.as_ref()),
            MdEvent::SoftBreak | MdEvent::HardBreak => self.flush_line(),
            MdEvent::Rule => {
                self.flush_line();
                self.lines.push(line_plain(
                    "────────────────────────────────".to_string(),
                    Style::default().fg(FAINT),
                ));
            }
            MdEvent::Html(text) | MdEvent::InlineHtml(text) => self.push_text(text.as_ref()),
            MdEvent::FootnoteReference(text) => self.push_text(text.as_ref()),
            MdEvent::TaskListMarker(checked) => {
                self.ensure_prefix();
                let marker = if checked { "☑ " } else { "☐ " };
                self.push_span(marker.to_string(), Style::default().fg(ACCENT));
            }
            _ => {}
        }
    }

    fn start_tag(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Paragraph => {}
            Tag::Heading { level, .. } => {
                self.flush_line();
                self.heading = Some(level);
            }
            Tag::BlockQuote(_) => {
                self.flush_line();
                self.quote_depth += 1;
            }
            Tag::List(start) => {
                self.flush_line();
                self.list_stack.push(ListState::new(start));
            }
            Tag::Item => {
                self.flush_line();
                self.item_prefix = Some(self.next_item_prefix());
            }
            Tag::CodeBlock(kind) => {
                self.flush_line();
                self.code_block = Some(CodeFence::new(kind));
            }
            Tag::Emphasis => self.inline_style.emphasis += 1,
            Tag::Strong => self.inline_style.strong += 1,
            Tag::Strikethrough => self.inline_style.strike += 1,
            Tag::Link { .. } => self.inline_style.link += 1,
            _ => {}
        }
    }

    fn end_tag(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph => {
                self.flush_line();
            }
            TagEnd::Heading(_) => {
                self.flush_line();
                self.heading = None;
                self.lines.push(blank_line());
            }
            TagEnd::BlockQuote(_) => {
                self.flush_line();
                self.quote_depth = self.quote_depth.saturating_sub(1);
                self.lines.push(blank_line());
            }
            TagEnd::List(_) => {
                self.flush_line();
                self.list_stack.pop();
                self.lines.push(blank_line());
            }
            TagEnd::Item => {
                self.flush_line();
                self.item_prefix = None;
            }
            TagEnd::CodeBlock => {
                if let Some(block) = self.code_block.take() {
                    self.emit_code_block(block);
                    self.lines.push(blank_line());
                }
            }
            TagEnd::Emphasis => {
                self.inline_style.emphasis = self.inline_style.emphasis.saturating_sub(1)
            }
            TagEnd::Strong => self.inline_style.strong = self.inline_style.strong.saturating_sub(1),
            TagEnd::Strikethrough => {
                self.inline_style.strike = self.inline_style.strike.saturating_sub(1)
            }
            TagEnd::Link => self.inline_style.link = self.inline_style.link.saturating_sub(1),
            _ => {}
        }
    }

    fn emit_code_block(&mut self, block: CodeFence) {
        let label = if block.language.is_empty() {
            "code".to_string()
        } else {
            block.language
        };
        self.lines.push(line_plain(
            format!("  {label}"),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ));

        let content = if block.content.is_empty() {
            String::new()
        } else {
            block.content
        };
        for raw_line in content.lines() {
            self.lines.push(line_plain(
                format!("  {raw_line}"),
                Style::default().fg(TEXT),
            ));
        }
        if content.is_empty() || content.ends_with('\n') {
            self.lines
                .push(line_plain("  ".to_string(), Style::default().fg(TEXT)));
        }
    }

    fn next_item_prefix(&mut self) -> String {
        if let Some(list) = self.list_stack.last_mut() {
            list.take_prefix()
        } else {
            "• ".to_string()
        }
    }

    fn push_text(&mut self, text: &str) {
        if let Some(block) = &mut self.code_block {
            block.content.push_str(text);
            return;
        }

        for (idx, part) in text.split('\n').enumerate() {
            if idx > 0 {
                self.flush_line();
            }
            if !part.is_empty() {
                self.ensure_prefix();
                self.push_span(part.to_string(), self.current_style());
            }
        }
    }

    fn push_inline_code(&mut self, text: &str) {
        self.ensure_prefix();
        self.push_span(
            format!(" {text} "),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        );
    }

    fn current_style(&self) -> Style {
        let mut style = Style::default().fg(TEXT);
        if self.inline_style.emphasis > 0 {
            style = style.add_modifier(Modifier::ITALIC);
        }
        if self.inline_style.strong > 0 {
            style = style.add_modifier(Modifier::BOLD);
        }
        if self.inline_style.strike > 0 {
            style = style.add_modifier(Modifier::CROSSED_OUT);
        }
        if self.inline_style.link > 0 {
            style = style.fg(LINK).add_modifier(Modifier::UNDERLINED);
        }
        if let Some(level) = self.heading {
            style = heading_style(level);
        }
        style
    }

    fn ensure_prefix(&mut self) {
        if !self.current_spans.is_empty() || !self.current_plain.is_empty() {
            return;
        }
        if self.quote_depth > 0 {
            let prefix = format!("{} ", "▎".repeat(self.quote_depth));
            self.push_span(prefix, Style::default().fg(QUOTE));
        }
        if let Some(prefix) = self.item_prefix.take() {
            self.push_span(prefix, Style::default().fg(ACCENT));
        }
    }

    fn push_span(&mut self, text: String, style: Style) {
        self.current_plain.push_str(&text);
        self.current_spans.push(Span::styled(text, style));
    }

    fn flush_line(&mut self) {
        if self.current_spans.is_empty() {
            if !self.lines.last().is_some_and(|line| line.plain.is_empty()) {
                self.lines.push(blank_line());
            }
            self.current_plain.clear();
            return;
        }
        let line = StyledLine {
            line: Line::from(std::mem::take(&mut self.current_spans)),
            plain: std::mem::take(&mut self.current_plain),
        };
        self.lines.push(line);
    }
}

#[derive(Default)]
pub(super) struct InlineStyle {
    emphasis: usize,
    strong: usize,
    strike: usize,
    link: usize,
}

pub(super) struct ListState {
    next_number: Option<u64>,
}

impl ListState {
    fn new(start: Option<u64>) -> Self {
        Self { next_number: start }
    }

    fn take_prefix(&mut self) -> String {
        match self.next_number {
            Some(number) => {
                self.next_number = Some(number + 1);
                format!("{number}. ")
            }
            None => "• ".to_string(),
        }
    }
}

pub(super) struct CodeFence {
    language: String,
    content: String,
}

impl CodeFence {
    fn new(kind: CodeBlockKind<'_>) -> Self {
        let language = match kind {
            CodeBlockKind::Indented => String::new(),
            CodeBlockKind::Fenced(name) => name.to_string(),
        };
        Self {
            language,
            content: String::new(),
        }
    }
}

pub(super) fn heading_style(level: HeadingLevel) -> Style {
    match level {
        HeadingLevel::H1 => Style::default()
            .fg(ACCENT)
            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
        HeadingLevel::H2 => Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        HeadingLevel::H3 => Style::default().fg(ASSISTANT).add_modifier(Modifier::BOLD),
        _ => Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
    }
}

pub(super) fn blank_line() -> StyledLine {
    line_plain(String::new(), Style::default())
}

pub(super) fn line_plain(text: String, style: Style) -> StyledLine {
    StyledLine {
        plain: text.clone(),
        line: Line::from(Span::styled(text, style)),
    }
}

pub(super) fn line_with_plain(spans: Vec<Span<'static>>) -> StyledLine {
    let mut plain = String::new();
    for span in &spans {
        plain.push_str(span.content.as_ref());
    }
    StyledLine {
        line: Line::from(spans),
        plain,
    }
}

pub(super) fn push_styled_line(lines: &mut Vec<StyledLine>, text: impl Into<String>, style: Style) {
    lines.push(line_plain(text.into(), style));
}

pub(super) fn compact_styled_lines(lines: &mut Vec<StyledLine>) {
    let mut compacted = Vec::with_capacity(lines.len());
    let mut last_was_blank = true;

    for line in lines.drain(..) {
        let is_blank = line.plain.trim().is_empty();
        if is_blank && last_was_blank {
            continue;
        }
        last_was_blank = is_blank;
        compacted.push(line);
    }

    while compacted
        .last()
        .is_some_and(|line| line.plain.trim().is_empty())
    {
        compacted.pop();
    }

    *lines = compacted;
}
