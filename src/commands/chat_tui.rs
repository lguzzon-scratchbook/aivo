use std::env;
use std::io::{self, Stdout};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use anyhow::Result;
use chrono::{DateTime, Local, Utc};
use crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event, KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use pulldown_cmark::{
    CodeBlockKind, Event as MdEvent, HeadingLevel, Options as MdOptions, Parser, Tag, TagEnd,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{
    Block, Borders, Clear, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap,
};
use ratatui::{Frame, Terminal};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::task::JoinHandle;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::style::spinner_frame;
use crate::tui::matches_fuzzy;

use super::*;

const TEXT: Color = Color::Rgb(224, 225, 221);
const MUTED: Color = Color::Rgb(136, 142, 139);
const FAINT: Color = Color::Rgb(92, 99, 102);
const ACCENT: Color = Color::Rgb(208, 180, 132);
const ASSISTANT: Color = Color::Rgb(174, 202, 161);
const USER: Color = Color::Rgb(166, 193, 226);
const LINK: Color = Color::Rgb(142, 181, 219);
const QUOTE: Color = Color::Rgb(143, 164, 146);
const ERROR: Color = Color::Rgb(230, 134, 128);
const EMPTY_STATE_BOTTOM_GAP: u16 = 1;
const TRANSCRIPT_BOTTOM_PADDING: u16 = 1;
const COMPOSER_PREFIX_WIDTH: u16 = 2;

const COMMAND_MENU_MAX_ROWS: usize = 7;
const PICKER_ROW_PREFIX_WIDTH: usize = 2;
const SELECT_WARM: Color = Color::Rgb(255, 228, 194);

#[derive(Clone, Copy)]
struct SlashCommandSpec {
    name: &'static str,
    help_label: &'static str,
    description: &'static str,
    takes_argument: bool,
}

impl SlashCommandSpec {
    fn command_label(self) -> String {
        format!("/{}", self.name)
    }

    fn insertion_text(self) -> String {
        let suffix = if self.takes_argument { " " } else { "" };
        format!("/{}{}", self.name, suffix)
    }
}

const SLASH_COMMANDS: &[SlashCommandSpec] = &[
    SlashCommandSpec {
        name: "new",
        help_label: "/new",
        description: "start a fresh chat",
        takes_argument: false,
    },
    SlashCommandSpec {
        name: "exit",
        help_label: "/exit",
        description: "leave chat",
        takes_argument: false,
    },
    SlashCommandSpec {
        name: "resume",
        help_label: "/resume [query]",
        description: "resume a saved chat",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "model",
        help_label: "/model [name]",
        description: "switch model",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "key",
        help_label: "/key [id|name]",
        description: "switch saved key",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "help",
        help_label: "/help",
        description: "open help",
        takes_argument: false,
    },
];

pub(super) struct ChatTuiParams {
    pub session_store: SessionStore,
    pub cache: ModelsCache,
    pub client: Client,
    pub key: ApiKey,
    pub copilot_tm: Option<Arc<CopilotTokenManager>>,
    pub cwd: String,
    pub raw_model: String,
    pub model: String,
    pub initial_session: String,
    pub initial_history: Vec<ChatMessage>,
    pub startup_notice: Option<String>,
}

#[derive(Clone)]
struct SessionPreview {
    key_id: String,
    key_name: String,
    base_url: String,
    session_id: String,
    raw_model: String,
    updated_at: String,
    title: String,
    preview_text: String,
}

fn decrypt_to_chat_messages(
    state: &crate::services::session_store::ChatSessionState,
) -> Result<Vec<ChatMessage>> {
    let messages = state
        .decrypt_messages()?
        .into_iter()
        .map(|m| ChatMessage {
            role: m.role,
            content: m.content,
            reasoning_content: m.reasoning_content,
        })
        .collect();
    Ok(messages)
}

impl SessionPreview {
    fn from_index_entry(
        entry: crate::services::session_store::SessionIndexEntry,
        key: &ApiKey,
    ) -> Self {
        Self {
            key_id: key.id.clone(),
            key_name: key.display_name().to_string(),
            base_url: key.base_url.clone(),
            session_id: entry.session_id,
            raw_model: entry.model,
            updated_at: entry.updated_at,
            title: entry.title,
            preview_text: entry.preview,
        }
    }

    fn search_text(&self) -> String {
        format!(
            "{} {} {} {} {} {}",
            self.session_id,
            self.title,
            self.preview_text,
            self.key_name,
            self.raw_model,
            self.base_url
        )
    }
}

#[derive(Clone)]
struct LoadedSession {
    key_id: String,
    session_id: String,
    raw_model: String,
    messages: Vec<ChatMessage>,
}

impl LoadedSession {
    fn from_state(state: crate::services::session_store::ChatSessionState) -> Result<Self> {
        let messages = decrypt_to_chat_messages(&state)?;

        Ok(Self {
            key_id: state.key_id,
            session_id: state.session_id,
            raw_model: state.model,
            messages,
        })
    }
}

#[derive(Clone)]
enum Overlay {
    None,
    Help,
    Picker(Box<PickerState>),
}

impl Overlay {
    fn blocks_input(&self) -> bool {
        !matches!(self, Self::None)
    }
}

#[derive(Clone)]
enum PickerValue {
    Model(String),
    Key(ApiKey),
    Session(SessionPreview),
}

#[derive(Clone)]
struct PickerEntry {
    label: String,
    search_text: String,
    value: PickerValue,
}

impl PickerEntry {
    fn row_height(&self) -> usize {
        1
    }
}

#[derive(Clone)]
enum ModelSelectionTarget {
    CurrentChat,
    KeySwitch(ApiKey),
}

#[derive(Clone)]
enum PickerKind {
    Model {
        target: ModelSelectionTarget,
        auto_accept_exact: bool,
    },
    Key,
    Session,
}

#[derive(Clone)]
struct PickerState {
    title: &'static str,
    query: String,
    items: Vec<PickerEntry>,
    loading: bool,
    selected: usize,
    kind: PickerKind,
    pending_delete: Option<DeleteConfirmTarget>,
}

#[derive(Clone, Default)]
struct PickerHitbox {
    overlay_area: Rect,
    list_area: Rect,
    row_to_filtered_index: Vec<Option<usize>>,
}

#[derive(Clone)]
struct LoadingResume {
    request_id: u64,
    preview: SessionPreview,
}

#[derive(Clone, PartialEq, Eq)]
struct DeleteConfirmTarget {
    key_id: String,
    session_id: String,
}

#[derive(Clone)]
struct ResumeRestoreState {
    key: ApiKey,
    copilot_tm: Option<Arc<CopilotTokenManager>>,
    raw_model: String,
    model: String,
    format: ChatFormat,
    history: Vec<ChatMessage>,
    draft: String,
    cursor: usize,
    command_menu: CommandMenuState,
    draft_history_index: Option<usize>,
    draft_history_stash: Option<String>,
    session_id: String,
    notice: Option<(Color, String)>,
    show_reasoning: bool,
    pending_response: String,
    pending_reasoning: String,
    pending_submit: Option<String>,
    last_usage: Option<TokenUsage>,
    context_tokens: u64,
    follow_output: bool,
    transcript_scroll: usize,
}

#[derive(Clone, Default)]
struct CommandMenuState {
    query: String,
    selected: usize,
    dismissed: bool,
    placement: Option<CommandMenuPlacement>,
    last_row_count: usize,
}

impl CommandMenuState {
    fn reset(&mut self) {
        self.query.clear();
        self.selected = 0;
        self.dismissed = false;
        self.placement = None;
        self.last_row_count = 0;
    }
}

struct VisibleCommandMenu {
    entries: Vec<&'static SlashCommandSpec>,
    selected: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CommandMenuPlacement {
    Above,
    Below,
}

impl ResumeRestoreState {
    fn capture(app: &ChatTuiApp) -> Self {
        Self {
            key: app.key.clone(),
            copilot_tm: app.copilot_tm.clone(),
            raw_model: app.raw_model.clone(),
            model: app.model.clone(),
            format: app.format.clone(),
            history: app.history.clone(),
            draft: app.draft.clone(),
            cursor: app.cursor,
            command_menu: app.command_menu.clone(),
            draft_history_index: app.draft_history_index,
            draft_history_stash: app.draft_history_stash.clone(),
            session_id: app.session_id.clone(),
            notice: app.notice.clone(),
            show_reasoning: app.show_reasoning,
            pending_response: app.pending_response.clone(),
            pending_reasoning: app.pending_reasoning.clone(),
            pending_submit: app.pending_submit.clone(),
            last_usage: app.last_usage,
            context_tokens: app.context_tokens,
            follow_output: app.follow_output,
            transcript_scroll: app.transcript_scroll,
        }
    }
}

impl PickerState {
    fn loading(title: &'static str, query: String, kind: PickerKind) -> Self {
        Self {
            title,
            query,
            items: Vec::new(),
            loading: true,
            selected: 0,
            kind,
            pending_delete: None,
        }
    }

    fn ready(
        title: &'static str,
        query: String,
        items: Vec<PickerEntry>,
        kind: PickerKind,
    ) -> Self {
        Self {
            title,
            query,
            items,
            loading: false,
            selected: 0,
            kind,
            pending_delete: None,
        }
    }

    fn filtered_items(&self) -> Vec<(usize, &PickerEntry)> {
        self.items
            .iter()
            .enumerate()
            .filter(|(_, item)| matches_fuzzy(&self.query, &item.search_text))
            .collect()
    }

    fn exact_match_index(&self) -> Option<usize> {
        let PickerKind::Model {
            auto_accept_exact, ..
        } = &self.kind
        else {
            return None;
        };
        if !*auto_accept_exact || self.query.is_empty() {
            return None;
        }
        self.filtered_items().iter().position(
            |(_, item)| matches!(&item.value, PickerValue::Model(model) if model == &self.query),
        )
    }

    fn visible_items(&self, max_rows: usize) -> Vec<(usize, &PickerEntry)> {
        let filtered = self.filtered_items();
        if filtered.is_empty() || max_rows == 0 {
            return Vec::new();
        }

        let selected = self.selected.min(filtered.len().saturating_sub(1));
        let mut start = selected;
        let mut used_rows = filtered[selected].1.row_height();

        while start > 0 {
            let next_height = filtered[start - 1].1.row_height();
            if used_rows + next_height > max_rows {
                break;
            }
            used_rows += next_height;
            start -= 1;
        }

        let mut end = selected + 1;
        while end < filtered.len() {
            let next_height = filtered[end].1.row_height();
            if used_rows + next_height > max_rows {
                break;
            }
            used_rows += next_height;
            end += 1;
        }

        filtered[start..end]
            .iter()
            .enumerate()
            .map(|(offset, (_, item))| (start + offset, *item))
            .collect()
    }

    fn select_prev(&mut self) {
        let len = self.filtered_items().len();
        if len == 0 {
            self.selected = 0;
        } else if self.selected == 0 {
            self.selected = len - 1;
        } else {
            self.selected -= 1;
        }
    }

    fn select_next(&mut self) {
        let len = self.filtered_items().len();
        if len > 0 {
            self.selected = if self.selected + 1 >= len {
                0
            } else {
                self.selected + 1
            };
        }
    }

    fn clear_pending_delete(&mut self) {
        self.pending_delete = None;
    }

    fn selected_delete_target(&self) -> Option<DeleteConfirmTarget> {
        let (_, item) = self.filtered_items().get(self.selected).copied()?;
        match &item.value {
            PickerValue::Session(session) => Some(DeleteConfirmTarget {
                key_id: session.key_id.clone(),
                session_id: session.session_id.clone(),
            }),
            _ => None,
        }
    }

    fn arm_or_confirm_delete(&mut self) -> bool {
        let Some(target) = self.selected_delete_target() else {
            return false;
        };
        if self.pending_delete.as_ref() == Some(&target) {
            self.pending_delete = None;
            true
        } else {
            self.pending_delete = Some(target);
            false
        }
    }

    fn delete_is_armed_for_selected(&self) -> bool {
        self.selected_delete_target()
            .is_some_and(|target| self.pending_delete.as_ref() == Some(&target))
    }

    fn delete_is_armed_for_session(&self, preview: &SessionPreview) -> bool {
        self.pending_delete.as_ref().is_some_and(|target| {
            target.key_id == preview.key_id && target.session_id == preview.session_id
        })
    }
}

enum SubmitAction {
    Send(String),
    Command(SlashCommand),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SlashCommand {
    New,
    Exit,
    Resume(Option<String>),
    Model(Option<String>),
    Key(Option<String>),
    Help,
}

enum RuntimeEvent {
    Delta(ChatResponseChunk),
    Finished {
        result: std::result::Result<ChatTurnResult, String>,
        format: ChatFormat,
    },
    ModelsLoaded(std::result::Result<Vec<String>, String>),
    ResumeLoaded {
        request_id: u64,
        result: std::result::Result<LoadedSession, String>,
    },
}

struct ChatTuiApp {
    session_store: SessionStore,
    cache: ModelsCache,
    client: Client,
    key: ApiKey,
    copilot_tm: Option<Arc<CopilotTokenManager>>,
    cwd: String,
    raw_model: String,
    model: String,
    format: ChatFormat,
    history: Vec<ChatMessage>,
    draft: String,
    cursor: usize,
    command_menu: CommandMenuState,
    draft_history: Vec<String>,
    draft_history_index: Option<usize>,
    draft_history_stash: Option<String>,
    session_id: String,
    overlay: Overlay,
    notice: Option<(Color, String)>,
    show_reasoning: bool,
    pending_response: String,
    pending_reasoning: String,
    pending_submit: Option<String>,
    sending: bool,
    request_started_at: Option<Instant>,
    last_usage: Option<TokenUsage>,
    context_tokens: u64,
    follow_output: bool,
    transcript_scroll: usize,
    transcript_width: u16,
    transcript_view_height: u16,
    tx: UnboundedSender<RuntimeEvent>,
    rx: UnboundedReceiver<RuntimeEvent>,
    response_task: Option<JoinHandle<()>>,
    resume_task: Option<JoinHandle<()>>,
    resume_request_id: u64,
    loading_resume: Option<LoadingResume>,
    resume_restore_state: Option<ResumeRestoreState>,
    reduce_motion: bool,
    frame_tick: usize,
    picker_hitbox: Option<PickerHitbox>,
}

impl ChatTuiApp {
    async fn new(params: ChatTuiParams) -> Result<Self> {
        let (tx, rx) = mpsc::unbounded_channel();
        let startup_notice = params
            .startup_notice
            .map(|message| (MUTED, message))
            .or(Some((MUTED, "Ready".to_string())));

        Ok(Self {
            session_store: params.session_store,
            cache: params.cache,
            client: params.client,
            key: params.key,
            copilot_tm: params.copilot_tm,
            cwd: params.cwd,
            raw_model: params.raw_model,
            model: params.model,
            format: ChatFormat::OpenAI,
            history: params.initial_history,
            draft: String::new(),
            cursor: 0,
            command_menu: CommandMenuState::default(),
            draft_history: load_persisted_draft_history(),
            draft_history_index: None,
            draft_history_stash: None,
            session_id: params.initial_session,
            overlay: Overlay::None,
            notice: startup_notice,
            show_reasoning: true,
            pending_response: String::new(),
            pending_reasoning: String::new(),
            pending_submit: None,
            sending: false,
            request_started_at: None,
            last_usage: None,
            context_tokens: 0,
            follow_output: true,
            transcript_scroll: 0,
            transcript_width: 0,
            transcript_view_height: 0,
            tx,
            rx,
            response_task: None,
            resume_task: None,
            resume_request_id: 0,
            loading_resume: None,
            resume_restore_state: None,
            reduce_motion: reduce_motion_requested(),
            frame_tick: 0,
            picker_hitbox: None,
        })
    }

    fn persist_draft_history(&self) {
        let _ = save_persisted_draft_history(&self.draft_history);
    }

    fn is_busy(&self) -> bool {
        self.sending || self.loading_resume.is_some()
    }

    fn should_show_input_cursor(&self) -> bool {
        !self.overlay.blocks_input() && !self.is_busy()
    }

    fn abort_resume_task(&mut self) {
        if let Some(task) = self.resume_task.take() {
            task.abort();
        }
    }

    fn discard_resume_state(&mut self) {
        self.abort_resume_task();
        self.loading_resume = None;
        self.resume_restore_state = None;
    }

    fn restore_resume_state(&mut self, state: ResumeRestoreState) {
        self.key = state.key;
        self.copilot_tm = state.copilot_tm;
        self.raw_model = state.raw_model;
        self.model = state.model;
        self.format = state.format;
        self.history = state.history;
        self.draft = state.draft;
        self.cursor = state.cursor;
        self.command_menu = state.command_menu;
        self.draft_history_index = state.draft_history_index;
        self.draft_history_stash = state.draft_history_stash;
        self.session_id = state.session_id;
        self.notice = state.notice;
        self.show_reasoning = state.show_reasoning;
        self.pending_response = state.pending_response;
        self.pending_reasoning = state.pending_reasoning;
        self.pending_submit = state.pending_submit;
        self.last_usage = state.last_usage;
        self.context_tokens = state.context_tokens;
        self.follow_output = state.follow_output;
        self.transcript_scroll = state.transcript_scroll;
        self.loading_resume = None;
        self.resume_restore_state = None;
        self.request_started_at = None;
        self.sending = false;
    }

    fn cancel_resume_load(&mut self) {
        self.abort_resume_task();
        self.loading_resume = None;
        if let Some(state) = self.resume_restore_state.take() {
            self.restore_resume_state(state);
        }
        self.notice = Some((MUTED, "Resume cancelled".to_string()));
    }

    fn clear_for_resume_loading(&mut self) {
        self.history.clear();
        self.draft.clear();
        self.cursor = 0;
        self.command_menu.reset();
        self.draft_history_index = None;
        self.draft_history_stash = None;
        self.pending_response.clear();
        self.pending_reasoning.clear();
        self.pending_submit = None;
        self.format = ChatFormat::OpenAI;
        self.last_usage = None;
        self.context_tokens = 0;
        self.follow_output = true;
        self.transcript_scroll = 0;
        self.request_started_at = None;
        self.sending = false;
        self.notice = None;
    }

    fn cursor_left(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let mut pos = self.cursor - 1;
        while pos > 0 && !self.draft.is_char_boundary(pos) {
            pos -= 1;
        }
        self.cursor = pos;
    }

    fn cursor_right(&mut self) {
        if self.cursor >= self.draft.len() {
            return;
        }
        let mut pos = self.cursor + 1;
        while pos < self.draft.len() && !self.draft.is_char_boundary(pos) {
            pos += 1;
        }
        self.cursor = pos;
    }

    fn cursor_home(&mut self) {
        let before = &self.draft[..self.cursor];
        self.cursor = before.rfind('\n').map(|pos| pos + 1).unwrap_or(0);
    }

    fn cursor_end(&mut self) {
        let after = &self.draft[self.cursor..];
        self.cursor = after
            .find('\n')
            .map(|pos| self.cursor + pos)
            .unwrap_or(self.draft.len());
    }

    fn cursor_word_left(&mut self) {
        let chars: Vec<(usize, char)> = self.draft[..self.cursor].char_indices().collect();
        let mut i = chars.len();
        while i > 0 && chars[i - 1].1.is_whitespace() {
            i -= 1;
        }
        while i > 0 && !chars[i - 1].1.is_whitespace() {
            i -= 1;
        }
        self.cursor = chars.get(i).map(|(pos, _)| *pos).unwrap_or(0);
    }

    fn cursor_word_right(&mut self) {
        let rest = &self.draft[self.cursor..];
        let chars: Vec<(usize, char)> = rest.char_indices().collect();
        let mut i = 0;
        while i < chars.len() && chars[i].1.is_whitespace() {
            i += 1;
        }
        while i < chars.len() && !chars[i].1.is_whitespace() {
            i += 1;
        }
        if i >= chars.len() {
            self.cursor = self.draft.len();
        } else {
            self.cursor += chars[i].0;
        }
    }

    fn cursor_up(&mut self) {
        let before = &self.draft[..self.cursor];
        let Some(prev_nl) = before.rfind('\n') else {
            return;
        };
        let col = before[prev_nl + 1..].chars().count();
        let before_prev = &before[..prev_nl];
        let prev_line_start = before_prev.rfind('\n').map(|pos| pos + 1).unwrap_or(0);
        let prev_line_len = before_prev[prev_line_start..].chars().count();
        let target_col = col.min(prev_line_len);
        self.cursor = prev_line_start;
        for _ in 0..target_col {
            self.cursor_right();
        }
    }

    fn cursor_down(&mut self) {
        let before = &self.draft[..self.cursor];
        let col = if let Some(prev_nl) = before.rfind('\n') {
            before[prev_nl + 1..].chars().count()
        } else {
            before.chars().count()
        };
        let after = &self.draft[self.cursor..];
        let Some(next_nl_offset) = after.find('\n') else {
            return;
        };
        let next_line_start = self.cursor + next_nl_offset + 1;
        let after_next = &self.draft[next_line_start..];
        let next_line_len = after_next
            .find('\n')
            .map(|pos| after_next[..pos].chars().count())
            .unwrap_or_else(|| after_next.chars().count());
        let target_col = col.min(next_line_len);
        self.cursor = next_line_start;
        for _ in 0..target_col {
            self.cursor_right();
        }
    }

    fn insert_char_at_cursor(&mut self, ch: char) {
        self.draft.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
    }

    fn delete_char_before_cursor(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let mut start = self.cursor - 1;
        while start > 0 && !self.draft.is_char_boundary(start) {
            start -= 1;
        }
        self.draft.remove(start);
        self.cursor = start;
    }

    fn delete_char_at_cursor(&mut self) {
        if self.cursor >= self.draft.len() {
            return;
        }
        self.draft.remove(self.cursor);
    }

    fn delete_word_backward(&mut self) {
        let old_cursor = self.cursor;
        self.cursor_word_left();
        self.draft.drain(self.cursor..old_cursor);
    }

    fn kill_to_end_of_line(&mut self) {
        let after = &self.draft[self.cursor..];
        let end = after
            .find('\n')
            .map(|pos| self.cursor + pos)
            .unwrap_or(self.draft.len());
        if end == self.cursor && end < self.draft.len() {
            self.draft.remove(self.cursor);
        } else {
            self.draft.drain(self.cursor..end);
        }
    }

    fn active_command_query(&self) -> Option<&str> {
        if self.overlay.blocks_input()
            || self.is_busy()
            || !self.draft.starts_with('/')
            || self.draft.starts_with("//")
            || self.draft.contains('\n')
            || self.draft.contains(' ')
        {
            return None;
        }
        Some(&self.draft[1..])
    }

    fn visible_command_menu(&self) -> Option<VisibleCommandMenu> {
        let query = self.active_command_query()?;
        if self.command_menu.dismissed {
            return None;
        }
        let entries = filter_slash_commands(query);
        let selected = if entries.is_empty() {
            None
        } else {
            Some(
                self.command_menu
                    .selected
                    .min(entries.len().saturating_sub(1)),
            )
        };
        Some(VisibleCommandMenu { entries, selected })
    }

    fn sync_command_menu_state(&mut self) {
        let Some(query) = self.active_command_query() else {
            self.command_menu.reset();
            return;
        };
        let query = query.to_string();

        if self.command_menu.query != query {
            if self.command_menu.dismissed {
                self.command_menu.placement = None;
            }
            self.command_menu.query = query.clone();
            self.command_menu.selected = 0;
            self.command_menu.dismissed = false;
        }

        let matches = filter_slash_commands(&query);
        if matches.is_empty() {
            self.command_menu.selected = 0;
        } else {
            self.command_menu.selected = self.command_menu.selected.min(matches.len() - 1);
        }
    }

    fn select_previous_command(&mut self) {
        let Some(menu) = self.visible_command_menu() else {
            return;
        };
        let Some(selected) = menu.selected else {
            return;
        };
        self.command_menu.selected = if selected == 0 {
            menu.entries.len() - 1
        } else {
            selected - 1
        };
    }

    fn select_next_command(&mut self) {
        let Some(menu) = self.visible_command_menu() else {
            return;
        };
        let Some(selected) = menu.selected else {
            return;
        };
        self.command_menu.selected = if selected + 1 >= menu.entries.len() {
            0
        } else {
            selected + 1
        };
    }

    fn dismiss_command_menu(&mut self) -> bool {
        if self.active_command_query().is_none() || self.command_menu.dismissed {
            return false;
        }
        self.command_menu.dismissed = true;
        self.command_menu.placement = None;
        true
    }

    fn selected_command(&self) -> Option<SlashCommandSpec> {
        let menu = self.visible_command_menu()?;
        let selected = menu.selected?;
        menu.entries.get(selected).copied().copied()
    }

    fn insert_selected_command(&mut self) -> bool {
        let Some(command) = self.selected_command() else {
            return false;
        };
        self.draft = command.insertion_text();
        self.cursor = self.draft.len();
        self.command_menu.selected = 0;
        self.command_menu.dismissed = true;
        self.command_menu.placement = None;
        true
    }

    async fn execute_selected_command(&mut self) -> Result<bool> {
        let Some(command) = self.selected_command() else {
            return Ok(false);
        };
        self.draft = command.command_label();
        self.cursor = self.draft.len();
        self.command_menu.reset();
        self.submit_draft().await
    }

    fn composer_border_color(&self) -> Color {
        FAINT
    }

    async fn handle_runtime_events(&mut self) -> Result<()> {
        while let Ok(event) = self.rx.try_recv() {
            match event {
                RuntimeEvent::Delta(delta) => match delta {
                    ChatResponseChunk::Content(text) => self.pending_response.push_str(&text),
                    ChatResponseChunk::Reasoning(text) => {
                        self.pending_reasoning.push_str(&text);
                    }
                },
                RuntimeEvent::Finished { result, format } => {
                    self.sending = false;
                    self.request_started_at = None;
                    self.response_task = None;
                    self.format = format;
                    match result {
                        Ok(turn) => {
                            let content = if self.pending_response.is_empty() {
                                turn.content.clone()
                            } else {
                                self.pending_response.clone()
                            };
                            let reasoning_content = if self.pending_reasoning.is_empty() {
                                turn.reasoning_content.clone()
                            } else {
                                Some(self.pending_reasoning.clone())
                            };
                            self.pending_submit = None;
                            self.pending_response.clear();
                            self.pending_reasoning.clear();
                            self.history.push(ChatMessage {
                                role: "assistant".to_string(),
                                content,
                                reasoning_content,
                            });
                            if let Some(usage) = turn.usage {
                                self.session_store
                                    .record_tokens(
                                        &self.key.id,
                                        Some(&self.raw_model),
                                        usage.prompt_tokens,
                                        usage.completion_tokens,
                                    )
                                    .await?;
                                self.context_tokens = usage.prompt_tokens + usage.completion_tokens;
                                self.last_usage = Some(usage);
                            } else {
                                self.context_tokens = estimate_context_tokens(&self.history);
                                self.last_usage = None;
                            }
                            self.persist_history().await?;
                            self.notice = None;
                        }
                        Err(err) => {
                            self.pending_response.clear();
                            self.pending_reasoning.clear();
                            if self
                                .history
                                .last()
                                .is_some_and(|message| message.role == "user")
                            {
                                self.history.pop();
                            }
                            if let Some(submitted) = self.pending_submit.take()
                                && self.draft.is_empty()
                            {
                                self.draft = submitted;
                            }
                            self.notice = Some((ERROR, err));
                        }
                    }
                }
                RuntimeEvent::ModelsLoaded(result) => match result {
                    Ok(models) => {
                        let auto_accept = if let Overlay::Picker(picker) = &mut self.overlay {
                            if matches!(picker.kind, PickerKind::Model { .. }) {
                                picker.items = models
                                    .into_iter()
                                    .map(|model| PickerEntry {
                                        search_text: model.clone(),
                                        label: model.clone(),
                                        value: PickerValue::Model(model),
                                    })
                                    .collect();
                                picker.loading = false;
                                picker.selected = 0;
                                picker.exact_match_index()
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                        if let Some(index) = auto_accept {
                            self.activate_picker_selection(index).await?;
                        }
                    }
                    Err(err) => {
                        self.overlay = Overlay::None;
                        self.notice = Some((ERROR, err));
                    }
                },
                RuntimeEvent::ResumeLoaded { request_id, result } => {
                    let Some(loading) = &self.loading_resume else {
                        continue;
                    };
                    if loading.request_id != request_id {
                        continue;
                    }

                    self.resume_task = None;
                    match result {
                        Ok(session) => {
                            self.apply_loaded_session(session).await?;
                            self.loading_resume = None;
                            self.resume_restore_state = None;
                            self.notice = None;
                        }
                        Err(err) => {
                            self.loading_resume = None;
                            if let Some(state) = self.resume_restore_state.take() {
                                self.restore_resume_state(state);
                            }
                            self.notice = Some((ERROR, err));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn toggle_reasoning_visibility(&mut self) {
        self.show_reasoning = !self.show_reasoning;
        self.notice = Some((
            MUTED,
            if self.show_reasoning {
                "Thinking blocks shown".to_string()
            } else {
                "Thinking blocks hidden".to_string()
            },
        ));
    }

    async fn run(&mut self) -> Result<()> {
        let mut terminal = setup_terminal()?;
        let run_result = loop {
            self.frame_tick = self.frame_tick.wrapping_add(1);

            if let Err(err) = self.handle_runtime_events().await {
                break Err(err);
            }

            if let Err(err) = terminal.draw(|frame| self.render(frame)) {
                break Err(err.into());
            }

            match event::poll(Duration::from_millis(0)) {
                Ok(true) => match event::read() {
                    Ok(Event::Key(key)) => match self.handle_key(key).await {
                        Ok(true) => break Ok(()),
                        Ok(false) => {}
                        Err(err) => break Err(err),
                    },
                    Ok(Event::Mouse(mouse)) => match self.handle_mouse(mouse).await {
                        Ok(true) => break Ok(()),
                        Ok(false) => {}
                        Err(err) => break Err(err),
                    },
                    Ok(Event::Resize(_, _)) => {}
                    Ok(Event::Paste(text)) => {
                        if !self.overlay.blocks_input() && !self.is_busy() {
                            for ch in text.chars() {
                                self.insert_char_at_cursor(ch);
                            }
                            self.sync_command_menu_state();
                        }
                    }
                    Ok(_) => {}
                    Err(err) => break Err(err.into()),
                },
                Ok(false) => {}
                Err(err) => break Err(err.into()),
            }

            tokio::time::sleep(Duration::from_millis(16)).await;
        };

        self.discard_resume_state();
        if let Some(task) = self.response_task.take() {
            task.abort();
        }
        restore_terminal(terminal)?;
        run_result
    }

    async fn handle_key(&mut self, key: KeyEvent) -> Result<bool> {
        if matches!(key.code, KeyCode::Char('c')) && key.modifiers.contains(KeyModifiers::CONTROL) {
            return Ok(true);
        }

        let command_menu_shortcuts_active =
            self.active_command_query().is_some() && !self.command_menu.dismissed;

        let mut picker_submit = None;
        let mut picker_delete = None;
        match &mut self.overlay {
            Overlay::Help => {
                if matches!(key.code, KeyCode::Esc | KeyCode::Enter | KeyCode::F(1)) {
                    self.overlay = Overlay::None;
                }
                return Ok(false);
            }
            Overlay::Picker(picker) => {
                if picker.loading {
                    if matches!(key.code, KeyCode::Esc) {
                        self.overlay = Overlay::None;
                    }
                    return Ok(false);
                }

                match key.code {
                    KeyCode::Esc => self.overlay = Overlay::None,
                    KeyCode::Up => {
                        picker.clear_pending_delete();
                        picker.select_prev();
                    }
                    KeyCode::Char('p') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        picker.clear_pending_delete();
                        picker.select_prev();
                    }
                    KeyCode::Down => {
                        picker.clear_pending_delete();
                        picker.select_next();
                    }
                    KeyCode::Char('n') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        picker.clear_pending_delete();
                        picker.select_next();
                    }
                    KeyCode::Backspace => {
                        picker.clear_pending_delete();
                        picker.query.pop();
                        picker.selected = 0;
                    }
                    KeyCode::Enter => {
                        if picker.delete_is_armed_for_selected() {
                            picker_delete = Some(picker.selected);
                        } else {
                            picker_submit = Some(picker.selected);
                        }
                    }
                    KeyCode::Char('d')
                        if key.modifiers.contains(KeyModifiers::CONTROL)
                            && matches!(picker.kind, PickerKind::Session) =>
                    {
                        if picker.arm_or_confirm_delete() {
                            picker_delete = Some(picker.selected);
                        }
                    }
                    KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                        picker.clear_pending_delete();
                        picker.query.push(ch);
                        picker.selected = 0;
                    }
                    _ => {}
                }
                if picker_submit.is_none() && picker_delete.is_none() {
                    return Ok(false);
                }
            }
            Overlay::None => {}
        }

        if let Some(selected) = picker_delete {
            return self.delete_picker_selection(selected).await;
        }

        if let Some(selected) = picker_submit {
            return self.activate_picker_selection(selected).await;
        }

        if is_help_shortcut(key) {
            self.open_help_overlay();
            return Ok(false);
        }

        match key.code {
            KeyCode::Esc if self.loading_resume.is_some() => {
                self.cancel_resume_load();
                return Ok(false);
            }
            KeyCode::Esc if self.sending => {
                self.interrupt_inflight_request().await?;
                return Ok(false);
            }
            KeyCode::PageUp => {
                self.scroll_up();
                return Ok(false);
            }
            KeyCode::PageDown => {
                self.scroll_down();
                return Ok(false);
            }
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.scroll_up();
                return Ok(false);
            }
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.scroll_down();
                return Ok(false);
            }
            KeyCode::Up if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.scroll_up_lines(3);
                return Ok(false);
            }
            KeyCode::Down if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.scroll_down_lines(3);
                return Ok(false);
            }
            KeyCode::Home if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.scroll_to_top();
                return Ok(false);
            }
            KeyCode::End if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.scroll_to_bottom();
                return Ok(false);
            }
            KeyCode::Up if self.sending => {
                self.scroll_up_lines(3);
                return Ok(false);
            }
            KeyCode::Down if self.sending => {
                self.scroll_down_lines(3);
                return Ok(false);
            }
            KeyCode::Char('p')
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && self.loading_resume.is_none()
                    && !command_menu_shortcuts_active =>
            {
                self.history_prev();
                return Ok(false);
            }
            KeyCode::Char('n')
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && self.loading_resume.is_none()
                    && !command_menu_shortcuts_active =>
            {
                self.history_next();
                return Ok(false);
            }
            KeyCode::Char('r')
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && self.loading_resume.is_none() =>
            {
                self.open_resume_picker(None).await?;
                return Ok(false);
            }
            KeyCode::Char('m')
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && self.loading_resume.is_none() =>
            {
                self.open_model_picker(None, ModelSelectionTarget::CurrentChat, false);
                return Ok(false);
            }
            KeyCode::Char('t')
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && self.loading_resume.is_none() =>
            {
                self.toggle_reasoning_visibility();
                return Ok(false);
            }
            _ => {}
        }

        if self.is_busy() {
            return Ok(false);
        }

        let command_menu_visible = self.visible_command_menu().is_some();

        if matches!(key.code, KeyCode::Esc) && self.dismiss_command_menu() {
            return Ok(false);
        }

        if matches!(key.code, KeyCode::Char('p'))
            && key.modifiers.contains(KeyModifiers::CONTROL)
            && command_menu_visible
        {
            self.select_previous_command();
            return Ok(false);
        }

        if matches!(key.code, KeyCode::Char('n'))
            && key.modifiers.contains(KeyModifiers::CONTROL)
            && command_menu_visible
        {
            self.select_next_command();
            return Ok(false);
        }

        if matches!(key.code, KeyCode::Enter)
            && !key.modifiers.contains(KeyModifiers::CONTROL)
            && command_menu_visible
        {
            return self.execute_selected_command().await;
        }

        if matches!(key.code, KeyCode::Tab) && self.insert_selected_command() {
            return Ok(false);
        }

        match key.code {
            KeyCode::Enter if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                return self.submit_draft().await;
            }
            KeyCode::Tab => {}
            KeyCode::Char('j') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.push_newline();
                self.sync_command_menu_state();
            }
            KeyCode::Char('l') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.draft.clear();
                self.cursor = 0;
                self.command_menu.reset();
            }
            KeyCode::Char('w') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.leave_history_navigation();
                self.delete_word_backward();
                self.sync_command_menu_state();
            }
            KeyCode::Char('k') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.kill_to_end_of_line();
                self.sync_command_menu_state();
            }
            KeyCode::Char('a') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.cursor_home();
            }
            KeyCode::Char('e') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.cursor_end();
            }
            KeyCode::Char('b') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.cursor_left();
            }
            KeyCode::Char('f') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.cursor_right();
            }
            KeyCode::Backspace if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.leave_history_navigation();
                self.delete_word_backward();
                self.sync_command_menu_state();
            }
            KeyCode::Backspace => {
                self.leave_history_navigation();
                self.delete_char_before_cursor();
                self.sync_command_menu_state();
            }
            KeyCode::Delete => {
                self.delete_char_at_cursor();
                self.sync_command_menu_state();
            }
            KeyCode::Left if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.cursor_word_left();
            }
            KeyCode::Right if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.cursor_word_right();
            }
            KeyCode::Left => {
                self.cursor_left();
            }
            KeyCode::Right => {
                self.cursor_right();
            }
            KeyCode::Home => {
                self.cursor_home();
            }
            KeyCode::End => {
                self.cursor_end();
            }
            KeyCode::Up if command_menu_visible => {
                self.select_previous_command();
            }
            KeyCode::Down if command_menu_visible => {
                self.select_next_command();
            }
            KeyCode::Up => {
                if !self.draft.contains('\n') {
                    self.history_prev();
                } else {
                    self.cursor_up();
                }
            }
            KeyCode::Down => {
                if !self.draft.contains('\n') {
                    self.history_next();
                } else {
                    self.cursor_down();
                }
            }
            KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.leave_history_navigation();
                self.insert_char_at_cursor(ch);
                self.sync_command_menu_state();
            }
            _ => {}
        }

        Ok(false)
    }

    async fn handle_mouse(&mut self, mouse: MouseEvent) -> Result<bool> {
        match (&self.overlay, mouse.kind) {
            (Overlay::Picker(picker), MouseEventKind::ScrollUp) if !picker.loading => {
                if let Overlay::Picker(picker) = &mut self.overlay {
                    picker.select_prev();
                }
            }
            (Overlay::Picker(picker), MouseEventKind::ScrollDown) if !picker.loading => {
                if let Overlay::Picker(picker) = &mut self.overlay {
                    picker.select_next();
                }
            }
            (Overlay::Picker(picker), MouseEventKind::Down(MouseButton::Left))
                if !picker.loading =>
            {
                if let Some(hitbox) = &self.picker_hitbox {
                    let point = (mouse.column, mouse.row);
                    if rect_contains(hitbox.list_area, point) {
                        let row = usize::from(mouse.row.saturating_sub(hitbox.list_area.y));
                        if let Some(Some(filtered_index)) = hitbox.row_to_filtered_index.get(row) {
                            return self.activate_picker_selection(*filtered_index).await;
                        }
                    } else if !rect_contains(hitbox.overlay_area, point) {
                        self.overlay = Overlay::None;
                    }
                }
            }
            (Overlay::None, MouseEventKind::ScrollUp) => self.scroll_up_lines(3),
            (Overlay::None, MouseEventKind::ScrollDown) => self.scroll_down_lines(3),
            _ => {}
        }

        Ok(false)
    }

    async fn submit_draft(&mut self) -> Result<bool> {
        let action = match self.prepare_submit_action() {
            Ok(action) => action,
            Err(err) => {
                self.notice = Some((ERROR, err.to_string()));
                return Ok(false);
            }
        };
        let Some(action) = action else {
            return Ok(false);
        };

        match action {
            SubmitAction::Send(input) => {
                self.send_user_message(input);
                Ok(false)
            }
            SubmitAction::Command(command) => {
                self.draft.clear();
                self.cursor = 0;
                self.command_menu.reset();
                self.draft_history_index = None;
                self.draft_history_stash = None;
                self.execute_slash_command(command).await
            }
        }
    }

    fn prepare_submit_action(&self) -> Result<Option<SubmitAction>> {
        let trimmed = self.draft.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }
        if self.draft.contains('\n') {
            return Ok(Some(SubmitAction::Send(trimmed.to_string())));
        }
        if let Some(escaped) = trimmed.strip_prefix("//") {
            return Ok(Some(SubmitAction::Send(format!("/{escaped}"))));
        }
        if let Some(command) = trimmed.strip_prefix('/') {
            return Ok(Some(SubmitAction::Command(parse_slash_command(command)?)));
        }
        Ok(Some(SubmitAction::Send(trimmed.to_string())))
    }

    fn send_user_message(&mut self, input: String) {
        self.record_draft_history(&input);
        self.draft.clear();
        self.cursor = 0;
        self.command_menu.reset();
        self.overlay = Overlay::None;
        self.notice = None;
        self.last_usage = None;
        self.pending_response.clear();
        self.pending_reasoning.clear();
        self.pending_submit = Some(input.clone());
        self.request_started_at = Some(Instant::now());
        self.history.push(ChatMessage {
            role: "user".to_string(),
            content: input,
            reasoning_content: None,
        });
        trim_history(&mut self.history, MAX_HISTORY_MESSAGES);
        self.sending = true;
        self.follow_output = true;

        let tx = self.tx.clone();
        let client = self.client.clone();
        let key = self.key.clone();
        let model = self.model.clone();
        let history = self.history.clone();
        let copilot_tm = self.copilot_tm.clone();
        let mut format = self.format.clone();

        self.response_task = Some(tokio::spawn(async move {
            let spinning = Arc::new(AtomicBool::new(false));
            let mut parser = ThinkTagParser::new();
            let result = {
                let mut on_chunk = |chunk: ChatResponseChunk| -> Result<()> {
                    match chunk {
                        ChatResponseChunk::Content(text) => {
                            for c in parser.feed(&text) {
                                tx.send(RuntimeEvent::Delta(c)).ok();
                            }
                        }
                        other => {
                            tx.send(RuntimeEvent::Delta(other)).ok();
                        }
                    }
                    Ok(())
                };

                send_message_turn(
                    &client,
                    &key,
                    copilot_tm.as_deref(),
                    &model,
                    &history,
                    &mut format,
                    &spinning,
                    &mut on_chunk,
                )
                .await
            };

            for chunk in parser.flush() {
                tx.send(RuntimeEvent::Delta(chunk)).ok();
            }

            let result = result.map_err(|err| err.to_string());

            tx.send(RuntimeEvent::Finished { result, format }).ok();
        }));
    }

    async fn execute_slash_command(&mut self, command: SlashCommand) -> Result<bool> {
        match command {
            SlashCommand::New => {
                self.start_new_chat();
                Ok(false)
            }
            SlashCommand::Exit => Ok(true),
            SlashCommand::Resume(query) => {
                self.open_resume_picker(query).await?;
                Ok(false)
            }
            SlashCommand::Model(query) => {
                let auto_accept_exact = query.is_some();
                self.open_model_picker(query, ModelSelectionTarget::CurrentChat, auto_accept_exact);
                Ok(false)
            }
            SlashCommand::Key(query) => {
                self.open_or_switch_key(query).await?;
                Ok(false)
            }
            SlashCommand::Help => {
                self.open_help_overlay();
                Ok(false)
            }
        }
    }

    fn push_newline(&mut self) {
        if !self.draft.is_empty() {
            self.leave_history_navigation();
            self.insert_char_at_cursor('\n');
        }
    }

    fn start_new_chat(&mut self) {
        self.discard_resume_state();
        self.cancel_inflight_request();
        self.overlay = Overlay::None;
        self.history.clear();
        self.draft.clear();
        self.cursor = 0;
        self.command_menu.reset();
        self.draft_history_index = None;
        self.draft_history_stash = None;
        self.pending_response.clear();
        self.pending_reasoning.clear();
        self.pending_submit = None;
        self.sending = false;
        self.request_started_at = None;
        self.session_id = new_chat_session_id();
        self.format = ChatFormat::OpenAI;
        self.last_usage = None;
        self.context_tokens = 0;
        self.follow_output = true;
        self.notice = None;
    }

    fn cancel_inflight_request(&mut self) {
        if let Some(task) = self.response_task.take() {
            task.abort();
        }
        restore_cancelled_submission(&mut self.history, &mut self.draft, &mut self.pending_submit);
        self.cursor = self.draft.len();
        self.sync_command_menu_state();
        self.sending = false;
        self.request_started_at = None;
        self.pending_response.clear();
        self.pending_reasoning.clear();
        self.follow_output = true;
        self.notice = Some((MUTED, "Request cancelled".to_string()));
    }

    async fn interrupt_inflight_request(&mut self) -> Result<()> {
        if self.pending_response.is_empty() && self.pending_reasoning.is_empty() {
            self.cancel_inflight_request();
            return Ok(());
        }

        if let Some(task) = self.response_task.take() {
            task.abort();
        }

        let partial = std::mem::take(&mut self.pending_response);
        let reasoning = std::mem::take(&mut self.pending_reasoning);
        self.pending_submit = None;
        self.cursor = self.draft.len();
        self.sync_command_menu_state();
        self.sending = false;
        self.request_started_at = None;
        self.follow_output = true;
        self.history.push(ChatMessage {
            role: "assistant".to_string(),
            content: partial,
            reasoning_content: normalize_reasoning_content(reasoning),
        });
        self.context_tokens = estimate_context_tokens(&self.history);
        self.last_usage = None;
        self.persist_history().await?;
        self.notice = Some((MUTED, "Response interrupted".to_string()));
        Ok(())
    }

    fn record_draft_history(&mut self, input: &str) {
        if input.is_empty() {
            return;
        }
        self.draft_history.push(input.to_string());
        self.draft_history_index = None;
        self.draft_history_stash = None;
    }

    fn history_prev(&mut self) {
        if self.draft_history.is_empty() {
            return;
        }

        let next_index = match self.draft_history_index {
            Some(index) => index.saturating_sub(1),
            None => {
                self.draft_history_stash = Some(self.draft.clone());
                self.draft_history.len().saturating_sub(1)
            }
        };

        self.draft_history_index = Some(next_index);
        self.draft = self.draft_history[next_index].clone();
        self.cursor = self.draft.len();
        self.sync_command_menu_state();
    }

    fn history_next(&mut self) {
        let Some(index) = self.draft_history_index else {
            return;
        };

        if index + 1 < self.draft_history.len() {
            let next_index = index + 1;
            self.draft_history_index = Some(next_index);
            self.draft = self.draft_history[next_index].clone();
            self.cursor = self.draft.len();
            self.sync_command_menu_state();
            return;
        }

        self.draft_history_index = None;
        self.draft = self.draft_history_stash.take().unwrap_or_default();
        self.cursor = self.draft.len();
        self.sync_command_menu_state();
    }

    fn leave_history_navigation(&mut self) {
        if self.draft_history_index.is_some() && self.draft_history_stash.is_none() {
            self.draft_history_stash = Some(self.draft.clone());
        }
        self.draft_history_index = None;
    }

    fn open_model_picker(
        &mut self,
        query: Option<String>,
        target: ModelSelectionTarget,
        auto_accept_exact: bool,
    ) {
        self.prepare_for_model_picker();
        let query = query.unwrap_or_default();
        self.overlay = Overlay::Picker(Box::new(PickerState::loading(
            "Select model",
            query,
            PickerKind::Model {
                target,
                auto_accept_exact,
            },
        )));
        let tx = self.tx.clone();
        let client = self.client.clone();
        let key = match self.current_model_picker_key() {
            Some(key) => key,
            None => return,
        };
        let cache = self.cache.clone();

        tokio::spawn(async move {
            let models = fetch_models_for_select(&client, &key, &cache).await;
            if models.is_empty() {
                tx.send(RuntimeEvent::ModelsLoaded(Err(
                    "No models available for this provider".to_string(),
                )))
                .ok();
            } else {
                tx.send(RuntimeEvent::ModelsLoaded(Ok(models))).ok();
            }
        });
    }

    fn prepare_for_model_picker(&mut self) {
        if self.sending {
            self.cancel_inflight_request();
        }
    }

    async fn apply_model(&mut self, raw_model: String) -> Result<()> {
        self.session_store
            .set_chat_model(&self.key.id, &raw_model)
            .await?;
        self.session_store
            .record_selection(&self.key.id, "chat", Some(&raw_model))
            .await?;

        self.raw_model = raw_model.clone();
        self.model = ChatCommand::transform_model_for_provider(&self.key.base_url, &raw_model);
        self.draft_history_index = None;
        self.draft_history_stash = None;
        self.notice = None;

        if !self.history.is_empty() {
            self.persist_history().await?;
        }
        Ok(())
    }

    async fn complete_key_switch(&mut self, key: ApiKey, raw_model: String) -> Result<()> {
        self.key = key;
        self.raw_model = raw_model.clone();
        self.model = ChatCommand::transform_model_for_provider(&self.key.base_url, &raw_model);
        self.copilot_tm = copilot_token_manager_for_key(&self.key);
        self.session_store
            .set_chat_model(&self.key.id, &raw_model)
            .await?;
        self.session_store
            .record_selection(&self.key.id, "chat", Some(&raw_model))
            .await?;

        self.start_new_chat();
        Ok(())
    }

    async fn open_or_switch_key(&mut self, query: Option<String>) -> Result<()> {
        if let Some(query) = query {
            if let Some(key) = self.resolve_key_exact(&query).await? {
                self.begin_key_switch(key).await?;
                return Ok(());
            }
            self.open_key_picker(Some(query)).await?;
            return Ok(());
        }

        self.open_key_picker(None).await
    }

    async fn begin_key_switch(&mut self, mut key: ApiKey) -> Result<()> {
        SessionStore::decrypt_key_secret(&mut key)?;
        if let Some(raw_model) = self.session_store.get_chat_model(&key.id).await? {
            self.complete_key_switch(key, raw_model).await?;
        } else {
            self.overlay = Overlay::None;
            self.open_model_picker(None, ModelSelectionTarget::KeySwitch(key), false);
        }
        Ok(())
    }

    async fn open_key_picker(&mut self, query: Option<String>) -> Result<()> {
        let keys = self.session_store.get_keys().await?;
        if keys.is_empty() {
            self.notice = Some((ERROR, "No saved keys".to_string()));
            return Ok(());
        }

        let items = keys
            .into_iter()
            .map(|key| PickerEntry {
                label: format!("{} · {}", key.display_name(), key.base_url),
                search_text: key_search_text(&key),
                value: PickerValue::Key(key),
            })
            .collect();

        self.overlay = Overlay::Picker(Box::new(PickerState::ready(
            "Keys",
            query.unwrap_or_default(),
            items,
            PickerKind::Key,
        )));
        Ok(())
    }

    async fn open_resume_picker(&mut self, query: Option<String>) -> Result<()> {
        let sessions = load_resume_snapshots(&self.session_store, &self.cwd).await?;

        if let Some(query) = &query
            && let Some(snapshot) = sessions.iter().find(|session| session.session_id == *query)
        {
            self.begin_resume_load(snapshot.clone());
            return Ok(());
        }

        let items = sessions
            .into_iter()
            .map(|session| PickerEntry {
                label: session.title.clone(),
                search_text: session.search_text(),
                value: PickerValue::Session(session),
            })
            .collect();

        self.overlay = Overlay::Picker(Box::new(PickerState::ready(
            "Sessions",
            query.unwrap_or_default(),
            items,
            PickerKind::Session,
        )));
        Ok(())
    }

    fn open_help_overlay(&mut self) {
        self.overlay = Overlay::Help;
    }

    async fn activate_picker_selection(&mut self, filtered_index: usize) -> Result<bool> {
        let (kind, value) = {
            let Overlay::Picker(picker) = &self.overlay else {
                return Ok(false);
            };
            let Some((original_index, _)) = picker.filtered_items().get(filtered_index).copied()
            else {
                return Ok(false);
            };
            (
                picker.kind.clone(),
                picker.items[original_index].value.clone(),
            )
        };

        self.overlay = Overlay::None;

        match (kind, value) {
            (PickerKind::Model { target, .. }, PickerValue::Model(model)) => match target {
                ModelSelectionTarget::CurrentChat => self.apply_model(model).await?,
                ModelSelectionTarget::KeySwitch(key) => {
                    self.complete_key_switch(key, model).await?
                }
            },
            (PickerKind::Key, PickerValue::Key(key)) => {
                self.begin_key_switch(key).await?;
            }
            (PickerKind::Session, PickerValue::Session(session)) => {
                self.begin_resume_load(session);
            }
            _ => {}
        }

        Ok(false)
    }

    async fn delete_picker_selection(&mut self, filtered_index: usize) -> Result<bool> {
        let session = {
            let Overlay::Picker(picker) = &self.overlay else {
                return Ok(false);
            };
            let Some((_, item)) = picker.filtered_items().get(filtered_index).copied() else {
                return Ok(false);
            };
            match &item.value {
                PickerValue::Session(session) => session.clone(),
                _ => return Ok(false),
            }
        };

        let removed = self
            .session_store
            .delete_chat_session(&session.session_id)
            .await?;
        if !removed {
            self.notice = Some((ERROR, "Saved chat no longer exists".to_string()));
            return Ok(false);
        }

        if let Overlay::Picker(picker) = &mut self.overlay {
            picker.clear_pending_delete();
            picker.items.retain(|item| {
                !matches!(
                    &item.value,
                    PickerValue::Session(existing)
                        if existing.key_id == session.key_id && existing.session_id == session.session_id
                )
            });

            let filtered_len = picker.filtered_items().len();
            if filtered_len == 0 {
                self.overlay = Overlay::None;
                self.notice = Some((MUTED, "Saved chat deleted".to_string()));
                return Ok(false);
            }

            picker.selected = picker.selected.min(filtered_len.saturating_sub(1));
        }

        self.notice = Some((MUTED, "Saved chat deleted".to_string()));
        Ok(false)
    }

    async fn resolve_key_exact(&self, query: &str) -> Result<Option<ApiKey>> {
        let keys = self.session_store.get_keys().await?;

        if let Some(key) = keys.iter().find(|key| key.id == query).cloned() {
            return Ok(Some(key));
        }

        let name_matches = keys
            .into_iter()
            .filter(|key| key.name == query)
            .collect::<Vec<_>>();

        if name_matches.len() == 1 {
            Ok(name_matches.into_iter().next())
        } else {
            Ok(None)
        }
    }

    fn current_model_picker_key(&self) -> Option<ApiKey> {
        let Overlay::Picker(picker) = &self.overlay else {
            return None;
        };
        match &picker.kind {
            PickerKind::Model {
                target: ModelSelectionTarget::CurrentChat,
                ..
            } => Some(self.key.clone()),
            PickerKind::Model {
                target: ModelSelectionTarget::KeySwitch(key),
                ..
            } => Some(key.clone()),
            _ => None,
        }
    }

    async fn persist_history(&self) -> Result<()> {
        let stored = to_stored_messages(&self.history);
        let title = session_title_from_messages(&self.history, &self.raw_model);
        let preview = session_preview_text_from_messages(&self.history, &self.raw_model);
        self.session_store
            .save_chat_session_with_id(
                &self.key.id,
                &self.key.base_url,
                &self.cwd,
                &self.session_id,
                &self.raw_model,
                &stored,
                &title,
                &preview,
            )
            .await
    }

    fn begin_resume_load(&mut self, preview: SessionPreview) {
        self.discard_resume_state();
        self.overlay = Overlay::None;
        if self.sending {
            self.cancel_inflight_request();
        }

        self.resume_restore_state = Some(ResumeRestoreState::capture(self));
        self.clear_for_resume_loading();
        self.resume_request_id = self.resume_request_id.wrapping_add(1);
        let request_id = self.resume_request_id;
        self.loading_resume = Some(LoadingResume {
            request_id,
            preview: preview.clone(),
        });

        let session_store = self.session_store.clone();
        let cwd = self.cwd.clone();
        let tx = self.tx.clone();
        let task = tokio::spawn(async move {
            let result = load_resume_session(&session_store, &cwd, &preview).await;
            let _ = tx.send(RuntimeEvent::ResumeLoaded { request_id, result });
        });
        self.resume_task = Some(task);
    }

    async fn apply_loaded_session(&mut self, session: LoadedSession) -> Result<()> {
        if self.key.id != session.key_id {
            let key = self
                .session_store
                .get_key_by_id(&session.key_id)
                .await?
                .ok_or_else(|| anyhow::anyhow!("Saved key for this chat is no longer available"))?;
            self.key = key;
            self.copilot_tm = copilot_token_manager_for_key(&self.key);
        }

        self.overlay = Overlay::None;
        self.session_id = session.session_id;
        self.history = session.messages;
        self.draft.clear();
        self.cursor = 0;
        self.command_menu.reset();
        self.draft_history_index = None;
        self.draft_history_stash = None;
        self.pending_response.clear();
        self.pending_submit = None;
        self.format = ChatFormat::OpenAI;
        self.last_usage = None;
        self.context_tokens = estimate_context_tokens(&self.history);
        self.follow_output = true;
        self.transcript_scroll = 0;
        self.raw_model = session.raw_model.clone();
        self.model =
            ChatCommand::transform_model_for_provider(&self.key.base_url, &session.raw_model);
        self.session_store
            .set_chat_model(&self.key.id, &session.raw_model)
            .await?;
        self.session_store
            .record_selection(&self.key.id, "chat", Some(&session.raw_model))
            .await?;
        Ok(())
    }

    fn scroll_up(&mut self) {
        let step = usize::from(self.transcript_view_height.max(4) / 2);
        let max_scroll = self.max_scroll();
        if self.follow_output {
            self.transcript_scroll = max_scroll;
            self.follow_output = false;
        }
        self.transcript_scroll = self.transcript_scroll.saturating_sub(step.max(1));
    }

    fn scroll_down(&mut self) {
        let step = usize::from(self.transcript_view_height.max(4) / 2);
        let max_scroll = self.max_scroll();
        self.follow_output = false;
        self.transcript_scroll = (self.transcript_scroll + step.max(1)).min(max_scroll);
        if self.transcript_scroll >= max_scroll {
            self.follow_output = true;
        }
    }

    fn scroll_up_lines(&mut self, lines: usize) {
        let max_scroll = self.max_scroll();
        if self.follow_output {
            self.transcript_scroll = max_scroll;
            self.follow_output = false;
        }
        self.transcript_scroll = self.transcript_scroll.saturating_sub(lines);
    }

    fn scroll_down_lines(&mut self, lines: usize) {
        let max_scroll = self.max_scroll();
        self.follow_output = false;
        self.transcript_scroll = (self.transcript_scroll + lines).min(max_scroll);
        if self.transcript_scroll >= max_scroll {
            self.follow_output = true;
        }
    }

    fn scroll_to_top(&mut self) {
        self.transcript_scroll = 0;
        self.follow_output = false;
    }

    fn scroll_to_bottom(&mut self) {
        self.transcript_scroll = self.max_scroll();
        self.follow_output = true;
    }

    fn max_scroll(&self) -> usize {
        let total = self.estimated_transcript_height(self.transcript_width);
        total.saturating_sub(usize::from(self.transcript_view_height))
    }

    fn estimated_transcript_height(&self, width: u16) -> usize {
        let transcript = self.build_transcript();
        wrapped_text_line_count(transcript.text, width.max(1))
    }

    fn build_transcript(&self) -> RenderedTranscript {
        let mut lines = Vec::new();
        let mut previous_role: Option<&str> = None;

        if self.history.is_empty()
            && self.pending_response.is_empty()
            && self.pending_reasoning.is_empty()
            && !self.sending
        {
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
                "user" => render_user_message(&mut lines, &message.content),
                "assistant" => render_assistant_message(
                    &mut lines,
                    self.show_reasoning,
                    message.reasoning_content.as_deref(),
                    &message.content,
                ),
                other => render_system_message(&mut lines, other, &message.content),
            }
            previous_role = Some(message.role.as_str());
        }

        let has_visible_streaming = !self.pending_response.is_empty()
            || (!self.pending_reasoning.is_empty() && self.show_reasoning);
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
            render_assistant_streaming(
                &mut lines,
                self.show_reasoning,
                if self.pending_reasoning.is_empty() {
                    None
                } else {
                    Some(self.pending_reasoning.as_str())
                },
                &self.pending_response,
                self.sending,
                self.frame_tick,
                self.reduce_motion,
            );
        }

        if let Some(error) = error_notice(self.notice.as_ref()) {
            push_message_spacing(&mut lines);
            render_error_notice(&mut lines, error);
        }

        compact_styled_lines(&mut lines);
        lines.into()
    }

    fn transcript_intro_lines(&self) -> Vec<String> {
        vec![
            "AIVO Chat".to_string(),
            format!("{} · {}", self.raw_model, self.key.base_url),
            self.cwd.clone(),
        ]
    }

    fn empty_state_plain_lines(&self, width: u16) -> Vec<String> {
        if let Some(loading) = &self.loading_resume {
            vec![
                "Loading saved chat…".to_string(),
                loading.preview.title.clone(),
                plain_text_from_spans(&resume_metadata_spans(
                    &loading.preview,
                    width.saturating_sub(2).max(1),
                )),
                self.cwd.clone(),
            ]
        } else {
            self.transcript_intro_lines()
        }
    }

    fn render(&mut self, frame: &mut Frame<'_>) {
        let outer = frame.area();
        self.picker_hitbox = None;
        let composer_area = self.render_main(frame, outer);
        if let Some(menu) = self.visible_command_menu() {
            let current_rows = menu.entries.len().clamp(1, COMMAND_MENU_MAX_ROWS);
            let grew = current_rows > self.command_menu.last_row_count;
            let preferred = if grew {
                None
            } else {
                self.command_menu.placement
            };
            let (area, placement) =
                command_menu_area(composer_area, outer, menu.entries.len(), preferred);
            self.command_menu.placement = Some(placement);
            self.command_menu.last_row_count = current_rows;
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
    }

    fn render_main(&mut self, frame: &mut Frame<'_>, area: Rect) -> Rect {
        let composer_height = self.composer_height();
        let footer_height = 1u16;
        let max_transcript_height = area
            .height
            .saturating_sub(composer_height + footer_height)
            .max(1);
        let transcript_height =
            self.desired_transcript_height(area.width.max(1), max_transcript_height);
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

        let transcript = self.build_transcript();
        let transcript_area = chunks[0];
        let transcript_total_lines =
            wrapped_text_line_count(transcript.text.clone(), transcript_area.width.max(1));
        let transcript_line_height = transcript_total_lines as u16;
        let transcript_padding =
            if transcript_area.height > 2 && transcript_line_height > transcript_area.height {
                TRANSCRIPT_BOTTOM_PADDING
            } else {
                0
            };
        let transcript_content_area = Rect {
            x: transcript_area.x,
            y: transcript_area.y,
            width: transcript_area.width,
            height: transcript_area
                .height
                .saturating_sub(transcript_padding)
                .max(1),
        };
        let view_height = transcript_content_area.height.max(1);
        let width = transcript_content_area.width.max(1);
        self.transcript_width = width;
        self.transcript_view_height = view_height;
        let max_scroll = transcript_total_lines.saturating_sub(usize::from(view_height));
        if self.follow_output {
            self.transcript_scroll = max_scroll;
        } else {
            self.transcript_scroll = self.transcript_scroll.min(max_scroll);
        }

        frame.render_widget(Clear, chunks[0]);

        if self.history.is_empty()
            && self.pending_response.is_empty()
            && self.pending_reasoning.is_empty()
            && !self.sending
        {
            self.render_empty_state(frame, transcript_area);
        } else {
            let transcript_widget = Paragraph::new(transcript.text)
                .style(Style::default().fg(TEXT))
                .scroll(((self.transcript_scroll.min(u16::MAX as usize)) as u16, 0))
                .wrap(Wrap { trim: false });
            frame.render_widget(transcript_widget, transcript_content_area);
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
                frame.render_stateful_widget(
                    scrollbar,
                    transcript_content_area,
                    &mut scrollbar_state,
                );
            }
        }

        if transcript_padding > 0 {
            frame.render_widget(
                Clear,
                Rect {
                    x: transcript_area.x,
                    y: transcript_content_area
                        .y
                        .saturating_add(transcript_content_area.height),
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
                Style::default().fg(self.composer_border_color()),
            ))),
            composer_chunks[0],
        );

        let composer_area = composer_chunks[1];
        let composer = Paragraph::new(self.render_composer_text()).wrap(Wrap { trim: false });
        frame.render_widget(composer, composer_area);

        if self.should_show_input_cursor() {
            let (cursor_x, cursor_y) = cursor_position(
                &self.draft,
                self.cursor,
                composer_area.width.max(1),
                COMPOSER_PREFIX_WIDTH,
            );
            frame.set_cursor_position((
                composer_area.x + cursor_x,
                composer_area.y + cursor_y.min(composer_area.height.saturating_sub(1)),
            ));
        }

        self.render_footer(frame, chunks[2]);
        composer_area
    }

    fn desired_transcript_height(&self, width: u16, max_height: u16) -> u16 {
        let min_height = self.empty_state_height(width).clamp(1, max_height);
        if self.history.is_empty()
            && self.pending_response.is_empty()
            && self.pending_reasoning.is_empty()
            && !self.sending
        {
            return min_height;
        }

        (self.estimated_transcript_height(width) as u16).clamp(min_height, max_height)
    }

    fn empty_state_height(&self, width: u16) -> u16 {
        let content_width = width.saturating_sub(1).max(1);
        let intro_height = wrapped_text_line_count(
            plain_lines_to_text(self.empty_state_plain_lines(width)),
            content_width,
        ) as u16;

        let error_height = error_notice(self.notice.as_ref())
            .map(|error| {
                wrapped_text_line_count(format!("Error: {error}"), content_width) as u16 + 1
            })
            .unwrap_or(0);

        intro_height
            .saturating_add(EMPTY_STATE_BOTTOM_GAP)
            .saturating_add(error_height)
    }

    fn render_empty_state(&self, frame: &mut Frame<'_>, area: Rect) {
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
        if let Some(error) = error_notice(self.notice.as_ref()) {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                format!("Error: {error}"),
                Style::default().fg(ERROR),
            )));
        }

        frame.render_widget(
            Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }),
            content_area,
        );
    }

    fn render_composer_text(&self) -> Text<'static> {
        let prompt = if self.draft_history_index.is_some() {
            Span::styled(
                "^ ",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            )
        } else {
            Span::styled("> ", Style::default().fg(USER).add_modifier(Modifier::BOLD))
        };
        if self.draft.is_empty() {
            let placeholder = if self.loading_resume.is_some() {
                Span::styled("Resume loading…", Style::default().fg(FAINT))
            } else if self.sending {
                Span::styled("", Style::default())
            } else if self.has_reasoning_content() {
                Span::styled(
                    " Ask anything · / for commands · Ctrl+T toggle think",
                    Style::default().fg(FAINT),
                )
            } else {
                Span::styled(" Ask anything · / for commands", Style::default().fg(FAINT))
            };
            return Text::from(vec![Line::from(vec![prompt, placeholder])]);
        }

        let mut lines = Vec::new();
        for (index, line) in self.draft.split('\n').enumerate() {
            let prefix = if index == 0 {
                if self.draft_history_index.is_some() {
                    Span::styled(
                        "^ ",
                        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                    )
                } else {
                    Span::styled("> ", Style::default().fg(USER).add_modifier(Modifier::BOLD))
                }
            } else {
                Span::raw("  ")
            };
            lines.push(Line::from(vec![
                prefix,
                Span::styled(line.to_string(), Style::default().fg(TEXT)),
            ]));
        }

        Text::from(lines)
    }

    fn render_footer(&self, frame: &mut Frame<'_>, area: Rect) {
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

    fn render_command_menu(&self, frame: &mut Frame<'_>, area: Rect, menu: &VisibleCommandMenu) {
        frame.render_widget(Clear, area);
        let shell = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(FAINT));
        frame.render_widget(shell, area);

        let inner = area.inner(ratatui::layout::Margin {
            vertical: 1,
            horizontal: 1,
        });
        let footer_height = 1u16;
        let rows_area = Rect {
            x: inner.x,
            y: inner.y,
            width: inner.width,
            height: inner.height.saturating_sub(footer_height),
        };
        let footer_area = Rect::new(
            inner.x,
            inner.y + inner.height.saturating_sub(footer_height),
            inner.width,
            footer_height,
        );

        let lines = render_command_menu_rows(menu, rows_area.width);
        frame.render_widget(
            Paragraph::new(Text::from(lines))
                .style(Style::default().fg(TEXT))
                .wrap(Wrap { trim: false }),
            rows_area,
        );

        let footer_text = if menu.entries.is_empty() {
            "Esc close · Enter submit"
        } else {
            "Esc close · Enter run · Tab insert · ↑/↓ navigate"
        };
        frame.render_widget(
            Paragraph::new(footer_text).style(Style::default().fg(MUTED)),
            footer_area,
        );
    }

    fn footer_status_label(&self) -> (String, Color) {
        let color = MUTED;
        (
            format_token_count(self.context_tokens, self.last_usage),
            color,
        )
    }

    fn has_reasoning_content(&self) -> bool {
        !self.pending_reasoning.trim().is_empty()
            || self.history.iter().any(|message| {
                message
                    .reasoning_content
                    .as_deref()
                    .is_some_and(|text| !text.trim().is_empty())
            })
    }

    fn render_picker(&mut self, frame: &mut Frame<'_>, area: Rect, picker: &PickerState) {
        if matches!(picker.kind, PickerKind::Session) {
            self.render_session_picker(frame, area, picker);
            return;
        }

        frame.render_widget(Clear, area);
        let shell = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(FAINT));
        frame.render_widget(shell, area);

        let inner = area.inner(ratatui::layout::Margin {
            vertical: 1,
            horizontal: 2,
        });
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),
                Constraint::Min(6),
                Constraint::Length(1),
            ])
            .split(inner);

        let filtered_count = picker.filtered_items().len();
        let total_count = picker.items.len();
        let status_label = if picker.loading {
            "loading · esc".to_string()
        } else {
            format!(
                "{} · esc",
                format_picker_match_count(
                    filtered_count,
                    total_count,
                    picker_kind_noun(&picker.kind)
                )
            )
        };
        let status_width = display_width(&status_label) as u16;
        let title_width = display_width(picker.title) as u16;
        let middle_padding = chunks[0]
            .width
            .saturating_sub(title_width + status_width)
            .max(1);
        let header = Line::from(vec![
            Span::styled(
                picker.title,
                Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" ".repeat(usize::from(middle_padding))),
            Span::styled(status_label, Style::default().fg(MUTED)),
        ]);
        frame.render_widget(
            Paragraph::new(header),
            Rect::new(chunks[0].x, chunks[0].y, chunks[0].width, 1),
        );
        let search_line = if picker.query.is_empty() {
            Line::from(vec![
                Span::styled("/ ", Style::default().fg(ACCENT)),
                Span::styled(
                    picker_search_placeholder(&picker.kind),
                    Style::default().fg(MUTED).add_modifier(Modifier::ITALIC),
                ),
            ])
        } else {
            Line::from(vec![
                Span::styled("/ ", Style::default().fg(ACCENT)),
                Span::styled(
                    picker.query.clone(),
                    Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
                ),
            ])
        };
        frame.render_widget(
            Paragraph::new(search_line),
            Rect::new(chunks[0].x, chunks[0].y + 1, chunks[0].width, 1),
        );

        if picker.loading {
            frame.render_widget(
                Paragraph::new("Loading available models…").style(Style::default().fg(MUTED)),
                chunks[1],
            );
            return;
        }

        let visible = picker.visible_items(usize::from(chunks[1].height));
        let (lines, row_to_filtered_index) = if visible.is_empty() {
            (
                vec![Line::from(Span::styled(
                    "No matches",
                    Style::default().fg(MUTED),
                ))],
                Vec::new(),
            )
        } else {
            let mut lines = Vec::new();
            let mut row_to_filtered_index = Vec::new();

            for (filtered_index, item) in visible {
                let item_lines =
                    picker_entry_lines(item, filtered_index == picker.selected, chunks[1].width);
                row_to_filtered_index.extend(std::iter::repeat_n(filtered_index, item_lines.len()));
                lines.extend(item_lines);
            }

            (lines, row_to_filtered_index)
        };

        self.picker_hitbox = Some(PickerHitbox {
            overlay_area: area,
            list_area: chunks[1],
            row_to_filtered_index: row_to_filtered_index.into_iter().map(Some).collect(),
        });

        frame.render_widget(
            Paragraph::new(Text::from(lines))
                .style(Style::default().fg(TEXT))
                .wrap(Wrap { trim: false }),
            chunks[1],
        );
        frame.render_widget(
            Paragraph::new("Type to filter · Up/Down wrap · Enter open · Esc close")
                .style(Style::default().fg(MUTED)),
            chunks[2],
        );
    }

    fn render_session_picker(&mut self, frame: &mut Frame<'_>, area: Rect, picker: &PickerState) {
        frame.render_widget(Clear, area);
        let shell = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(FAINT));
        frame.render_widget(shell, area);

        let inner = area.inner(ratatui::layout::Margin {
            vertical: 1,
            horizontal: 2,
        });
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),
                Constraint::Min(6),
                Constraint::Length(1),
            ])
            .split(inner);

        let filtered_count = picker.filtered_items().len();
        let total_count = picker.items.len();
        let status_label = format!(
            "{} · esc",
            format_session_match_count(filtered_count, total_count)
        );
        let search_placeholder = if picker.query.is_empty() {
            vec![Span::styled(
                "filter chats, keys, models",
                Style::default().fg(MUTED).add_modifier(Modifier::ITALIC),
            )]
        } else {
            vec![Span::styled(
                picker.query.clone(),
                Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
            )]
        };
        let search_width = chunks[0].width.max(1);
        let title_label = "Sessions";
        let esc_width = status_label.chars().count() as u16;
        let title_width = title_label.chars().count() as u16;
        let middle_padding = search_width.saturating_sub(title_width + esc_width).max(1);
        let mut header_spans = vec![Span::styled(
            title_label,
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        )];
        header_spans.push(Span::raw(" ".repeat(usize::from(middle_padding))));
        header_spans.push(Span::styled(status_label, Style::default().fg(MUTED)));
        frame.render_widget(Paragraph::new(Line::from(header_spans)), chunks[0]);

        let search_line = Line::from(
            std::iter::once(Span::styled("/ ", Style::default().fg(ACCENT)))
                .chain(search_placeholder)
                .collect::<Vec<_>>(),
        );
        frame.render_widget(
            Paragraph::new(search_line),
            Rect::new(chunks[0].x, chunks[0].y + 1, chunks[0].width, 1),
        );

        let (lines, row_to_filtered_index) =
            render_session_picker_rows(picker, usize::from(chunks[1].height), chunks[1].width);

        self.picker_hitbox = Some(PickerHitbox {
            overlay_area: area,
            list_area: chunks[1],
            row_to_filtered_index,
        });

        frame.render_widget(
            Paragraph::new(Text::from(lines))
                .style(Style::default().fg(TEXT))
                .wrap(Wrap { trim: false }),
            chunks[1],
        );
        let footer_text = if picker.pending_delete.is_some() {
            "Enter or Ctrl+D confirm delete · Esc cancel"
        } else {
            "Type to filter · Up/Down wrap · Enter open · Ctrl+D delete"
        };
        frame.render_widget(
            Paragraph::new(footer_text).style(Style::default().fg(MUTED)),
            chunks[2],
        );
    }

    fn render_help_overlay(&self, frame: &mut Frame<'_>, area: Rect) {
        frame.render_widget(Clear, area);
        let shell = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(FAINT))
            .title(Span::styled(
                "Help",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ));
        frame.render_widget(shell, area);

        let inner = area.inner(ratatui::layout::Margin {
            vertical: 1,
            horizontal: 2,
        });

        let cmd_style = Style::default().fg(ASSISTANT).add_modifier(Modifier::BOLD);
        let key_style = Style::default().fg(ACCENT).add_modifier(Modifier::BOLD);
        let mut lines = vec![
            Line::from(Span::styled(
                "Slash commands",
                Style::default().fg(MUTED).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
        ];
        for command in SLASH_COMMANDS {
            lines.push(Line::from(vec![
                Span::styled(command.help_label, cmd_style),
                Span::styled(
                    format!("  {}", command.description),
                    Style::default().fg(TEXT),
                ),
            ]));
        }
        lines.extend([
            Line::from(""),
            Line::from(Span::styled(
                "Keybindings",
                Style::default().fg(MUTED).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(vec![
                Span::styled("Enter", key_style),
                Span::styled(
                    "       send message / run command",
                    Style::default().fg(TEXT),
                ),
            ]),
            Line::from(vec![
                Span::styled("Ctrl+J", key_style),
                Span::styled("      insert newline", Style::default().fg(TEXT)),
            ]),
            Line::from(vec![
                Span::styled("↑/↓", key_style),
                Span::styled(
                    "         command list / history / line nav",
                    Style::default().fg(TEXT),
                ),
            ]),
            Line::from(vec![
                Span::styled("←/→", key_style),
                Span::styled("         move cursor", Style::default().fg(TEXT)),
            ]),
            Line::from(vec![
                Span::styled("Home/Ctrl+A", key_style),
                Span::styled("  line start", Style::default().fg(TEXT)),
            ]),
            Line::from(vec![
                Span::styled("End/Ctrl+E", key_style),
                Span::styled("   line end", Style::default().fg(TEXT)),
            ]),
            Line::from(vec![
                Span::styled("Ctrl+←/→", key_style),
                Span::styled("    word jump", Style::default().fg(TEXT)),
            ]),
            Line::from(vec![
                Span::styled("Ctrl+W", key_style),
                Span::styled("      delete word backward", Style::default().fg(TEXT)),
            ]),
            Line::from(vec![
                Span::styled("Ctrl+K", key_style),
                Span::styled("      kill to end of line", Style::default().fg(TEXT)),
            ]),
            Line::from(vec![
                Span::styled("Ctrl+L", key_style),
                Span::styled("      clear prompt", Style::default().fg(TEXT)),
            ]),
            Line::from(vec![
                Span::styled("Ctrl+T", key_style),
                Span::styled("      toggle thinking blocks", Style::default().fg(TEXT)),
            ]),
            Line::from(vec![
                Span::styled("Tab", key_style),
                Span::styled("         complete slash command", Style::default().fg(TEXT)),
            ]),
            Line::from(vec![
                Span::styled("PgUp/PgDn", key_style),
                Span::styled("   scroll half page", Style::default().fg(TEXT)),
            ]),
            Line::from(vec![
                Span::styled("Ctrl+↑/↓", key_style),
                Span::styled("    scroll 3 lines", Style::default().fg(TEXT)),
            ]),
            Line::from(vec![
                Span::styled("Mouse wheel", key_style),
                Span::styled(" scroll 3 lines", Style::default().fg(TEXT)),
            ]),
            Line::from(vec![
                Span::styled("Ctrl+Home/End", key_style),
                Span::styled(" jump to top/bottom", Style::default().fg(TEXT)),
            ]),
            Line::from(vec![
                Span::styled("Esc", key_style),
                Span::styled(
                    "         cancel request / close overlay",
                    Style::default().fg(TEXT),
                ),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                "//message sends a literal leading slash",
                Style::default().fg(MUTED),
            )),
            Line::from(Span::styled(
                "Esc closes this overlay",
                Style::default().fg(MUTED),
            )),
        ]);

        frame.render_widget(
            Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }),
            inner,
        );
    }

    fn composer_height(&self) -> u16 {
        let lines = self.draft.lines().count().max(1) as u16;
        (lines + 2).clamp(3, 9)
    }
}

pub(super) async fn run_chat_tui(params: ChatTuiParams) -> Result<()> {
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let mut stdout = io::stdout();
        let _ = execute!(
            stdout,
            LeaveAlternateScreen,
            DisableBracketedPaste,
            DisableMouseCapture
        );
        original_hook(info);
    }));
    let mut app = ChatTuiApp::new(params).await?;
    let result = app.run().await;
    app.persist_draft_history();
    result
}

fn setup_terminal() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    let result: Result<_> = (|| {
        let mut stdout = io::stdout();
        execute!(
            stdout,
            EnterAlternateScreen,
            EnableBracketedPaste,
            EnableMouseCapture
        )?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;
        terminal.clear()?;
        Ok(terminal)
    })();
    if result.is_err() {
        let _ = disable_raw_mode();
        let mut stdout = io::stdout();
        let _ = execute!(stdout, LeaveAlternateScreen, DisableMouseCapture);
    }
    result
}

fn restore_terminal(mut terminal: Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableBracketedPaste,
        DisableMouseCapture,
    )?;
    terminal.show_cursor()?;
    Ok(())
}

async fn load_session_snapshots(
    session_store: &SessionStore,
    key: &ApiKey,
    cwd: &str,
) -> Result<Vec<SessionPreview>> {
    Ok(session_store
        .list_chat_sessions(&key.id, &key.base_url, cwd)
        .await?
        .into_iter()
        .map(|entry| SessionPreview::from_index_entry(entry, key))
        .collect())
}

async fn load_resume_snapshots(
    session_store: &SessionStore,
    cwd: &str,
) -> Result<Vec<SessionPreview>> {
    let keys = session_store.get_keys().await?;
    let mut sessions = Vec::new();

    for key in keys {
        sessions.extend(load_session_snapshots(session_store, &key, cwd).await?);
    }

    sessions.sort_by(|left, right| right.updated_at.cmp(&left.updated_at));
    Ok(sessions)
}

async fn load_resume_session(
    session_store: &SessionStore,
    _cwd: &str,
    preview: &SessionPreview,
) -> std::result::Result<LoadedSession, String> {
    let session = session_store
        .get_chat_session(&preview.session_id)
        .await
        .map_err(|err| err.to_string())?
        .ok_or_else(|| "Saved chat is no longer available".to_string())?;

    LoadedSession::from_state(session).map_err(|err| err.to_string())
}

fn load_persisted_draft_history() -> Vec<String> {
    let path = draft_history_path();
    load_persisted_draft_history_from_path(&path)
}

fn load_persisted_draft_history_from_path(path: &Path) -> Vec<String> {
    let Ok(data) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let Ok(plain) = crate::services::session_store::decrypt(&data) else {
        return Vec::new();
    };

    plain
        .lines()
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn save_persisted_draft_history(history: &[String]) -> io::Result<()> {
    let path = draft_history_path();
    save_persisted_draft_history_to_path(&path, history)
}

fn save_persisted_draft_history_to_path(path: &Path, history: &[String]) -> io::Result<()> {
    if history.is_empty() {
        return Ok(());
    }

    let joined = history.join("\n");
    let encrypted = crate::services::session_store::encrypt(&joined).map_err(io::Error::other)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, encrypted)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

fn draft_history_path() -> PathBuf {
    crate::services::system_env::home_dir()
        .map(|path| path.join(".config").join("aivo").join("chat_history"))
        .unwrap_or_else(|| PathBuf::from(".config/aivo/chat_history"))
}

struct StyledLine {
    line: Line<'static>,
    plain: String,
}

struct RenderedTranscript {
    text: Text<'static>,
}

impl From<Vec<StyledLine>> for RenderedTranscript {
    fn from(lines: Vec<StyledLine>) -> Self {
        let text = Text::from(lines.into_iter().map(|line| line.line).collect::<Vec<_>>());
        Self { text }
    }
}

fn push_message_spacing(lines: &mut Vec<StyledLine>) {
    if !lines.is_empty() {
        lines.push(blank_line());
    }
}

fn push_transcript_intro(lines: &mut Vec<StyledLine>, raw_model: &str, base_url: &str, cwd: &str) {
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

fn should_add_message_spacing(previous_role: Option<&str>, next_role: &str) -> bool {
    previous_role.is_some() && !next_role.is_empty()
}

fn render_user_message(lines: &mut Vec<StyledLine>, content: &str) {
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
}

fn render_reasoning_block(lines: &mut Vec<StyledLine>, reasoning: &str, show_reasoning: bool) {
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

fn normalized_reasoning_lines(reasoning: &str) -> Vec<String> {
    let mut lines = Vec::new();

    for raw_line in reasoning.lines() {
        let trimmed = raw_line.trim();
        if !trimmed.is_empty() {
            lines.push(trimmed.to_string());
        }
    }

    lines
}

fn extend_without_leading_blank(lines: &mut Vec<StyledLine>, mut rendered: Vec<StyledLine>) {
    while rendered
        .first()
        .is_some_and(|line| line.plain.trim().is_empty())
    {
        rendered.remove(0);
    }
    lines.extend(rendered);
}

fn render_assistant_message(
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

fn render_assistant_streaming(
    lines: &mut Vec<StyledLine>,
    show_reasoning: bool,
    reasoning: Option<&str>,
    content: &str,
    _sending: bool,
    _frame_tick: usize,
    _reduce_motion: bool,
) {
    if let Some(reasoning) = reasoning.filter(|text| !text.trim().is_empty()) {
        render_reasoning_block(lines, reasoning, show_reasoning);
        if show_reasoning && !content.is_empty() {
            push_styled_line(lines, "", Style::default());
        }
    }

    let rendered = render_markdown_lines(content);
    if !rendered.is_empty() {
        extend_without_leading_blank(lines, rendered);
    }
}

fn render_pending_status(
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

fn format_request_elapsed(elapsed: Duration) -> String {
    format!("{}s", elapsed.as_secs())
}

fn spinner_frame_indexed(frame_tick: usize, reduce_motion: bool) -> &'static str {
    if reduce_motion {
        return spinner_frame(0);
    }
    spinner_frame(frame_tick / 5)
}

fn error_notice(notice: Option<&(Color, String)>) -> Option<&str> {
    notice
        .filter(|(color, _)| *color == ERROR)
        .map(|(_, text)| text.as_str())
}

fn rect_contains(area: Rect, point: (u16, u16)) -> bool {
    let (x, y) = point;
    x >= area.x
        && x < area.x.saturating_add(area.width)
        && y >= area.y
        && y < area.y.saturating_add(area.height)
}

fn render_error_notice(lines: &mut Vec<StyledLine>, error: &str) {
    push_styled_line(lines, format!("Error: {error}"), Style::default().fg(ERROR));
}

fn render_system_message(lines: &mut Vec<StyledLine>, role: &str, content: &str) {
    push_styled_line(
        lines,
        role.to_string(),
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
    );
    if !content.is_empty() {
        extend_without_leading_blank(lines, render_markdown_lines(content));
    }
}

fn render_markdown_lines(content: &str) -> Vec<StyledLine> {
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

struct MarkdownRenderer {
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
struct InlineStyle {
    emphasis: usize,
    strong: usize,
    strike: usize,
    link: usize,
}

struct ListState {
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

struct CodeFence {
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

fn heading_style(level: HeadingLevel) -> Style {
    match level {
        HeadingLevel::H1 => Style::default()
            .fg(ACCENT)
            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
        HeadingLevel::H2 => Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        HeadingLevel::H3 => Style::default().fg(ASSISTANT).add_modifier(Modifier::BOLD),
        _ => Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
    }
}

fn blank_line() -> StyledLine {
    line_plain(String::new(), Style::default())
}

fn line_plain(text: String, style: Style) -> StyledLine {
    StyledLine {
        plain: text.clone(),
        line: Line::from(Span::styled(text, style)),
    }
}

fn line_with_plain(spans: Vec<Span<'static>>) -> StyledLine {
    let mut plain = String::new();
    for span in &spans {
        plain.push_str(span.content.as_ref());
    }
    StyledLine {
        line: Line::from(spans),
        plain,
    }
}

fn push_styled_line(lines: &mut Vec<StyledLine>, text: impl Into<String>, style: Style) {
    lines.push(line_plain(text.into(), style));
}

fn compact_styled_lines(lines: &mut Vec<StyledLine>) {
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

fn format_token_count(tokens: u64, usage: Option<TokenUsage>) -> String {
    if let Some(usage) = usage {
        let total = usage.prompt_tokens.saturating_add(usage.completion_tokens);
        let label = if total == 1 { "token" } else { "tokens" };
        return format!("{} {}", format_token_count_value(total), label);
    }
    if tokens == 0 {
        "0 tokens".to_string()
    } else {
        let label = if tokens == 1 { "token" } else { "tokens" };
        format!("~{} {}", format_token_count_value(tokens), label)
    }
}

fn format_token_count_value(tokens: u64) -> String {
    if tokens < 1_000 {
        return tokens.to_string();
    }

    let rounded_tenths = (tokens + 50) / 100;
    let thousands = rounded_tenths / 10;
    let tenths = rounded_tenths % 10;
    if tenths == 0 {
        format!("{thousands}k")
    } else {
        format!("{thousands}.{tenths}k")
    }
}

fn estimate_context_tokens(history: &[ChatMessage]) -> u64 {
    let total_chars: usize = history
        .iter()
        .map(|m| m.role.len() + m.content.len() + 20)
        .sum();
    (total_chars / 4) as u64
}

fn build_footer_text(model: &str, base_url: &str, cwd: &str, width: u16) -> String {
    let host = footer_host_label(base_url);
    let cwd_label = footer_cwd_label(cwd);
    let candidates = [
        format!("{model} · {host} · {cwd_label}"),
        format!("{model} · {host}"),
        model.to_string(),
    ];

    candidates
        .into_iter()
        .find(|candidate| candidate.chars().count() <= usize::from(width.max(1)))
        .unwrap_or_else(|| truncate_for_width(model, width))
}

fn restore_cancelled_submission(
    history: &mut Vec<ChatMessage>,
    draft: &mut String,
    pending_submit: &mut Option<String>,
) {
    if let Some(submitted) = pending_submit.take()
        && draft.is_empty()
    {
        *draft = submitted;
    }

    if history.last().is_some_and(|message| message.role == "user") {
        history.pop();
    }
}

fn footer_host_label(base_url: &str) -> String {
    if base_url == "copilot" {
        return "copilot".to_string();
    }

    let trimmed = base_url.trim().trim_end_matches('/');
    let without_scheme = trimmed
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(trimmed);
    without_scheme
        .split('/')
        .next()
        .filter(|host| !host.is_empty())
        .unwrap_or(trimmed)
        .to_string()
}

fn footer_cwd_label(cwd: &str) -> String {
    Path::new(cwd)
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or(cwd)
        .to_string()
}

fn session_title_from_messages(messages: &[ChatMessage], raw_model: &str) -> String {
    let last_user = messages
        .iter()
        .rev()
        .find(|message| message.role == "user" && !message.content.trim().is_empty())
        .map(|message| first_non_empty_line(&message.content));
    let fallback = messages
        .iter()
        .rev()
        .find(|message| !message.content.trim().is_empty())
        .map(|message| first_non_empty_line(&message.content));

    last_user
        .or(fallback)
        .filter(|title| !title.is_empty())
        .unwrap_or_else(|| raw_model.to_string())
}

fn session_preview_text_from_messages(messages: &[ChatMessage], raw_model: &str) -> String {
    let snippets = messages
        .iter()
        .rev()
        .filter(|message| !message.content.trim().is_empty())
        .take(2)
        .map(|message| collapse_whitespace(&message.content))
        .collect::<Vec<_>>();

    let joined = snippets
        .into_iter()
        .rev()
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" · ");

    if !joined.is_empty() {
        joined
    } else {
        raw_model.to_string()
    }
}

fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn plain_text_from_spans(spans: &[Span<'static>]) -> String {
    let mut plain = String::new();
    for span in spans {
        plain.push_str(span.content.as_ref());
    }
    plain
}

fn plain_lines_to_text(lines: Vec<String>) -> Text<'static> {
    Text::from(lines.into_iter().map(Line::from).collect::<Vec<_>>())
}

// Keep transcript height calculations aligned with Ratatui's own word-wrapping rules.
fn wrapped_text_line_count(text: impl Into<Text<'static>>, width: u16) -> usize {
    if width == 0 {
        return 0;
    }

    Paragraph::new(text)
        .wrap(Wrap { trim: false })
        .line_count(width)
}

fn metadata_text_len(value: &str) -> usize {
    value.chars().count()
}

fn resume_metadata_values(
    preview: &SessionPreview,
    width: u16,
) -> (String, String, Option<String>) {
    const SEPARATOR_LEN: usize = 3;

    let time_value = format_time_ago_short(&preview.updated_at);
    let key_value = preview.key_name.clone();
    let available = usize::from(width.max(1));

    let mut used = metadata_text_len(&time_value) + SEPARATOR_LEN + metadata_text_len(&key_value);
    let full_model_len = SEPARATOR_LEN + metadata_text_len(&preview.raw_model);

    if used + full_model_len <= available {
        return (time_value, key_value, Some(preview.raw_model.clone()));
    }

    used += SEPARATOR_LEN;
    if used >= available {
        return (time_value, key_value, None);
    }

    let model_width = available.saturating_sub(used) as u16;
    (
        time_value,
        key_value,
        Some(truncate_for_width(&preview.raw_model, model_width.max(1))),
    )
}

fn push_resume_metadata_segment(spans: &mut Vec<Span<'static>>, value: String, color: Color) {
    if !spans.is_empty() {
        spans.push(Span::styled(" · ", Style::default().fg(FAINT)));
    }
    spans.push(Span::styled(value, Style::default().fg(color)));
}

fn resume_metadata_spans(preview: &SessionPreview, width: u16) -> Vec<Span<'static>> {
    let (time_value, key_value, model_value) = resume_metadata_values(preview, width);
    let mut spans = Vec::new();
    push_resume_metadata_segment(&mut spans, time_value, ACCENT);
    push_resume_metadata_segment(&mut spans, key_value, USER);
    if let Some(model_value) = model_value {
        push_resume_metadata_segment(&mut spans, model_value, ASSISTANT);
    }
    spans
}

fn format_session_group_label(updated_at: &str) -> String {
    let parsed = DateTime::parse_from_rfc3339(updated_at)
        .map(|value| value.with_timezone(&Local))
        .ok();
    let Some(parsed) = parsed else {
        return updated_at.to_string();
    };
    let today = Local::now().date_naive();
    if parsed.date_naive() == today {
        "Today".to_string()
    } else {
        parsed.format("%a %b %d %Y").to_string()
    }
}

fn format_session_time(updated_at: &str) -> String {
    DateTime::parse_from_rfc3339(updated_at)
        .map(|value| value.with_timezone(&Local).format("%-I:%M %p").to_string())
        .unwrap_or_else(|_| updated_at.to_string())
}

fn format_session_match_count(filtered: usize, total: usize) -> String {
    if total == 0 {
        return "0 chats".to_string();
    }
    if filtered == total {
        return format!("{total} chats");
    }
    format!("{filtered}/{total}")
}

fn format_picker_match_count(filtered: usize, total: usize, noun: &str) -> String {
    if total == 0 {
        return format!("0 {noun}");
    }
    if filtered == total {
        return format!("{total} {noun}");
    }
    format!("{filtered}/{total}")
}

fn display_width(text: &str) -> usize {
    UnicodeWidthStr::width(text)
}

fn truncate_for_display_width(text: &str, max_width: usize) -> String {
    if max_width == 0 {
        return String::new();
    }
    if display_width(text) <= max_width {
        return text.to_string();
    }
    if max_width == 1 {
        return "…".to_string();
    }

    let mut result = String::new();
    let mut used = 0;
    let limit = max_width - 1;
    for ch in text.chars() {
        let width = UnicodeWidthChar::width(ch).unwrap_or(0);
        if used + width > limit {
            break;
        }
        used += width;
        result.push(ch);
    }
    result.push('…');
    result
}

fn filter_slash_commands(query: &str) -> Vec<&'static SlashCommandSpec> {
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

fn command_menu_area(
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
    let height = row_count.saturating_add(3).min(frame_area.height.max(4));
    let above_space = composer_area.y.saturating_sub(frame_area.y);
    let below_space = frame_area
        .y
        .saturating_add(frame_area.height)
        .saturating_sub(composer_area.y.saturating_add(composer_area.height));
    let placement = preferred_placement.unwrap_or({
        if above_space >= height || above_space >= below_space {
            CommandMenuPlacement::Above
        } else {
            CommandMenuPlacement::Below
        }
    });
    let y = match placement {
        CommandMenuPlacement::Above => composer_area.y.saturating_sub(height).max(frame_area.y),
        CommandMenuPlacement::Below => composer_area.y.saturating_add(composer_area.height).min(
            frame_area
                .y
                .saturating_add(frame_area.height.saturating_sub(height)),
        ),
    };
    (Rect::new(composer_area.x, y, width, height), placement)
}

fn command_menu_item_line(
    command: SlashCommandSpec,
    label: &str,
    selected: bool,
    width: u16,
    label_column_width: usize,
) -> Line<'static> {
    const SELECT_TEXT: Color = SELECT_WARM;
    const COLUMN_GAP: usize = 2;

    let content_width = usize::from(width.max(1))
        .saturating_sub(PICKER_ROW_PREFIX_WIDTH)
        .max(1);
    let description_width = content_width
        .saturating_sub(label_column_width)
        .saturating_sub(COLUMN_GAP);
    let rendered_label = truncate_for_display_width(label, label_column_width.max(1));
    let rendered_description = if description_width >= 8 {
        truncate_for_display_width(command.description, description_width)
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

fn render_command_menu_rows(menu: &VisibleCommandMenu, width: u16) -> Vec<Line<'static>> {
    if menu.entries.is_empty() {
        return vec![Line::from(Span::styled(
            "No matching command",
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
    let content_width = usize::from(width.max(1))
        .saturating_sub(PICKER_ROW_PREFIX_WIDTH)
        .max(1);
    let labels: Vec<String> = menu.entries[start..end]
        .iter()
        .map(|command| command.command_label())
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
        .map(|(index, (command, label))| {
            command_menu_item_line(
                **command,
                label,
                start + index == selected,
                width,
                label_column_width,
            )
        })
        .collect()
}

fn picker_kind_noun(kind: &PickerKind) -> &'static str {
    match kind {
        PickerKind::Key => "keys",
        PickerKind::Model { .. } => "models",
        PickerKind::Session => "chats",
    }
}

fn picker_search_placeholder(kind: &PickerKind) -> &'static str {
    match kind {
        PickerKind::Key => "filter key name or endpoint",
        PickerKind::Model { .. } => "filter model names",
        PickerKind::Session => "filter saved chats",
    }
}

fn key_search_text(key: &ApiKey) -> String {
    format!(
        "{} {} {}",
        key.id,
        key.display_name(),
        footer_host_label(&key.base_url)
    )
}

fn key_picker_item_line(key: &ApiKey, selected: bool, width: u16) -> Line<'static> {
    const SELECT_BG: Color = Color::Rgb(78, 108, 136);
    const SELECT_TEXT: Color = Color::Rgb(242, 245, 247);
    const SELECT_ACCENT: Color = SELECT_WARM;

    const SEPARATOR: &str = " · ";

    let name = key.display_name().to_string();
    let endpoint = key.base_url.clone();
    let content_width = usize::from(width.max(1))
        .saturating_sub(PICKER_ROW_PREFIX_WIDTH)
        .max(1);
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

fn session_picker_item_lines(
    preview: &SessionPreview,
    selected: bool,
    armed_delete: bool,
    width: u16,
) -> Vec<Line<'static>> {
    const SELECT_BG: Color = Color::Rgb(78, 108, 136);
    const SELECT_TEXT: Color = Color::Rgb(242, 245, 247);
    const SELECT_TIME: Color = SELECT_WARM;
    const DELETE_BG: Color = Color::Rgb(104, 63, 63);
    const DELETE_TEXT: Color = Color::Rgb(255, 241, 233);
    const DELETE_TIME: Color = Color::Rgb(255, 198, 176);

    let time = format_session_time(&preview.updated_at);
    let content_width = usize::from(width.max(1))
        .saturating_sub(PICKER_ROW_PREFIX_WIDTH)
        .max(1);
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

fn render_session_picker_rows(
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

fn picker_entry_lines(item: &PickerEntry, selected: bool, width: u16) -> Vec<Line<'static>> {
    match &item.value {
        PickerValue::Session(preview) => session_picker_item_lines(preview, selected, false, width),
        PickerValue::Key(key) => vec![key_picker_item_line(key, selected, width)],
        _ => {
            const SELECT_BG: Color = Color::Rgb(78, 108, 136);
            const SELECT_TEXT: Color = Color::Rgb(242, 245, 247);
            const SELECT_ACCENT: Color = SELECT_WARM;

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

fn centered_rect(width_pct: u16, height_pct: u16, area: Rect) -> Rect {
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

fn cursor_position(text: &str, cursor: usize, width: u16, line_prefix_width: u16) -> (u16, u16) {
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

fn format_time_ago_short(updated_at: &str) -> String {
    let parsed = DateTime::parse_from_rfc3339(updated_at)
        .map(|value| value.with_timezone(&Utc))
        .ok();
    let Some(parsed) = parsed else {
        return updated_at.to_string();
    };
    let seconds = (Utc::now() - parsed).num_seconds().max(0);
    match seconds {
        0..=59 => "now".to_string(),
        60..=3599 => format!("{}m", seconds / 60),
        3600..=86_399 => format!("{}h", seconds / 3600),
        86_400..=604_799 => format!("{}d", seconds / 86_400),
        604_800..=2_592_000 => format!("{}w", seconds / 604_800),
        2_592_001..=31_535_999 => format!("{}mo", seconds / 2_592_000),
        _ => format!("{}y", seconds / 31_536_000),
    }
}

fn parse_slash_command(input: &str) -> Result<SlashCommand> {
    let trimmed = input.trim();
    let mut parts = trimmed.splitn(2, char::is_whitespace);
    let command = parts.next().unwrap_or_default();
    let argument = parts
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);

    match command {
        "new" => Ok(SlashCommand::New),
        "exit" => Ok(SlashCommand::Exit),
        "resume" => Ok(SlashCommand::Resume(argument)),
        "model" => Ok(SlashCommand::Model(argument)),
        "key" => Ok(SlashCommand::Key(argument)),
        "help" => Ok(SlashCommand::Help),
        "" => anyhow::bail!("Type a command after '/'"),
        other => anyhow::bail!("Unknown command '/{other}'"),
    }
}

fn reduce_motion_requested() -> bool {
    env::var("AIVO_REDUCE_MOTION")
        .ok()
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn is_help_shortcut(key: KeyEvent) -> bool {
    matches!(key.code, KeyCode::F(1))
}

fn truncate_for_width(text: &str, width: u16) -> String {
    truncate_for_display_width(text, usize::from(width))
}

fn first_non_empty_line(text: &str) -> String {
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("")
        .to_string()
}

fn copilot_token_manager_for_key(key: &ApiKey) -> Option<Arc<CopilotTokenManager>> {
    if key.base_url == "copilot" {
        Some(Arc::new(CopilotTokenManager::new(
            key.key.as_str().to_string(),
        )))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration as ChronoDuration;
    use tempfile::TempDir;

    #[test]
    fn test_wrapped_text_line_count_uses_ratatui_word_wrap() {
        assert_eq!(wrapped_text_line_count("", 10), 1);
        assert_eq!(wrapped_text_line_count("hello", 10), 1);
        assert_eq!(wrapped_text_line_count("abcdefghij", 5), 2);
        assert_eq!(wrapped_text_line_count("word word word", 8), 3);
    }

    #[test]
    fn test_matches_fuzzy() {
        assert!(matches_fuzzy("g4", "gpt-4o"));
        assert!(matches_fuzzy("", "anything"));
        assert!(!matches_fuzzy("xyz", "gpt-4o"));
    }

    #[test]
    fn test_format_time_ago_short() {
        let updated_at = (Utc::now() - ChronoDuration::minutes(5)).to_rfc3339();
        assert_eq!(format_time_ago_short(&updated_at), "5m");
    }

    #[test]
    fn test_cursor_position_multiline() {
        // cursor at end of text
        assert_eq!(cursor_position("hello", 5, 10, 2), (7, 0));
        assert_eq!(cursor_position("hello\nworld", 11, 10, 2), (7, 1));
        // cursor in middle
        assert_eq!(cursor_position("hello\nworld", 6, 10, 2), (2, 1));
        assert_eq!(cursor_position("hello\nworld", 0, 10, 2), (2, 0));
    }

    #[test]
    fn test_cursor_position_uses_display_width_for_cjk() {
        assert_eq!(
            cursor_position("最新的软件开发工具", "最新的软件开发工具".len(), 30, 2),
            (20, 0)
        );
    }

    #[test]
    fn test_cursor_position_wraps_after_prefix_width() {
        assert_eq!(cursor_position("abcdefgh", 8, 8, 2), (2, 1));
    }

    #[test]
    fn test_question_mark_is_not_help_shortcut() {
        let question = KeyEvent::new(KeyCode::Char('?'), KeyModifiers::NONE);
        let f1 = KeyEvent::new(KeyCode::F(1), KeyModifiers::NONE);
        assert!(!is_help_shortcut(question));
        assert!(is_help_shortcut(f1));
    }

    #[test]
    fn test_cursor_movement_basic() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        app.draft = "hello world".to_string();
        app.cursor = app.draft.len();

        // cursor_left moves back one char
        app.cursor_left();
        assert_eq!(app.cursor, 10); // before 'd'

        // cursor_right moves forward
        app.cursor_right();
        assert_eq!(app.cursor, 11); // end

        // cursor_home goes to start
        app.cursor_home();
        assert_eq!(app.cursor, 0);

        // cursor_end goes to end
        app.cursor_end();
        assert_eq!(app.cursor, 11);
    }

    #[test]
    fn test_cursor_insert_delete() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        app.draft = "hllo".to_string();
        app.cursor = 1; // after 'h'

        app.insert_char_at_cursor('e');
        assert_eq!(app.draft, "hello");
        assert_eq!(app.cursor, 2);

        app.cursor = app.draft.len();
        app.delete_char_before_cursor();
        assert_eq!(app.draft, "hell");
        assert_eq!(app.cursor, 4);

        app.cursor = 0;
        app.delete_char_at_cursor();
        assert_eq!(app.draft, "ell");
        assert_eq!(app.cursor, 0);
    }

    #[test]
    fn test_cursor_word_movement() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        app.draft = "hello world foo".to_string();
        app.cursor = app.draft.len();

        app.cursor_word_left();
        assert_eq!(app.cursor, 12); // start of 'foo'

        app.cursor_word_left();
        assert_eq!(app.cursor, 6); // start of 'world'

        app.cursor_word_right();
        assert_eq!(app.cursor, 11); // end of 'world'
    }

    #[test]
    fn test_visible_command_menu_filters_and_hides_escaped_slash() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        app.draft = "/mo".to_string();
        app.cursor = 3;
        app.sync_command_menu_state();
        let menu = app.visible_command_menu().unwrap();
        assert_eq!(menu.entries.len(), 1);
        assert_eq!(menu.entries[0].name, "model");
        assert_eq!(menu.selected, Some(0));

        app.draft = "//literal".to_string();
        app.sync_command_menu_state();
        assert!(app.visible_command_menu().is_none());

        app.draft = "/model claude".to_string();
        app.sync_command_menu_state();
        assert!(app.visible_command_menu().is_none());
    }

    #[test]
    fn test_filter_slash_commands_prefers_prefix_matches() {
        let matches = filter_slash_commands("m");
        assert_eq!(matches.first().map(|command| command.name), Some("model"));
    }

    #[test]
    fn test_command_menu_area_prefers_below_when_above_space_is_tight() {
        let composer = Rect::new(2, 1, 40, 2);
        let frame = Rect::new(0, 0, 80, 20);
        let (area, placement) = command_menu_area(composer, frame, 6, None);
        assert_eq!(placement, CommandMenuPlacement::Below);
        assert!(area.y >= composer.y + composer.height);
    }

    #[test]
    fn test_command_menu_area_respects_sticky_placement() {
        let composer = Rect::new(2, 1, 40, 2);
        let frame = Rect::new(0, 0, 80, 20);
        let (area, placement) =
            command_menu_area(composer, frame, 6, Some(CommandMenuPlacement::Above));
        assert_eq!(placement, CommandMenuPlacement::Above);
        assert_eq!(area.y, frame.y);
    }

    #[test]
    fn test_command_menu_query_change_keeps_existing_placement_until_reopen() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        app.command_menu.placement = Some(CommandMenuPlacement::Above);
        app.draft = "/m".to_string();
        app.cursor = app.draft.len();
        app.command_menu.query = "m".to_string();
        app.command_menu.dismissed = false;

        app.draft = "/mo".to_string();
        app.cursor = app.draft.len();
        app.sync_command_menu_state();

        assert_eq!(
            app.command_menu.placement,
            Some(CommandMenuPlacement::Above)
        );

        app.dismiss_command_menu();
        app.draft = "/mod".to_string();
        app.cursor = app.draft.len();
        app.sync_command_menu_state();

        assert_eq!(app.command_menu.placement, None);
    }

    #[test]
    fn test_bare_slash_filtering_keeps_detected_placement() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        app.draft = "/".to_string();
        app.cursor = app.draft.len();
        app.sync_command_menu_state();
        app.command_menu.placement = Some(CommandMenuPlacement::Below);

        app.draft = "/new".to_string();
        app.cursor = app.draft.len();
        app.sync_command_menu_state();

        assert_eq!(
            app.command_menu.placement,
            Some(CommandMenuPlacement::Below)
        );
    }

    #[test]
    fn test_insert_selected_command_uses_argument_space_when_needed() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        app.draft = "/m".to_string();
        app.cursor = app.draft.len();
        app.sync_command_menu_state();

        assert!(app.insert_selected_command());
        assert_eq!(app.draft, "/model ");
        assert_eq!(app.cursor, app.draft.len());
        assert!(app.visible_command_menu().is_none());
    }

    #[test]
    fn test_insert_selected_command_omits_space_for_zero_arg_command() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        app.draft = "/ex".to_string();
        app.cursor = app.draft.len();
        app.sync_command_menu_state();

        assert!(app.insert_selected_command());
        assert_eq!(app.draft, "/exit");
        assert_eq!(app.cursor, app.draft.len());
        assert!(app.visible_command_menu().is_none());
    }

    #[tokio::test]
    async fn test_ctrl_p_navigate_command_menu_before_history() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        app.draft_history = vec!["previous prompt".to_string()];
        app.draft = "/".to_string();
        app.cursor = app.draft.len();
        app.sync_command_menu_state();

        app.handle_key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL))
            .await
            .unwrap();

        let menu = app.visible_command_menu().unwrap();
        assert_eq!(menu.selected, Some(menu.entries.len() - 1));
        assert_eq!(app.draft, "/");
        assert!(app.draft_history_index.is_none());
    }

    #[tokio::test]
    async fn test_escape_dismisses_command_menu_until_query_changes() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        app.draft = "/mo".to_string();
        app.cursor = app.draft.len();
        app.sync_command_menu_state();
        assert!(app.visible_command_menu().is_some());

        app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
            .await
            .unwrap();

        assert!(app.visible_command_menu().is_none());

        app.handle_key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE))
            .await
            .unwrap();
        assert!(app.visible_command_menu().is_none());

        app.handle_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE))
            .await
            .unwrap();
        let menu = app.visible_command_menu().unwrap();
        assert_eq!(menu.entries[0].name, "model");
    }

    #[tokio::test]
    async fn test_enter_executes_selected_command() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        app.draft = "/".to_string();
        app.cursor = app.draft.len();
        app.sync_command_menu_state();

        app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
            .await
            .unwrap();
        let should_exit = app
            .handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
            .await
            .unwrap();

        assert!(app.draft.is_empty());
        assert_eq!(app.cursor, 0);
        assert!(should_exit);
        assert!(matches!(app.overlay, Overlay::None));
    }

    #[test]
    fn test_render_command_menu_rows_shows_empty_state() {
        let menu = VisibleCommandMenu {
            entries: Vec::new(),
            selected: None,
        };
        let lines = render_command_menu_rows(&menu, 32);
        let plain = plain_text_from_spans(&lines[0].spans);
        assert_eq!(plain, "No matching command");
    }

    #[test]
    fn test_render_command_menu_rows_aligns_description_column() {
        let menu = VisibleCommandMenu {
            entries: vec![&SLASH_COMMANDS[0], &SLASH_COMMANDS[2]],
            selected: Some(0),
        };
        let lines = render_command_menu_rows(&menu, 48);
        let first = plain_text_from_spans(&lines[0].spans);
        let second = plain_text_from_spans(&lines[1].spans);
        let first_desc = first.find("start a fresh chat").unwrap();
        let second_desc = second.find("resume a saved chat").unwrap();
        assert_eq!(first_desc, second_desc);
    }

    #[test]
    fn test_delete_word_backward() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        app.draft = "hello world".to_string();
        app.cursor = app.draft.len();
        app.delete_word_backward();
        assert_eq!(app.draft, "hello ");
        assert_eq!(app.cursor, 6);
    }

    #[test]
    fn test_kill_to_end_of_line() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        app.draft = "hello\nworld".to_string();
        app.cursor = 2;
        app.kill_to_end_of_line();
        assert_eq!(app.draft, "he\nworld");
        assert_eq!(app.cursor, 2);
    }

    fn make_test_app(
        tx: tokio::sync::mpsc::UnboundedSender<RuntimeEvent>,
        rx: tokio::sync::mpsc::UnboundedReceiver<RuntimeEvent>,
    ) -> ChatTuiApp {
        ChatTuiApp {
            session_store: SessionStore::new(),
            cache: ModelsCache::new(),
            client: reqwest::Client::new(),
            key: ApiKey::new_with_protocol(
                "test".to_string(),
                "test".to_string(),
                "https://api.anthropic.com".to_string(),
                None,
                String::new(),
            ),
            copilot_tm: None,
            cwd: String::new(),
            raw_model: String::new(),
            model: String::new(),
            format: ChatFormat::OpenAI,
            history: Vec::new(),
            draft: String::new(),
            cursor: 0,
            command_menu: CommandMenuState::default(),
            draft_history: Vec::new(),
            draft_history_index: None,
            draft_history_stash: None,
            session_id: String::new(),
            overlay: Overlay::None,
            notice: None,
            show_reasoning: true,
            pending_response: String::new(),
            pending_reasoning: String::new(),
            pending_submit: None,
            sending: false,
            request_started_at: None,
            last_usage: None,
            context_tokens: 0,
            follow_output: true,
            transcript_scroll: 0,
            transcript_width: 0,
            transcript_view_height: 0,
            tx,
            rx,
            response_task: None,
            resume_task: None,
            resume_request_id: 0,
            loading_resume: None,
            resume_restore_state: None,
            reduce_motion: false,
            frame_tick: 0,
            picker_hitbox: None,
        }
    }

    #[test]
    fn test_markdown_renderer_formats_code_and_lists() {
        let lines = render_markdown_lines("## Title\n\n- one\n- two\n\n```rust\nlet x = 1;\n```");
        let plain = lines
            .into_iter()
            .map(|line| line.plain)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(plain.contains("Title"));
        assert!(plain.contains("• one"));
        assert!(plain.contains("rust"));
        assert!(plain.contains("let x = 1;"));
    }

    #[test]
    fn test_render_assistant_streaming_does_not_append_cursor_glyph() {
        let mut lines = Vec::new();
        render_assistant_streaming(&mut lines, true, None, "- item", true, 0, false);

        let plain = lines
            .into_iter()
            .map(|line| line.plain)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!plain.contains('▋'));
        assert!(plain.contains("• item"));
    }

    #[test]
    fn test_build_transcript_shows_streaming_reasoning_before_content() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        app.sending = true;
        app.pending_reasoning = "Inspecting the request".to_string();

        let transcript = app.build_transcript();
        let plain = transcript
            .text
            .lines
            .iter()
            .map(|line| plain_text_from_spans(&line.spans))
            .collect::<Vec<_>>()
            .join("\n");

        assert!(plain.contains("Thinking"));
        assert!(plain.contains("Inspecting the request"));
        assert!(!plain.contains("esc to interrupt"));
    }

    #[test]
    fn test_build_transcript_hides_reasoning_when_hidden() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        app.sending = true;
        app.show_reasoning = false;
        app.pending_reasoning = "Inspecting the request".to_string();

        let transcript = app.build_transcript();
        let plain = transcript
            .text
            .lines
            .iter()
            .map(|line| plain_text_from_spans(&line.spans))
            .collect::<Vec<_>>()
            .join("\n");

        assert!(!plain.contains("Inspecting the request"));
        assert!(!plain.contains("Thinking hidden"));
    }

    #[test]
    fn test_hidden_reasoning_hint_moves_to_composer_placeholder() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        app.history.push(ChatMessage {
            role: "assistant".to_string(),
            content: "answer".to_string(),
            reasoning_content: Some("private reasoning".to_string()),
        });

        let line = app.render_composer_text().lines[0].clone();
        let plain = plain_text_from_spans(&line.spans);

        assert_eq!(
            plain,
            ">  Ask anything · / for commands · Ctrl+T toggle think"
        );
    }

    #[test]
    fn test_normalized_reasoning_lines_trims_and_removes_blank_runs() {
        assert_eq!(
            normalized_reasoning_lines("\nalpha\n\n\nbeta\n\n"),
            vec!["alpha".to_string(), "beta".to_string()]
        );
    }

    #[test]
    fn test_truncate_for_width() {
        assert_eq!(truncate_for_width("hello", 10), "hello");
        assert_eq!(truncate_for_width("hello world", 6), "hello…");
    }

    #[test]
    fn test_build_footer_text_prefers_whole_segments() {
        assert_eq!(
            build_footer_text("gpt-4o", "https://openrouter.ai/api/v1", "/tmp/project", 80),
            "gpt-4o · openrouter.ai · project"
        );
        assert_eq!(
            build_footer_text("gpt-4o", "https://openrouter.ai/api/v1", "/tmp/project", 22),
            "gpt-4o · openrouter.ai"
        );
        assert_eq!(
            build_footer_text("gpt-4o", "https://openrouter.ai/api/v1", "/tmp/project", 6),
            "gpt-4o"
        );
    }

    #[test]
    fn test_footer_status_label_stays_token_count_while_streaming() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        app.sending = true;
        app.request_started_at = Some(Instant::now() - Duration::from_secs(12));
        app.context_tokens = 5_120;

        let (label, color) = app.footer_status_label();
        assert_eq!(label, "~5.1k tokens");
        assert_eq!(color, MUTED);
    }

    #[test]
    fn test_format_token_count_with_usage_shows_total() {
        assert_eq!(
            format_token_count(
                999,
                Some(TokenUsage {
                    prompt_tokens: 24,
                    completion_tokens: 11
                })
            ),
            "35 tokens"
        );
        assert_eq!(
            format_token_count(
                5_120,
                Some(TokenUsage {
                    prompt_tokens: 5_000,
                    completion_tokens: 120
                })
            ),
            "5.1k tokens"
        );
    }

    #[test]
    fn test_format_token_count_marks_estimates() {
        assert_eq!(format_token_count(0, None), "0 tokens");
        assert_eq!(format_token_count(105, None), "~105 tokens");
        assert_eq!(format_token_count(5_000, None), "~5k tokens");
        assert_eq!(format_token_count(12_345, None), "~12.3k tokens");
    }

    #[test]
    fn test_transcript_intro_lines_use_model_and_base_url() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        app.raw_model = "claude-sonnet-4".to_string();
        app.key = ApiKey::new_with_protocol(
            "prod".to_string(),
            "test".to_string(),
            "https://openrouter.ai/api/v1".to_string(),
            None,
            String::new(),
        );
        app.cwd = "/tmp/project".to_string();

        assert_eq!(
            app.transcript_intro_lines(),
            vec![
                "AIVO Chat".to_string(),
                "claude-sonnet-4 · https://openrouter.ai/api/v1".to_string(),
                "/tmp/project".to_string(),
            ]
        );
    }

    #[test]
    fn test_format_session_match_count() {
        assert_eq!(format_session_match_count(0, 0), "0 chats");
        assert_eq!(format_session_match_count(4, 4), "4 chats");
        assert_eq!(format_session_match_count(2, 5), "2/5");
    }

    #[test]
    fn test_truncate_for_display_width_handles_wide_text() {
        let truncated = truncate_for_display_width("你好🙂 hello", 8);
        assert!(display_width(&truncated) <= 8);
        assert!(truncated.ends_with('…'));
    }

    #[test]
    fn test_session_picker_item_line_fits_mixed_width_preview() {
        let preview = SessionPreview {
            key_id: "key-1".to_string(),
            key_name: "prod".to_string(),
            base_url: "https://api.example.com".to_string(),
            session_id: "session-1234".to_string(),
            raw_model: "deepseek".to_string(),
            updated_at: (Utc::now() - ChronoDuration::minutes(5)).to_rfc3339(),
            title: "hi".to_string(),
            preview_text: "hi · Hi there! ✨ 想聊点什么？还是需要我帮忙呢？ 我随时待命～ 😊🌟"
                .to_string(),
        };

        let line = session_picker_item_lines(&preview, true, false, 64)
            .into_iter()
            .next()
            .unwrap();
        let plain = plain_text_from_spans(&line.spans);
        assert!(display_width(&plain) <= 64);
    }

    #[test]
    fn test_key_picker_item_line_fits_modal_width() {
        let key = ApiKey::new_with_protocol(
            "deepseek".to_string(),
            "deepseek".to_string(),
            "https://api.cloudflare.com/client/v4/accounts/long/endpoint".to_string(),
            None,
            "sk-test".to_string(),
        );

        let line = key_picker_item_line(&key, true, 36);
        let plain = plain_text_from_spans(&line.spans);
        assert!(display_width(&plain) <= 36);
        assert!(plain.contains("deepseek"));
    }

    #[test]
    fn test_key_search_text_uses_host_not_full_path() {
        let key = ApiKey::new_with_protocol(
            "testgw".to_string(),
            "testgw".to_string(),
            "https://api.ai.example-gateway.net/endpoint".to_string(),
            None,
            "sk-test".to_string(),
        );

        let search = key_search_text(&key);
        assert!(search.contains("testgw"));
        assert!(search.contains("api.ai.example-gateway.net"));
        assert!(!search.contains("/endpoint"));
    }

    #[test]
    fn test_key_filter_does_not_match_across_full_url_path() {
        let unrelated = "groq groq api.groq.com";
        let target = "testgw testgw api.ai.example-gateway.net";

        assert!(matches_fuzzy("gapn", target));
        assert!(!matches_fuzzy("gapn", unrelated));
    }

    #[test]
    fn test_error_notice_only_returns_errors() {
        let error = (ERROR, "boom".to_string());
        let info = (MUTED, "ok".to_string());

        assert_eq!(error_notice(Some(&error)), Some("boom"));
        assert_eq!(error_notice(Some(&info)), None);
    }

    #[test]
    fn test_picker_visible_items_track_selection_for_single_line_rows() {
        let picker = PickerState {
            title: "Select model",
            query: String::new(),
            items: (0..6)
                .map(|index| PickerEntry {
                    label: format!("item-{index}"),
                    search_text: format!("item-{index}"),
                    value: PickerValue::Model(format!("item-{index}")),
                })
                .collect(),
            loading: false,
            selected: 4,
            kind: PickerKind::Session,
            pending_delete: None,
        };

        let visible = picker.visible_items(3);
        assert_eq!(visible.len(), 3);
        assert_eq!(visible[0].0, 2);
        assert_eq!(visible[2].0, 4);
    }

    #[test]
    fn test_picker_navigation_wraps() {
        let mut picker = PickerState {
            title: "Select model",
            query: String::new(),
            items: (0..3)
                .map(|index| PickerEntry {
                    label: format!("item-{index}"),
                    search_text: format!("item-{index}"),
                    value: PickerValue::Model(format!("item-{index}")),
                })
                .collect(),
            loading: false,
            selected: 0,
            kind: PickerKind::Session,
            pending_delete: None,
        };

        picker.select_prev();
        assert_eq!(picker.selected, 2);

        picker.select_next();
        assert_eq!(picker.selected, 0);
    }

    #[test]
    fn test_picker_visible_items_respect_single_line_session_rows() {
        let preview = SessionPreview {
            key_id: "key-1".to_string(),
            key_name: "prod".to_string(),
            base_url: "https://api.example.com".to_string(),
            session_id: "session-1234".to_string(),
            raw_model: "claude".to_string(),
            updated_at: (Utc::now() - ChronoDuration::hours(2)).to_rfc3339(),
            title: "Deploy status".to_string(),
            preview_text: "Deploy status for api gateway after rollout".to_string(),
        };
        let picker = PickerState {
            title: "Resume",
            query: String::new(),
            items: vec![
                PickerEntry {
                    label: "one".to_string(),
                    search_text: "one".to_string(),
                    value: PickerValue::Session(preview.clone()),
                },
                PickerEntry {
                    label: "two".to_string(),
                    search_text: "two".to_string(),
                    value: PickerValue::Session(preview.clone()),
                },
                PickerEntry {
                    label: "three".to_string(),
                    search_text: "three".to_string(),
                    value: PickerValue::Session(preview),
                },
            ],
            loading: false,
            selected: 2,
            kind: PickerKind::Session,
            pending_delete: None,
        };

        let visible = picker.visible_items(4);
        assert_eq!(visible.len(), 3);
        assert_eq!(visible[0].0, 0);
        assert_eq!(visible[2].0, 2);
    }

    #[test]
    fn test_rect_contains() {
        let area = Rect::new(10, 4, 8, 3);
        assert!(rect_contains(area, (10, 4)));
        assert!(rect_contains(area, (17, 6)));
        assert!(!rect_contains(area, (18, 6)));
        assert!(!rect_contains(area, (17, 7)));
    }

    #[test]
    fn test_format_request_elapsed() {
        assert_eq!(format_request_elapsed(Duration::from_secs(54)), "54s");
    }

    #[test]
    fn test_parse_slash_command_with_argument() {
        assert_eq!(
            parse_slash_command("model claude-sonnet-4").unwrap(),
            SlashCommand::Model(Some("claude-sonnet-4".to_string()))
        );
        assert_eq!(
            parse_slash_command("resume").unwrap(),
            SlashCommand::Resume(None)
        );
    }

    #[test]
    fn test_parse_slash_command_unknown() {
        let err = parse_slash_command("wat").unwrap_err().to_string();
        assert!(err.contains("Unknown command"));
    }

    #[test]
    fn test_restore_cancelled_submission_puts_prompt_back() {
        let mut history = vec![ChatMessage {
            role: "user".to_string(),
            content: "draft".to_string(),
            reasoning_content: None,
        }];
        let mut draft = String::new();
        let mut pending_submit = Some("draft".to_string());

        restore_cancelled_submission(&mut history, &mut draft, &mut pending_submit);

        assert!(history.is_empty());
        assert_eq!(draft, "draft");
        assert!(pending_submit.is_none());
    }

    #[test]
    fn test_prepare_for_model_picker_cancels_inflight_request() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        app.history.push(ChatMessage {
            role: "user".to_string(),
            content: "draft".to_string(),
            reasoning_content: None,
        });
        app.pending_submit = Some("draft".to_string());
        app.pending_response = "partial".to_string();
        app.sending = true;
        app.request_started_at = Some(Instant::now());

        app.prepare_for_model_picker();

        assert!(!app.sending);
        assert!(app.pending_response.is_empty());
        assert_eq!(app.draft, "draft");
        assert!(app.history.is_empty());
        assert!(app.request_started_at.is_none());
        assert_eq!(
            app.notice.as_ref().map(|(_, text)| text.as_str()),
            Some("Request cancelled")
        );
    }

    #[tokio::test]
    async fn test_interrupt_inflight_request_keeps_partial_response() {
        let temp_dir = TempDir::new().unwrap();
        let store = SessionStore::with_path(temp_dir.path().join("config.json"));
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        app.session_store = store;
        app.cwd = "/tmp/demo".to_string();
        app.session_id = "session-123".to_string();
        app.history.push(ChatMessage {
            role: "user".to_string(),
            content: "draft".to_string(),
            reasoning_content: None,
        });
        app.pending_submit = Some("draft".to_string());
        app.pending_response = "partial".to_string();
        app.sending = true;
        app.request_started_at = Some(Instant::now());

        app.interrupt_inflight_request().await.unwrap();

        assert!(!app.sending);
        assert!(app.pending_response.is_empty());
        assert!(app.pending_submit.is_none());
        assert!(app.draft.is_empty());
        assert_eq!(app.history.len(), 2);
        assert_eq!(app.history[1].role, "assistant");
        assert_eq!(app.history[1].content, "partial");
        assert_eq!(
            app.notice.as_ref().map(|(_, text)| text.as_str()),
            Some("Response interrupted")
        );
    }

    #[test]
    fn test_empty_composer_placeholder_reserves_cursor_cell() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let app = make_test_app(tx, rx);
        let line = app.render_composer_text().lines[0].clone();
        let plain = plain_text_from_spans(&line.spans);
        assert_eq!(plain, ">  Ask anything · / for commands");
    }

    #[test]
    fn test_overlay_hides_input_cursor() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        assert!(app.should_show_input_cursor());

        app.overlay = Overlay::Picker(Box::new(PickerState::loading(
            "Select model",
            String::new(),
            PickerKind::Model {
                target: ModelSelectionTarget::CurrentChat,
                auto_accept_exact: false,
            },
        )));

        assert!(!app.should_show_input_cursor());
    }

    #[test]
    fn test_persisted_draft_history_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("chat_history");
        let history = vec!["first".to_string(), "second".to_string()];

        save_persisted_draft_history_to_path(&path, &history).unwrap();

        assert_eq!(load_persisted_draft_history_from_path(&path), history);
    }

    #[test]
    fn test_session_preview_uses_last_user_message() {
        let preview = SessionPreview {
            key_id: "key-1".to_string(),
            key_name: "prod".to_string(),
            base_url: "https://api.example.com".to_string(),
            session_id: "session".to_string(),
            raw_model: "claude".to_string(),
            updated_at: (Utc::now() - ChronoDuration::hours(2)).to_rfc3339(),
            title: session_title_from_messages(
                &[
                    ChatMessage {
                        role: "assistant".to_string(),
                        content: "Hi".to_string(),
                        reasoning_content: None,
                    },
                    ChatMessage {
                        role: "user".to_string(),
                        content: "What is the deployment status for api gateway?".to_string(),
                        reasoning_content: None,
                    },
                ],
                "claude",
            ),
            preview_text: "What is the deployment status for api gateway?".to_string(),
        };

        assert_eq!(
            preview.title,
            "What is the deployment status for api gateway?".to_string()
        );
    }

    #[test]
    fn test_session_preview_text_uses_two_latest_turns() {
        let preview = session_preview_text_from_messages(
            &[
                ChatMessage {
                    role: "user".to_string(),
                    content: "hello".to_string(),
                    reasoning_content: None,
                },
                ChatMessage {
                    role: "assistant".to_string(),
                    content: "hi there".to_string(),
                    reasoning_content: None,
                },
            ],
            "claude",
        );

        assert_eq!(preview, "hello · hi there");
    }

    #[test]
    fn test_resume_metadata_spans_drop_labels_and_id() {
        let preview = SessionPreview {
            key_id: "key-1".to_string(),
            key_name: "prod".to_string(),
            base_url: "https://api.example.com".to_string(),
            session_id: "session-1234".to_string(),
            raw_model: "claude-sonnet-4-extended".to_string(),
            updated_at: (Utc::now() - ChronoDuration::hours(2)).to_rfc3339(),
            title: "Deploy status".to_string(),
            preview_text: "Deploy status for api gateway after rollout".to_string(),
        };

        let plain = plain_text_from_spans(&resume_metadata_spans(&preview, 40));
        assert!(plain.contains("2h"));
        assert!(plain.contains("prod"));
        assert!(plain.contains("claude"));
        assert!(!plain.contains("time"));
        assert!(!plain.contains("key"));
        assert!(!plain.contains("model"));
        assert!(!plain.contains("session-1"));
    }

    #[test]
    fn test_session_picker_item_line_shows_two_turn_preview() {
        let preview = SessionPreview {
            key_id: "key-1".to_string(),
            key_name: "prod".to_string(),
            base_url: "https://api.example.com".to_string(),
            session_id: "session-1234".to_string(),
            raw_model: "claude-sonnet-4-extended".to_string(),
            updated_at: (Utc::now() - ChronoDuration::hours(2)).to_rfc3339(),
            title: "Deploy status".to_string(),
            preview_text:
                "What is the deployment status for api gateway after the canary rollout finished?"
                    .to_string(),
        };

        let lines = session_picker_item_lines(&preview, false, false, 32);
        let first = plain_text_from_spans(&lines[0].spans);

        assert!(first.contains("What is"));
        assert!(first.chars().any(|ch| ch.is_ascii_digit()));
        assert!(!first.contains("key"));
    }

    #[tokio::test]
    async fn test_begin_resume_load_clears_transcript_before_result() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        app.history.push(ChatMessage {
            role: "user".to_string(),
            content: "old".to_string(),
            reasoning_content: None,
        });
        app.pending_response = "pending".to_string();
        app.draft = "draft".to_string();
        let preview = SessionPreview {
            key_id: app.key.id.clone(),
            key_name: app.key.display_name().to_string(),
            base_url: app.key.base_url.clone(),
            session_id: "session-1234".to_string(),
            raw_model: "claude".to_string(),
            updated_at: (Utc::now() - ChronoDuration::hours(2)).to_rfc3339(),
            title: "Deploy status".to_string(),
            preview_text: "Deploy status for api gateway after rollout".to_string(),
        };

        app.begin_resume_load(preview.clone());

        assert!(app.history.is_empty());
        assert!(app.pending_response.is_empty());
        assert!(app.draft.is_empty());
        assert_eq!(
            app.loading_resume
                .as_ref()
                .map(|loading| loading.preview.title.clone()),
            Some(preview.title)
        );
    }

    #[tokio::test]
    async fn test_delete_picker_selection_removes_saved_chat() {
        let temp_dir = TempDir::new().unwrap();
        let store = SessionStore::with_path(temp_dir.path().join("config.json"));
        let key_id = store
            .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
            .await
            .unwrap();
        store
            .save_chat_session_with_id(
                &key_id,
                "https://api.example.com",
                "/tmp/demo",
                "session-1234",
                "claude",
                &[
                    crate::services::session_store::StoredChatMessage {
                        role: "user".to_string(),
                        content: "hello".to_string(),
                        reasoning_content: None,
                        id: None,
                        timestamp: None,
                        attachments: None,
                    },
                    crate::services::session_store::StoredChatMessage {
                        role: "assistant".to_string(),
                        content: "hi there".to_string(),
                        reasoning_content: None,
                        id: None,
                        timestamp: None,
                        attachments: None,
                    },
                ],
                "hello",
                "hello · hi there",
            )
            .await
            .unwrap();

        let preview = SessionPreview {
            key_id: key_id.clone(),
            key_name: "prod".to_string(),
            base_url: "https://api.example.com".to_string(),
            session_id: "session-1234".to_string(),
            raw_model: "claude".to_string(),
            updated_at: (Utc::now() - ChronoDuration::minutes(5)).to_rfc3339(),
            title: "hello".to_string(),
            preview_text: "hello · hi there".to_string(),
        };

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        app.session_store = store.clone();
        app.cwd = "/tmp/demo".to_string();
        app.overlay = Overlay::Picker(Box::new(PickerState::ready(
            "Sessions",
            String::new(),
            vec![PickerEntry {
                label: preview.title.clone(),
                search_text: preview.search_text(),
                value: PickerValue::Session(preview),
            }],
            PickerKind::Session,
        )));

        app.delete_picker_selection(0).await.unwrap();

        assert!(matches!(app.overlay, Overlay::None));
        assert_eq!(
            app.notice.as_ref().map(|(_, text)| text.as_str()),
            Some("Saved chat deleted")
        );
        let saved = app
            .session_store
            .get_chat_session("session-1234")
            .await
            .unwrap();
        assert!(saved.is_none());
    }

    #[tokio::test]
    async fn test_ctrl_d_requires_confirmation_before_delete() {
        let temp_dir = TempDir::new().unwrap();
        let store = SessionStore::with_path(temp_dir.path().join("config.json"));
        let key_id = store
            .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
            .await
            .unwrap();
        store
            .save_chat_session_with_id(
                &key_id,
                "https://api.example.com",
                "/tmp/demo",
                "session-1234",
                "claude",
                &[crate::services::session_store::StoredChatMessage {
                    role: "user".to_string(),
                    content: "hello".to_string(),
                    reasoning_content: None,
                    id: None,
                    timestamp: None,
                    attachments: None,
                }],
                "hello",
                "hello",
            )
            .await
            .unwrap();

        let preview = SessionPreview {
            key_id: key_id.clone(),
            key_name: "prod".to_string(),
            base_url: "https://api.example.com".to_string(),
            session_id: "session-1234".to_string(),
            raw_model: "claude".to_string(),
            updated_at: (Utc::now() - ChronoDuration::minutes(5)).to_rfc3339(),
            title: "hello".to_string(),
            preview_text: "hello".to_string(),
        };

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        app.session_store = store.clone();
        app.cwd = "/tmp/demo".to_string();
        app.overlay = Overlay::Picker(Box::new(PickerState::ready(
            "Sessions",
            String::new(),
            vec![PickerEntry {
                label: preview.title.clone(),
                search_text: preview.search_text(),
                value: PickerValue::Session(preview),
            }],
            PickerKind::Session,
        )));

        app.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL))
            .await
            .unwrap();

        let saved = app
            .session_store
            .get_chat_session("session-1234")
            .await
            .unwrap();
        assert!(saved.is_some());
        let Overlay::Picker(picker) = &app.overlay else {
            panic!("expected picker overlay");
        };
        assert!(picker.pending_delete.is_some());

        app.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL))
            .await
            .unwrap();

        let saved = app
            .session_store
            .get_chat_session("session-1234")
            .await
            .unwrap();
        assert!(saved.is_none());
    }

    #[tokio::test]
    async fn test_resume_loaded_failure_restores_previous_state() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx.clone(), rx);
        app.history.push(ChatMessage {
            role: "user".to_string(),
            content: "old".to_string(),
            reasoning_content: None,
        });
        let preview = SessionPreview {
            key_id: app.key.id.clone(),
            key_name: app.key.display_name().to_string(),
            base_url: app.key.base_url.clone(),
            session_id: "session-1234".to_string(),
            raw_model: "claude".to_string(),
            updated_at: (Utc::now() - ChronoDuration::hours(2)).to_rfc3339(),
            title: "Deploy status".to_string(),
            preview_text: "Deploy status for api gateway after rollout".to_string(),
        };

        app.begin_resume_load(preview);
        let request_id = app.loading_resume.as_ref().unwrap().request_id;
        tx.send(RuntimeEvent::ResumeLoaded {
            request_id,
            result: Err("boom".to_string()),
        })
        .unwrap();

        app.handle_runtime_events().await.unwrap();

        assert_eq!(app.history.len(), 1);
        assert_eq!(app.history[0].content, "old");
        assert!(app.loading_resume.is_none());
        assert_eq!(
            app.notice.as_ref().map(|(_, text)| text.as_str()),
            Some("boom")
        );
    }
}
