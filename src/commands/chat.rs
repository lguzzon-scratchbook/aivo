/**
 * ChatCommand handler for interactive REPL with streaming API responses.
 * Tries OpenAI-compatible /v1/chat/completions first; falls back to
 * Anthropic's /v1/messages format if the provider returns 404/405.
 */
use std::io::{self, IsTerminal, Read, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use crate::tui::FuzzySelect;
use anyhow::Result;
use console::{Key, Term};
use reqwest::{Client, StatusCode};
use rustyline::{
    Context, Editor, Helper,
    completion::{Completer, Pair},
    error::ReadlineError,
    highlight::Highlighter,
    hint::Hinter,
    history::History,
    validate::Validator,
};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

use crate::commands::models::fetch_models_for_select;
use crate::commands::normalize_base_url;
use crate::errors::ExitCode;
use crate::services::copilot_auth::{
    COPILOT_EDITOR_VERSION, COPILOT_INTEGRATION_ID, COPILOT_OPENAI_INTENT, CopilotTokenManager,
};
use crate::services::http_utils::sse_data_payload;
use crate::services::model_names;
use crate::services::models_cache::ModelsCache;
use crate::services::session_store::{ApiKey, SessionStore, StoredChatMessage};
use crate::style;

const CMD_EXIT: &str = "/exit";
const CMD_HELP: &str = "/help";
const CMD_MODEL: &str = "/model";
const CMD_NEW: &str = "/new";
const CMD_RESUME: &str = "/resume";
/// Maximum number of messages to keep in chat history.
/// When exceeded, the oldest messages are dropped (keeping any system message).
const MAX_HISTORY_MESSAGES: usize = 50;
/// Retry budget for transient HTTP failures.
const MAX_REQUEST_ATTEMPTS: usize = 3;

struct ChatHelper {
    commands: Vec<&'static str>,
}

impl ChatHelper {
    fn new() -> Self {
        Self {
            commands: vec![CMD_EXIT, CMD_HELP, CMD_MODEL, CMD_NEW, CMD_RESUME],
        }
    }
}

impl Completer for ChatHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        if !line.starts_with('/') {
            return Ok((0, vec![]));
        }
        let prefix = &line[..pos];
        let completions = self
            .commands
            .iter()
            .filter(|&&cmd| cmd.starts_with(prefix))
            .map(|&cmd| Pair {
                display: cmd.to_string(),
                replacement: cmd.to_string(),
            })
            .collect();
        Ok((0, completions))
    }
}

impl Hinter for ChatHelper {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, _ctx: &Context<'_>) -> Option<String> {
        // Only hint when cursor is at end of input and input starts with /
        if pos < line.len() || !line.starts_with('/') {
            return None;
        }
        self.commands
            .iter()
            .find(|&&cmd| cmd.starts_with(line) && cmd != line)
            .map(|&cmd| cmd[pos..].to_string())
    }
}

impl Highlighter for ChatHelper {
    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        Cow::Owned(crate::style::dim(hint))
    }
}

impl Validator for ChatHelper {}

impl Helper for ChatHelper {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Clone)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct ChatChunk {
    choices: Vec<ChunkChoice>,
}

#[derive(Debug, Deserialize)]
struct ChunkChoice {
    delta: ChunkDelta,
}

#[derive(Debug, Deserialize)]
struct ChunkDelta {
    content: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct TokenUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
}

#[derive(Debug, Default)]
struct ChatTurnResult {
    content: String,
    usage: Option<TokenUsage>,
}

/// Which API format the provider speaks
#[derive(Debug, Clone, PartialEq)]
enum ChatFormat {
    /// OpenAI-compatible: POST /v1/chat/completions
    OpenAI,
    /// Anthropic native: POST /v1/messages
    Anthropic,
}

// Anthropic response structs

#[derive(Deserialize)]
struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    delta: Option<AnthropicDelta>,
}

#[derive(Deserialize)]
struct AnthropicDelta {
    text: Option<String>,
}

/// ChatCommand provides an interactive REPL for chatting with AI models
pub struct ChatCommand {
    session_store: SessionStore,
    cache: ModelsCache,
}

impl ChatCommand {
    pub fn new(session_store: SessionStore, cache: ModelsCache) -> Self {
        Self {
            session_store,
            cache,
        }
    }

    /// Resolves the model to use: --model flag > persisted per-key > None
    /// Returns None when the picker should be shown (no flag, no persisted, or --model with no value).
    async fn resolve_model(
        &self,
        key_id: &str,
        flag_model: Option<String>,
    ) -> Result<Option<String>> {
        match flag_model {
            // --model with no value → force picker (bypass persisted model)
            Some(ref m) if m.is_empty() => Ok(None),
            // --model <value> → use it and save
            Some(model) => {
                let current = self.session_store.get_chat_model(key_id).await?;
                if current.as_deref() != Some(&model) {
                    self.session_store.set_chat_model(key_id, &model).await?;
                }
                Ok(Some(model))
            }
            None => self.session_store.get_chat_model(key_id).await,
        }
    }

    /// Fetches the model list (cache-first) with a spinner for network fetches.
    async fn fetch_models_for_select(&self, client: &Client, key: &ApiKey) -> Vec<String> {
        fetch_models_for_select(client, key, &self.cache).await
    }

    /// Transforms model names for OpenRouter compatibility
    /// OpenRouter uses dots in version numbers: 4.6 instead of 4-6
    fn transform_model_for_provider(base_url: &str, model: &str) -> String {
        model_names::transform_model_for_provider(base_url, model)
    }

    pub async fn execute(
        &self,
        model: Option<String>,
        one_shot: Option<String>,
        key_override: Option<ApiKey>,
    ) -> ExitCode {
        match self.execute_internal(model, one_shot, key_override).await {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                ExitCode::UserError
            }
        }
    }

    async fn execute_internal(
        &self,
        model_flag: Option<String>,
        one_shot: Option<String>,
        key_override: Option<ApiKey>,
    ) -> Result<ExitCode> {
        let explicit_model_requested = model_flag.is_some();
        let key = match key_override {
            Some(k) => k,
            None => match self.session_store.get_active_key().await? {
                Some(k) => k,
                None => {
                    eprintln!(
                        "{} No API key configured. Run 'aivo keys add' first.",
                        style::red("Error:")
                    );
                    return Ok(ExitCode::AuthError);
                }
            },
        };

        let client = crate::services::http_utils::router_http_client();

        let mut raw_model = match self.resolve_model(&key.id, model_flag).await? {
            Some(m) => m,
            None => {
                // No model set for this key — prompt user to select one
                let models_list = self.fetch_models_for_select(&client, &key).await;

                if models_list.is_empty() {
                    anyhow::bail!(
                        "No model configured and could not fetch model list. Use --model <name> to specify one."
                    );
                }

                match FuzzySelect::new()
                    .with_prompt("Select model")
                    .items(&models_list)
                    .default(0)
                    .interact_opt()
                    .ok()
                    .flatten()
                    .map(|idx| models_list[idx].clone())
                {
                    Some(selected) => {
                        self.session_store
                            .set_chat_model(&key.id, &selected)
                            .await?;
                        selected
                    }
                    None => return Ok(ExitCode::Success),
                }
            }
        };
        let mut model = Self::transform_model_for_provider(&key.base_url, &raw_model);
        let cwd =
            crate::services::system_env::current_dir_string().unwrap_or_else(|| ".".to_string());

        // Create once so its token cache is reused across messages in the session.
        let copilot_tm = if key.base_url == "copilot" {
            Some(CopilotTokenManager::new(key.key.as_str().to_string()))
        } else {
            None
        };

        if let Some(input) = one_shot {
            let input = input.trim().to_string();
            if input.is_empty() {
                anyhow::bail!("Message for -x/--execute cannot be empty");
            }

            let stdin_context = read_stdin_if_piped()?;
            let one_shot_input = compose_one_shot_prompt(&input, stdin_context.as_deref());

            let history = vec![ChatMessage {
                role: "user".to_string(),
                content: one_shot_input,
            }];
            let mut format = ChatFormat::OpenAI;
            self.session_store
                .record_selection(&key.id, "chat", Some(&raw_model))
                .await?;
            let (spinning, spinner_handle) = style::start_spinner(None);
            let result = send_message_turn(
                &client,
                &key,
                copilot_tm.as_ref(),
                &model,
                &history,
                &mut format,
                &spinning,
            )
            .await;
            style::stop_spinner(&spinning);
            let _ = spinner_handle.await;

            match result {
                Ok(turn) => {
                    if let Some(usage) = turn.usage {
                        self.session_store
                            .record_tokens(
                                &key.id,
                                Some(&raw_model),
                                usage.prompt_tokens,
                                usage.completion_tokens,
                            )
                            .await?;
                    }
                    println!();
                    return Ok(ExitCode::Success);
                }
                Err(e) => return Err(e),
            }
        }

        eprintln!(
            "{} model: {} {}",
            style::success_symbol(),
            style::cyan(&model),
            style::dim(format!("({})", key.base_url))
        );
        eprintln!(
            "{}",
            style::dim("Type a message, or use /help for chat commands.")
        );
        let mut history: Vec<ChatMessage> = Vec::new();
        let mut format = ChatFormat::OpenAI;
        let mut session_id = new_chat_session_id();
        let prompt = format!("{} ", style::cyan(">"));

        let mut rl = Editor::<ChatHelper, rustyline::history::DefaultHistory>::new()
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        rl.set_helper(Some(ChatHelper::new()));

        let history_path: PathBuf = crate::services::system_env::home_dir()
            .map(|p| p.join(".config").join("aivo").join("chat_history"))
            .unwrap_or_else(|| PathBuf::from(".config/aivo/chat_history"));
        if let Ok(data) = std::fs::read_to_string(&history_path)
            && let Ok(plain) = crate::services::session_store::decrypt(&data)
        {
            for line in plain.lines() {
                if !line.is_empty() {
                    let _ = rl.add_history_entry(line);
                }
            }
        }

        if let Some((saved_session, saved_messages)) = self
            .session_store
            .get_chat_session(&key.id, &key.base_url, &cwd)
            .await?
            .and_then(|session| {
                let msgs = session.decrypt_messages().ok()?;
                (!msgs.is_empty()).then_some((session, msgs))
            })
        {
            let resume = prompt_yes_no(
                &format!(
                    "Resume your last session in this directory? ({} messages)",
                    saved_messages.len()
                ),
                true,
            )?;
            if resume {
                session_id = saved_session.session_id.clone();
                if !explicit_model_requested {
                    raw_model = saved_session.model.clone();
                    model = Self::transform_model_for_provider(&key.base_url, &raw_model);
                }
                history = saved_messages
                    .into_iter()
                    .map(|message| ChatMessage {
                        role: message.role,
                        content: message.content,
                    })
                    .collect();
                eprintln!(
                    "{} Resumed {} messages",
                    style::success_symbol(),
                    history.len()
                );
                eprintln!("{}", style::dim(format!("Model: {}", model)));
            }
        }

        self.session_store
            .record_selection(&key.id, "chat", Some(&raw_model))
            .await?;

        loop {
            let input = match rl.readline(&prompt) {
                Ok(line) => line,
                Err(ReadlineError::Eof | ReadlineError::Interrupted) => {
                    eprintln!();
                    break;
                }
                Err(_) => break,
            };

            let input = input.trim().to_string();

            if input.is_empty() {
                continue;
            }

            rl.add_history_entry(&input)
                .map_err(|e| anyhow::anyhow!("{}", e))?;

            if input == CMD_EXIT {
                break;
            }

            if input == CMD_HELP {
                print_chat_commands();
                continue;
            }

            if input == CMD_NEW {
                history.clear();
                session_id = new_chat_session_id();
                format = ChatFormat::OpenAI;
                eprintln!("{}", style::dim("Started a new chat."));
                continue;
            }

            if input == CMD_RESUME {
                let sessions = self
                    .session_store
                    .list_chat_sessions(&key.id, &key.base_url, &cwd)
                    .await?;
                if sessions.is_empty() {
                    eprintln!("{}", style::dim("No saved chats in this directory yet."));
                    continue;
                }

                let items = sessions
                    .iter()
                    .map(format_session_choice)
                    .collect::<Vec<_>>();
                let current_idx = sessions
                    .iter()
                    .position(|session| session.session_id == session_id)
                    .unwrap_or(0);

                let selected = FuzzySelect::new()
                    .with_prompt("Resume chat")
                    .items(&items)
                    .default(current_idx)
                    .interact_opt()
                    .ok()
                    .flatten();

                if let Some(idx) = selected {
                    let session = &sessions[idx];
                    session_id = session.session_id.clone();
                    raw_model = session.model.clone();
                    model = Self::transform_model_for_provider(&key.base_url, &raw_model);
                    self.session_store
                        .set_chat_model(&key.id, &raw_model)
                        .await?;
                    history = session
                        .decrypt_messages()?
                        .into_iter()
                        .map(|message| ChatMessage {
                            role: message.role,
                            content: message.content,
                        })
                        .collect();
                    format = ChatFormat::OpenAI;
                    eprintln!(
                        "{}",
                        style::dim(format!("Resumed {} messages · {}", history.len(), model))
                    );
                }
                continue;
            }

            if input == CMD_MODEL {
                let models_list = self.fetch_models_for_select(&client, &key).await;
                if models_list.is_empty() {
                    eprintln!("{}", style::dim(format!("Keeping {}", model)));
                    continue;
                }

                let current_idx = models_list
                    .iter()
                    .position(|m| m == &raw_model)
                    .unwrap_or(0);
                if let Some(idx) = FuzzySelect::new()
                    .with_prompt("Select model")
                    .items(&models_list)
                    .default(current_idx)
                    .interact_opt()
                    .ok()
                    .flatten()
                {
                    let raw = models_list[idx].clone();
                    self.session_store.set_chat_model(&key.id, &raw).await?;
                    self.session_store
                        .record_selection(&key.id, "chat", Some(&raw))
                        .await?;
                    raw_model = raw.clone();
                    model = Self::transform_model_for_provider(&key.base_url, &raw);
                    eprintln!("  model: {}", style::cyan(&model));
                    if !history.is_empty() {
                        self.session_store
                            .save_chat_session_with_id(
                                &key.id,
                                &key.base_url,
                                &cwd,
                                &session_id,
                                &raw_model,
                                &to_stored_messages(&history),
                            )
                            .await?;
                    }
                }
                continue;
            }

            if input.starts_with('/') {
                let cmd = input.split_whitespace().next().unwrap_or(&input);
                eprintln!(
                    "{} Unknown command: {}",
                    style::yellow("Warning:"),
                    style::cyan(cmd)
                );
                continue;
            }

            // Add user message to history and trim if needed
            history.push(ChatMessage {
                role: "user".to_string(),
                content: input,
            });
            trim_history(&mut history, MAX_HISTORY_MESSAGES);

            // Start loading spinner
            let (spinning, spinner_handle) = style::start_spinner(None);

            // Stream response, auto-detecting provider format
            let result = send_message_turn(
                &client,
                &key,
                copilot_tm.as_ref(),
                &model,
                &history,
                &mut format,
                &spinning,
            )
            .await;

            style::stop_spinner(&spinning);
            let _ = spinner_handle.await;
            match result {
                Ok(turn) => {
                    // Ensure newline after streamed response
                    println!();
                    history.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: turn.content,
                    });
                    if let Some(usage) = turn.usage {
                        self.session_store
                            .record_tokens(
                                &key.id,
                                Some(&raw_model),
                                usage.prompt_tokens,
                                usage.completion_tokens,
                            )
                            .await?;
                    }
                    self.session_store
                        .save_chat_session_with_id(
                            &key.id,
                            &key.base_url,
                            &cwd,
                            &session_id,
                            &raw_model,
                            &to_stored_messages(&history),
                        )
                        .await?;
                }
                Err(e) => {
                    eprintln!("\n{} {}", style::red("Error:"), e);
                    // Remove the failed user message so user can retry
                    history.pop();
                }
            }
        }

        if !rl.history().is_empty() {
            let joined = rl
                .history()
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            if let Ok(encrypted) = crate::services::session_store::encrypt(&joined) {
                if let Some(parent) = history_path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                if std::fs::write(&history_path, &encrypted).is_ok() {
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        let _ = std::fs::set_permissions(
                            &history_path,
                            std::fs::Permissions::from_mode(0o600),
                        );
                    }
                }
            }
        }
        Ok(ExitCode::Success)
    }

    pub fn print_help() {
        println!(
            "{} aivo chat [--model <model>] [-x <message>]",
            style::bold("Usage:")
        );
        println!();
        println!(
            "{}",
            style::dim("Start an interactive chat REPL with streaming responses.")
        );
        println!(
            "{}",
            style::dim("Uses the active API key to call the chat completions endpoint.")
        );
        println!();
        println!("{}", style::bold("Options:"));
        println!(
            "  {}  {}",
            style::cyan("-m, --model <model>"),
            style::dim("Specify AI model (saved for next session)")
        );
        println!(
            "  {}  {}",
            style::cyan("-k, --key <id|name>"),
            style::dim("Select API key by ID or name")
        );
        println!(
            "  {}  {}",
            style::cyan("-x, --execute <message>"),
            style::dim("Send one message and exit (uses piped stdin as context)")
        );
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo chat"));
        println!("  {}", style::dim("aivo chat --model gpt-4o"));
        println!("  {}", style::dim("aivo chat -m claude-sonnet-4-5"));
        println!(
            "  {}",
            style::dim("aivo chat -x \"Explain Rust lifetimes\"")
        );
        println!(
            "  {}",
            style::dim("git diff --cached | aivo chat -x \"Summarize changes in one sentence\"")
        );
    }
}

async fn send_message_turn(
    client: &Client,
    key: &ApiKey,
    copilot_tm: Option<&CopilotTokenManager>,
    model: &str,
    history: &[ChatMessage],
    format: &mut ChatFormat,
    spinning: &Arc<AtomicBool>,
) -> Result<ChatTurnResult> {
    if let Some(tm) = copilot_tm {
        return send_copilot_request(client, tm, model, history, spinning).await;
    }

    match format {
        ChatFormat::OpenAI => {
            match send_chat_request(client, key, model, history, spinning).await {
                ok @ Ok(_) => ok,
                Err(e) if is_format_mismatch(&e) => {
                    // Provider doesn't speak OpenAI format; try Anthropic
                    match send_anthropic_request(client, key, model, history, spinning).await {
                        Ok(content) => {
                            eprintln!("{}", style::dim("  (using Anthropic messages format)"));
                            *format = ChatFormat::Anthropic;
                            Ok(content)
                        }
                        Err(_) => Err(e), // both failed; report original error
                    }
                }
                Err(e) => Err(e),
            }
        }
        ChatFormat::Anthropic => {
            send_anthropic_request(client, key, model, history, spinning).await
        }
    }
}

fn read_stdin_if_piped() -> Result<Option<String>> {
    if io::stdin().is_terminal() {
        return Ok(None);
    }

    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf)?;
    if buf.trim().is_empty() {
        Ok(None)
    } else {
        Ok(Some(buf))
    }
}

fn compose_one_shot_prompt(prompt: &str, stdin_context: Option<&str>) -> String {
    match stdin_context.map(str::trim).filter(|c| !c.is_empty()) {
        Some(ctx) => format!("{prompt}\n\nContext from stdin:\n{ctx}"),
        None => prompt.to_string(),
    }
}

fn to_stored_messages(history: &[ChatMessage]) -> Vec<StoredChatMessage> {
    history
        .iter()
        .map(|message| StoredChatMessage {
            role: message.role.clone(),
            content: message.content.clone(),
        })
        .collect()
}

fn print_chat_commands() {
    let rows = [
        (CMD_HELP, "Show chat commands"),
        (CMD_MODEL, "Pick a different model"),
        (CMD_NEW, "Start a fresh chat"),
        (CMD_RESUME, "Resume a saved chat"),
        (CMD_EXIT, "Leave chat"),
    ];
    let width = rows.iter().map(|(cmd, _)| cmd.len()).max().unwrap_or(0);

    eprintln!("{}", style::bold("Commands"));
    for (cmd, description) in rows {
        eprintln!(
            "  {}  {}",
            style::cyan(format!("{cmd:<width$}")),
            style::dim(description)
        );
    }
}

fn new_chat_session_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .to_string()
}

fn format_session_choice(session: &crate::services::session_store::ChatSessionState) -> String {
    let updated = session.updated_at.split('T').collect::<Vec<_>>();
    let updated = if updated.len() == 2 {
        let time = updated[1]
            .split('.')
            .next()
            .unwrap_or(updated[1])
            .trim_end_matches('Z');
        format!("{} {}", updated[0], time)
    } else {
        session.updated_at.clone()
    };

    format!(
        "{} · {} messages · {}",
        session.model,
        session.message_count(),
        updated
    )
}

fn prompt_yes_no(prompt: &str, default_yes: bool) -> io::Result<bool> {
    let suffix = if default_yes { "[Y/n]" } else { "[y/N]" };
    let term = Term::stdout();
    term.write_str(&format!("{} {} ", prompt, suffix))?;

    loop {
        match term.read_key()? {
            Key::Enter => {
                term.write_line(if default_yes { "y" } else { "n" })?;
                return Ok(default_yes);
            }
            Key::Char('y') | Key::Char('Y') => {
                term.write_line("y")?;
                return Ok(true);
            }
            Key::Char('n') | Key::Char('N') | Key::Escape => {
                term.write_line("n")?;
                return Ok(false);
            }
            _ => {}
        }
    }
}


/// Extracts assistant text from OpenAI-compatible non-streaming chat responses.
fn extract_openai_message_content(body: &serde_json::Value) -> String {
    if let Some(content) = body["choices"][0]["message"]["content"].as_str() {
        return content.to_string();
    }

    // Some providers return content as an array of typed parts.
    body["choices"][0]["message"]["content"]
        .as_array()
        .iter()
        .flat_map(|parts| parts.iter())
        .filter_map(|part| {
            part.get("text")
                .and_then(|v| v.as_str())
                .or_else(|| part.get("content").and_then(|v| v.as_str()))
        })
        .collect()
}

fn extract_openai_usage(body: &serde_json::Value) -> Option<TokenUsage> {
    let usage = body.get("usage")?;
    Some(TokenUsage {
        prompt_tokens: usage
            .get("prompt_tokens")
            .and_then(|value| value.as_u64())
            .unwrap_or(0),
        completion_tokens: usage
            .get("completion_tokens")
            .and_then(|value| value.as_u64())
            .unwrap_or(0),
    })
}

fn extract_anthropic_usage(body: &serde_json::Value) -> Option<TokenUsage> {
    let usage = body.get("usage")?;
    Some(TokenUsage {
        prompt_tokens: usage
            .get("input_tokens")
            .and_then(|value| value.as_u64())
            .unwrap_or(0),
        completion_tokens: usage
            .get("output_tokens")
            .and_then(|value| value.as_u64())
            .unwrap_or(0),
    })
}

fn parse_openai_usage_chunk(data: &str) -> Option<TokenUsage> {
    let value = serde_json::from_str::<serde_json::Value>(data).ok()?;
    extract_openai_usage(&value)
}

fn parse_anthropic_usage_chunk(data: &str) -> Option<TokenUsage> {
    let value = serde_json::from_str::<serde_json::Value>(data).ok()?;
    extract_anthropic_usage(&value)
}

fn should_retry_status(status: StatusCode) -> bool {
    status.is_server_error()
        || status == StatusCode::TOO_MANY_REQUESTS
        || status == StatusCode::REQUEST_TIMEOUT
}

fn should_retry_error(err: &reqwest::Error) -> bool {
    err.is_timeout() || err.is_connect() || err.is_request() || err.is_body()
}

fn retry_delay(attempt: usize, retry_after: Option<&reqwest::header::HeaderValue>) -> Duration {
    if let Some(seconds) = retry_after
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<u64>().ok())
    {
        return Duration::from_secs(seconds.min(30));
    }
    let exp = 250u64.saturating_mul(1u64 << (attempt.saturating_sub(1).min(4)));
    Duration::from_millis(exp.min(4000))
}

async fn send_with_retry<F>(mut build_request: F) -> Result<reqwest::Response>
where
    F: FnMut() -> reqwest::RequestBuilder,
{
    let mut last_err: Option<anyhow::Error> = None;

    for attempt in 1..=MAX_REQUEST_ATTEMPTS {
        match build_request().send().await {
            Ok(response) => {
                if should_retry_status(response.status()) && attempt < MAX_REQUEST_ATTEMPTS {
                    let delay = retry_delay(
                        attempt,
                        response.headers().get(reqwest::header::RETRY_AFTER),
                    );
                    let _ = response.bytes().await;
                    tokio::time::sleep(delay).await;
                    continue;
                }
                return Ok(response);
            }
            Err(err) => {
                if should_retry_error(&err) && attempt < MAX_REQUEST_ATTEMPTS {
                    tokio::time::sleep(retry_delay(attempt, None)).await;
                    continue;
                }
                last_err = Some(err.into());
                break;
            }
        }
    }

    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("Request failed")))
}

/// Sends a chat completion request and prints the response.
/// Tries streaming first; falls back to non-streaming if the server returns a 5xx error.
/// Returns the full assistant message content.
async fn send_chat_request(
    client: &Client,
    key: &ApiKey,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
) -> Result<ChatTurnResult> {
    let base = normalize_base_url(&key.base_url);
    let url = format!("{}/v1/chat/completions", base);

    // Try streaming first; fall back to non-streaming on server errors
    let request = ChatRequest {
        model: model.to_string(),
        messages: messages.to_vec(),
        stream: true,
    };

    let mut response = send_with_retry(|| {
        client
            .post(&url)
            .header("Authorization", format!("Bearer {}", key.key.as_str()))
            .header("Content-Type", "application/json")
            .header("User-Agent", format!("aivo/{}", crate::version::VERSION))
            .json(&request)
    })
    .await?;

    // If the server can't handle streaming, fall back to non-streaming.
    // Note: 404 is NOT included here — it means wrong endpoint, not streaming unsupported.
    // The caller detects 404 and switches to a different API format instead.
    if response.status().is_server_error() {
        return send_non_streaming(client, &url, key, model, messages, spinning).await;
    }

    if !response.status().is_success() {
        style::stop_spinner(spinning);
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("API returned {} — {}", status, body);
    }

    let mut full_content = String::new();
    let mut usage = None;
    let mut line_buf = String::new();
    let mut done = false;

    while !done {
        let Some(chunk) = response.chunk().await? else {
            break;
        };
        let text = String::from_utf8_lossy(&chunk);
        line_buf.push_str(&text);

        while let Some(pos) = line_buf.find('\n') {
            let line = line_buf[..pos].trim_end_matches('\r').to_string();
            line_buf = line_buf[pos + 1..].to_string();

            if let Some(data) = sse_data_payload(&line) {
                if data.trim() == "[DONE]" {
                    done = true;
                    break;
                }
                if let Some(tokens) = parse_openai_usage_chunk(data) {
                    usage = Some(tokens);
                }
                if let Some(content) = parse_sse_chunk(data) {
                    style::stop_spinner(spinning);
                    print!("{}", content);
                    io::stdout().flush()?;
                    full_content.push_str(&content);
                }
            }
        }
    }

    let tail = line_buf.trim();
    if !tail.is_empty() {
        if let Some(data) = sse_data_payload(tail) {
            if let Some(tokens) = parse_openai_usage_chunk(data) {
                usage = Some(tokens);
            }
            if data.trim() != "[DONE]"
                && let Some(content) = parse_sse_chunk(data)
            {
                style::stop_spinner(spinning);
                print!("{}", content);
                io::stdout().flush()?;
                full_content.push_str(&content);
            }
        } else if full_content.is_empty()
            && let Ok(resp) = serde_json::from_str::<serde_json::Value>(tail)
        {
            let content = extract_openai_message_content(&resp);
            if !content.is_empty() {
                style::stop_spinner(spinning);
                print!("{}", content);
                io::stdout().flush()?;
                full_content = content;
            }
        }
    }

    if full_content.is_empty() {
        return send_non_streaming(client, &url, key, model, messages, spinning).await;
    }

    Ok(ChatTurnResult {
        content: full_content,
        usage,
    })
}

/// Non-streaming fallback for gateways that don't support SSE streaming.
async fn send_non_streaming(
    client: &Client,
    url: &str,
    key: &ApiKey,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
) -> Result<ChatTurnResult> {
    let request = ChatRequest {
        model: model.to_string(),
        messages: messages.to_vec(),
        stream: false,
    };

    let response = send_with_retry(|| {
        client
            .post(url)
            .header("Authorization", format!("Bearer {}", key.key.as_str()))
            .header("Content-Type", "application/json")
            .header("User-Agent", format!("aivo/{}", crate::version::VERSION))
            .json(&request)
    })
    .await?;

    if !response.status().is_success() {
        style::stop_spinner(spinning);
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("API returned {} — {}", status, body);
    }

    let body: serde_json::Value = response.json().await?;
    let content = extract_openai_message_content(&body);
    let usage = extract_openai_usage(&body);

    if content.is_empty() {
        style::stop_spinner(spinning);
        anyhow::bail!("Provider returned an empty response");
    }

    style::stop_spinner(spinning);
    print!("{}", content);
    io::stdout().flush()?;

    Ok(ChatTurnResult { content, usage })
}

/// Sends a chat request via GitHub Copilot (token exchange + Copilot API).
async fn send_copilot_request(
    client: &Client,
    tm: &CopilotTokenManager,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
) -> Result<ChatTurnResult> {
    let (copilot_token, api_endpoint) = tm.get_token().await?;
    let url = format!("{}/chat/completions", api_endpoint.trim_end_matches('/'));

    let request = ChatRequest {
        model: model.to_string(),
        messages: messages.to_vec(),
        stream: true,
    };

    let mut response = send_with_retry(|| {
        client
            .post(&url)
            .header("Authorization", format!("Bearer {}", copilot_token))
            .header("Content-Type", "application/json")
            .header("Editor-Version", COPILOT_EDITOR_VERSION)
            .header("Copilot-Integration-Id", COPILOT_INTEGRATION_ID)
            .header("Openai-Intent", COPILOT_OPENAI_INTENT)
            .json(&request)
    })
    .await?;

    if response.status().is_server_error() {
        return send_copilot_non_streaming(client, &url, &copilot_token, model, messages, spinning)
            .await;
    }

    if !response.status().is_success() {
        style::stop_spinner(spinning);
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("API returned {} — {}", status, body);
    }

    let mut full_content = String::new();
    let mut usage = None;
    let mut line_buf = String::new();
    let mut done = false;

    while !done {
        let Some(chunk) = response.chunk().await? else {
            break;
        };
        let text = String::from_utf8_lossy(&chunk);
        line_buf.push_str(&text);

        while let Some(pos) = line_buf.find('\n') {
            let line = line_buf[..pos].trim_end_matches('\r').to_string();
            line_buf = line_buf[pos + 1..].to_string();

            if let Some(data) = sse_data_payload(&line) {
                if data.trim() == "[DONE]" {
                    done = true;
                    break;
                }
                if let Some(tokens) = parse_openai_usage_chunk(data) {
                    usage = Some(tokens);
                }
                if let Some(content) = parse_sse_chunk(data) {
                    style::stop_spinner(spinning);
                    print!("{}", content);
                    io::stdout().flush()?;
                    full_content.push_str(&content);
                }
            }
        }
    }

    let tail = line_buf.trim();
    if !tail.is_empty() {
        if let Some(data) = sse_data_payload(tail) {
            if let Some(tokens) = parse_openai_usage_chunk(data) {
                usage = Some(tokens);
            }
            if data.trim() != "[DONE]"
                && let Some(content) = parse_sse_chunk(data)
            {
                style::stop_spinner(spinning);
                print!("{}", content);
                io::stdout().flush()?;
                full_content.push_str(&content);
            }
        } else if full_content.is_empty()
            && let Ok(resp) = serde_json::from_str::<serde_json::Value>(tail)
        {
            let content = extract_openai_message_content(&resp);
            if !content.is_empty() {
                style::stop_spinner(spinning);
                print!("{}", content);
                io::stdout().flush()?;
                full_content = content;
            }
        }
    }

    if full_content.is_empty() {
        return send_copilot_non_streaming(client, &url, &copilot_token, model, messages, spinning)
            .await;
    }

    Ok(ChatTurnResult {
        content: full_content,
        usage,
    })
}

async fn send_copilot_non_streaming(
    client: &Client,
    url: &str,
    copilot_token: &str,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
) -> Result<ChatTurnResult> {
    let request = ChatRequest {
        model: model.to_string(),
        messages: messages.to_vec(),
        stream: false,
    };

    let response = send_with_retry(|| {
        client
            .post(url)
            .header("Authorization", format!("Bearer {}", copilot_token))
            .header("Content-Type", "application/json")
            .header("Editor-Version", COPILOT_EDITOR_VERSION)
            .header("Copilot-Integration-Id", COPILOT_INTEGRATION_ID)
            .header("Openai-Intent", COPILOT_OPENAI_INTENT)
            .json(&request)
    })
    .await?;

    if !response.status().is_success() {
        style::stop_spinner(spinning);
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("API returned {} — {}", status, body);
    }

    let body: serde_json::Value = response.json().await?;
    let content = extract_openai_message_content(&body);
    let usage = extract_openai_usage(&body);

    if content.is_empty() {
        style::stop_spinner(spinning);
        anyhow::bail!("Provider returned an empty response");
    }

    style::stop_spinner(spinning);
    print!("{}", content);
    io::stdout().flush()?;

    Ok(ChatTurnResult { content, usage })
}

/// Parses a single SSE data chunk and extracts the content delta
pub fn parse_sse_chunk(data: &str) -> Option<String> {
    let chunk: ChatChunk = serde_json::from_str(data).ok()?;
    chunk.choices.first()?.delta.content.clone()
}

/// Trims chat history to keep at most `max_messages` messages.
/// If there's a system message at the start, it's always preserved.
/// Drops the oldest non-system messages first.
fn trim_history(history: &mut Vec<ChatMessage>, max_messages: usize) {
    if history.len() <= max_messages {
        return;
    }

    let has_system = history.first().is_some_and(|m| m.role == "system");

    if has_system {
        // Keep the system message + last (max_messages - 1) messages
        let keep_from = history.len() - (max_messages - 1);
        let system_msg = history[0].clone();
        let kept: Vec<ChatMessage> = std::iter::once(system_msg)
            .chain(history[keep_from..].iter().cloned())
            .collect();
        *history = kept;
    } else {
        // Keep the last max_messages messages
        let keep_from = history.len() - max_messages;
        *history = history[keep_from..].to_vec();
    }
}

/// Returns true when the error indicates the endpoint doesn't exist,
/// meaning we should try a different API format.
fn is_format_mismatch(e: &anyhow::Error) -> bool {
    let msg = e.to_string().to_lowercase();
    msg.contains("404")
        || msg.contains("405")
        || (msg.contains("not found")
            && (msg.contains("endpoint") || msg.contains("route") || msg.contains("path")))
        || (msg.contains("method not allowed")
            && (msg.contains("endpoint") || msg.contains("route") || msg.contains("path")))
}

/// Sends a request using Anthropic's native /v1/messages API.
/// Tries streaming first; falls back to non-streaming on server errors.
async fn send_anthropic_request(
    client: &Client,
    key: &ApiKey,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
) -> Result<ChatTurnResult> {
    let base = normalize_base_url(&key.base_url);
    let url = format!("{}/v1/messages", base);

    let request = serde_json::json!({
        "model": model,
        "messages": messages,
        "max_tokens": 8096,
        "stream": true,
    });

    let mut response = send_with_retry(|| {
        client
            .post(&url)
            // Send both auth headers: gateways vary on which they accept
            .header("Authorization", format!("Bearer {}", key.key.as_str()))
            .header("x-api-key", key.key.as_str())
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .header("User-Agent", format!("aivo/{}", crate::version::VERSION))
            .json(&request)
    })
    .await?;

    if response.status().is_server_error() || response.status() == reqwest::StatusCode::NOT_FOUND {
        return send_anthropic_non_streaming(client, &url, key, model, messages, spinning).await;
    }

    if !response.status().is_success() {
        style::stop_spinner(spinning);
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("API returned {} — {}", status, body);
    }

    let mut full_content = String::new();
    let mut usage = None;
    let mut line_buf = String::new();

    while let Some(chunk) = response.chunk().await? {
        let text = String::from_utf8_lossy(&chunk);
        line_buf.push_str(&text);

        while let Some(pos) = line_buf.find('\n') {
            let line = line_buf[..pos].trim_end_matches('\r').to_string();
            line_buf = line_buf[pos + 1..].to_string();

            if let Some(data) = sse_data_payload(&line) {
                if let Some(tokens) = parse_anthropic_usage_chunk(data) {
                    usage = Some(tokens);
                }
                if let Some(text) = parse_anthropic_chunk(data) {
                    style::stop_spinner(spinning);
                    print!("{}", text);
                    io::stdout().flush()?;
                    full_content.push_str(&text);
                }
            }
        }
    }

    if full_content.is_empty() {
        let tail = line_buf.trim();
        if let Some(data) = sse_data_payload(tail) {
            if let Some(tokens) = parse_anthropic_usage_chunk(data) {
                usage = Some(tokens);
            }
            if let Some(text) = parse_anthropic_chunk(data) {
                style::stop_spinner(spinning);
                print!("{}", text);
                io::stdout().flush()?;
                full_content.push_str(&text);
            }
        }
    }

    // If streaming produced no content, fall back to non-streaming
    if full_content.is_empty() {
        return send_anthropic_non_streaming(client, &url, key, model, messages, spinning).await;
    }

    Ok(ChatTurnResult {
        content: full_content,
        usage,
    })
}

/// Non-streaming fallback for Anthropic-format providers.
async fn send_anthropic_non_streaming(
    client: &Client,
    url: &str,
    key: &ApiKey,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
) -> Result<ChatTurnResult> {
    let request = serde_json::json!({
        "model": model,
        "messages": messages,
        "max_tokens": 8096,
        "stream": false,
    });

    let response = send_with_retry(|| {
        client
            .post(url)
            // Send both auth headers: gateways vary on which they accept
            .header("Authorization", format!("Bearer {}", key.key.as_str()))
            .header("x-api-key", key.key.as_str())
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .header("User-Agent", format!("aivo/{}", crate::version::VERSION))
            .json(&request)
    })
    .await?;

    if !response.status().is_success() {
        style::stop_spinner(spinning);
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("API returned {} — {}", status, body);
    }

    let body: serde_json::Value = response.json().await?;
    let usage = extract_anthropic_usage(&body);

    // Try Anthropic format: content[].text
    let content: String = body["content"]
        .as_array()
        .iter()
        .flat_map(|arr| arr.iter())
        .filter(|c| c["type"].as_str() == Some("text"))
        .filter_map(|c| c["text"].as_str())
        .collect();

    if content.is_empty() {
        style::stop_spinner(spinning);
        anyhow::bail!("Provider returned an empty response");
    }

    style::stop_spinner(spinning);
    print!("{}", content);
    io::stdout().flush()?;

    Ok(ChatTurnResult { content, usage })
}

/// Parses an Anthropic SSE data line and returns the text delta if present.
pub fn parse_anthropic_chunk(data: &str) -> Option<String> {
    let event: AnthropicStreamEvent = serde_json::from_str(data).ok()?;
    if event.event_type == "content_block_delta" {
        event.delta?.text
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustyline::Context;
    use rustyline::history::DefaultHistory;

    fn make_history() -> DefaultHistory {
        DefaultHistory::new()
    }

    #[test]
    fn test_completer_exit_prefix() {
        let h = ChatHelper::new();
        let hist = make_history();
        let ctx = Context::new(&hist);
        let (start, completions) = h.complete("/e", 2, &ctx).unwrap();
        assert_eq!(start, 0);
        assert!(completions.iter().any(|p| p.replacement == "/exit"));
    }

    #[test]
    fn test_completer_model_prefix() {
        let h = ChatHelper::new();
        let hist = make_history();
        let ctx = Context::new(&hist);
        let (start, completions) = h.complete("/m", 2, &ctx).unwrap();
        assert_eq!(start, 0);
        assert!(completions.iter().any(|p| p.replacement == "/model"));
    }

    #[test]
    fn test_completer_no_match_for_normal_text() {
        let h = ChatHelper::new();
        let hist = make_history();
        let ctx = Context::new(&hist);
        let (_, completions) = h.complete("hello", 5, &ctx).unwrap();
        assert!(completions.is_empty());
    }

    #[test]
    fn test_completer_full_slash_prefix_returns_all() {
        let h = ChatHelper::new();
        let hist = make_history();
        let ctx = Context::new(&hist);
        let (_, completions) = h.complete("/", 1, &ctx).unwrap();
        assert_eq!(completions.len(), h.commands.len());
    }

    #[test]
    fn test_hinter_shows_remainder_for_partial_exit() {
        let h = ChatHelper::new();
        let hist = make_history();
        let ctx = Context::new(&hist);
        let hint = h.hint("/e", 2, &ctx);
        assert_eq!(hint.as_deref(), Some("xit"));
    }

    #[test]
    fn test_hinter_shows_remainder_for_partial_model() {
        let h = ChatHelper::new();
        let hist = make_history();
        let ctx = Context::new(&hist);
        let hint = h.hint("/m", 2, &ctx);
        assert_eq!(hint.as_deref(), Some("odel"));
    }

    #[test]
    fn test_hinter_no_hint_for_complete_command() {
        let h = ChatHelper::new();
        let hist = make_history();
        let ctx = Context::new(&hist);
        assert!(h.hint("/exit", 5, &ctx).is_none());
    }

    #[test]
    fn test_hinter_no_hint_for_normal_text() {
        let h = ChatHelper::new();
        let hist = make_history();
        let ctx = Context::new(&hist);
        assert!(h.hint("hello", 5, &ctx).is_none());
    }

    #[test]
    fn test_hinter_no_hint_when_cursor_not_at_end() {
        let h = ChatHelper::new();
        let hist = make_history();
        let ctx = Context::new(&hist);
        // cursor at pos 2, line is longer — mid-edit, no hint
        assert!(h.hint("/exit", 2, &ctx).is_none());
    }

    #[test]
    fn test_parse_sse_chunk_with_content() {
        let data = r#"{"id":"chatcmpl-1","choices":[{"delta":{"content":"Hello"}}]}"#;
        assert_eq!(parse_sse_chunk(data), Some("Hello".to_string()));
    }

    #[test]
    fn test_parse_sse_chunk_empty_delta() {
        let data = r#"{"id":"chatcmpl-1","choices":[{"delta":{}}]}"#;
        assert_eq!(parse_sse_chunk(data), None);
    }

    #[test]
    fn test_parse_sse_chunk_invalid_json() {
        assert_eq!(parse_sse_chunk("not json"), None);
    }

    #[test]
    fn test_parse_sse_chunk_no_choices() {
        let data = r#"{"id":"chatcmpl-1","choices":[]}"#;
        assert_eq!(parse_sse_chunk(data), None);
    }

    #[test]
    fn test_sse_data_payload_with_optional_space() {
        assert_eq!(
            sse_data_payload(r#"data: {"choices":[]}"#),
            Some(r#"{"choices":[]}"#)
        );
        assert_eq!(
            sse_data_payload(r#"data:{"choices":[]}"#),
            Some(r#"{"choices":[]}"#)
        );
    }

    #[test]
    fn test_extract_openai_message_content_string_and_parts() {
        let text = serde_json::json!({
            "choices": [{"message": {"content": "hello"}}]
        });
        assert_eq!(extract_openai_message_content(&text), "hello");

        let parts = serde_json::json!({
            "choices": [{
                "message": {
                    "content": [
                        {"type":"text", "text":"hello "},
                        {"type":"text", "text":"world"}
                    ]
                }
            }]
        });
        assert_eq!(extract_openai_message_content(&parts), "hello world");
    }

    #[test]
    fn test_compose_one_shot_prompt_without_stdin() {
        let out = compose_one_shot_prompt("Summarize in one sentence", None);
        assert_eq!(out, "Summarize in one sentence");
    }

    #[test]
    fn test_compose_one_shot_prompt_with_stdin_context() {
        let out = compose_one_shot_prompt("Summarize in one sentence", Some("diff --git a b"));
        assert!(out.contains("Summarize in one sentence"));
        assert!(out.contains("Context from stdin:"));
        assert!(out.contains("diff --git a b"));
    }

    #[test]
    fn test_compose_one_shot_prompt_ignores_empty_stdin() {
        let out = compose_one_shot_prompt("Summarize", Some("   \n  "));
        assert_eq!(out, "Summarize");
    }

    #[test]
    fn test_should_retry_status() {
        assert!(should_retry_status(StatusCode::TOO_MANY_REQUESTS));
        assert!(should_retry_status(StatusCode::INTERNAL_SERVER_ERROR));
        assert!(!should_retry_status(StatusCode::BAD_REQUEST));
    }

    #[test]
    fn test_chat_message_serialization() {
        let msg = ChatMessage {
            role: "user".to_string(),
            content: "hello".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"hello\""));
    }

    #[test]
    fn test_parse_anthropic_chunk_with_text() {
        let data = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
        assert_eq!(parse_anthropic_chunk(data), Some("Hello".to_string()));
    }

    #[test]
    fn test_parse_anthropic_chunk_non_delta_event() {
        let data = r#"{"type":"message_start","message":{"id":"msg_1"}}"#;
        assert_eq!(parse_anthropic_chunk(data), None);
    }

    #[test]
    fn test_parse_anthropic_chunk_ping() {
        let data = r#"{"type":"ping"}"#;
        assert_eq!(parse_anthropic_chunk(data), None);
    }

    #[test]
    fn test_parse_anthropic_chunk_invalid_json() {
        assert_eq!(parse_anthropic_chunk("not json"), None);
    }

    #[test]
    fn test_is_format_mismatch_404() {
        let e = anyhow::anyhow!("API returned 404 Not Found — endpoint missing");
        assert!(is_format_mismatch(&e));
    }

    #[test]
    fn test_is_format_mismatch_405() {
        let e = anyhow::anyhow!("API returned 405 Method Not Allowed");
        assert!(is_format_mismatch(&e));
    }

    #[test]
    fn test_is_format_mismatch_endpoint_text() {
        let e = anyhow::anyhow!("route not found for requested endpoint");
        assert!(is_format_mismatch(&e));
    }

    #[test]
    fn test_is_format_mismatch_other_errors() {
        let e = anyhow::anyhow!("API returned 401 Unauthorized");
        assert!(!is_format_mismatch(&e));
        let e = anyhow::anyhow!("API returned 429 Too Many Requests");
        assert!(!is_format_mismatch(&e));
    }
}
