/**
 * ChatCommand handler. Interactive sessions launch the full-screen TUI
 * (chat_tui). One-shot queries (-x flag) stream directly to stdout using
 * OpenAI-compatible /v1/chat/completions, falling back to Anthropic
 * /v1/messages on 404/405.
 */
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use std::io::{self, IsTerminal, Read, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use crate::tui::FuzzySelect;
use anyhow::Result;
use chrono::Utc;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};

use crate::commands::models::fetch_models_for_select;
use crate::commands::normalize_base_url;
use crate::errors::ExitCode;
use crate::services::anthropic_route_pipeline::inject_cache_control_on_last_block;
use crate::services::copilot_auth::{
    COPILOT_EDITOR_VERSION, COPILOT_INTEGRATION_ID, COPILOT_OPENAI_INTENT, CopilotTokenManager,
};
use crate::services::http_utils::{parse_token_u64, sse_data_payload};
use crate::services::model_names;
use crate::services::models_cache::ModelsCache;
use crate::services::session_store::{
    ApiKey, AttachmentStorage, MessageAttachment, SessionStore, StoredChatMessage,
};
use crate::style;

#[path = "chat_tui.rs"]
mod chat_tui;
#[path = "chat_tui_format.rs"]
mod chat_tui_format;

/// Maximum number of messages to keep in chat history.
/// When exceeded, the oldest messages are dropped (keeping any system message).
const MAX_HISTORY_MESSAGES: usize = 50;
/// Retry budget for transient HTTP failures.
const MAX_REQUEST_ATTEMPTS: usize = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing, skip_deserializing)]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<MessageAttachment>,
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
    reasoning_content: Option<String>,
    reasoning: Option<String>,
    thinking: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ChatResponseChunk {
    Content(String),
    Reasoning(String),
}

#[derive(Debug, Default, PartialEq, Eq)]
struct AssistantResponse {
    content: String,
    reasoning_content: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct TokenUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    cache_read_input_tokens: u64,
    cache_creation_input_tokens: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct TokenUsageUpdate {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
}

impl TokenUsageUpdate {
    fn is_empty(self) -> bool {
        self.prompt_tokens.is_none()
            && self.completion_tokens.is_none()
            && self.cache_read_input_tokens.is_none()
            && self.cache_creation_input_tokens.is_none()
    }
}

impl TokenUsage {
    fn total_tokens(self) -> u64 {
        self.prompt_tokens.saturating_add(self.completion_tokens)
    }
}

#[derive(Debug, Default)]
struct ChatTurnResult {
    content: String,
    reasoning_content: Option<String>,
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
    thinking: Option<String>,
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
        attachments: Vec<String>,
        key_override: Option<ApiKey>,
    ) -> ExitCode {
        match self
            .execute_internal(model, one_shot, attachments, key_override)
            .await
        {
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
        attachments: Vec<String>,
        key_override: Option<ApiKey>,
    ) -> Result<ExitCode> {
        let mut key = match key_override {
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

        let raw_model = match self.resolve_model(&key.id, model_flag).await? {
            Some(m) => m,
            None => {
                ensure_picker_terminal("model", "--model <name>")?;
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
        let model = Self::transform_model_for_provider(&key.base_url, &raw_model);
        let cwd =
            crate::services::system_env::current_dir_string().unwrap_or_else(|| ".".to_string());
        let pending_attachments = build_pending_attachments(&attachments)?;

        // Resolve the "ollama" sentinel to the actual local URL before any HTTP calls.
        if key.base_url == "ollama" {
            crate::services::ollama::ensure_ready().await?;
            crate::services::ollama::ensure_model(&raw_model).await?;
            key.base_url = crate::services::ollama::ollama_openai_base_url();
        }

        // Create once so its token cache is reused across messages in the session.
        let copilot_tm = if key.base_url == "copilot" {
            Some(Arc::new(CopilotTokenManager::new(
                key.key.as_str().to_string(),
            )))
        } else {
            None
        };

        if let Some(input) = one_shot {
            let one_shot_input = if input.is_empty() {
                sanitize_one_shot_message(read_one_shot_message_from_stdin()?)?
            } else {
                let input = sanitize_one_shot_message(input)?;
                let stdin_context = read_stdin_if_piped()?;
                compose_one_shot_prompt(&input, stdin_context.as_deref())
            };
            let one_shot_attachments = materialize_attachments(&pending_attachments).await?;

            let history = vec![ChatMessage {
                role: "user".to_string(),
                content: one_shot_input,
                reasoning_content: None,
                attachments: one_shot_attachments,
            }];
            let mut format = ChatFormat::OpenAI;
            self.session_store
                .record_selection(&key.id, "chat", Some(&raw_model))
                .await?;
            let (spinning, spinner_handle) = style::start_spinner(None);
            let mut current_section: Option<&'static str> = None;
            let result = send_message_turn(
                &client,
                &key,
                copilot_tm.as_deref(),
                &model,
                &history,
                &mut format,
                &spinning,
                &mut |chunk| {
                    match chunk {
                        ChatResponseChunk::Reasoning(text) => {
                            if current_section != Some("thinking") {
                                if current_section.is_some() {
                                    print!("\n\n");
                                }
                                println!("Thinking:");
                                current_section = Some("thinking");
                            }
                            print!("{text}");
                        }
                        ChatResponseChunk::Content(text) => {
                            if current_section == Some("thinking") {
                                print!("\n\nAnswer:\n");
                            }
                            current_section = Some("answer");
                            print!("{text}");
                        }
                    }
                    io::stdout().flush()?;
                    Ok(())
                },
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
                                usage.cache_read_input_tokens,
                                usage.cache_creation_input_tokens,
                            )
                            .await?;
                    }
                    println!();
                    return Ok(ExitCode::Success);
                }
                Err(e) => return Err(e),
            }
        }

        if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
            anyhow::bail!(
                "Interactive chat now uses a full-screen TUI. Run it in a terminal, or use -x/--execute for non-interactive mode."
            );
        }

        let initial_session = new_chat_session_id();
        let initial_history = Vec::new();
        let startup_notice = attachment_notice(&pending_attachments);

        self.session_store
            .record_selection(&key.id, "chat", Some(&raw_model))
            .await?;

        chat_tui::run_chat_tui(chat_tui::ChatTuiParams {
            session_store: self.session_store.clone(),
            cache: self.cache.clone(),
            client,
            key,
            copilot_tm,
            cwd,
            raw_model,
            model,
            initial_session,
            initial_history,
            initial_draft_attachments: pending_attachments,
            startup_notice,
        })
        .await?;

        Ok(ExitCode::Success)
    }

    pub fn print_help() {
        println!(
            "{} aivo chat [--model <model>] [-x <message>] [--attach <path> ...]",
            style::bold("Usage:")
        );
        println!();
        println!(
            "{}",
            style::dim("Start the interactive full-screen chat TUI with streaming responses.")
        );
        println!(
            "{}",
            style::dim(
                "Uses the active API key and opens a transcript/composer interface in your terminal."
            )
        );
        println!(
            "{}",
            style::dim(
                "Slash commands are available inside chat: /new, /resume, /model, /key, /attach, /detach, /clear, /help, /exit."
            )
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
            style::dim("Select API key by ID or name (-k opens key picker)")
        );
        println!(
            "  {}  {}",
            style::cyan("-x, --execute <message>"),
            style::dim("Send one message and exit (-x with no value reads stdin until Ctrl-D)")
        );
        println!(
            "  {}  {}",
            style::cyan("--attach <path>"),
            style::dim("Queue a text file or image for the next message")
        );
        println!();
        println!("{}", style::bold("Slash Commands:"));
        println!(
            "  {}  {}",
            style::cyan("/new"),
            style::dim("Start a fresh chat with the current key and model")
        );
        println!(
            "  {}  {}",
            style::cyan("/resume [query]"),
            style::dim("Resume a saved chat from this directory")
        );
        println!(
            "  {}  {}",
            style::cyan("/model [name]"),
            style::dim("Switch the current chat model")
        );
        println!(
            "  {}  {}",
            style::cyan("/key [id|name]"),
            style::dim("Switch to another saved key for this chat")
        );
        println!(
            "  {}  {}",
            style::cyan("/attach <path>"),
            style::dim("Attach a text file or image to the next message")
        );
        println!(
            "  {}  {}",
            style::cyan("/detach <n>"),
            style::dim("Remove one queued attachment by number")
        );
        println!(
            "  {}  {}",
            style::cyan("/clear"),
            style::dim("Clear queued attachments from the composer")
        );
        println!(
            "  {}  {}",
            style::cyan("/help / /exit"),
            style::dim("Open command help / leave chat")
        );
        println!(
            "  {}  {}",
            style::cyan("//message"),
            style::dim("Send a literal leading slash")
        );
        println!();
        println!("{}", style::bold("Keys:"));
        println!(
            "  {}  {}",
            style::cyan("Enter / Ctrl+J"),
            style::dim("Send message / insert newline")
        );
        println!(
            "  {}  {}",
            style::cyan("Ctrl+V"),
            style::dim("Paste system clipboard (text or image)")
        );
        println!(
            "  {}  {}",
            style::cyan("Ctrl+R / F1"),
            style::dim("Open resume picker / show help")
        );
        println!(
            "  {}  {}",
            style::cyan("Ctrl+P / Ctrl+N"),
            style::dim("Previous / next input")
        );
        println!(
            "  {}  {}",
            style::cyan("Ctrl+M"),
            style::dim("Change model")
        );
        println!(
            "  {}  {}",
            style::cyan("Ctrl+T"),
            style::dim("Show / hide thinking blocks")
        );
        println!(
            "  {}  {}",
            style::cyan("AIVO_REDUCE_MOTION=1"),
            style::dim("Disable chat TUI motion effects")
        );
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo chat"));
        println!("  {}", style::dim("aivo chat --model gpt-4o"));
        println!("  {}", style::dim("aivo chat -m claude-sonnet-4-5"));
        println!(
            "  {}",
            style::dim("aivo chat --attach README.md --attach screenshot.png")
        );
        println!(
            "  {}",
            style::dim("aivo chat -x \"Explain Rust lifetimes\"")
        );
        println!("  {}", style::dim("aivo chat -x"));
        println!("  {}", style::dim("aivo -x \"Summarize this repository\""));
        println!(
            "  {}",
            style::dim("git diff --cached | aivo chat -x \"Summarize changes in one sentence\"")
        );
    }
}

#[allow(clippy::too_many_arguments)]
async fn send_message_turn<F>(
    client: &Client,
    key: &ApiKey,
    copilot_tm: Option<&CopilotTokenManager>,
    model: &str,
    history: &[ChatMessage],
    format: &mut ChatFormat,
    spinning: &Arc<AtomicBool>,
    on_chunk: &mut F,
) -> Result<ChatTurnResult>
where
    F: FnMut(ChatResponseChunk) -> Result<()>,
{
    if let Some(tm) = copilot_tm {
        return send_copilot_request(client, tm, model, history, spinning, on_chunk).await;
    }

    match format {
        ChatFormat::OpenAI => {
            match send_chat_request(client, key, model, history, spinning, on_chunk).await {
                ok @ Ok(_) => ok,
                Err(e) if is_format_mismatch(&e) => {
                    // Provider doesn't speak OpenAI format; try Anthropic
                    match send_anthropic_request(client, key, model, history, spinning, on_chunk)
                        .await
                    {
                        Ok(content) => {
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
            send_anthropic_request(client, key, model, history, spinning, on_chunk).await
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

fn read_one_shot_message_from_stdin() -> Result<String> {
    if io::stdin().is_terminal() {
        eprintln!(
            "{}",
            style::dim("Enter message, then press Ctrl-D to send.")
        );
    }

    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf)?;
    Ok(buf)
}

fn compose_one_shot_prompt(prompt: &str, stdin_context: Option<&str>) -> String {
    match stdin_context.map(str::trim).filter(|c| !c.is_empty()) {
        Some(ctx) => format!("{prompt}\n\nContext from stdin:\n{ctx}"),
        None => prompt.to_string(),
    }
}

fn sanitize_one_shot_message(message: String) -> Result<String> {
    if message.trim().is_empty() {
        anyhow::bail!("Message for -x/--execute cannot be empty");
    }
    Ok(message)
}

fn ensure_picker_terminal(kind: &str, explicit_flag: &str) -> Result<()> {
    if io::stderr().is_terminal() {
        return Ok(());
    }

    anyhow::bail!(
        "Cannot open {kind} picker without a terminal. Run in a terminal or pass {explicit_flag}."
    );
}

fn attachment_notice(attachments: &[MessageAttachment]) -> Option<String> {
    if attachments.is_empty() {
        None
    } else {
        Some(format!(
            "{} attachment{} queued. Press Enter to send or use /attach to add more.",
            attachments.len(),
            if attachments.len() == 1 { "" } else { "s" }
        ))
    }
}

fn build_pending_attachments(paths: &[String]) -> Result<Vec<MessageAttachment>> {
    paths
        .iter()
        .map(|path| build_pending_attachment(path))
        .collect()
}

fn build_pending_attachment(path: &str) -> Result<MessageAttachment> {
    let expanded = crate::services::system_env::expand_tilde(path);
    let file_path = expanded.as_path();
    ensure_attachment_exists(file_path)?;
    let mime_type = guess_attachment_mime_type(file_path)?;
    Ok(MessageAttachment {
        name: attachment_name(file_path),
        mime_type,
        storage: AttachmentStorage::FileRef {
            path: path.to_string(),
        },
    })
}

fn ensure_attachment_exists(path: &Path) -> Result<()> {
    let metadata = std::fs::metadata(path)
        .map_err(|err| anyhow::anyhow!("Failed to read attachment '{}': {err}", path.display()))?;
    if !metadata.is_file() {
        anyhow::bail!("Attachment '{}' is not a file", path.display());
    }
    Ok(())
}

fn attachment_name(path: &Path) -> String {
    match path
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
    {
        Some(name) => name.to_string(),
        None => path.to_string_lossy().into_owned(),
    }
}

fn guess_attachment_mime_type(path: &Path) -> Result<String> {
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    let mime = match extension.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "json" => "application/json",
        "md" => "text/markdown",
        "html" => "text/html",
        "css" => "text/css",
        "csv" => "text/csv",
        "xml" => "application/xml",
        "yaml" | "yml" => "application/yaml",
        "toml" => "application/toml",
        "" => "text/plain",
        _ => "text/plain",
    };
    Ok(mime.to_string())
}

async fn materialize_attachments(
    attachments: &[MessageAttachment],
) -> Result<Vec<MessageAttachment>> {
    let mut resolved = Vec::with_capacity(attachments.len());
    for attachment in attachments {
        resolved.push(materialize_attachment(attachment).await?);
    }
    Ok(resolved)
}

async fn materialize_attachment(attachment: &MessageAttachment) -> Result<MessageAttachment> {
    match &attachment.storage {
        AttachmentStorage::Inline { .. } => Ok(attachment.clone()),
        AttachmentStorage::FileRef { path } => {
            let storage = if attachment.mime_type.starts_with("image/") {
                let bytes = tokio::fs::read(path)
                    .await
                    .map_err(|err| anyhow::anyhow!("Failed to read image '{}': {err}", path))?;
                AttachmentStorage::Inline {
                    data: BASE64.encode(bytes),
                }
            } else {
                let text = tokio::fs::read_to_string(path).await.map_err(|err| {
                    anyhow::anyhow!(
                        "Failed to read text attachment '{}': {err}. Files must be UTF-8.",
                        path
                    )
                })?;
                AttachmentStorage::Inline { data: text }
            };
            Ok(MessageAttachment {
                name: attachment.name.clone(),
                mime_type: attachment.mime_type.clone(),
                storage,
            })
        }
    }
}

fn format_text_attachment_content(name: &str, content: &str) -> String {
    format!("[Attached file: {name}]\n{content}")
}

fn to_stored_messages(history: &[ChatMessage]) -> Vec<StoredChatMessage> {
    history
        .iter()
        .map(|message| StoredChatMessage {
            role: message.role.clone(),
            content: message.content.clone(),
            reasoning_content: message.reasoning_content.clone(),
            id: Some(new_chat_session_id()),
            timestamp: Some(Utc::now().to_rfc3339()),
            attachments: (!message.attachments.is_empty()).then(|| message.attachments.clone()),
        })
        .collect()
}

fn new_chat_session_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

fn normalize_reasoning_content(reasoning: String) -> Option<String> {
    if reasoning.trim().is_empty() {
        None
    } else {
        Some(reasoning)
    }
}

fn extract_reasoning_part(part: &serde_json::Value) -> Option<String> {
    part.get("thinking")
        .and_then(|v| v.as_str())
        .or_else(|| part.get("reasoning_content").and_then(|v| v.as_str()))
        .or_else(|| part.get("reasoning").and_then(|v| v.as_str()))
        .or_else(|| part.get("text").and_then(|v| v.as_str()))
        .or_else(|| part.get("content").and_then(|v| v.as_str()))
        .map(ToString::to_string)
}

fn extract_openai_message(body: &serde_json::Value) -> AssistantResponse {
    let message = &body["choices"][0]["message"];
    let mut content_parts = Vec::new();
    let mut reasoning_parts = Vec::new();

    if let Some(reasoning) = message.get("reasoning_content").and_then(|v| v.as_str()) {
        reasoning_parts.push(reasoning.to_string());
    }

    if let Some(content) = message.get("content").and_then(|v| v.as_str()) {
        content_parts.push(content.to_string());
    } else if let Some(parts) = message.get("content").and_then(|v| v.as_array()) {
        for part in parts {
            let part_type = part.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if matches!(part_type, "reasoning" | "thinking") {
                if let Some(reasoning) = extract_reasoning_part(part) {
                    reasoning_parts.push(reasoning);
                }
                continue;
            }

            if let Some(text) = part
                .get("text")
                .and_then(|v| v.as_str())
                .or_else(|| part.get("content").and_then(|v| v.as_str()))
            {
                content_parts.push(text.to_string());
            }
        }
    }

    AssistantResponse {
        content: content_parts.concat(),
        reasoning_content: normalize_reasoning_content(reasoning_parts.join("")),
    }
}

fn extract_openai_usage(body: &serde_json::Value) -> Option<TokenUsage> {
    let update = extract_openai_usage_update(body)?;
    Some(TokenUsage {
        prompt_tokens: update.prompt_tokens.unwrap_or(0),
        completion_tokens: update.completion_tokens.unwrap_or(0),
        cache_read_input_tokens: update.cache_read_input_tokens.unwrap_or(0),
        cache_creation_input_tokens: update.cache_creation_input_tokens.unwrap_or(0),
    })
}

fn extract_anthropic_usage(body: &serde_json::Value) -> Option<TokenUsage> {
    let update = extract_anthropic_usage_update(body)?;
    Some(TokenUsage {
        prompt_tokens: update.prompt_tokens.unwrap_or(0),
        completion_tokens: update.completion_tokens.unwrap_or(0),
        cache_read_input_tokens: update.cache_read_input_tokens.unwrap_or(0),
        cache_creation_input_tokens: update.cache_creation_input_tokens.unwrap_or(0),
    })
}

fn extract_openai_usage_update(body: &serde_json::Value) -> Option<TokenUsageUpdate> {
    let usage = body.get("usage")?;
    let update = TokenUsageUpdate {
        prompt_tokens: usage.get("prompt_tokens").and_then(parse_token_u64),
        completion_tokens: usage.get("completion_tokens").and_then(parse_token_u64),
        cache_read_input_tokens: usage
            .get("cache_read_input_tokens")
            .and_then(parse_token_u64)
            .or_else(|| {
                usage
                    .get("prompt_tokens_details")
                    .and_then(|details| details.get("cached_tokens"))
                    .and_then(parse_token_u64)
            }),
        cache_creation_input_tokens: usage
            .get("cache_creation_input_tokens")
            .and_then(parse_token_u64),
    };
    if update.is_empty() {
        None
    } else {
        Some(update)
    }
}

fn extract_anthropic_usage_update(body: &serde_json::Value) -> Option<TokenUsageUpdate> {
    let usage = body.get("usage")?;
    let raw_input = usage.get("input_tokens").and_then(parse_token_u64);
    let cache_read = usage
        .get("cache_read_input_tokens")
        .and_then(parse_token_u64);
    let cache_creation = usage
        .get("cache_creation_input_tokens")
        .and_then(parse_token_u64);
    // Normalize: Anthropic's input_tokens excludes cache, so add cache to get total input
    let prompt_tokens = raw_input.map(|it| {
        it.saturating_add(cache_read.unwrap_or(0))
            .saturating_add(cache_creation.unwrap_or(0))
    });
    let update = TokenUsageUpdate {
        prompt_tokens,
        completion_tokens: usage.get("output_tokens").and_then(parse_token_u64),
        cache_read_input_tokens: cache_read,
        cache_creation_input_tokens: cache_creation,
    };
    if update.is_empty() {
        None
    } else {
        Some(update)
    }
}

fn parse_openai_usage_chunk(data: &str) -> Option<TokenUsageUpdate> {
    let value = serde_json::from_str::<serde_json::Value>(data).ok()?;
    extract_openai_usage_update(&value)
}

fn parse_anthropic_usage_chunk(data: &str) -> Option<TokenUsageUpdate> {
    let value = serde_json::from_str::<serde_json::Value>(data).ok()?;
    extract_anthropic_usage_update(&value)
}

fn merge_token_usage(usage: &mut Option<TokenUsage>, update: TokenUsageUpdate) {
    let current = usage.get_or_insert_with(TokenUsage::default);
    if let Some(tokens) = update.prompt_tokens {
        current.prompt_tokens = tokens;
    }
    if let Some(tokens) = update.completion_tokens {
        current.completion_tokens = tokens;
    }
    if let Some(tokens) = update.cache_read_input_tokens {
        current.cache_read_input_tokens = tokens;
    }
    if let Some(tokens) = update.cache_creation_input_tokens {
        current.cache_creation_input_tokens = tokens;
    }
}

fn build_openai_chat_request(
    model: &str,
    messages: &[ChatMessage],
    stream: bool,
) -> Result<serde_json::Value> {
    let mut encoded_messages = Vec::with_capacity(messages.len());
    for message in messages {
        encoded_messages.push(build_openai_message(message)?);
    }

    Ok(serde_json::json!({
        "model": model,
        "messages": encoded_messages,
        "stream": stream,
    }))
}

/// Returns the inline data for a materialized attachment, or fails if it is still a FileRef.
fn require_inline(attachment: &MessageAttachment) -> Result<&str> {
    match &attachment.storage {
        AttachmentStorage::Inline { data } => Ok(data),
        AttachmentStorage::FileRef { path } => anyhow::bail!(
            "Attachment '{}' is unresolved. Expected inline data before sending.",
            path
        ),
    }
}

fn build_openai_message(message: &ChatMessage) -> Result<serde_json::Value> {
    if message.attachments.is_empty() {
        return Ok(serde_json::json!({
            "role": message.role,
            "content": message.content,
        }));
    }

    let mut parts = Vec::new();
    if !message.content.is_empty() {
        parts.push(serde_json::json!({
            "type": "text",
            "text": message.content,
        }));
    }

    for attachment in &message.attachments {
        let data = require_inline(attachment)?;
        if attachment.mime_type.starts_with("image/") {
            parts.push(serde_json::json!({
                "type": "image_url",
                "image_url": {
                    "url": format!("data:{};base64,{}", attachment.mime_type, data),
                },
            }));
        } else {
            parts.push(serde_json::json!({
                "type": "text",
                "text": format_text_attachment_content(&attachment.name, data),
            }));
        }
    }

    Ok(serde_json::json!({
        "role": message.role,
        "content": parts,
    }))
}

fn build_anthropic_request(
    model: &str,
    messages: &[ChatMessage],
    stream: bool,
) -> Result<serde_json::Value> {
    let mut system_parts = Vec::new();
    let mut encoded_messages = Vec::new();

    for message in messages {
        if message.role == "system" {
            if !message.content.is_empty() {
                system_parts.push(message.content.clone());
            }
            continue;
        }

        let role = if message.role == "assistant" {
            "assistant"
        } else {
            "user"
        };
        encoded_messages.push(serde_json::json!({
            "role": role,
            "content": build_anthropic_content(message)?,
        }));
    }

    let mut request = serde_json::json!({
        "model": model,
        "messages": encoded_messages,
        "max_tokens": 8096,
        "stream": stream,
    });
    if !system_parts.is_empty() {
        request["system"] = serde_json::json!([{
            "type": "text",
            "text": system_parts.join("\n\n"),
            "cache_control": {"type": "ephemeral"}
        }]);
    }

    // Add cache_control to the last user message for Anthropic prompt caching
    for msg in encoded_messages.iter_mut().rev() {
        if msg["role"] != "user" {
            continue;
        }
        if let Some(content) = msg.get_mut("content") {
            inject_cache_control_on_last_block(content);
        }
        break;
    }

    request["messages"] = serde_json::json!(encoded_messages);
    Ok(request)
}

fn build_anthropic_content(message: &ChatMessage) -> Result<serde_json::Value> {
    if message.attachments.is_empty() {
        return Ok(serde_json::Value::String(message.content.clone()));
    }

    let mut blocks = Vec::new();
    if !message.content.is_empty() {
        blocks.push(serde_json::json!({
            "type": "text",
            "text": message.content,
        }));
    }

    for attachment in &message.attachments {
        let data = require_inline(attachment)?;
        if attachment.mime_type.starts_with("image/") {
            blocks.push(serde_json::json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": attachment.mime_type,
                    "data": data,
                },
            }));
        } else {
            blocks.push(serde_json::json!({
                "type": "text",
                "text": format_text_attachment_content(&attachment.name, data),
            }));
        }
    }

    Ok(serde_json::Value::Array(blocks))
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
async fn send_chat_request<F>(
    client: &Client,
    key: &ApiKey,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
    on_chunk: &mut F,
) -> Result<ChatTurnResult>
where
    F: FnMut(ChatResponseChunk) -> Result<()>,
{
    let base = normalize_base_url(&key.base_url);
    let url = format!("{}/v1/chat/completions", base);

    // Try streaming first; fall back to non-streaming on server errors
    let request = build_openai_chat_request(model, messages, true)?;

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
        return send_non_streaming(client, &url, key, model, messages, spinning, on_chunk).await;
    }

    if !response.status().is_success() {
        style::stop_spinner(spinning);
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("API returned {} — {}", status, body);
    }

    let mut full_content = String::new();
    let mut full_reasoning = String::new();
    let mut usage = None;
    let mut line_buf = String::new();
    let mut done = false;

    while !done {
        let chunk_result = response.chunk().await;
        let Some(chunk) = (match chunk_result {
            Ok(c) => c,
            Err(_) if !full_content.is_empty() || !full_reasoning.is_empty() => {
                // Stream error after content was received — use what we have.
                break;
            }
            Err(e) => return Err(e.into()),
        }) else {
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
                    merge_token_usage(&mut usage, tokens);
                }
                if let Some(chunk) = parse_sse_chunk(data) {
                    style::stop_spinner(spinning);
                    match &chunk {
                        ChatResponseChunk::Content(content) => full_content.push_str(content),
                        ChatResponseChunk::Reasoning(reasoning) => {
                            full_reasoning.push_str(reasoning);
                        }
                    }
                    on_chunk(chunk)?;
                }
            }
        }
    }

    let tail = line_buf.trim();
    if !tail.is_empty() {
        if let Some(data) = sse_data_payload(tail) {
            if let Some(tokens) = parse_openai_usage_chunk(data) {
                merge_token_usage(&mut usage, tokens);
            }
            if data.trim() != "[DONE]"
                && let Some(chunk) = parse_sse_chunk(data)
            {
                style::stop_spinner(spinning);
                match &chunk {
                    ChatResponseChunk::Content(content) => full_content.push_str(content),
                    ChatResponseChunk::Reasoning(reasoning) => full_reasoning.push_str(reasoning),
                }
                on_chunk(chunk)?;
            }
        } else if full_content.is_empty()
            && let Ok(resp) = serde_json::from_str::<serde_json::Value>(tail)
        {
            let response = extract_openai_message(&resp);
            if !response.content.is_empty() || response.reasoning_content.is_some() {
                style::stop_spinner(spinning);
                if let Some(reasoning) = response.reasoning_content.clone() {
                    on_chunk(ChatResponseChunk::Reasoning(reasoning.clone()))?;
                    full_reasoning = reasoning;
                }
                if !response.content.is_empty() {
                    on_chunk(ChatResponseChunk::Content(response.content.clone()))?;
                    full_content = response.content;
                }
            }
        }
    }

    if full_content.is_empty() && full_reasoning.is_empty() {
        return send_non_streaming(client, &url, key, model, messages, spinning, on_chunk).await;
    }

    Ok(ChatTurnResult {
        content: full_content,
        reasoning_content: normalize_reasoning_content(full_reasoning),
        usage,
    })
}

/// Non-streaming fallback for gateways that don't support SSE streaming.
async fn send_non_streaming<F>(
    client: &Client,
    url: &str,
    key: &ApiKey,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
    on_chunk: &mut F,
) -> Result<ChatTurnResult>
where
    F: FnMut(ChatResponseChunk) -> Result<()>,
{
    let request = build_openai_chat_request(model, messages, false)?;

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
    let response = extract_openai_message(&body);
    let usage = extract_openai_usage(&body);

    if response.content.is_empty() && response.reasoning_content.is_none() {
        style::stop_spinner(spinning);
        anyhow::bail!("Provider returned an empty response");
    }

    style::stop_spinner(spinning);
    if let Some(reasoning) = response.reasoning_content.clone() {
        on_chunk(ChatResponseChunk::Reasoning(reasoning))?;
    }
    if !response.content.is_empty() {
        on_chunk(ChatResponseChunk::Content(response.content.clone()))?;
    }

    Ok(ChatTurnResult {
        content: response.content,
        reasoning_content: response.reasoning_content,
        usage,
    })
}

/// Sends a chat request via GitHub Copilot (token exchange + Copilot API).
async fn send_copilot_request<F>(
    client: &Client,
    tm: &CopilotTokenManager,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
    on_chunk: &mut F,
) -> Result<ChatTurnResult>
where
    F: FnMut(ChatResponseChunk) -> Result<()>,
{
    let (copilot_token, api_endpoint) = tm.get_token().await?;
    let url = format!("{}/chat/completions", api_endpoint.trim_end_matches('/'));

    let request = build_openai_chat_request(model, messages, true)?;

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
        return send_copilot_non_streaming(
            client,
            &url,
            &copilot_token,
            model,
            messages,
            spinning,
            on_chunk,
        )
        .await;
    }

    if !response.status().is_success() {
        style::stop_spinner(spinning);
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("API returned {} — {}", status, body);
    }

    let mut full_content = String::new();
    let mut full_reasoning = String::new();
    let mut usage = None;
    let mut line_buf = String::new();
    let mut done = false;

    while !done {
        let chunk_result = response.chunk().await;
        let Some(chunk) = (match chunk_result {
            Ok(c) => c,
            Err(_) if !full_content.is_empty() || !full_reasoning.is_empty() => {
                // Stream error after content was received — use what we have.
                break;
            }
            Err(e) => return Err(e.into()),
        }) else {
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
                    merge_token_usage(&mut usage, tokens);
                }
                if let Some(chunk) = parse_sse_chunk(data) {
                    style::stop_spinner(spinning);
                    match &chunk {
                        ChatResponseChunk::Content(content) => full_content.push_str(content),
                        ChatResponseChunk::Reasoning(reasoning) => {
                            full_reasoning.push_str(reasoning);
                        }
                    }
                    on_chunk(chunk)?;
                }
            }
        }
    }

    let tail = line_buf.trim();
    if !tail.is_empty() {
        if let Some(data) = sse_data_payload(tail) {
            if let Some(tokens) = parse_openai_usage_chunk(data) {
                merge_token_usage(&mut usage, tokens);
            }
            if data.trim() != "[DONE]"
                && let Some(chunk) = parse_sse_chunk(data)
            {
                style::stop_spinner(spinning);
                match &chunk {
                    ChatResponseChunk::Content(content) => full_content.push_str(content),
                    ChatResponseChunk::Reasoning(reasoning) => full_reasoning.push_str(reasoning),
                }
                on_chunk(chunk)?;
            }
        } else if full_content.is_empty()
            && let Ok(resp) = serde_json::from_str::<serde_json::Value>(tail)
        {
            let response = extract_openai_message(&resp);
            if !response.content.is_empty() || response.reasoning_content.is_some() {
                style::stop_spinner(spinning);
                if let Some(reasoning) = response.reasoning_content.clone() {
                    on_chunk(ChatResponseChunk::Reasoning(reasoning.clone()))?;
                    full_reasoning = reasoning;
                }
                if !response.content.is_empty() {
                    on_chunk(ChatResponseChunk::Content(response.content.clone()))?;
                    full_content = response.content;
                }
            }
        }
    }

    if full_content.is_empty() && full_reasoning.is_empty() {
        return send_copilot_non_streaming(
            client,
            &url,
            &copilot_token,
            model,
            messages,
            spinning,
            on_chunk,
        )
        .await;
    }

    Ok(ChatTurnResult {
        content: full_content,
        reasoning_content: normalize_reasoning_content(full_reasoning),
        usage,
    })
}

async fn send_copilot_non_streaming<F>(
    client: &Client,
    url: &str,
    copilot_token: &str,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
    on_chunk: &mut F,
) -> Result<ChatTurnResult>
where
    F: FnMut(ChatResponseChunk) -> Result<()>,
{
    let request = build_openai_chat_request(model, messages, false)?;

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
    let response = extract_openai_message(&body);
    let usage = extract_openai_usage(&body);

    if response.content.is_empty() && response.reasoning_content.is_none() {
        style::stop_spinner(spinning);
        anyhow::bail!("Provider returned an empty response");
    }

    style::stop_spinner(spinning);
    if let Some(reasoning) = response.reasoning_content.clone() {
        on_chunk(ChatResponseChunk::Reasoning(reasoning))?;
    }
    if !response.content.is_empty() {
        on_chunk(ChatResponseChunk::Content(response.content.clone()))?;
    }

    Ok(ChatTurnResult {
        content: response.content,
        reasoning_content: response.reasoning_content,
        usage,
    })
}

/// Parses a single SSE data chunk and extracts either a content or reasoning delta.
fn parse_sse_chunk(data: &str) -> Option<ChatResponseChunk> {
    let chunk: ChatChunk = serde_json::from_str(data).ok()?;
    let delta = &chunk.choices.first()?.delta;
    delta
        .reasoning_content
        .clone()
        .or_else(|| delta.reasoning.clone())
        .or_else(|| delta.thinking.clone())
        .filter(|text| !text.is_empty())
        .map(ChatResponseChunk::Reasoning)
        .or_else(|| {
            delta
                .content
                .clone()
                .filter(|text| !text.is_empty())
                .map(ChatResponseChunk::Content)
        })
}

/// Extracts `<think>...</think>` blocks that some LLMs embed inline in their content stream,
/// re-routing them as `Reasoning` chunks. Handles partial tags split across streaming chunks.
pub(super) struct ThinkTagParser {
    inside_think: bool,
    buf: String,
}

impl ThinkTagParser {
    fn new() -> Self {
        Self {
            inside_think: false,
            buf: String::new(),
        }
    }

    /// Feed a content string; returns re-routed chunks to emit.
    fn feed(&mut self, text: &str) -> Vec<ChatResponseChunk> {
        self.buf.push_str(text);
        let mut out = Vec::new();
        loop {
            if self.inside_think {
                const CLOSE: &str = "</think>";
                if let Some(pos) = self.buf.find(CLOSE) {
                    let reasoning = self.buf[..pos].to_string();
                    if !reasoning.trim().is_empty() {
                        out.push(ChatResponseChunk::Reasoning(reasoning));
                    }
                    self.buf = self.buf[pos + CLOSE.len()..].to_string();
                    if self.buf.starts_with('\n') {
                        self.buf = self.buf[1..].to_string();
                    }
                    self.inside_think = false;
                } else {
                    let keep = longest_suffix_prefix_len(&self.buf, CLOSE);
                    let emit_end = self.buf.len() - keep;
                    if emit_end > 0 {
                        out.push(ChatResponseChunk::Reasoning(
                            self.buf[..emit_end].to_string(),
                        ));
                    }
                    self.buf = self.buf[emit_end..].to_string();
                    break;
                }
            } else {
                const OPEN: &str = "<think>";
                if let Some(pos) = self.buf.find(OPEN) {
                    if pos > 0 {
                        out.push(ChatResponseChunk::Content(self.buf[..pos].to_string()));
                    }
                    self.buf = self.buf[pos + OPEN.len()..].to_string();
                    self.inside_think = true;
                } else {
                    let keep = longest_suffix_prefix_len(&self.buf, OPEN);
                    let emit_end = self.buf.len() - keep;
                    if emit_end > 0 {
                        out.push(ChatResponseChunk::Content(self.buf[..emit_end].to_string()));
                    }
                    self.buf = self.buf[emit_end..].to_string();
                    break;
                }
            }
        }
        out
    }

    /// Flush remaining buffered content after the stream ends.
    fn flush(&mut self) -> Vec<ChatResponseChunk> {
        if self.buf.is_empty() {
            return Vec::new();
        }
        let text = std::mem::take(&mut self.buf);
        if self.inside_think {
            vec![ChatResponseChunk::Reasoning(text)]
        } else {
            vec![ChatResponseChunk::Content(text)]
        }
    }
}

/// Returns the length of the longest suffix of `text` that is also a prefix of `tag`.
/// Used to avoid splitting a potential partial tag across chunk boundaries.
fn longest_suffix_prefix_len(text: &str, tag: &str) -> usize {
    let text_bytes = text.as_bytes();
    let tag_bytes = tag.as_bytes();
    let max_check = tag_bytes.len().min(text_bytes.len());
    for len in (1..=max_check).rev() {
        if text_bytes[text_bytes.len() - len..] == tag_bytes[..len] {
            return len;
        }
    }
    0
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
async fn send_anthropic_request<F>(
    client: &Client,
    key: &ApiKey,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
    on_chunk: &mut F,
) -> Result<ChatTurnResult>
where
    F: FnMut(ChatResponseChunk) -> Result<()>,
{
    let base = normalize_base_url(&key.base_url);
    let url = format!("{}/v1/messages", base);

    let request = build_anthropic_request(model, messages, true)?;

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
        return send_anthropic_non_streaming(
            client, &url, key, model, messages, spinning, on_chunk,
        )
        .await;
    }

    if !response.status().is_success() {
        style::stop_spinner(spinning);
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("API returned {} — {}", status, body);
    }

    let mut full_content = String::new();
    let mut full_reasoning = String::new();
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
                    merge_token_usage(&mut usage, tokens);
                }
                if let Some(chunk) = parse_anthropic_chunk(data) {
                    style::stop_spinner(spinning);
                    match &chunk {
                        ChatResponseChunk::Content(text) => full_content.push_str(text),
                        ChatResponseChunk::Reasoning(reasoning) => {
                            full_reasoning.push_str(reasoning);
                        }
                    }
                    on_chunk(chunk)?;
                }
            }
        }
    }

    if full_content.is_empty() {
        let tail = line_buf.trim();
        if let Some(data) = sse_data_payload(tail) {
            if let Some(tokens) = parse_anthropic_usage_chunk(data) {
                merge_token_usage(&mut usage, tokens);
            }
            if let Some(chunk) = parse_anthropic_chunk(data) {
                style::stop_spinner(spinning);
                match &chunk {
                    ChatResponseChunk::Content(text) => full_content.push_str(text),
                    ChatResponseChunk::Reasoning(reasoning) => full_reasoning.push_str(reasoning),
                }
                on_chunk(chunk)?;
            }
        }
    }

    // If streaming produced no content, fall back to non-streaming
    if full_content.is_empty() && full_reasoning.is_empty() {
        return send_anthropic_non_streaming(
            client, &url, key, model, messages, spinning, on_chunk,
        )
        .await;
    }

    Ok(ChatTurnResult {
        content: full_content,
        reasoning_content: normalize_reasoning_content(full_reasoning),
        usage,
    })
}

/// Non-streaming fallback for Anthropic-format providers.
async fn send_anthropic_non_streaming<F>(
    client: &Client,
    url: &str,
    key: &ApiKey,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
    on_chunk: &mut F,
) -> Result<ChatTurnResult>
where
    F: FnMut(ChatResponseChunk) -> Result<()>,
{
    let request = build_anthropic_request(model, messages, false)?;

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

    let mut content_parts = Vec::new();
    let mut reasoning_parts = Vec::new();
    for block in body["content"].as_array().into_iter().flatten() {
        match block.get("type").and_then(|v| v.as_str()).unwrap_or("") {
            "text" => {
                if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                    content_parts.push(text.to_string());
                }
            }
            "thinking" => {
                if let Some(reasoning) = block
                    .get("thinking")
                    .and_then(|v| v.as_str())
                    .or_else(|| block.get("text").and_then(|v| v.as_str()))
                {
                    reasoning_parts.push(reasoning.to_string());
                }
            }
            _ => {}
        }
    }

    let content = content_parts.concat();
    let reasoning_content = normalize_reasoning_content(reasoning_parts.join(""));

    if content.is_empty() && reasoning_content.is_none() {
        style::stop_spinner(spinning);
        anyhow::bail!("Provider returned an empty response");
    }

    style::stop_spinner(spinning);
    if let Some(reasoning) = reasoning_content.clone() {
        on_chunk(ChatResponseChunk::Reasoning(reasoning))?;
    }
    if !content.is_empty() {
        on_chunk(ChatResponseChunk::Content(content.clone()))?;
    }

    Ok(ChatTurnResult {
        content,
        reasoning_content,
        usage,
    })
}

/// Parses an Anthropic SSE data line and returns either a text or thinking delta.
fn parse_anthropic_chunk(data: &str) -> Option<ChatResponseChunk> {
    let event: AnthropicStreamEvent = serde_json::from_str(data).ok()?;
    if event.event_type == "content_block_delta" {
        let delta = event.delta?;
        delta
            .thinking
            .filter(|text| !text.is_empty())
            .map(ChatResponseChunk::Reasoning)
            .or_else(|| {
                delta
                    .text
                    .filter(|text| !text.is_empty())
                    .map(ChatResponseChunk::Content)
            })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sse_chunk_with_content() {
        let data = r#"{"id":"chatcmpl-1","choices":[{"delta":{"content":"Hello"}}]}"#;
        assert_eq!(
            parse_sse_chunk(data),
            Some(ChatResponseChunk::Content("Hello".to_string()))
        );
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
    fn test_extract_openai_message_string_and_parts() {
        let text = serde_json::json!({
            "choices": [{"message": {"content": "hello"}}]
        });
        assert_eq!(
            extract_openai_message(&text),
            AssistantResponse {
                content: "hello".to_string(),
                reasoning_content: None,
            }
        );

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
        assert_eq!(
            extract_openai_message(&parts),
            AssistantResponse {
                content: "hello world".to_string(),
                reasoning_content: None,
            }
        );
    }

    #[test]
    fn test_extract_openai_message_reasoning_content() {
        let body = serde_json::json!({
            "choices": [{
                "message": {
                    "content": "answer",
                    "reasoning_content": "step by step"
                }
            }]
        });

        assert_eq!(
            extract_openai_message(&body),
            AssistantResponse {
                content: "answer".to_string(),
                reasoning_content: Some("step by step".to_string()),
            }
        );
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
    fn test_sanitize_one_shot_message_rejects_whitespace() {
        let err = sanitize_one_shot_message(" \n\t ".to_string()).unwrap_err();
        assert!(err.to_string().contains("cannot be empty"));
    }

    #[test]
    fn test_sanitize_one_shot_message_preserves_content() {
        let out = sanitize_one_shot_message("hello\nworld\n".to_string()).unwrap();
        assert_eq!(out, "hello\nworld\n");
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
            reasoning_content: Some("hidden".to_string()),
            attachments: vec![],
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"hello\""));
        assert!(!json.contains("reasoning_content"));
    }

    #[test]
    fn test_build_openai_chat_request_encodes_file_and_image_attachments() {
        let request = build_openai_chat_request(
            "gpt-4o",
            &[ChatMessage {
                role: "user".to_string(),
                content: "Review these".to_string(),
                reasoning_content: None,
                attachments: vec![
                    MessageAttachment {
                        name: "notes.md".to_string(),
                        mime_type: "text/markdown".to_string(),
                        storage: AttachmentStorage::Inline {
                            data: "# hello".to_string(),
                        },
                    },
                    MessageAttachment {
                        name: "diagram.png".to_string(),
                        mime_type: "image/png".to_string(),
                        storage: AttachmentStorage::Inline {
                            data: "YWJj".to_string(),
                        },
                    },
                ],
            }],
            true,
        )
        .unwrap();

        let parts = request["messages"][0]["content"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "text");
        assert!(parts[1]["text"].as_str().unwrap().contains("notes.md"));
        assert_eq!(parts[2]["type"], "image_url");
        assert_eq!(parts[2]["image_url"]["url"], "data:image/png;base64,YWJj");
    }

    #[test]
    fn test_build_anthropic_request_encodes_image_attachment() {
        let request = build_anthropic_request(
            "claude-sonnet-4-5",
            &[ChatMessage {
                role: "user".to_string(),
                content: String::new(),
                reasoning_content: None,
                attachments: vec![MessageAttachment {
                    name: "diagram.png".to_string(),
                    mime_type: "image/png".to_string(),
                    storage: AttachmentStorage::Inline {
                        data: "YWJj".to_string(),
                    },
                }],
            }],
            false,
        )
        .unwrap();

        let blocks = request["messages"][0]["content"].as_array().unwrap();
        assert_eq!(blocks[0]["type"], "image");
        assert_eq!(blocks[0]["source"]["media_type"], "image/png");
        assert_eq!(blocks[0]["source"]["data"], "YWJj");
    }

    #[test]
    fn test_parse_anthropic_chunk_with_text() {
        let data = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
        assert_eq!(
            parse_anthropic_chunk(data),
            Some(ChatResponseChunk::Content("Hello".to_string()))
        );
    }

    #[test]
    fn test_parse_anthropic_chunk_with_thinking() {
        let data = r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Need to inspect files."}}"#;
        assert_eq!(
            parse_anthropic_chunk(data),
            Some(ChatResponseChunk::Reasoning(
                "Need to inspect files.".to_string()
            ))
        );
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
    fn test_merge_openai_stream_usage_across_chunks() {
        let mut usage = None;
        merge_token_usage(
            &mut usage,
            parse_openai_usage_chunk(r#"{"usage":{"prompt_tokens":24}}"#).unwrap(),
        );
        merge_token_usage(
            &mut usage,
            parse_openai_usage_chunk(r#"{"usage":{"completion_tokens":11}}"#).unwrap(),
        );

        assert_eq!(
            usage,
            Some(TokenUsage {
                prompt_tokens: 24,
                completion_tokens: 11,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
            })
        );
    }

    #[test]
    fn test_merge_anthropic_stream_usage_across_events() {
        let mut usage = None;
        merge_token_usage(
            &mut usage,
            parse_anthropic_usage_chunk(r#"{"usage":{"input_tokens":12}}"#).unwrap(),
        );
        merge_token_usage(
            &mut usage,
            parse_anthropic_usage_chunk(r#"{"usage":{"output_tokens":7}}"#).unwrap(),
        );

        assert_eq!(
            usage,
            Some(TokenUsage {
                prompt_tokens: 12,
                completion_tokens: 7,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
            })
        );
    }

    #[test]
    fn test_extract_openai_usage_accepts_numeric_strings() {
        let body = serde_json::json!({
            "usage": {
                "prompt_tokens": "10",
                "completion_tokens": "5"
            }
        });

        assert_eq!(
            extract_openai_usage(&body),
            Some(TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
            })
        );
    }

    #[test]
    fn test_extract_anthropic_usage_accepts_numeric_strings() {
        let body = serde_json::json!({
            "usage": {
                "input_tokens": "8",
                "output_tokens": "3",
                "cache_read_input_tokens": "90",
                "cache_creation_input_tokens": "15"
            }
        });

        assert_eq!(
            extract_anthropic_usage(&body),
            Some(TokenUsage {
                prompt_tokens: 113,
                completion_tokens: 3,
                cache_read_input_tokens: 90,
                cache_creation_input_tokens: 15,
            })
        );
    }

    #[test]
    fn test_extract_openai_usage_reads_cached_tokens_details() {
        let body = serde_json::json!({
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "prompt_tokens_details": {
                    "cached_tokens": 90
                }
            }
        });

        assert_eq!(
            extract_openai_usage(&body),
            Some(TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                cache_read_input_tokens: 90,
                cache_creation_input_tokens: 0,
            })
        );
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
