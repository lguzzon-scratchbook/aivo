/**
 * ChatCommand handler for interactive REPL with streaming API responses.
 * Tries OpenAI-compatible /v1/chat/completions first; falls back to
 * Anthropic's /v1/messages format if the provider returns 404/405.
 */
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crate::tui::FuzzySelect;
use anyhow::Result;
use futures_util::StreamExt;
use reqwest::Client;
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
use crate::services::model_names;
use crate::services::models_cache::ModelsCache;
use crate::services::session_store::{ApiKey, SessionStore};
use crate::style;

const CMD_EXIT: &str = "/exit";
const CMD_MODEL: &str = "/model";
const CMD_MODEL_ARG: &str = "/model ";
/// Maximum number of messages to keep in chat history.
/// When exceeded, the oldest messages are dropped (keeping any system message).
const MAX_HISTORY_MESSAGES: usize = 50;

struct ChatHelper {
    commands: Vec<&'static str>,
}

impl ChatHelper {
    fn new() -> Self {
        Self {
            commands: vec![CMD_EXIT, CMD_MODEL],
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

#[derive(Debug, Serialize)]
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

    pub async fn execute(&self, model: Option<String>, key_override: Option<ApiKey>) -> ExitCode {
        match self.execute_internal(model, key_override).await {
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
        key_override: Option<ApiKey>,
    ) -> Result<ExitCode> {
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

        let client = Client::new();

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

        eprintln!(
            "{} model: {} {}",
            style::success_symbol(),
            style::cyan(&model),
            style::dim(format!("({})", key.base_url))
        );
        eprintln!(
            "{}",
            style::dim(
                "Type /exit to quit, /model to pick a model, /model <name> to set directly. Ctrl+D also works."
            )
        );
        let mut history: Vec<ChatMessage> = Vec::new();
        let mut format = ChatFormat::OpenAI;
        let prompt = format!("{} ", style::cyan(">"));

        // Create once so its token cache is reused across messages in the session.
        let copilot_tm = if key.base_url == "copilot" {
            Some(CopilotTokenManager::new(key.key.as_str().to_string()))
        } else {
            None
        };

        let mut rl = Editor::<ChatHelper, rustyline::history::DefaultHistory>::new()
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        rl.set_helper(Some(ChatHelper::new()));

        let history_path: PathBuf = dirs::home_dir()
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

            if input == CMD_MODEL || input.starts_with(CMD_MODEL_ARG) {
                let selected_raw = if input == CMD_MODEL {
                    let models_list = self.fetch_models_for_select(&client, &key).await;
                    if models_list.is_empty() {
                        eprintln!("  model: {}", style::cyan(&model));
                        None
                    } else {
                        // Use raw_model (pre-transform) to find current selection in the list
                        let current_idx = models_list
                            .iter()
                            .position(|m| m == &raw_model)
                            .unwrap_or(0);
                        FuzzySelect::new()
                            .with_prompt("Select model")
                            .items(&models_list)
                            .default(current_idx)
                            .interact_opt()
                            .ok()
                            .flatten()
                            .map(|idx| models_list[idx].clone())
                    }
                } else {
                    input
                        .strip_prefix(CMD_MODEL_ARG)
                        .map(|s| s.trim())
                        .filter(|s| !s.is_empty())
                        .map(|s| s.to_string())
                };

                if let Some(raw) = selected_raw {
                    self.session_store.set_chat_model(&key.id, &raw).await?;
                    raw_model = raw.clone();
                    model = Self::transform_model_for_provider(&key.base_url, &raw);
                    eprintln!("  model: {}", style::cyan(&model));
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
            let result = if let Some(ref tm) = copilot_tm {
                send_copilot_request(&client, tm, &model, &history, &spinning).await
            } else {
                match format {
                    ChatFormat::OpenAI => {
                        match send_chat_request(&client, &key, &model, &history, &spinning).await {
                            ok @ Ok(_) => ok,
                            Err(e) if is_format_mismatch(&e) => {
                                // Provider doesn't speak OpenAI format; try Anthropic
                                match send_anthropic_request(
                                    &client, &key, &model, &history, &spinning,
                                )
                                .await
                                {
                                    Ok(content) => {
                                        eprintln!(
                                            "{}",
                                            style::dim("  (using Anthropic messages format)")
                                        );
                                        format = ChatFormat::Anthropic;
                                        Ok(content)
                                    }
                                    Err(_) => Err(e), // both failed; report original error
                                }
                            }
                            Err(e) => Err(e),
                        }
                    }
                    ChatFormat::Anthropic => {
                        send_anthropic_request(&client, &key, &model, &history, &spinning).await
                    }
                }
            };

            stop_spinner(&spinning);
            let _ = spinner_handle.await;
            match result {
                Ok(assistant_content) => {
                    // Ensure newline after streamed response
                    println!();
                    history.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: assistant_content,
                    });
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
        println!("{} aivo chat [--model <model>]", style::bold("Usage:"));
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
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo chat"));
        println!("  {}", style::dim("aivo chat --model gpt-4o"));
        println!("  {}", style::dim("aivo chat -m claude-sonnet-4-5"));
    }
}

/// Stops the spinner and clears its character from the line.
fn stop_spinner(spinning: &Arc<AtomicBool>) {
    style::stop_spinner(spinning);
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
) -> Result<String> {
    let base = normalize_base_url(&key.base_url);
    let url = format!("{}/v1/chat/completions", base);

    // Try streaming first; fall back to non-streaming on server errors
    let request = ChatRequest {
        model: model.to_string(),
        messages: messages.to_vec(),
        stream: true,
    };

    let response = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", key.key.as_str()))
        .header("Content-Type", "application/json")
        .header("User-Agent", format!("aivo/{}", crate::version::VERSION))
        .json(&request)
        .send()
        .await?;

    // If the server can't handle streaming, fall back to non-streaming.
    // Note: 404 is NOT included here — it means wrong endpoint, not streaming unsupported.
    // The caller detects 404 and switches to a different API format instead.
    if response.status().is_server_error() {
        return send_non_streaming(client, &url, key, model, messages, spinning).await;
    }

    if !response.status().is_success() {
        stop_spinner(spinning);
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("API returned {} — {}", status, body);
    }

    let mut full_content = String::new();
    let mut line_buf = String::new();

    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let text = String::from_utf8_lossy(&chunk);
        line_buf.push_str(&text);

        while let Some(pos) = line_buf.find('\n') {
            let line = line_buf[..pos].trim_end_matches('\r').to_string();
            line_buf = line_buf[pos + 1..].to_string();

            if let Some(data) = line.strip_prefix("data: ") {
                if data.trim() == "[DONE]" {
                    break;
                }
                if let Some(content) = parse_sse_chunk(data) {
                    stop_spinner(spinning);
                    print!("{}", content);
                    io::stdout().flush()?;
                    full_content.push_str(&content);
                }
            }
        }
    }

    // If we got no streaming data, the response might be non-streaming JSON
    if full_content.is_empty()
        && !line_buf.is_empty()
        && let Ok(resp) = serde_json::from_str::<serde_json::Value>(&line_buf)
        && let Some(content) = resp["choices"][0]["message"]["content"].as_str()
    {
        stop_spinner(spinning);
        print!("{}", content);
        io::stdout().flush()?;
        full_content = content.to_string();
    }

    Ok(full_content)
}

/// Non-streaming fallback for gateways that don't support SSE streaming.
async fn send_non_streaming(
    client: &Client,
    url: &str,
    key: &ApiKey,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
) -> Result<String> {
    let request = ChatRequest {
        model: model.to_string(),
        messages: messages.to_vec(),
        stream: false,
    };

    let response = client
        .post(url)
        .header("Authorization", format!("Bearer {}", key.key.as_str()))
        .header("Content-Type", "application/json")
        .header("User-Agent", format!("aivo/{}", crate::version::VERSION))
        .json(&request)
        .send()
        .await?;

    if !response.status().is_success() {
        stop_spinner(spinning);
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("API returned {} — {}", status, body);
    }

    let body: serde_json::Value = response.json().await?;
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();

    stop_spinner(spinning);
    print!("{}", content);
    io::stdout().flush()?;

    Ok(content)
}

/// Sends a chat request via GitHub Copilot (token exchange + Copilot API).
async fn send_copilot_request(
    client: &Client,
    tm: &CopilotTokenManager,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
) -> Result<String> {
    let (copilot_token, api_endpoint) = tm.get_token().await?;
    let url = format!("{}/chat/completions", api_endpoint.trim_end_matches('/'));

    let request = ChatRequest {
        model: model.to_string(),
        messages: messages.to_vec(),
        stream: true,
    };

    let response = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", copilot_token))
        .header("Content-Type", "application/json")
        .header("Editor-Version", COPILOT_EDITOR_VERSION)
        .header("Copilot-Integration-Id", COPILOT_INTEGRATION_ID)
        .header("Openai-Intent", COPILOT_OPENAI_INTENT)
        .json(&request)
        .send()
        .await?;

    if !response.status().is_success() {
        stop_spinner(spinning);
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("API returned {} — {}", status, body);
    }

    let mut full_content = String::new();
    let mut line_buf = String::new();

    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let text = String::from_utf8_lossy(&chunk);
        line_buf.push_str(&text);

        while let Some(pos) = line_buf.find('\n') {
            let line = line_buf[..pos].trim_end_matches('\r').to_string();
            line_buf = line_buf[pos + 1..].to_string();

            if let Some(data) = line.strip_prefix("data: ") {
                if data.trim() == "[DONE]" {
                    break;
                }
                if let Some(content) = parse_sse_chunk(data) {
                    stop_spinner(spinning);
                    print!("{}", content);
                    io::stdout().flush()?;
                    full_content.push_str(&content);
                }
            }
        }
    }

    if full_content.is_empty()
        && !line_buf.is_empty()
        && let Ok(resp) = serde_json::from_str::<serde_json::Value>(&line_buf)
        && let Some(content) = resp["choices"][0]["message"]["content"].as_str()
    {
        stop_spinner(spinning);
        print!("{}", content);
        io::stdout().flush()?;
        full_content = content.to_string();
    }

    Ok(full_content)
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
    let msg = e.to_string();
    msg.contains("404") || msg.contains("405")
}

/// Sends a request using Anthropic's native /v1/messages API.
/// Tries streaming first; falls back to non-streaming on server errors.
async fn send_anthropic_request(
    client: &Client,
    key: &ApiKey,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
) -> Result<String> {
    let base = normalize_base_url(&key.base_url);
    let url = format!("{}/v1/messages", base);

    let request = serde_json::json!({
        "model": model,
        "messages": messages,
        "max_tokens": 8096,
        "stream": true,
    });

    let response = client
        .post(&url)
        // Send both auth headers: gateways vary on which they accept
        .header("Authorization", format!("Bearer {}", key.key.as_str()))
        .header("x-api-key", key.key.as_str())
        .header("anthropic-version", "2023-06-01")
        .header("Content-Type", "application/json")
        .header("User-Agent", format!("aivo/{}", crate::version::VERSION))
        .json(&request)
        .send()
        .await?;

    if response.status().is_server_error() || response.status() == reqwest::StatusCode::NOT_FOUND {
        return send_anthropic_non_streaming(client, &url, key, model, messages, spinning).await;
    }

    if !response.status().is_success() {
        stop_spinner(spinning);
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("API returned {} — {}", status, body);
    }

    let mut full_content = String::new();
    let mut line_buf = String::new();

    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let text = String::from_utf8_lossy(&chunk);
        line_buf.push_str(&text);

        while let Some(pos) = line_buf.find('\n') {
            let line = line_buf[..pos].trim_end_matches('\r').to_string();
            line_buf = line_buf[pos + 1..].to_string();

            if let Some(data) = line.strip_prefix("data: ")
                && let Some(text) = parse_anthropic_chunk(data)
            {
                stop_spinner(spinning);
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

    Ok(full_content)
}

/// Non-streaming fallback for Anthropic-format providers.
async fn send_anthropic_non_streaming(
    client: &Client,
    url: &str,
    key: &ApiKey,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
) -> Result<String> {
    let request = serde_json::json!({
        "model": model,
        "messages": messages,
        "max_tokens": 8096,
        "stream": false,
    });

    let response = client
        .post(url)
        // Send both auth headers: gateways vary on which they accept
        .header("Authorization", format!("Bearer {}", key.key.as_str()))
        .header("x-api-key", key.key.as_str())
        .header("anthropic-version", "2023-06-01")
        .header("Content-Type", "application/json")
        .header("User-Agent", format!("aivo/{}", crate::version::VERSION))
        .json(&request)
        .send()
        .await?;

    if !response.status().is_success() {
        stop_spinner(spinning);
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("API returned {} — {}", status, body);
    }

    let body: serde_json::Value = response.json().await?;

    // Try Anthropic format: content[].text
    let content: String = body["content"]
        .as_array()
        .iter()
        .flat_map(|arr| arr.iter())
        .filter(|c| c["type"].as_str() == Some("text"))
        .filter_map(|c| c["text"].as_str())
        .collect();

    if content.is_empty() {
        stop_spinner(spinning);
        anyhow::bail!("Provider returned an empty response");
    }

    stop_spinner(spinning);
    print!("{}", content);
    io::stdout().flush()?;

    Ok(content)
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
        assert_eq!(completions.len(), 2);
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
    fn test_is_format_mismatch_other_errors() {
        let e = anyhow::anyhow!("API returned 401 Unauthorized");
        assert!(!is_format_mismatch(&e));
        let e = anyhow::anyhow!("API returned 429 Too Many Requests");
        assert!(!is_format_mismatch(&e));
    }
}
