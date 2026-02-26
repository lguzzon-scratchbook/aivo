/**
 * ChatCommand handler for interactive REPL with streaming API responses.
 * Makes direct HTTP calls to OpenAI-compatible /v1/chat/completions endpoint.
 */
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use futures_util::StreamExt;
use reqwest::Client;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use serde::{Deserialize, Serialize};

use crate::errors::ExitCode;
use crate::services::session_store::{ApiKey, SessionStore};
use crate::style;

const DEFAULT_MODEL: &str = "gpt-4o";

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

/// ChatCommand provides an interactive REPL for chatting with AI models
pub struct ChatCommand {
    session_store: SessionStore,
}

impl ChatCommand {
    pub fn new(session_store: SessionStore) -> Self {
        Self { session_store }
    }

    /// Resolves the model to use: --model flag > persisted > default
    async fn resolve_model(&self, flag_model: Option<String>) -> Result<String> {
        if let Some(model) = flag_model {
            // Save as the new default
            self.session_store.set_chat_model(&model).await?;
            return Ok(model);
        }

        if let Some(saved) = self.session_store.get_chat_model().await? {
            return Ok(saved);
        }

        Ok(DEFAULT_MODEL.to_string())
    }

    pub async fn execute(&self, model: Option<String>) -> ExitCode {
        match self.execute_internal(model).await {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                ExitCode::UserError
            }
        }
    }

    async fn execute_internal(&self, model_flag: Option<String>) -> Result<ExitCode> {
        let key = match self.session_store.get_active_key().await? {
            Some(k) => k,
            None => {
                eprintln!(
                    "{} No API key configured. Run 'aivo keys add' first.",
                    style::red("Error:")
                );
                return Ok(ExitCode::AuthError);
            }
        };

        let model = self.resolve_model(model_flag).await?;

        eprintln!(
            "{} model: {} {}",
            style::success_symbol(),
            style::cyan(&model),
            style::dim(format!("({})", key.base_url))
        );
        eprintln!(
            "{}",
            style::dim("Type 'exit' to end. Ctrl+D also works.")
        );

        let client = Client::new();
        let mut history: Vec<ChatMessage> = Vec::new();
        let prompt = format!("{} ", style::cyan(">"));

        let mut rl = DefaultEditor::new().map_err(|e| anyhow::anyhow!("{}", e))?;

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

            if input == "exit" || input == "quit" {
                break;
            }

            // Add user message to history
            history.push(ChatMessage {
                role: "user".to_string(),
                content: input,
            });

            // Start loading spinner
            let spinning = Arc::new(AtomicBool::new(true));
            let spinning_clone = spinning.clone();
            let spinner_handle = tokio::task::spawn_blocking(move || {
                let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
                let mut i = 0;
                while spinning_clone.load(Ordering::Relaxed) {
                    eprint!("\r{}", style::dim(frames[i % frames.len()]));
                    let _ = io::stderr().flush();
                    std::thread::sleep(std::time::Duration::from_millis(80));
                    i += 1;
                }
            });

            // Stream response (retry once on transient errors)
            let result =
                match send_chat_request(&client, &key, &model, &history, &spinning).await {
                    Ok(content) => Ok(content),
                    Err(_) => {
                        send_chat_request(&client, &key, &model, &history, &spinning).await
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
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo chat"));
        println!("  {}", style::dim("aivo chat --model gpt-4o"));
        println!("  {}", style::dim("aivo chat -m claude-sonnet-4-5"));
    }
}

/// Stops the spinner and clears its character from the line.
fn stop_spinner(spinning: &Arc<AtomicBool>) {
    if spinning.swap(false, Ordering::Relaxed) {
        // Wait longer than one spinner frame (80ms) so the thread exits its loop
        std::thread::sleep(std::time::Duration::from_millis(100));
        eprint!("\r \r");
        let _ = io::stderr().flush();
    }
}

/// Sends a chat completion request and prints the response.
/// Tries streaming first; falls back to non-streaming if the server returns 503.
/// Returns the full assistant message content.
async fn send_chat_request(
    client: &Client,
    key: &ApiKey,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
) -> Result<String> {
    let base = key.base_url.trim_end_matches('/');
    let base = base.strip_suffix("/v1").unwrap_or(base);
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

    // If streaming is not supported, fall back to non-streaming
    if response.status().is_server_error() || response.status() == reqwest::StatusCode::NOT_FOUND {
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
    if full_content.is_empty() && !line_buf.is_empty() {
        if let Ok(resp) = serde_json::from_str::<serde_json::Value>(&line_buf) {
            if let Some(content) = resp["choices"][0]["message"]["content"].as_str() {
                stop_spinner(spinning);
                print!("{}", content);
                io::stdout().flush()?;
                full_content = content.to_string();
            }
        }
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

/// Parses a single SSE data chunk and extracts the content delta
pub fn parse_sse_chunk(data: &str) -> Option<String> {
    let chunk: ChatChunk = serde_json::from_str(data).ok()?;
    chunk.choices.first()?.delta.content.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
