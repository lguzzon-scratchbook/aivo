/**
 * OpenAI-Compatible Router Service
 *
 * Acts as an HTTP proxy that intercepts Claude Code requests (Anthropic format)
 * and routes them to OpenAI-compatible providers (like Cloudflare Workers AI),
 * handling all necessary API transformations.
 *
 * Flow:
 * Claude Code (Anthropic /v1/messages) → Router → OpenAI /v1/chat/completions → Cloudflare
 */
use anyhow::Result;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;

use crate::services::http_utils::{self, router_http_client};

#[derive(Clone)]
pub struct OpenAIRouterConfig {
    /// The target OpenAI-compatible provider base URL (e.g., Cloudflare)
    pub target_base_url: String,
    /// API key for the target provider
    pub target_api_key: String,
    /// Optional model prefix to add (e.g., "@cf/" for Cloudflare)
    pub model_prefix: Option<String>,
    /// Whether the provider requires `reasoning_content` on assistant tool-call turns (e.g., Moonshot)
    pub requires_reasoning_content: bool,
}

pub struct OpenAIRouter {
    config: OpenAIRouterConfig,
}

impl OpenAIRouter {
    pub fn new(config: OpenAIRouterConfig) -> Self {
        Self { config }
    }

    /// Binds to a random available port and starts the router in the background.
    /// Returns the actual port number so callers can set ANTHROPIC_BASE_URL.
    pub async fn start_background(&self) -> Result<(u16, tokio::task::JoinHandle<Result<()>>)> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let port = listener.local_addr()?.port();
        let config = self.config.clone();
        let handle = tokio::spawn(async move { run_openai_router(listener, config).await });
        Ok((port, handle))
    }
}

async fn run_openai_router(
    listener: tokio::net::TcpListener,
    config: OpenAIRouterConfig,
) -> Result<()> {
    let config = Arc::new(config);

    loop {
        let (mut socket, _) = listener.accept().await?;
        let config = config.clone();

        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;

            let request_bytes = match http_utils::read_full_request(&mut socket).await {
                Ok(b) => b,
                Err(_) => return,
            };

            let request = String::from_utf8_lossy(&request_bytes);

            // Route Anthropic /v1/messages to OpenAI /v1/chat/completions
            let response = if request.starts_with("POST /v1/messages") {
                match handle_anthropic_to_openai(&request, &config).await {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("Router error: {}", e);
                        http_utils::http_error_response(500, "Internal Server Error")
                    }
                }
            } else {
                http_utils::http_response(404, "application/json", "{\"error\":\"Not found\"}")
            };

            let _ = socket.write_all(response.as_bytes()).await;
        });
    }
}

/// Apply an optional prefix to a model name, skipping if the prefix is already present.
fn apply_model_prefix(model: &str, prefix: Option<&str>) -> String {
    match prefix {
        Some(p) if !model.starts_with(p) => format!("{}{}", p, model),
        _ => model.to_string(),
    }
}

/// Convert Anthropic /v1/messages request to OpenAI /v1/chat/completions
async fn handle_anthropic_to_openai(
    request: &str,
    config: &Arc<OpenAIRouterConfig>,
) -> Result<String> {
    let body_str = http_utils::extract_request_body(request)?;

    let body: Value = serde_json::from_str(body_str)?;
    let mut simplified = anthropic_to_openai(&body, config.requires_reasoning_content);

    // Transform model name: add prefix if configured (e.g., "@cf/" for Cloudflare)
    if let Some(model) = simplified.get_mut("model")
        && let Some(model_str) = model.as_str()
    {
        *model = Value::String(apply_model_prefix(
            model_str,
            config.model_prefix.as_deref(),
        ));
    }

    // Build target URL
    let client = router_http_client();
    let base = config.target_base_url.trim_end_matches('/');
    let url = if base.ends_with("/v1") {
        format!("{}/chat/completions", base)
    } else {
        format!("{}/v1/chat/completions", base)
    };

    let response = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", config.target_api_key))
        .header("Content-Type", "application/json")
        .header("User-Agent", "aivo-router/1.0")
        .json(&simplified)
        .send()
        .await?;

    let status_code = response.status().as_u16();
    // Check Content-Type header before consuming body to reliably detect streaming responses
    let is_streaming = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(|ct| ct.contains("text/event-stream"))
        .unwrap_or(false);
    let response_body = response.text().await?;

    // Check if streaming (header check is primary; body prefix is a fallback for providers
    // that don't set Content-Type correctly)
    if status_code == 200 && (is_streaming || response_body.starts_with("data:")) {
        // Convert OpenAI SSE stream to Anthropic SSE stream (text + tool_calls)
        let anthropic_sse = convert_openai_sse_to_anthropic(&response_body, status_code)?;
        return Ok(format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n{}",
            anthropic_sse
        ));
    }

    // Non-streaming: convert JSON response
    let anthropic_response = convert_openai_to_anthropic(&response_body, status_code)?;

    Ok(http_utils::http_json_response(
        status_code,
        &anthropic_response,
    ))
}

fn anthropic_to_openai(body: &Value, requires_reasoning_content: bool) -> Value {
    let mut messages: Vec<Value> = Vec::new();

    // System message
    if let Some(system) = body.get("system") {
        let system_text = match system {
            Value::String(s) => s.clone(),
            Value::Array(blocks) => blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        };
        if !system_text.is_empty() {
            messages.push(json!({"role": "system", "content": system_text}));
        }
    }

    // Convert messages
    if let Some(msgs) = body.get("messages").and_then(|m| m.as_array()) {
        for msg in msgs {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let content = msg.get("content");

            match content {
                Some(Value::String(text)) => {
                    messages.push(json!({"role": role, "content": text}));
                }
                Some(Value::Array(blocks)) => {
                    convert_content_blocks(blocks, role, &mut messages, requires_reasoning_content);
                }
                _ => {
                    messages.push(json!({"role": role, "content": ""}));
                }
            }
        }
    }

    let model = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("gpt-4o");
    let mut req = json!({
        "model": model,
        "messages": messages,
        "stream": body.get("stream").cloned().unwrap_or(json!(false)),
    });

    // max_tokens
    if let Some(mt) = body.get("max_tokens") {
        req["max_tokens"] = mt.clone();
    }

    // temperature
    if let Some(t) = body.get("temperature") {
        req["temperature"] = t.clone();
    }

    // top_p
    if let Some(tp) = body.get("top_p") {
        req["top_p"] = tp.clone();
    }

    // stop_sequences → stop
    if let Some(ss) = body.get("stop_sequences") {
        req["stop"] = ss.clone();
    }

    // tools
    if let Some(tools) = body.get("tools").and_then(|t| t.as_array()) {
        let openai_tools: Vec<Value> = tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "function": {
                        "name": tool.get("name").cloned().unwrap_or_default(),
                        "description": tool.get("description").cloned().unwrap_or(json!("")),
                        "parameters": tool.get("input_schema").cloned().unwrap_or(json!({})),
                    }
                })
            })
            .collect();
        if !openai_tools.is_empty() {
            req["tools"] = Value::Array(openai_tools);
        }
    }

    // tool_choice: convert Anthropic format → OpenAI format
    if let Some(tc) = body.get("tool_choice") {
        match tc.get("type").and_then(|t| t.as_str()) {
            Some("auto") => {
                req["tool_choice"] = json!("auto");
            }
            Some("any") => {
                req["tool_choice"] = json!("required");
            }
            Some("tool") => {
                if let Some(name) = tc.get("name").and_then(|n| n.as_str()) {
                    req["tool_choice"] = json!({"type": "function", "function": {"name": name}});
                }
            }
            _ => {}
        }
    }

    req
}

fn convert_content_blocks(
    blocks: &[Value],
    role: &str,
    messages: &mut Vec<Value>,
    requires_reasoning_content: bool,
) {
    let mut text_parts: Vec<String> = Vec::new();
    let mut thinking_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut tool_results: Vec<(String, String)> = Vec::new();

    for block in blocks {
        let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");

        match block_type {
            "text" => {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    text_parts.push(text.to_string());
                }
            }
            "thinking" => {
                if let Some(thinking) = block
                    .get("thinking")
                    .and_then(|t| t.as_str())
                    .or_else(|| block.get("text").and_then(|t| t.as_str()))
                {
                    thinking_parts.push(thinking.to_string());
                }
            }
            "tool_use" => {
                let id = block
                    .get("id")
                    .and_then(|i| i.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_string();
                let input = block.get("input").cloned().unwrap_or(json!({}));

                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": serde_json::to_string(&input).unwrap_or_else(|_| "{}".to_string()),
                    }
                }));
            }
            "tool_result" => {
                let tool_use_id = block
                    .get("tool_use_id")
                    .and_then(|i| i.as_str())
                    .unwrap_or("")
                    .to_string();
                let content = match block.get("content") {
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::Array(blocks)) => blocks
                        .iter()
                        .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                        .collect::<Vec<_>>()
                        .join("\n"),
                    Some(v) => v.to_string(),
                    None => String::new(),
                };
                tool_results.push((tool_use_id, content));
            }
            _ => {}
        }
    }

    if !tool_results.is_empty() {
        // Tool results → individual tool messages
        for (tool_use_id, content) in tool_results {
            messages.push(json!({
                "role": "tool",
                "tool_call_id": tool_use_id,
                "content": content,
            }));
        }
    } else if !tool_calls.is_empty() {
        // Assistant message with tool calls
        let content = if text_parts.is_empty() {
            Value::Null
        } else {
            Value::String(text_parts.join("\n"))
        };
        let mut msg = json!({"role": role, "tool_calls": tool_calls});
        if !content.is_null() {
            msg["content"] = content;
        }
        if role == "assistant" && requires_reasoning_content {
            // Some providers (e.g. Moonshot) require non-empty reasoning_content on assistant
            // tool-call turns. Only inject when the config requests it to avoid 400s elsewhere.
            let reasoning_content = if !thinking_parts.is_empty() {
                thinking_parts.join("\n")
            } else {
                let text = text_parts.join("\n");
                if text.is_empty() {
                    " ".to_string() // single-space sentinel satisfies non-empty requirement
                } else {
                    text
                }
            };
            msg["reasoning_content"] = Value::String(reasoning_content);
        } else if !thinking_parts.is_empty() {
            msg["reasoning_content"] = Value::String(thinking_parts.join("\n"));
        }
        messages.push(msg);
    } else {
        // Plain text message
        let mut msg = json!({"role": role, "content": text_parts.join("\n")});
        if !thinking_parts.is_empty() {
            msg["reasoning_content"] = Value::String(thinking_parts.join("\n"));
        }
        messages.push(msg);
    }
}

/// Convert OpenAI /v1/chat/completions response to Anthropic /v1/messages format
fn convert_openai_to_anthropic(response_body: &str, status_code: u16) -> Result<String> {
    // If error status, return as-is
    if status_code >= 400 {
        return Ok(response_body.to_string());
    }

    let openai_resp: Value = serde_json::from_str(response_body)?;
    let choices = openai_resp
        .get("choices")
        .and_then(|c| c.as_array())
        .cloned()
        .unwrap_or_default();

    // Merge all choices: providers may split text and tool_calls across choices
    let mut content: Vec<Value> = Vec::new();
    let mut final_finish_reason = "stop";

    for choice in &choices {
        let message = choice.get("message").cloned().unwrap_or(json!({}));
        let finish_reason = choice
            .get("finish_reason")
            .and_then(|r| r.as_str())
            .unwrap_or("stop");

        // tool_calls finish_reason takes priority
        if finish_reason == "tool_calls" {
            final_finish_reason = "tool_calls";
        } else if final_finish_reason != "tool_calls" {
            final_finish_reason = finish_reason;
        }

        // Text content
        if let Some(text) = message.get("content").and_then(|c| c.as_str())
            && !text.is_empty()
        {
            content.push(json!({"type": "text", "text": text}));
        }

        // Tool calls
        if let Some(tool_calls) = message.get("tool_calls").and_then(|t| t.as_array()) {
            for tc in tool_calls {
                let id = tc
                    .get("id")
                    .and_then(|i| i.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = tc
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                    .unwrap_or("");
                let args_str = tc
                    .get("function")
                    .and_then(|f| f.get("arguments"))
                    .and_then(|a| a.as_str())
                    .unwrap_or("{}");
                let input: Value = serde_json::from_str(args_str).unwrap_or(json!({}));

                content.push(json!({
                    "type": "tool_use",
                    "id": id,
                    "name": name,
                    "input": input,
                }));
            }
        }
    }

    let stop_reason = match final_finish_reason {
        "stop" => "end_turn",
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        "content_filter" => "end_turn",
        _ => "end_turn",
    };

    // If no content at all, add empty text block
    if content.is_empty() {
        content.push(json!({"type": "text", "text": ""}));
    }

    let prompt_tokens = openai_resp
        .get("usage")
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let completion_tokens = openai_resp
        .get("usage")
        .and_then(|u| u.get("completion_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let mut anthropic_resp = json!({
        "id": openai_resp.get("id").and_then(|i| i.as_str()).unwrap_or("msg_default"),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": openai_resp.get("model").and_then(|m| m.as_str()).unwrap_or("unknown"),
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens
        }
    });

    if let Some(created) = openai_resp.get("created") {
        anthropic_resp["created"] = created.clone();
    }

    Ok(anthropic_resp.to_string())
}

#[derive(Default)]
struct StreamToolBlock {
    anthropic_idx: usize,
    id: String,
    name: String,
    opened: bool,
    pending_args: String,
}

fn append_sse_event(output: &mut String, event: &str, data: Value) {
    output.push_str(&format!("event: {event}\ndata: {data}\n\n"));
}

fn ensure_message_start(
    output: &mut String,
    started: &mut bool,
    message_id: &str,
    model: &str,
    input_tokens: u64,
) {
    if *started {
        return;
    }
    append_sse_event(
        output,
        "message_start",
        json!({
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": 0
                }
            }
        }),
    );
    *started = true;
}

#[allow(clippy::too_many_arguments)]
fn emit_tool_delta(
    output: &mut String,
    block_count: &mut usize,
    tool_blocks: &mut HashMap<usize, StreamToolBlock>,
    openai_idx: usize,
    id: Option<&str>,
    name: Option<&str>,
    args_fragment: Option<&str>,
    saw_tool_use: &mut bool,
) {
    let block = tool_blocks.entry(openai_idx).or_insert_with(|| {
        let idx = *block_count;
        *block_count += 1;
        StreamToolBlock {
            anthropic_idx: idx,
            ..Default::default()
        }
    });

    if let Some(v) = id
        && !v.is_empty()
    {
        block.id = v.to_string();
    }
    if let Some(v) = name
        && !v.is_empty()
    {
        block.name = v.to_string();
    }

    if let Some(fragment) = args_fragment
        && !fragment.is_empty()
    {
        if block.opened {
            append_sse_event(
                output,
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": block.anthropic_idx,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": fragment
                    }
                }),
            );
        } else {
            block.pending_args.push_str(fragment);
        }
    }

    if !block.opened && !block.name.is_empty() {
        if block.id.is_empty() {
            block.id = format!("toolu_{}", uuid_simple());
        }
        append_sse_event(
            output,
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": block.anthropic_idx,
                "content_block": {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name
                }
            }),
        );
        block.opened = true;
        *saw_tool_use = true;

        if !block.pending_args.is_empty() {
            append_sse_event(
                output,
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": block.anthropic_idx,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": block.pending_args
                    }
                }),
            );
            block.pending_args.clear();
        }
    }
}

fn map_openai_finish_reason(reason: &str) -> &'static str {
    match reason {
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        _ => "end_turn",
    }
}

#[allow(clippy::too_many_arguments)]
fn finalize_stream_message(
    output: &mut String,
    message_started: &mut bool,
    message_id: &str,
    model: &str,
    input_tokens: u64,
    output_tokens: u64,
    text_block_idx: &mut Option<usize>,
    tool_blocks: &mut HashMap<usize, StreamToolBlock>,
    stop_reason: &str,
) {
    ensure_message_start(output, message_started, message_id, model, input_tokens);

    if let Some(idx) = text_block_idx.take() {
        append_sse_event(
            output,
            "content_block_stop",
            json!({
                "type": "content_block_stop",
                "index": idx
            }),
        );
    }

    let mut ordered_tool_idxs = tool_blocks
        .values()
        .filter(|b| b.opened)
        .map(|b| b.anthropic_idx)
        .collect::<Vec<_>>();
    ordered_tool_idxs.sort_unstable();
    for idx in ordered_tool_idxs {
        append_sse_event(
            output,
            "content_block_stop",
            json!({
                "type": "content_block_stop",
                "index": idx
            }),
        );
    }

    append_sse_event(
        output,
        "message_delta",
        json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": null
            },
            "usage": {
                "output_tokens": output_tokens
            }
        }),
    );
    append_sse_event(
        output,
        "message_stop",
        json!({
            "type": "message_stop"
        }),
    );
}

/// Convert OpenAI SSE streaming response to Anthropic SSE format.
fn convert_openai_sse_to_anthropic(response_body: &str, status_code: u16) -> Result<String> {
    if status_code >= 400 {
        return Ok(format!("data: {}\n\ndata: [DONE]\n\n", response_body));
    }

    let mut sse_output = String::new();
    let mut message_started = false;
    let mut finished = false;

    let mut block_count = 0usize;
    let mut text_block_idx: Option<usize> = None;
    let mut tool_blocks: HashMap<usize, StreamToolBlock> = HashMap::new();

    let mut message_id = "msg".to_string();
    let mut model = "claude".to_string();
    let mut input_tokens = 0u64;
    let mut output_tokens = 0u64;
    let mut saw_tool_use = false;

    for line in response_body.lines() {
        if !line.starts_with("data: ") {
            continue;
        }

        let data = line.strip_prefix("data: ").unwrap_or("");
        if data == "[DONE]" {
            if !finished {
                let fallback_stop = if saw_tool_use { "tool_use" } else { "end_turn" };
                finalize_stream_message(
                    &mut sse_output,
                    &mut message_started,
                    &message_id,
                    &model,
                    input_tokens,
                    output_tokens,
                    &mut text_block_idx,
                    &mut tool_blocks,
                    fallback_stop,
                );
                finished = true;
            }
            continue;
        }

        let chunk = match serde_json::from_str::<Value>(data) {
            Ok(v) => v,
            Err(_) => continue,
        };

        if let Some(v) = chunk.get("id").and_then(|v| v.as_str())
            && !v.is_empty()
        {
            message_id = v.to_string();
        }
        if let Some(v) = chunk.get("model").and_then(|v| v.as_str())
            && !v.is_empty()
        {
            model = v.to_string();
        }
        if let Some(usage) = chunk.get("usage") {
            if let Some(v) = usage.get("prompt_tokens").and_then(|v| v.as_u64()) {
                input_tokens = v;
            }
            if let Some(v) = usage.get("completion_tokens").and_then(|v| v.as_u64()) {
                output_tokens = v;
            }
        }

        let choices = chunk
            .get("choices")
            .and_then(|c| c.as_array())
            .cloned()
            .unwrap_or_default();
        for choice in choices {
            let delta = choice.get("delta").cloned().unwrap_or(json!({}));

            // Text delta
            if let Some(text) = delta.get("content").and_then(|v| v.as_str())
                && !text.is_empty()
            {
                ensure_message_start(
                    &mut sse_output,
                    &mut message_started,
                    &message_id,
                    &model,
                    input_tokens,
                );
                if text_block_idx.is_none() {
                    let idx = block_count;
                    block_count += 1;
                    text_block_idx = Some(idx);
                    append_sse_event(
                        &mut sse_output,
                        "content_block_start",
                        json!({
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {
                                "type": "text",
                                "text": ""
                            }
                        }),
                    );
                }
                append_sse_event(
                    &mut sse_output,
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta",
                        "index": text_block_idx.unwrap_or(0),
                        "delta": {
                            "type": "text_delta",
                            "text": text
                        }
                    }),
                );
            }

            // OpenAI legacy function_call delta
            if let Some(function_call) = delta.get("function_call") {
                ensure_message_start(
                    &mut sse_output,
                    &mut message_started,
                    &message_id,
                    &model,
                    input_tokens,
                );
                emit_tool_delta(
                    &mut sse_output,
                    &mut block_count,
                    &mut tool_blocks,
                    0,
                    function_call.get("id").and_then(|v| v.as_str()),
                    function_call.get("name").and_then(|v| v.as_str()),
                    function_call.get("arguments").and_then(|v| v.as_str()),
                    &mut saw_tool_use,
                );
            }

            // OpenAI modern tool_calls delta (possibly split by index across chunks)
            if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                ensure_message_start(
                    &mut sse_output,
                    &mut message_started,
                    &message_id,
                    &model,
                    input_tokens,
                );
                for tc in tool_calls {
                    let openai_idx = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    emit_tool_delta(
                        &mut sse_output,
                        &mut block_count,
                        &mut tool_blocks,
                        openai_idx,
                        tc.get("id").and_then(|v| v.as_str()),
                        tc.get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|v| v.as_str()),
                        tc.get("function")
                            .and_then(|f| f.get("arguments"))
                            .and_then(|v| v.as_str()),
                        &mut saw_tool_use,
                    );
                }
            }

            if !finished
                && let Some(finish_reason) = choice.get("finish_reason").and_then(|v| v.as_str())
                && !finish_reason.is_empty()
            {
                finalize_stream_message(
                    &mut sse_output,
                    &mut message_started,
                    &message_id,
                    &model,
                    input_tokens,
                    output_tokens,
                    &mut text_block_idx,
                    &mut tool_blocks,
                    map_openai_finish_reason(finish_reason),
                );
                finished = true;
            }
        }
    }

    Ok(sse_output)
}

/// Generate a collision-resistant unique ID using a monotonic counter + timestamp.
fn uuid_simple() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!(
        "{:x}{:x}{:x}",
        duration.as_secs(),
        duration.subsec_nanos(),
        count
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_openai_to_anthropic() {
        let openai_resp = r#"{
            "id": "chatcmpl-123",
            "created": 1700000000,
            "model": "gpt-4",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                },
                "finish_reason": "stop",
                "index": 0
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }"#;

        let result = convert_openai_to_anthropic(openai_resp, 200).unwrap();
        let parsed: Value = serde_json::from_str(&result).unwrap();

        assert_eq!(parsed["type"], "message");
        assert_eq!(parsed["role"], "assistant");
        assert!(parsed["content"].is_array());
        assert_eq!(parsed["usage"]["input_tokens"], 10);
        assert_eq!(parsed["usage"]["output_tokens"], 5);
    }

    #[test]
    fn test_anthropic_to_openai_preserves_fields_and_tools() {
        let body = json!({
            "model": "gpt-4o-mini",
            "system": [{"type": "text", "text": "You are helpful."}],
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "hello"}]
            }],
            "max_tokens": 128,
            "temperature": 0.2,
            "top_p": 0.9,
            "stop_sequences": ["END"],
            "tools": [{
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}
            }],
            "tool_choice": {"type": "tool", "name": "read_file"},
            "stream": true
        });

        let req = anthropic_to_openai(&body, false);
        let messages = req["messages"].as_array().unwrap();

        assert_eq!(req["model"], "gpt-4o-mini");
        assert_eq!(req["stream"], true);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are helpful.");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "hello");
        assert_eq!(req["max_tokens"], 128);
        assert_eq!(req["temperature"], 0.2);
        assert_eq!(req["top_p"], 0.9);
        assert_eq!(req["stop"][0], "END");
        assert_eq!(req["tools"][0]["type"], "function");
        assert_eq!(req["tools"][0]["function"]["name"], "read_file");
        assert_eq!(
            req["tool_choice"],
            json!({"type": "function", "function": {"name": "read_file"}})
        );
    }

    #[test]
    fn test_convert_openai_to_anthropic_merges_text_and_tool_calls() {
        let openai_resp = r#"{
            "id": "chatcmpl-456",
            "created": 1700000001,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Let me check."},
                    "finish_reason": "stop"
                },
                {
                    "index": 1,
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
                        }]
                    },
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 7,
                "total_tokens": 19
            }
        }"#;

        let result = convert_openai_to_anthropic(openai_resp, 200).unwrap();
        let parsed: Value = serde_json::from_str(&result).unwrap();
        let content = parsed["content"].as_array().unwrap();

        assert_eq!(parsed["stop_reason"], "tool_use");
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "Let me check.");
        assert_eq!(content[1]["type"], "tool_use");
        assert_eq!(content[1]["id"], "call_1");
        assert_eq!(content[1]["name"], "get_weather");
        assert_eq!(content[1]["input"]["city"], "Paris");
    }

    #[test]
    fn test_anthropic_to_openai_maps_thinking_to_reasoning_content_for_tool_calls() {
        let body = json!({
            "model": "kimi-k2.5",
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Need to inspect files first."},
                    {"type": "tool_use", "id": "toolu_1", "name": "list_files", "input": {"path": "."}}
                ]
            }]
        });

        let req = anthropic_to_openai(&body, true);
        let messages = req["messages"].as_array().unwrap();

        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(
            messages[0]["reasoning_content"],
            "Need to inspect files first."
        );
        assert_eq!(
            messages[0]["tool_calls"][0]["function"]["name"],
            "list_files"
        );
    }

    #[test]
    fn test_anthropic_to_openai_sets_reasoning_content_for_assistant_tool_calls_without_thinking() {
        let body = json!({
            "model": "kimi-k2.5",
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "list_files", "input": {"path": "."}}
                ]
            }]
        });

        let req = anthropic_to_openai(&body, true);
        let messages = req["messages"].as_array().unwrap();

        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[0]["reasoning_content"], " ");
        assert_eq!(
            messages[0]["tool_calls"][0]["function"]["name"],
            "list_files"
        );
    }

    #[test]
    fn test_convert_openai_sse_to_anthropic_text() {
        let sse = "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"hello \"},\"finish_reason\":null}]}\n\
data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"world\"},\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":4}}\n\
data: [DONE]\n";
        let result = convert_openai_sse_to_anthropic(sse, 200).unwrap();
        assert!(result.contains("event: message_start"));
        assert!(result.contains("\"type\":\"text_delta\""));
        assert!(result.contains("\"text\":\"hello \""));
        assert!(result.contains("\"text\":\"world\""));
        assert!(result.contains("\"stop_reason\":\"end_turn\""));
        assert!(result.contains("event: message_stop"));
    }

    #[test]
    fn test_convert_openai_sse_to_anthropic_split_tool_calls() {
        let sse = "data: {\"id\":\"chatcmpl_2\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"list_files\"}}]},\"finish_reason\":null}]}\n\
data: {\"id\":\"chatcmpl_2\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"path\\\":\\\".\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\
data: [DONE]\n";
        let result = convert_openai_sse_to_anthropic(sse, 200).unwrap();
        assert!(result.contains("\"type\":\"tool_use\""));
        assert!(result.contains("\"id\":\"call_1\""));
        assert!(result.contains("\"name\":\"list_files\""));
        assert!(result.contains("\"type\":\"input_json_delta\""));
        assert!(result.contains("\"partial_json\":\"{\\\"path\\\":\\\".\\\"}\""));
        assert!(result.contains("\"stop_reason\":\"tool_use\""));
    }

    #[test]
    fn test_model_prefix() {
        // Test the production helper directly to catch regressions in the real code path
        assert_eq!(
            apply_model_prefix("glm-4.7-flash", Some("@cf/")),
            "@cf/glm-4.7-flash"
        );
        // Prefix already present — must not double-add
        assert_eq!(
            apply_model_prefix("@cf/llama-3.1-8b", Some("@cf/")),
            "@cf/llama-3.1-8b"
        );
        // No prefix configured
        assert_eq!(apply_model_prefix("llama-3.1-8b", None), "llama-3.1-8b");
    }
}
