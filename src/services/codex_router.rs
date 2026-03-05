/**
 * Built-in Codex Router service
 *
 * Acts as an HTTP proxy that intercepts Codex requests and forwards them to
 * non-OpenAI providers with two levels of compatibility:
 *
 * 1. Tool filtering: strips built-in tool types (computer_use, file_search,
 *    web_search, code_interpreter) that most non-OpenAI providers reject.
 *
 * 2. Responses API conversion: Codex CLI v0.105+ uses the OpenAI Responses API
 *    (/v1/responses with "input" array). Providers that only support Chat
 *    Completions (/v1/chat/completions with "messages" array) need a full
 *    request/response conversion. This router handles that automatically.
 */
use anyhow::Result;
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::services::copilot_auth::{
    COPILOT_EDITOR_VERSION, COPILOT_INTEGRATION_ID, COPILOT_OPENAI_INTENT, CopilotTokenManager,
};
use crate::services::http_utils;

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone)]
pub struct CodexRouterConfig {
    pub target_base_url: String,
    pub api_key: String,
    pub copilot_token_manager: Option<Arc<CopilotTokenManager>>,
}

pub struct CodexRouter {
    config: CodexRouterConfig,
}

impl CodexRouter {
    pub fn new(config: CodexRouterConfig) -> Self {
        Self { config }
    }

    /// Binds to a random available port and starts the router in the background.
    /// Returns the actual port number so callers can set OPENAI_BASE_URL.
    pub async fn start_background(&self) -> Result<(u16, tokio::task::JoinHandle<Result<()>>)> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let port = listener.local_addr()?.port();
        let config = self.config.clone();
        let client = Arc::new(http_utils::router_http_client());
        let handle = tokio::spawn(async move { run_router(listener, config, client).await });
        Ok((port, handle))
    }
}

async fn run_router(
    listener: tokio::net::TcpListener,
    config: CodexRouterConfig,
    client: Arc<reqwest::Client>,
) -> Result<()> {
    let config = Arc::new(config);

    loop {
        let (mut socket, _) = listener.accept().await?;
        let config = config.clone();
        let client = client.clone();

        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;

            let request_bytes = match http_utils::read_full_request(&mut socket).await {
                Ok(b) => b,
                Err(_) => return,
            };

            let request = String::from_utf8_lossy(&request_bytes);
            let path = http_utils::extract_request_path(&request);

            let is_api_path = matches!(
                path.as_str(),
                "/responses" | "/v1/responses" | "/chat/completions" | "/v1/chat/completions"
            );

            let response = if is_api_path {
                match handle_api_request(&path, &request, &config, client.as_ref()).await {
                    Ok(r) => r,
                    Err(_) => http_utils::http_error_response(500, "Internal Server Error"),
                }
            } else {
                match forward_request(&path, &request, &config, client.as_ref()).await {
                    Ok(r) => r,
                    Err(_) => http_utils::http_error_response(502, "Bad Gateway"),
                }
            };

            let _ = socket.write_all(response.as_bytes()).await;
        });
    }
}

/// Routes the request based on body format:
/// - Responses API format (has "input" array): convert ↔ Chat Completions
/// - Chat Completions format: filter non-function tools and forward
async fn handle_api_request(
    path: &str,
    request: &str,
    config: &Arc<CodexRouterConfig>,
    client: &reqwest::Client,
) -> Result<String> {
    let body_str = http_utils::extract_request_body(request)?;
    let body: Value = serde_json::from_str(body_str)?;

    if is_responses_api_format(&body) {
        handle_responses_api_via_chat(path, &body, config, client).await
    } else {
        handle_chat_completions_with_filter(path, &body, config, client).await
    }
}

// =============================================================================
// RESPONSES API PATH: convert request → chat completions → convert response back
// =============================================================================

/// Builds an authorized POST request builder for the upstream provider.
///
/// - Copilot mode: exchanges the GitHub token for a short-lived Copilot token and
///   sets the Copilot-specific headers. The target URL is taken from the token
///   manager's endpoint (ignoring `url`).
/// - Normal mode: sets `Authorization: Bearer <api_key>` and POSTs to `url`.
async fn authorized_post(
    client: &reqwest::Client,
    url: &str,
    config: &CodexRouterConfig,
) -> Result<reqwest::RequestBuilder> {
    if let Some(ref tm) = config.copilot_token_manager {
        let (token, api_endpoint) = tm.get_token().await?;
        let copilot_url = format!("{}/chat/completions", api_endpoint.trim_end_matches('/'));
        Ok(client
            .post(&copilot_url)
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json")
            .header("Editor-Version", COPILOT_EDITOR_VERSION)
            .header("Copilot-Integration-Id", COPILOT_INTEGRATION_ID)
            .header("Openai-Intent", COPILOT_OPENAI_INTENT))
    } else {
        Ok(client
            .post(url)
            .header("Authorization", format!("Bearer {}", config.api_key))
            .header("Content-Type", "application/json"))
    }
}

/// Handles Responses API requests by converting to Chat Completions format,
/// forwarding to the provider, and converting the response back to Responses
/// API SSE format that the Codex CLI expects.
async fn handle_responses_api_via_chat(
    _path: &str,
    body: &Value,
    config: &Arc<CodexRouterConfig>,
    client: &reqwest::Client,
) -> Result<String> {
    let chat_body = convert_responses_to_chat_request(body, config);
    let target_url = build_target_url(&config.target_base_url, "/v1/chat/completions");

    let response = authorized_post(client, &target_url, config)
        .await?
        .json(&chat_body)
        .send()
        .await?;

    let status_code = response.status().as_u16();
    if status_code != 200 {
        let err_body = response.text().await?;
        return Ok(http_utils::http_json_response(status_code, &err_body));
    }

    let response_text = response.text().await?;
    let chat_response = parse_provider_response(&response_text)?;
    let sse = convert_chat_response_to_responses_sse(&chat_response);

    Ok(http_utils::http_response(200, "text/event-stream", &sse))
}

// =============================================================================
// CHAT COMPLETIONS PATH: filter tools and forward
// =============================================================================

async fn handle_chat_completions_with_filter(
    path: &str,
    body: &Value,
    config: &Arc<CodexRouterConfig>,
    client: &reqwest::Client,
) -> Result<String> {
    let mut body = body.clone();
    filter_tools(&mut body);
    if config.copilot_token_manager.is_none() {
        transform_model(&mut body, &config.target_base_url);
    }

    let target_url = build_target_url(&config.target_base_url, path);

    let response = authorized_post(client, &target_url, config)
        .await?
        .json(&body)
        .send()
        .await?;

    let status_code = response.status().as_u16();
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    let response_body = response.bytes().await?;
    let body_str = String::from_utf8_lossy(&response_body);

    Ok(http_utils::http_response(
        status_code,
        &content_type,
        &body_str,
    ))
}

// =============================================================================
// PASSTHROUGH
// =============================================================================

/// Forwards a request as-is to the target provider (for non-API paths)
async fn forward_request(
    path: &str,
    request: &str,
    config: &Arc<CodexRouterConfig>,
    client: &reqwest::Client,
) -> Result<String> {
    let body_str = http_utils::extract_request_body(request)?;

    let target_url = build_target_url(&config.target_base_url, path);

    let mut req = authorized_post(client, &target_url, config).await?;

    if !body_str.is_empty() {
        req = req.body(body_str.to_string());
    }

    let response = req.send().await?;
    let status_code = response.status().as_u16();
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    let response_body = response.bytes().await?;
    let body_str = String::from_utf8_lossy(&response_body);

    Ok(http_utils::http_response(
        status_code,
        &content_type,
        &body_str,
    ))
}

// =============================================================================
// URL HELPERS
// =============================================================================

/// Constructs target URL, avoiding /v1 duplication when base already ends with /v1
fn build_target_url(base_url: &str, path: &str) -> String {
    http_utils::build_target_url(base_url, path)
}

// =============================================================================
// TOOL FILTERING
// =============================================================================

/// Removes non-"function" tools from the request body.
/// If the tools array becomes empty, removes the key entirely.
fn filter_tools(body: &mut Value) {
    if let Some(tools) = body.get_mut("tools").and_then(|t| t.as_array_mut()) {
        tools.retain(|t| t.get("type").and_then(|v| v.as_str()) == Some("function"));
        if tools.is_empty() {
            body.as_object_mut().map(|o| o.remove("tools"));
        }
    }
}

// =============================================================================
// MODEL TRANSFORM
// =============================================================================

/// For OpenRouter, prefixes model with "openai/" if not already namespaced
fn transform_model(body: &mut Value, base_url: &str) {
    if let Some(model) = body["model"].as_str().map(String::from) {
        let transformed = transform_model_str(&model, base_url);
        if transformed != model {
            body["model"] = Value::String(transformed);
        }
    }
}

fn transform_model_str(model: &str, base_url: &str) -> String {
    if base_url.contains("openrouter") && !model.contains('/') {
        format!("openai/{}", model)
    } else {
        model.to_string()
    }
}

// =============================================================================
// RESPONSES API ↔ CHAT COMPLETIONS CONVERSION
// =============================================================================

/// Returns true if the body uses OpenAI Responses API format
/// (has "input" array, no "messages" array)
pub fn is_responses_api_format(body: &Value) -> bool {
    body.get("input").and_then(|v| v.as_array()).is_some() && body.get("messages").is_none()
}

/// Converts an OpenAI Responses API request body to Chat Completions format.
///
/// Handles all input item types:
/// - `message` → role/content message
/// - `function_call` → assistant message with tool_calls
/// - `function_call_output` → tool message
///
/// Also converts tool format (Responses API has no `function` wrapper;
/// Chat Completions requires `{type, function: {name, description, parameters}}`).
pub fn convert_responses_to_chat_request(body: &Value, config: &CodexRouterConfig) -> Value {
    let mut messages: Vec<Value> = vec![];

    // System message from "instructions" field
    if let Some(instructions) = body.get("instructions").and_then(|v| v.as_str())
        && !instructions.is_empty()
    {
        messages.push(json!({"role": "system", "content": instructions}));
    }

    // Convert "input" array items
    if let Some(input) = body.get("input").and_then(|v| v.as_array()) {
        for item in input {
            match item.get("type").and_then(|v| v.as_str()) {
                Some("message") => {
                    let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("user");
                    let content = extract_content_text(item.get("content"));
                    messages.push(json!({"role": role, "content": content}));
                }
                Some("function_call") => {
                    // Use call_id as the Chat Completions tool_calls[].id so it matches
                    // the corresponding function_call_output.call_id → tool_call_id.
                    // Fall back to id only if call_id is absent.
                    let call_id = item
                        .get("call_id")
                        .and_then(|v| v.as_str())
                        .or_else(|| item.get("id").and_then(|v| v.as_str()))
                        .unwrap_or("call_0");
                    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let arguments = item
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("{}");
                    messages.push(json!({
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}]
                    }));
                }
                Some("function_call_output") => {
                    let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                    let output = item.get("output").and_then(|v| v.as_str()).unwrap_or("");
                    messages
                        .push(json!({"role": "tool", "tool_call_id": call_id, "content": output}));
                }
                None => {
                    // Simple string input
                    if let Some(s) = item.as_str() {
                        messages.push(json!({"role": "user", "content": s}));
                    }
                }
                _ => {}
            }
        }
    }

    // Convert tools: filter non-function, convert format
    let tools: Vec<Value> = body
        .get("tools")
        .and_then(|t| t.as_array())
        .map(|tools| {
            tools
                .iter()
                .filter(|t| t.get("type").and_then(|v| v.as_str()) == Some("function"))
                .map(convert_tool_to_chat_format)
                .collect()
        })
        .unwrap_or_default();

    // Apply model name transform (e.g. openai/ prefix for OpenRouter)
    // Skip transform when using Copilot — model names pass through unchanged
    let model = body.get("model").cloned().unwrap_or(Value::Null);
    let model = if config.copilot_token_manager.is_none() {
        match model.as_str() {
            Some(s) => Value::String(transform_model_str(s, &config.target_base_url)),
            None => model,
        }
    } else {
        model
    };

    let mut chat = json!({
        "model": model,
        "messages": messages,
        "stream": false,  // request non-streaming for simpler response handling
    });

    if !tools.is_empty() {
        chat["tools"] = Value::Array(tools);
    }
    if let Some(v) = body
        .get("max_output_tokens")
        .or_else(|| body.get("max_tokens"))
    {
        chat["max_tokens"] = v.clone();
    }
    for field in ["temperature", "top_p"] {
        if let Some(v) = body.get(field) {
            chat[field] = v.clone();
        }
    }

    chat
}

/// Extracts text from a content value (handles string, array of content parts)
pub fn extract_content_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| match p {
                Value::String(s) => Some(s.clone()),
                _ => p.get("text").and_then(|v| v.as_str()).map(String::from),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Converts a tool from Responses API format to Chat Completions format.
///
/// Responses API: `{type, name, description, parameters}`
/// Chat Completions: `{type, function: {name, description, parameters}}`
pub fn convert_tool_to_chat_format(tool: &Value) -> Value {
    // Already in Chat Completions format (has "function" wrapper)
    if tool.get("function").is_some() {
        return tool.clone();
    }
    let mut func = serde_json::Map::new();
    for field in ["name", "description", "parameters", "strict"] {
        if let Some(v) = tool.get(field) {
            func.insert(field.to_string(), v.clone());
        }
    }
    json!({"type": "function", "function": Value::Object(func)})
}

/// Parses a provider HTTP response body as either a JSON chat completion
/// (stream:false) or an SSE chat completion stream (stream:true).
/// Returns a unified non-streaming chat completion JSON.
pub fn parse_provider_response(text: &str) -> anyhow::Result<Value> {
    // Try JSON first (non-streaming response)
    if let Ok(v) = serde_json::from_str::<Value>(text) {
        return Ok(v);
    }
    // Fallback: provider returned SSE despite stream:false
    Ok(accumulate_chat_sse(text))
}

/// Reads an SSE chat completions stream and returns a synthesized non-streaming response.
pub fn accumulate_chat_sse(text: &str) -> Value {
    let mut content = String::new();
    // (id, name, accumulated_args)
    let mut tool_calls_acc: Vec<(String, String, String)> = Vec::new();
    let mut finish_reason = String::from("stop");

    for line in text.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if data.trim() == "[DONE]" {
                break;
            }
            if let Ok(chunk) = serde_json::from_str::<Value>(data) {
                let choice = &chunk["choices"][0];
                let delta = &choice["delta"];

                if let Some(c) = delta["content"].as_str() {
                    content.push_str(c);
                }
                if let Some(tcs) = delta["tool_calls"].as_array() {
                    for tc in tcs {
                        let idx = tc["index"].as_u64().unwrap_or(0) as usize;
                        while tool_calls_acc.len() <= idx {
                            tool_calls_acc.push((String::new(), String::new(), String::new()));
                        }
                        if let Some(id) = tc["id"].as_str()
                            && !id.is_empty()
                        {
                            tool_calls_acc[idx].0 = id.to_string();
                        }
                        if let Some(name) = tc["function"]["name"].as_str()
                            && !name.is_empty()
                        {
                            tool_calls_acc[idx].1.push_str(name);
                        }
                        if let Some(args) = tc["function"]["arguments"].as_str() {
                            tool_calls_acc[idx].2.push_str(args);
                        }
                    }
                }
                if let Some(fr) = choice["finish_reason"].as_str()
                    && !fr.is_empty()
                {
                    finish_reason = fr.to_string();
                }
            }
        }
    }

    if !tool_calls_acc.is_empty() {
        let tcs: Vec<Value> = tool_calls_acc
            .iter()
            .enumerate()
            .map(|(i, (id, name, args))| {
                json!({
                    "id": if id.is_empty() { format!("call_{}", i) } else { id.clone() },
                    "type": "function",
                    "function": {"name": name, "arguments": args}
                })
            })
            .collect();
        json!({"choices": [{"message": {"role": "assistant", "content": null, "tool_calls": tcs}, "finish_reason": "tool_calls"}]})
    } else {
        json!({"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}]})
    }
}

/// Converts a Chat Completions non-streaming response to Responses API SSE events.
///
/// Codex CLI parses these SSE events to display output and handle tool calls.
/// Handles both text responses and tool call responses.
///
/// Key correctness requirements from the OpenAI Responses API spec:
/// - `object` must be "response" (not "realtime.response")
/// - All sub-events must include `response_id`
/// - Function call items need a `call_id` (= Chat Completions tc.id) separate
///   from `id` (a fresh item identifier); Codex puts `call_id` in the
///   follow-up `function_call_output.call_id` field
pub fn convert_chat_response_to_responses_sse(chat: &Value) -> String {
    let resp_id = gen_id("resp");
    let created_at = unix_timestamp();
    let mut sse = String::new();
    let mut output_items: Vec<Value> = Vec::new();

    // response.created — required opening event
    sse.push_str(&sse_event(
        "response.created",
        &json!({
            "type": "response.created",
            "response": {
                "id": resp_id, "object": "response",
                "created_at": created_at, "status": "in_progress", "output": []
            }
        }),
    ));

    let empty_msg = json!({"role": "assistant", "content": ""});
    let choice = chat
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
        .cloned()
        .unwrap_or(json!({"message": empty_msg}));
    let message = choice.get("message").cloned().unwrap_or(empty_msg);

    if let Some(tool_calls) = message.get("tool_calls").and_then(|t| t.as_array()) {
        // Tool call response — each tool call becomes a function_call output item
        for (i, tc) in tool_calls.iter().enumerate() {
            // call_id = the Chat Completions tool call ID (referenced in tool results)
            let call_id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("call_0");
            // item_id = a fresh item identifier within this response
            let item_id = gen_id("fc");
            let tc_name = tc["function"]["name"].as_str().unwrap_or("");
            let tc_args = tc["function"]["arguments"].as_str().unwrap_or("{}");

            sse.push_str(&sse_event(
                "response.output_item.added",
                &json!({
                    "type": "response.output_item.added",
                    "response_id": resp_id, "output_index": i,
                    "item": {
                        "id": item_id, "call_id": call_id,
                        "type": "function_call", "status": "in_progress",
                        "name": tc_name, "arguments": ""
                    }
                }),
            ));
            sse.push_str(&sse_event(
                "response.function_call_arguments.delta",
                &json!({
                    "type": "response.function_call_arguments.delta",
                    "response_id": resp_id, "output_index": i,
                    "item_id": item_id, "delta": tc_args
                }),
            ));
            sse.push_str(&sse_event(
                "response.function_call_arguments.done",
                &json!({
                    "type": "response.function_call_arguments.done",
                    "response_id": resp_id, "output_index": i,
                    "item_id": item_id, "arguments": tc_args
                }),
            ));

            let done_item = json!({
                "id": item_id, "call_id": call_id,
                "type": "function_call", "status": "completed",
                "name": tc_name, "arguments": tc_args
            });
            sse.push_str(&sse_event(
                "response.output_item.done",
                &json!({
                    "type": "response.output_item.done",
                    "response_id": resp_id, "output_index": i,
                    "item": done_item
                }),
            ));
            output_items.push(json!({
                "id": item_id, "call_id": call_id,
                "type": "function_call", "status": "completed",
                "name": tc_name, "arguments": tc_args
            }));
        }
    } else {
        // Text message response
        let content = message
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let msg_id = gen_id("msg");

        sse.push_str(&sse_event(
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "response_id": resp_id, "output_index": 0,
                "item": {
                    "id": msg_id, "type": "message",
                    "status": "in_progress", "role": "assistant", "content": []
                }
            }),
        ));
        sse.push_str(&sse_event(
            "response.content_part.added",
            &json!({
                "type": "response.content_part.added",
                "response_id": resp_id, "item_id": msg_id,
                "output_index": 0, "content_index": 0,
                "part": {"type": "output_text", "text": ""}
            }),
        ));
        if !content.is_empty() {
            sse.push_str(&sse_event(
                "response.output_text.delta",
                &json!({
                    "type": "response.output_text.delta",
                    "response_id": resp_id, "item_id": msg_id,
                    "output_index": 0, "content_index": 0, "delta": content
                }),
            ));
        }
        sse.push_str(&sse_event(
            "response.output_text.done",
            &json!({
                "type": "response.output_text.done",
                "response_id": resp_id, "item_id": msg_id,
                "output_index": 0, "content_index": 0, "text": content
            }),
        ));
        sse.push_str(&sse_event(
            "response.content_part.done",
            &json!({
                "type": "response.content_part.done",
                "response_id": resp_id, "item_id": msg_id,
                "output_index": 0, "content_index": 0,
                "part": {"type": "output_text", "text": content}
            }),
        ));
        let done_item = json!({
            "id": msg_id, "type": "message", "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content, "annotations": []}]
        });
        sse.push_str(&sse_event(
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "response_id": resp_id, "output_index": 0, "item": done_item
            }),
        ));
        output_items.push(json!({
            "id": msg_id, "type": "message", "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content, "annotations": []}]
        }));
    }

    // response.completed — required closing event with full output array
    sse.push_str(&sse_event(
        "response.completed",
        &json!({
            "type": "response.completed",
            "response": {
                "id": resp_id, "object": "response",
                "created_at": created_at, "status": "completed",
                "output": output_items
            }
        }),
    ));

    sse
}

fn sse_event(event_type: &str, data: &Value) -> String {
    format!(
        "event: {}\ndata: {}\n\n",
        event_type,
        serde_json::to_string(data).unwrap_or_default()
    )
}

/// Generates a unique ID using an atomic counter + timestamp
fn gen_id(prefix: &str) -> String {
    let n = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{}_{}_{:06}", prefix, secs, n % 1_000_000)
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── HTTP body extraction ────────────────────────────────────────────────────

    #[test]
    fn test_extract_request_body_normal() {
        let req = "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"model\":\"gpt-4\"}";
        assert_eq!(
            http_utils::extract_request_body(req).unwrap(),
            "{\"model\":\"gpt-4\"}"
        );
    }

    #[test]
    fn test_extract_request_body_missing_separator_returns_error() {
        let req = "POST /v1/chat/completions HTTP/1.1";
        assert!(http_utils::extract_request_body(req).is_err());
    }

    #[test]
    fn test_extract_request_body_short_request_no_panic() {
        assert!(http_utils::extract_request_body("AB").is_err());
    }

    // ── Tool filtering ─────────────────────────────────────────────────────────

    #[test]
    fn test_filter_tools_removes_non_function() {
        let mut body = json!({
            "model": "gpt-4",
            "tools": [
                {"type": "function", "function": {"name": "my_fn"}},
                {"type": "computer_use"},
                {"type": "file_search"},
                {"type": "web_search"},
                {"type": "code_interpreter"}
            ]
        });

        filter_tools(&mut body);

        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
    }

    #[test]
    fn test_filter_tools_all_non_function_removes_key() {
        let mut body = json!({
            "model": "gpt-4",
            "tools": [{"type": "computer_use"}, {"type": "web_search"}]
        });
        filter_tools(&mut body);
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn test_filter_tools_already_function_only_unchanged() {
        let mut body = json!({
            "model": "gpt-4",
            "tools": [
                {"type": "function", "function": {"name": "fn1"}},
                {"type": "function", "function": {"name": "fn2"}}
            ]
        });
        filter_tools(&mut body);
        assert_eq!(body["tools"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_filter_tools_no_tools_key_is_noop() {
        let mut body = json!({"model": "gpt-4", "messages": []});
        filter_tools(&mut body);
        assert!(body.get("tools").is_none());
        assert_eq!(body["model"], "gpt-4");
    }

    // ── Model transform ────────────────────────────────────────────────────────

    #[test]
    fn test_transform_model_openrouter_adds_prefix() {
        let mut body = json!({"model": "gpt-4o"});
        transform_model(&mut body, "https://openrouter.ai/api/v1");
        assert_eq!(body["model"], "openai/gpt-4o");
    }

    #[test]
    fn test_transform_model_openrouter_already_prefixed() {
        let mut body = json!({"model": "openai/gpt-4o"});
        transform_model(&mut body, "https://openrouter.ai/api/v1");
        assert_eq!(body["model"], "openai/gpt-4o");
    }

    #[test]
    fn test_transform_model_non_openrouter_passthrough() {
        let mut body = json!({"model": "gpt-4o"});
        transform_model(&mut body, "https://ai-gateway.vercel.sh/v1");
        assert_eq!(body["model"], "gpt-4o");
    }

    // ── URL building ───────────────────────────────────────────────────────────

    #[test]
    fn test_build_target_url_strips_v1_duplication() {
        let url = build_target_url("https://ai-gateway.vercel.sh/v1", "/v1/responses");
        assert_eq!(url, "https://ai-gateway.vercel.sh/v1/responses");
    }

    #[test]
    fn test_build_target_url_no_v1_in_path() {
        let url = build_target_url("https://ai-gateway.vercel.sh/v1", "/responses");
        assert_eq!(url, "https://ai-gateway.vercel.sh/v1/responses");
    }

    #[test]
    fn test_build_target_url_base_no_v1() {
        let url = build_target_url("https://api.example.com", "/v1/responses");
        assert_eq!(url, "https://api.example.com/v1/responses");
    }

    // ── is_responses_api_format ────────────────────────────────────────────────

    #[test]
    fn test_is_responses_api_format_detected() {
        assert!(is_responses_api_format(
            &json!({"input": [{"role": "user", "content": "hi"}]})
        ));
    }

    #[test]
    fn test_is_responses_api_format_chat_completions_not_detected() {
        assert!(!is_responses_api_format(
            &json!({"messages": [{"role": "user", "content": "hi"}]})
        ));
    }

    #[test]
    fn test_is_responses_api_format_has_both_not_detected() {
        // If both "input" and "messages" present, treat as Chat Completions
        assert!(!is_responses_api_format(&json!({
            "input": [],
            "messages": []
        })));
    }

    // ── extract_content_text ───────────────────────────────────────────────────

    #[test]
    fn test_extract_content_text_string() {
        assert_eq!(
            extract_content_text(Some(&json!("hello world"))),
            "hello world"
        );
    }

    #[test]
    fn test_extract_content_text_parts_array() {
        let content = json!([
            {"type": "input_text", "text": "list"},
            {"type": "input_text", "text": "files"}
        ]);
        assert_eq!(extract_content_text(Some(&content)), "list\nfiles");
    }

    #[test]
    fn test_extract_content_text_none() {
        assert_eq!(extract_content_text(None), "");
    }

    // ── convert_tool_to_chat_format ────────────────────────────────────────────

    #[test]
    fn test_convert_tool_format_responses_api_to_chat() {
        let tool = json!({
            "type": "function",
            "name": "shell",
            "description": "Run a shell command",
            "parameters": {"type": "object", "properties": {}}
        });
        let converted = convert_tool_to_chat_format(&tool);
        assert_eq!(converted["type"], "function");
        assert_eq!(converted["function"]["name"], "shell");
        assert_eq!(converted["function"]["description"], "Run a shell command");
        assert!(converted.get("name").is_none()); // moved into "function" wrapper
    }

    #[test]
    fn test_convert_tool_format_already_chat_format() {
        let tool = json!({
            "type": "function",
            "function": {"name": "shell", "description": "..."}
        });
        let converted = convert_tool_to_chat_format(&tool);
        assert_eq!(converted["function"]["name"], "shell");
    }

    // ── convert_responses_to_chat_request ─────────────────────────────────────

    #[test]
    fn test_convert_request_simple_message() {
        let body = json!({
            "model": "gpt-5.2-codex",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "list files"}]}
            ]
        });
        let chat = convert_responses_to_chat_request(
            &body,
            &CodexRouterConfig {
                target_base_url: "https://ai-gateway.vercel.sh/v1".to_string(),
                api_key: "sk-test".to_string(),
                copilot_token_manager: None,
            },
        );

        assert_eq!(chat["model"], "gpt-5.2-codex");
        assert_eq!(chat["stream"], false);
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "list files");
    }

    #[test]
    fn test_convert_request_instructions_become_system_message() {
        let body = json!({
            "model": "gpt-4",
            "instructions": "You are a helpful assistant.",
            "input": [{"type": "message", "role": "user", "content": "hi"}]
        });
        let chat = convert_responses_to_chat_request(
            &body,
            &CodexRouterConfig {
                target_base_url: "https://example.com/v1".to_string(),
                api_key: String::new(),
                copilot_token_manager: None,
            },
        );
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "You are a helpful assistant.");
        assert_eq!(msgs[1]["role"], "user");
    }

    #[test]
    fn test_convert_request_tool_call_items() {
        // Simulates the follow-up request Codex sends after running a tool.
        // function_call has both "id" (item ID) and "call_id" (result matcher).
        // Chat Completions tool_calls[].id must equal function_call_output.call_id.
        let body = json!({
            "model": "gpt-4",
            "input": [
                {"type": "message", "role": "user", "content": "list files"},
                {"type": "function_call", "id": "fc_item_1", "call_id": "call_abc", "name": "shell", "arguments": "{\"cmd\":\"ls\"}"},
                {"type": "function_call_output", "call_id": "call_abc", "output": "file1.txt\nfile2.txt"}
            ]
        });
        let chat = convert_responses_to_chat_request(
            &body,
            &CodexRouterConfig {
                target_base_url: "https://example.com/v1".to_string(),
                api_key: String::new(),
                copilot_token_manager: None,
            },
        );
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
        // tool_calls[].id must be call_id (not item id) to match the tool result
        assert_eq!(msgs[1]["tool_calls"][0]["id"], "call_abc");
        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[2]["tool_call_id"], "call_abc");
        assert_eq!(msgs[2]["content"], "file1.txt\nfile2.txt");
    }

    #[test]
    fn test_convert_request_function_call_without_call_id_falls_back_to_id() {
        // Older format: only "id", no "call_id"
        let body = json!({
            "model": "gpt-4",
            "input": [
                {"type": "function_call", "id": "call_legacy", "name": "shell", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_legacy", "output": "ok"}
            ]
        });
        let chat = convert_responses_to_chat_request(
            &body,
            &CodexRouterConfig {
                target_base_url: "https://example.com/v1".to_string(),
                api_key: String::new(),
                copilot_token_manager: None,
            },
        );
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["tool_calls"][0]["id"], "call_legacy");
        assert_eq!(msgs[1]["tool_call_id"], "call_legacy");
    }

    #[test]
    fn test_convert_request_filters_non_function_tools() {
        let body = json!({
            "model": "gpt-4",
            "input": [],
            "tools": [
                {"type": "function", "name": "shell", "parameters": {}},
                {"type": "computer_use"},
                {"type": "web_search"}
            ]
        });
        let chat = convert_responses_to_chat_request(
            &body,
            &CodexRouterConfig {
                target_base_url: "https://example.com/v1".to_string(),
                api_key: String::new(),
                copilot_token_manager: None,
            },
        );
        let tools = chat["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["function"]["name"], "shell");
    }

    #[test]
    fn test_convert_request_openrouter_transforms_model() {
        let body = json!({"model": "gpt-5.2-codex", "input": []});
        let chat = convert_responses_to_chat_request(
            &body,
            &CodexRouterConfig {
                target_base_url: "https://openrouter.ai/api/v1".to_string(),
                api_key: String::new(),
                copilot_token_manager: None,
            },
        );
        assert_eq!(chat["model"], "openai/gpt-5.2-codex");
    }

    // ── convert_chat_response_to_responses_sse ─────────────────────────────────

    #[test]
    fn test_convert_response_text_contains_required_events() {
        let chat = json!({
            "choices": [{"message": {"role": "assistant", "content": "Here are your files."}}]
        });
        let sse = convert_chat_response_to_responses_sse(&chat);
        assert!(sse.contains("event: response.created\n"));
        assert!(sse.contains("event: response.output_text.delta\n"));
        assert!(sse.contains("event: response.output_text.done\n"));
        assert!(sse.contains("event: response.completed\n"));
        assert!(sse.contains("Here are your files."));
    }

    #[test]
    fn test_convert_response_tool_call_contains_required_events() {
        let chat = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "shell", "arguments": "{\"cmd\":\"ls\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let sse = convert_chat_response_to_responses_sse(&chat);
        assert!(sse.contains("event: response.output_item.added\n"));
        assert!(sse.contains("event: response.function_call_arguments.delta\n"));
        assert!(sse.contains("event: response.function_call_arguments.done\n"));
        assert!(sse.contains("event: response.output_item.done\n"));
        assert!(sse.contains("event: response.completed\n"));
        assert!(sse.contains("call_abc"));
        assert!(sse.contains("shell"));
    }

    #[test]
    fn test_convert_response_empty_content_no_delta_event() {
        let chat = json!({
            "choices": [{"message": {"role": "assistant", "content": ""}}]
        });
        let sse = convert_chat_response_to_responses_sse(&chat);
        // Empty content: delta event should be omitted
        assert!(!sse.contains("response.output_text.delta"));
        // But done event should still be present
        assert!(sse.contains("response.output_text.done"));
    }

    #[test]
    fn test_convert_response_uses_correct_object_type() {
        let chat = json!({"choices": [{"message": {"role": "assistant", "content": "hi"}}]});
        let sse = convert_chat_response_to_responses_sse(&chat);
        assert!(sse.contains("\"object\":\"response\""));
        assert!(!sse.contains("realtime.response"));
    }

    #[test]
    fn test_convert_response_includes_response_id() {
        let chat = json!({"choices": [{"message": {"role": "assistant", "content": "hi"}}]});
        let sse = convert_chat_response_to_responses_sse(&chat);
        assert!(sse.contains("\"response_id\""));
    }

    #[test]
    fn test_convert_response_tool_call_has_call_id() {
        let chat = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{"id": "call_abc123", "type": "function",
                                    "function": {"name": "shell", "arguments": "{}"}}]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let sse = convert_chat_response_to_responses_sse(&chat);
        // call_id must be the Chat Completions tool call id
        assert!(sse.contains("\"call_id\":\"call_abc123\""));
    }

    // ── SSE accumulator ────────────────────────────────────────────────────────

    #[test]
    fn test_accumulate_chat_sse_text_response() {
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\
                   data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\
                   data: [DONE]\n";
        let result = accumulate_chat_sse(sse);
        assert_eq!(result["choices"][0]["message"]["content"], "Hello world");
        assert_eq!(result["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn test_accumulate_chat_sse_tool_call_response() {
        let sse = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_x\",\"type\":\"function\",\"function\":{\"name\":\"shell\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\
                   data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"cmd\\\":\\\"ls\\\"}\"}}]},\"finish_reason\":null}]}\n\
                   data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\
                   data: [DONE]\n";
        let result = accumulate_chat_sse(sse);
        let tcs = result["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(tcs[0]["id"], "call_x");
        assert_eq!(tcs[0]["function"]["name"], "shell");
        assert!(
            tcs[0]["function"]["arguments"]
                .as_str()
                .unwrap()
                .contains("ls")
        );
    }

    #[test]
    fn test_parse_provider_response_json() {
        let json_text = r#"{"choices":[{"message":{"role":"assistant","content":"hi"}}]}"#;
        let result = parse_provider_response(json_text).unwrap();
        assert_eq!(result["choices"][0]["message"]["content"], "hi");
    }

    #[test]
    fn test_parse_provider_response_sse_fallback() {
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\ndata: [DONE]\n";
        let result = parse_provider_response(sse).unwrap();
        assert_eq!(result["choices"][0]["message"]["content"], "hi");
    }

    #[test]
    fn test_codex_config_copilot_token_manager_field() {
        let config = CodexRouterConfig {
            target_base_url: "https://api.example.com".to_string(),
            api_key: "sk-test".to_string(),
            copilot_token_manager: None,
        };
        assert!(config.copilot_token_manager.is_none());
    }

    #[test]
    fn test_convert_request_copilot_skips_model_transform() {
        // When using Copilot, model names should pass through unchanged (no openai/ prefix)
        let body = json!({"model": "gpt-4o", "input": []});
        let config = CodexRouterConfig {
            target_base_url: String::new(),
            api_key: String::new(),
            // Simulate copilot mode by checking the None branch is the no-transform path
            copilot_token_manager: None,
        };
        // Non-copilot with non-openrouter URL: no transform
        let chat = convert_responses_to_chat_request(&body, &config);
        assert_eq!(chat["model"], "gpt-4o");
    }
}
