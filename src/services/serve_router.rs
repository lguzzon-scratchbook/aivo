//! Serve Router — exposes a local OpenAI-compatible HTTP API.
//!
//! Clients send OpenAI-format requests; this router transforms them to whatever
//! protocol the active upstream provider requires, forwards them, and returns
//! OpenAI-format responses.

use anyhow::Result;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::commands::models::fetch_models;
use crate::services::copilot_auth::CopilotTokenManager;
use crate::services::http_utils::{self, router_http_client};
use crate::services::model_names::{copilot_model_name, transform_model_for_openrouter};
use crate::services::openai_anthropic_bridge::{
    OpenAIToAnthropicChatConfig, convert_anthropic_to_openai_chat_response,
    convert_openai_chat_response_to_sse, convert_openai_chat_to_anthropic_request,
};
use crate::services::openai_gemini_bridge::{
    OpenAIToGeminiConfig, build_google_generate_content_url,
    build_google_stream_generate_content_url, convert_gemini_to_openai_chat_response,
    convert_openai_chat_to_gemini_request,
};
use crate::services::provider_protocol::ProviderProtocol;
use crate::services::session_store::ApiKey;

pub struct ServeRouterConfig {
    pub upstream_base_url: String,
    pub upstream_api_key: String,
    pub upstream_protocol: ProviderProtocol,
    pub is_copilot: bool,
    pub is_openrouter: bool,
}

pub struct ServeRouter {
    config: ServeRouterConfig,
    key: ApiKey,
}

struct ServeState {
    config: Arc<ServeRouterConfig>,
    client: reqwest::Client,
    key: ApiKey,
    copilot_tokens: Option<Arc<CopilotTokenManager>>,
}

enum RouterResponse {
    Buffered {
        status: u16,
        content_type: String,
        body: Vec<u8>,
    },
    Streaming {
        status: u16,
        content_type: String,
        body: StreamingBody,
    },
}

enum StreamingBody {
    Upstream(reqwest::Response),
    Anthropic {
        upstream: reqwest::Response,
        converter: AnthropicToOpenAIStreamConverter,
    },
    Gemini {
        upstream: reqwest::Response,
        converter: GeminiToOpenAIStreamConverter,
    },
}

#[derive(Default)]
struct AnthropicToolCallState {
    id: String,
    name: String,
}

struct AnthropicToOpenAIStreamConverter {
    pending: String,
    id: String,
    model: String,
    fallback_model: String,
    created: u64,
    role_sent: bool,
    finished: bool,
    saw_tool_call: bool,
    tool_calls: HashMap<usize, AnthropicToolCallState>,
}

struct GeminiToOpenAIStreamConverter {
    pending: String,
    id: String,
    model: String,
    created: u64,
    role_sent: bool,
    finished: bool,
    saw_tool_call: bool,
    next_tool_index: usize,
}

impl ServeRouter {
    pub fn new(config: ServeRouterConfig, key: ApiKey) -> Self {
        Self { config, key }
    }

    /// Binds to the port eagerly (propagates "address already in use" immediately),
    /// then spawns the accept loop in the background and returns the join handle.
    pub async fn start_background(self, port: u16) -> Result<tokio::task::JoinHandle<Result<()>>> {
        let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", port)).await?;

        let copilot_tokens = if self.config.is_copilot {
            Some(Arc::new(CopilotTokenManager::new(
                self.config.upstream_api_key.clone(),
            )))
        } else {
            None
        };

        let state = Arc::new(ServeState {
            config: Arc::new(self.config),
            client: router_http_client(),
            key: self.key,
            copilot_tokens,
        });

        Ok(tokio::spawn(run_accept_loop(listener, state)))
    }
}

async fn run_accept_loop(listener: tokio::net::TcpListener, state: Arc<ServeState>) -> Result<()> {
    loop {
        let (mut socket, _) = listener.accept().await?;
        let state = state.clone();

        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;

            let request_bytes = match http_utils::read_full_request(&mut socket).await {
                Ok(b) => b,
                Err(err) => {
                    let response = http_utils::http_request_read_error_response(&err);
                    let _ = socket.write_all(response.as_bytes()).await;
                    return;
                }
            };

            let request = String::from_utf8_lossy(&request_bytes).into_owned();
            let path = http_utils::extract_request_path(&request);
            let path_no_query = path.split('?').next().unwrap_or(&path);

            let result = match path_no_query {
                "/v1/models" | "/models" => handle_models(&state).await,
                "/v1/chat/completions" => {
                    if !request.starts_with("POST ") {
                        Ok(buffered_response(
                            405,
                            "application/json",
                            br#"{"error":{"message":"Method not allowed"}}"#.to_vec(),
                        ))
                    } else {
                        handle_chat(&request, &state).await
                    }
                }
                _ => Ok(buffered_response(
                    404,
                    "application/json",
                    br#"{"error":{"message":"Not found"}}"#.to_vec(),
                )),
            };

            match result {
                Ok(response) => {
                    let _ = write_router_response(&mut socket, response).await;
                }
                Err(e) => {
                    let _ = socket
                        .write_all(http_utils::http_error_response(500, &e.to_string()).as_bytes())
                        .await;
                }
            }
        });
    }
}

async fn handle_models(state: &ServeState) -> Result<RouterResponse> {
    let models = fetch_models(&state.client, &state.key).await?;
    let data: Vec<Value> = models
        .into_iter()
        .map(|id| json!({"id": id, "object": "model", "owned_by": "aivo"}))
        .collect();
    let resp = json!({"object": "list", "data": data});
    Ok(buffered_response(
        200,
        "application/json",
        resp.to_string().into_bytes(),
    ))
}

async fn handle_chat(request: &str, state: &ServeState) -> Result<RouterResponse> {
    let body_str = http_utils::extract_request_body(request)?;
    let mut body: Value = serde_json::from_str(body_str)?;

    let client_wants_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    match state.config.upstream_protocol {
        ProviderProtocol::Anthropic => {
            handle_chat_anthropic(&body, client_wants_stream, state).await
        }
        ProviderProtocol::Google => handle_chat_gemini(&mut body, client_wants_stream, state).await,
        ProviderProtocol::Openai => handle_chat_openai(&mut body, client_wants_stream, state).await,
    }
}

async fn handle_chat_anthropic(
    body: &Value,
    client_wants_stream: bool,
    state: &ServeState,
) -> Result<RouterResponse> {
    let fallback_model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("claude-sonnet-4-5")
        .to_string();

    let mut anthropic_req = convert_openai_chat_to_anthropic_request(
        body,
        &OpenAIToAnthropicChatConfig {
            default_model: "claude-sonnet-4-5",
        },
    );
    anthropic_req["stream"] = json!(client_wants_stream);

    let url = http_utils::build_target_url(&state.config.upstream_base_url, "/v1/messages");
    let response = state
        .client
        .post(&url)
        .header("x-api-key", state.config.upstream_api_key.as_str())
        .header("anthropic-version", "2023-06-01")
        .header("Content-Type", "application/json")
        .header("User-Agent", "aivo-serve/1.0")
        .json(&anthropic_req)
        .send()
        .await?;

    let status = response.status().as_u16();
    let content_type = http_utils::response_content_type(&response);

    if status >= 400 {
        return Ok(buffered_response(
            status,
            &content_type,
            response.bytes().await?.to_vec(),
        ));
    }

    if client_wants_stream && content_type.contains("text/event-stream") {
        return Ok(RouterResponse::Streaming {
            status,
            content_type: "text/event-stream".to_string(),
            body: StreamingBody::Anthropic {
                upstream: response,
                converter: AnthropicToOpenAIStreamConverter::new(&fallback_model),
            },
        });
    }

    let resp_body = response.text().await?;
    let anthropic_resp: Value = serde_json::from_str(&resp_body)?;
    let openai_resp = convert_anthropic_to_openai_chat_response(&anthropic_resp, &fallback_model);

    if client_wants_stream {
        Ok(buffered_response(
            200,
            "text/event-stream",
            convert_openai_chat_response_to_sse(&openai_resp).into_bytes(),
        ))
    } else {
        Ok(buffered_response(
            200,
            "application/json",
            openai_resp.to_string().into_bytes(),
        ))
    }
}

async fn handle_chat_gemini(
    body: &mut Value,
    client_wants_stream: bool,
    state: &ServeState,
) -> Result<RouterResponse> {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("gemini-2.5-pro")
        .to_string();

    let gemini_req = convert_openai_chat_to_gemini_request(
        body,
        &OpenAIToGeminiConfig {
            default_model: "gemini-2.5-pro",
        },
    );

    let url = if client_wants_stream {
        build_google_stream_generate_content_url(&state.config.upstream_base_url, &model)
    } else {
        build_google_generate_content_url(&state.config.upstream_base_url, &model)
    };
    let response = state
        .client
        .post(&url)
        .header("x-goog-api-key", state.config.upstream_api_key.as_str())
        .header("Content-Type", "application/json")
        .header("User-Agent", "aivo-serve/1.0")
        .json(&gemini_req)
        .send()
        .await?;

    let status = response.status().as_u16();
    let content_type = http_utils::response_content_type(&response);

    if status >= 400 {
        return Ok(buffered_response(
            status,
            &content_type,
            response.bytes().await?.to_vec(),
        ));
    }

    if client_wants_stream && content_type.contains("text/event-stream") {
        return Ok(RouterResponse::Streaming {
            status,
            content_type: "text/event-stream".to_string(),
            body: StreamingBody::Gemini {
                upstream: response,
                converter: GeminiToOpenAIStreamConverter::new(&model),
            },
        });
    }

    let resp_body = response.text().await?;
    let gemini_resp: Value = serde_json::from_str(&resp_body)?;
    let openai_resp = convert_gemini_to_openai_chat_response(&gemini_resp, &model);

    if client_wants_stream {
        Ok(buffered_response(
            200,
            "text/event-stream",
            convert_openai_chat_response_to_sse(&openai_resp).into_bytes(),
        ))
    } else {
        Ok(buffered_response(
            200,
            "application/json",
            openai_resp.to_string().into_bytes(),
        ))
    }
}

async fn handle_chat_openai(
    body: &mut Value,
    client_wants_stream: bool,
    state: &ServeState,
) -> Result<RouterResponse> {
    if state.config.is_openrouter {
        let normalized = body
            .get("model")
            .and_then(|v| v.as_str())
            .map(transform_model_for_openrouter);
        if let Some(n) = normalized {
            body["model"] = json!(n);
        }
    } else if state.config.is_copilot {
        let normalized = body
            .get("model")
            .and_then(|v| v.as_str())
            .map(copilot_model_name);
        if let Some(n) = normalized {
            body["model"] = json!(n);
        }
    }

    let url = http_utils::build_chat_completions_url(&state.config.upstream_base_url);
    let req = http_utils::authorized_openai_post(
        &state.client,
        &url,
        state.config.upstream_api_key.as_str(),
        state.copilot_tokens.as_deref(),
    )
    .await?;

    let response = req.json(&*body).send().await?;
    let status = response.status().as_u16();
    let content_type = http_utils::response_content_type(&response);

    if status >= 400 {
        return Ok(buffered_response(
            status,
            &content_type,
            response.bytes().await?.to_vec(),
        ));
    }

    if client_wants_stream && content_type.contains("text/event-stream") {
        return Ok(RouterResponse::Streaming {
            status,
            content_type,
            body: StreamingBody::Upstream(response),
        });
    }

    let resp_body = response.text().await?;

    if client_wants_stream && let Ok(openai_resp) = serde_json::from_str::<Value>(&resp_body) {
        return Ok(buffered_response(
            200,
            "text/event-stream",
            convert_openai_chat_response_to_sse(&openai_resp).into_bytes(),
        ));
    }

    Ok(buffered_response(
        status,
        &content_type,
        resp_body.into_bytes(),
    ))
}

fn buffered_response(status: u16, content_type: &str, body: Vec<u8>) -> RouterResponse {
    RouterResponse::Buffered {
        status,
        content_type: content_type.to_string(),
        body,
    }
}

async fn write_router_response(
    socket: &mut tokio::net::TcpStream,
    response: RouterResponse,
) -> Result<()> {
    use tokio::io::AsyncWriteExt;

    match response {
        RouterResponse::Buffered {
            status,
            content_type,
            body,
        } => {
            let headers = http_utils::http_response_head(status, &content_type, body.len());
            socket.write_all(headers.as_bytes()).await?;
            socket.write_all(&body).await?;
        }
        RouterResponse::Streaming {
            status,
            content_type,
            body,
        } => {
            let headers = http_utils::http_chunked_response_head(status, &content_type);
            socket.write_all(headers.as_bytes()).await?;

            match body {
                StreamingBody::Upstream(mut upstream) => {
                    while let Some(chunk) = upstream.chunk().await? {
                        write_chunk(socket, &chunk).await?;
                    }
                }
                StreamingBody::Anthropic {
                    mut upstream,
                    mut converter,
                } => {
                    while let Some(chunk) = upstream.chunk().await? {
                        let mapped = converter.push_bytes(&chunk)?;
                        if !mapped.is_empty() {
                            write_chunk(socket, mapped.as_bytes()).await?;
                        }
                    }
                    let tail = converter.finish()?;
                    if !tail.is_empty() {
                        write_chunk(socket, tail.as_bytes()).await?;
                    }
                }
                StreamingBody::Gemini {
                    mut upstream,
                    mut converter,
                } => {
                    while let Some(chunk) = upstream.chunk().await? {
                        let mapped = converter.push_bytes(&chunk)?;
                        if !mapped.is_empty() {
                            write_chunk(socket, mapped.as_bytes()).await?;
                        }
                    }
                    let tail = converter.finish()?;
                    if !tail.is_empty() {
                        write_chunk(socket, tail.as_bytes()).await?;
                    }
                }
            }

            socket.write_all(b"0\r\n\r\n").await?;
        }
    }

    Ok(())
}

async fn write_chunk(socket: &mut tokio::net::TcpStream, chunk: &[u8]) -> Result<()> {
    use tokio::io::AsyncWriteExt;

    if chunk.is_empty() {
        return Ok(());
    }

    let chunk_len = format!("{:X}\r\n", chunk.len());
    socket.write_all(chunk_len.as_bytes()).await?;
    socket.write_all(chunk).await?;
    socket.write_all(b"\r\n").await?;
    Ok(())
}

impl AnthropicToOpenAIStreamConverter {
    fn new(fallback_model: &str) -> Self {
        Self {
            pending: String::new(),
            id: "chatcmpl-aivo".to_string(),
            model: fallback_model.to_string(),
            fallback_model: fallback_model.to_string(),
            created: current_unix_ts(),
            role_sent: false,
            finished: false,
            saw_tool_call: false,
            tool_calls: HashMap::new(),
        }
    }

    fn push_bytes(&mut self, chunk: &[u8]) -> Result<String> {
        self.pending.push_str(&String::from_utf8_lossy(chunk));
        let mut output = String::new();

        while let Some(pos) = self.pending.find('\n') {
            let line = self.pending[..pos].trim_end_matches('\r').to_string();
            self.pending = self.pending[pos + 1..].to_string();
            self.process_line(&line, &mut output)?;
        }

        Ok(output)
    }

    fn finish(&mut self) -> Result<String> {
        let mut output = String::new();

        let tail = self.pending.trim_end_matches('\r').trim().to_string();
        self.pending.clear();
        if !tail.is_empty() {
            self.process_line(&tail, &mut output)?;
        }

        if !self.finished {
            let finish_reason = if self.saw_tool_call {
                "tool_calls"
            } else {
                "stop"
            };
            self.emit_finish(&mut output, finish_reason);
        }

        Ok(output)
    }

    fn process_line(&mut self, line: &str, output: &mut String) -> Result<()> {
        let Some(data) = sse_data_payload(line) else {
            return Ok(());
        };

        if data == "[DONE]" {
            if !self.finished {
                let finish_reason = if self.saw_tool_call {
                    "tool_calls"
                } else {
                    "stop"
                };
                self.emit_finish(output, finish_reason);
            }
            return Ok(());
        }

        let event: Value = match serde_json::from_str(data) {
            Ok(value) => value,
            Err(_) => return Ok(()),
        };

        match event.get("type").and_then(|v| v.as_str()).unwrap_or("") {
            "message_start" => {
                if let Some(message) = event.get("message") {
                    if let Some(id) = message.get("id").and_then(|v| v.as_str())
                        && !id.is_empty()
                    {
                        self.id = id.to_string();
                    }
                    if let Some(model) = message.get("model").and_then(|v| v.as_str())
                        && !model.is_empty()
                    {
                        self.model = model.to_string();
                    }
                }
            }
            "content_block_start" => {
                let index = event.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                if event
                    .get("content_block")
                    .and_then(|v| v.get("type"))
                    .and_then(|v| v.as_str())
                    == Some("tool_use")
                {
                    let block = event
                        .get("content_block")
                        .cloned()
                        .unwrap_or_else(|| json!({}));
                    let id = block
                        .get("id")
                        .and_then(|v| v.as_str())
                        .filter(|v| !v.is_empty())
                        .map(ToOwned::to_owned)
                        .unwrap_or_else(|| format!("call_{index}"));
                    let name = block
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    self.tool_calls.insert(
                        index,
                        AnthropicToolCallState {
                            id: id.clone(),
                            name: name.clone(),
                        },
                    );
                    self.saw_tool_call = true;
                    self.emit_role_if_needed(output);
                    output.push_str(&openai_sse_chunk(
                        &self.id,
                        self.created,
                        &self.model_name(),
                        json!({
                            "tool_calls": [{
                                "index": index,
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name
                                }
                            }]
                        }),
                        Value::Null,
                    ));
                }
            }
            "content_block_delta" => {
                let index = event.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let delta = event.get("delta").cloned().unwrap_or_else(|| json!({}));
                match delta.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                    "text_delta" => {
                        if let Some(text) = delta.get("text").and_then(|v| v.as_str())
                            && !text.is_empty()
                        {
                            self.emit_role_if_needed(output);
                            output.push_str(&openai_sse_chunk(
                                &self.id,
                                self.created,
                                &self.model_name(),
                                json!({ "content": text }),
                                Value::Null,
                            ));
                        }
                    }
                    "input_json_delta" => {
                        if let Some(partial_json) =
                            delta.get("partial_json").and_then(|v| v.as_str())
                        {
                            let (id, name) = {
                                let tool = self.tool_calls.entry(index).or_default();
                                let id = if tool.id.is_empty() {
                                    format!("call_{index}")
                                } else {
                                    tool.id.clone()
                                };
                                (id, tool.name.clone())
                            };
                            self.emit_role_if_needed(output);
                            output.push_str(&openai_sse_chunk(
                                &self.id,
                                self.created,
                                &self.model_name(),
                                json!({
                                    "tool_calls": [{
                                        "index": index,
                                        "id": id,
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": partial_json
                                        }
                                    }]
                                }),
                                Value::Null,
                            ));
                        }
                    }
                    _ => {}
                }
            }
            "message_delta" => {
                if let Some(stop_reason) = event
                    .get("delta")
                    .and_then(|v| v.get("stop_reason"))
                    .and_then(|v| v.as_str())
                {
                    self.emit_finish(output, map_anthropic_stop_reason(stop_reason));
                }
            }
            "message_stop" => {
                if !self.finished {
                    let finish_reason = if self.saw_tool_call {
                        "tool_calls"
                    } else {
                        "stop"
                    };
                    self.emit_finish(output, finish_reason);
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn emit_role_if_needed(&mut self, output: &mut String) {
        if self.role_sent {
            return;
        }
        self.role_sent = true;
        output.push_str(&openai_sse_chunk(
            &self.id,
            self.created,
            &self.model_name(),
            json!({ "role": "assistant" }),
            Value::Null,
        ));
    }

    fn emit_finish(&mut self, output: &mut String, finish_reason: &str) {
        if self.finished {
            return;
        }
        self.emit_role_if_needed(output);
        output.push_str(&openai_sse_chunk(
            &self.id,
            self.created,
            &self.model_name(),
            json!({}),
            json!(finish_reason),
        ));
        output.push_str("data: [DONE]\n\n");
        self.finished = true;
    }

    fn model_name(&self) -> String {
        if self.model.is_empty() {
            self.fallback_model.clone()
        } else {
            self.model.clone()
        }
    }
}

impl GeminiToOpenAIStreamConverter {
    fn new(model: &str) -> Self {
        Self {
            pending: String::new(),
            id: "chatcmpl-aivo".to_string(),
            model: model.to_string(),
            created: current_unix_ts(),
            role_sent: false,
            finished: false,
            saw_tool_call: false,
            next_tool_index: 0,
        }
    }

    fn push_bytes(&mut self, chunk: &[u8]) -> Result<String> {
        self.pending.push_str(&String::from_utf8_lossy(chunk));
        let mut output = String::new();

        while let Some(pos) = self.pending.find('\n') {
            let line = self.pending[..pos].trim_end_matches('\r').to_string();
            self.pending = self.pending[pos + 1..].to_string();
            self.process_line(&line, &mut output)?;
        }

        Ok(output)
    }

    fn finish(&mut self) -> Result<String> {
        let mut output = String::new();

        let tail = self.pending.trim_end_matches('\r').trim().to_string();
        self.pending.clear();
        if !tail.is_empty() {
            self.process_line(&tail, &mut output)?;
        }

        if !self.finished {
            let finish_reason = if self.saw_tool_call {
                "tool_calls"
            } else {
                "stop"
            };
            self.emit_finish(&mut output, finish_reason);
        }

        Ok(output)
    }

    fn process_line(&mut self, line: &str, output: &mut String) -> Result<()> {
        let Some(data) = sse_data_payload(line) else {
            return Ok(());
        };

        if data == "[DONE]" {
            if !self.finished {
                let finish_reason = if self.saw_tool_call {
                    "tool_calls"
                } else {
                    "stop"
                };
                self.emit_finish(output, finish_reason);
            }
            return Ok(());
        }

        let event: Value = match serde_json::from_str(data) {
            Ok(value) => value,
            Err(_) => return Ok(()),
        };

        if let Some(id) = event.get("responseId").and_then(|v| v.as_str())
            && !id.is_empty()
        {
            self.id = id.to_string();
        }

        let candidate = event
            .get("candidates")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .cloned()
            .unwrap_or_else(|| json!({}));

        if let Some(parts) = candidate
            .get("content")
            .and_then(|v| v.get("parts"))
            .and_then(|v| v.as_array())
        {
            for part in parts {
                if let Some(text) = part.get("text").and_then(|v| v.as_str())
                    && !text.is_empty()
                {
                    self.emit_role_if_needed(output);
                    output.push_str(&openai_sse_chunk(
                        &self.id,
                        self.created,
                        &self.model,
                        json!({ "content": text }),
                        Value::Null,
                    ));
                }

                if let Some(function_call) = part.get("functionCall") {
                    let index = self.next_tool_index;
                    self.next_tool_index += 1;
                    self.saw_tool_call = true;
                    self.emit_role_if_needed(output);
                    output.push_str(&openai_sse_chunk(
                        &self.id,
                        self.created,
                        &self.model,
                        json!({
                            "tool_calls": [{
                                "index": index,
                                "id": function_call
                                    .get("id")
                                    .cloned()
                                    .unwrap_or_else(|| json!(format!("call_{index}"))),
                                "type": "function",
                                "function": {
                                    "name": function_call.get("name").cloned().unwrap_or_else(|| json!("")),
                                    "arguments": serde_json::to_string(
                                        &function_call.get("args").cloned().unwrap_or_else(|| json!({}))
                                    ).unwrap_or_else(|_| "{}".to_string())
                                }
                            }]
                        }),
                        Value::Null,
                    ));
                }
            }
        }

        if let Some(reason) = candidate.get("finishReason").and_then(|v| v.as_str())
            && !reason.is_empty()
        {
            self.emit_finish(output, map_gemini_finish_reason(reason, self.saw_tool_call));
        }

        Ok(())
    }

    fn emit_role_if_needed(&mut self, output: &mut String) {
        if self.role_sent {
            return;
        }
        self.role_sent = true;
        output.push_str(&openai_sse_chunk(
            &self.id,
            self.created,
            &self.model,
            json!({ "role": "assistant" }),
            Value::Null,
        ));
    }

    fn emit_finish(&mut self, output: &mut String, finish_reason: &str) {
        if self.finished {
            return;
        }
        self.emit_role_if_needed(output);
        output.push_str(&openai_sse_chunk(
            &self.id,
            self.created,
            &self.model,
            json!({}),
            json!(finish_reason),
        ));
        output.push_str("data: [DONE]\n\n");
        self.finished = true;
    }
}

fn sse_data_payload(line: &str) -> Option<&str> {
    line.strip_prefix("data:").map(str::trim_start)
}

fn current_unix_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn openai_sse_chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: Value,
    finish_reason: Value,
) -> String {
    format!(
        "data: {}\n\n",
        json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }]
        })
    )
}

fn map_anthropic_stop_reason(stop_reason: &str) -> &'static str {
    match stop_reason {
        "tool_use" => "tool_calls",
        "max_tokens" => "length",
        _ => "stop",
    }
}

fn map_gemini_finish_reason(finish_reason: &str, saw_tool_call: bool) -> &'static str {
    if saw_tool_call {
        return "tool_calls";
    }

    match finish_reason {
        "MAX_TOKENS" => "length",
        "SAFETY" => "content_filter",
        _ => "stop",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_stream_converter_emits_openai_sse() {
        let mut converter = AnthropicToOpenAIStreamConverter::new("claude-sonnet-4-5");
        let input = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-sonnet-4-5\",\"usage\":{\"input_tokens\":12}}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"shell\",\"input\":{}}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"cmd\\\":\\\"ls\\\"}\"}}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":7}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );

        let mut output = converter.push_bytes(input.as_bytes()).unwrap();
        output.push_str(&converter.finish().unwrap());

        assert!(output.contains("\"role\":\"assistant\""));
        assert!(output.contains("\"content\":\"Hello\""));
        assert!(output.contains("\"tool_calls\":[{"));
        assert!(output.contains("\"id\":\"toolu_1\""));
        assert!(output.contains("\"index\":1"));
        assert!(output.contains("\"arguments\":\"{\\\"cmd\\\":\\\"ls\\\"}\""));
        assert!(output.contains("\"finish_reason\":\"tool_calls\""));
        assert!(output.contains("data: [DONE]"));
    }

    #[test]
    fn test_gemini_stream_converter_emits_openai_sse() {
        let mut converter = GeminiToOpenAIStreamConverter::new("gemini-2.5-pro");
        let input = concat!(
            "data: {\"responseId\":\"resp_1\",\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hel\"}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"lo\"}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"id\":\"call_1\",\"name\":\"shell\",\"args\":{\"cmd\":\"ls\"}}}],\"role\":\"model\"},\"finishReason\":\"STOP\"}]}\n\n",
        );

        let mut output = converter.push_bytes(input.as_bytes()).unwrap();
        output.push_str(&converter.finish().unwrap());

        assert!(output.contains("\"role\":\"assistant\""));
        assert!(output.contains("\"content\":\"Hel\""));
        assert!(output.contains("\"content\":\"lo\""));
        assert!(output.contains("\"tool_calls\":[{"));
        assert!(output.contains("\"id\":\"call_1\""));
        assert!(output.contains("\"index\":0"));
        assert!(output.contains("\"name\":\"shell\""));
        assert!(output.contains("\"finish_reason\":\"tool_calls\""));
        assert!(output.contains("data: [DONE]"));
    }
}
