//! Serve Router — exposes a local OpenAI-compatible HTTP API.
//!
//! Clients send OpenAI-format requests; this router transforms them to whatever
//! protocol the active upstream provider requires, forwards them, and returns
//! OpenAI-format responses.

use anyhow::Result;
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::commands::models::fetch_models;
use crate::services::copilot_auth::CopilotTokenManager;
use crate::services::http_utils::{self, router_http_client};
use crate::services::provider_protocol::{
    ProviderProtocol, fallback_protocols, is_protocol_mismatch,
};
use crate::services::responses_to_chat_router::{
    ResponsesToChatRouterConfig, convert_chat_response_to_responses_sse,
    convert_responses_to_chat_request,
};
use crate::services::serve_responses::{
    OpenAIToResponsesStreamConverter, convert_chat_response_to_responses_json,
    convert_chat_sse_to_responses_sse,
};
use crate::services::serve_upstream::{
    RouterResponse, StreamingBody, UpstreamRequestContext, send_anthropic_chat, send_gemini_chat,
    send_openai_chat,
};
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
    active_protocol: Arc<AtomicU8>,
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

        let initial_protocol = self.config.upstream_protocol;

        let state = Arc::new(ServeState {
            config: Arc::new(self.config),
            client: router_http_client(),
            key: self.key,
            copilot_tokens,
            active_protocol: Arc::new(AtomicU8::new(initial_protocol.to_u8())),
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
                        Ok(RouterResponse::buffered(
                            405,
                            "application/json",
                            br#"{"error":{"message":"Method not allowed"}}"#.to_vec(),
                        ))
                    } else {
                        handle_chat(&request, &state).await
                    }
                }
                "/v1/responses" | "/responses" => {
                    if !request.starts_with("POST ") {
                        Ok(RouterResponse::buffered(
                            405,
                            "application/json",
                            br#"{"error":{"message":"Method not allowed"}}"#.to_vec(),
                        ))
                    } else {
                        handle_responses(&request, &state).await
                    }
                }
                _ => Ok(RouterResponse::buffered(
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
    Ok(RouterResponse::buffered(
        200,
        "application/json",
        resp.to_string().into_bytes(),
    ))
}

async fn handle_chat(request: &str, state: &ServeState) -> Result<RouterResponse> {
    let body_str = http_utils::extract_request_body(request)?;
    let body: Value = serde_json::from_str(body_str)?;

    handle_chat_body(body, state).await
}

async fn handle_responses(request: &str, state: &ServeState) -> Result<RouterResponse> {
    let body_str = http_utils::extract_request_body(request)?;
    let body: Value = serde_json::from_str(body_str)?;
    let original_model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("gpt-4o")
        .to_string();
    let client_wants_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Use `actual_model` to pin the model name to the raw user-supplied value.  The config's
    // `target_protocol` is snapshotted here, before `handle_chat_body` runs the fallback loop;
    // if the loop switches protocol, any protocol-based model-name transformation done by
    // `convert_responses_to_chat_request` would have used the wrong protocol.  Setting
    // `actual_model` causes `select_model_for_protocol` to return it verbatim, so the model
    // field in `chat_body` is always the original string and `handle_chat_body` transforms it
    // for the protocol that is actually selected.
    let mut config = responses_router_config(state);
    config.actual_model = Some(original_model.clone());
    let mut chat_body = convert_responses_to_chat_request(&body, &config);
    chat_body["stream"] = json!(client_wants_stream);
    let chat_response = handle_chat_body(chat_body, state).await?;
    convert_chat_response_for_responses_route(chat_response, client_wants_stream, &original_model)
}

async fn handle_chat_body(body: Value, state: &ServeState) -> Result<RouterResponse> {
    let client_wants_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Skip fallback for copilot/openrouter — these have fixed protocols
    if state.config.is_copilot || state.config.is_openrouter {
        let mut body = body;
        return match ProviderProtocol::from_u8(state.active_protocol.load(Ordering::Relaxed)) {
            ProviderProtocol::Anthropic => {
                handle_chat_anthropic(&body, client_wants_stream, state).await
            }
            ProviderProtocol::Google => {
                handle_chat_gemini(&mut body, client_wants_stream, state).await
            }
            ProviderProtocol::Openai | ProviderProtocol::ResponsesApi => {
                handle_chat_openai(&mut body, client_wants_stream, state).await
            }
        };
    }

    let current = ProviderProtocol::from_u8(state.active_protocol.load(Ordering::Relaxed));
    let candidates: Vec<ProviderProtocol> = std::iter::once(current)
        .chain(fallback_protocols(current, &state.config.upstream_base_url))
        .collect();

    let mut last_response: Option<RouterResponse> = None;
    for (attempt, protocol) in candidates.into_iter().enumerate() {
        let mut body_clone = body.clone();
        let response = match protocol {
            ProviderProtocol::Anthropic => {
                handle_chat_anthropic(&body_clone, client_wants_stream, state).await?
            }
            ProviderProtocol::Google => {
                handle_chat_gemini(&mut body_clone, client_wants_stream, state).await?
            }
            ProviderProtocol::Openai | ProviderProtocol::ResponsesApi => {
                handle_chat_openai(&mut body_clone, client_wants_stream, state).await?
            }
        };

        let status = match &response {
            RouterResponse::Buffered { status, .. } => *status,
            // Streaming is only produced when the upstream returned 200 (see each handle_chat_* handler);
            // a protocol mismatch (404/405/415) always results in a Buffered error response.
            RouterResponse::Streaming { .. } => 200,
        };

        if is_protocol_mismatch(status) {
            last_response = Some(response);
            continue;
        }

        // Not a mismatch — return this response
        if attempt > 0 {
            state
                .active_protocol
                .store(protocol.to_u8(), Ordering::Relaxed);
            eprintln!("  \u{2022} Protocol auto-switched to {}", protocol.as_str());
        }
        return Ok(response);
    }

    Ok(last_response.unwrap_or(RouterResponse::buffered(
        503,
        "application/json",
        br#"{"error":{"message":"No compatible protocol found"}}"#.to_vec(),
    )))
}

fn responses_router_config(state: &ServeState) -> ResponsesToChatRouterConfig {
    ResponsesToChatRouterConfig {
        target_base_url: state.config.upstream_base_url.clone(),
        api_key: state.config.upstream_api_key.clone(),
        target_protocol: ProviderProtocol::from_u8(state.active_protocol.load(Ordering::Relaxed)),
        copilot_token_manager: state.copilot_tokens.clone(),
        model_prefix: None,
        requires_reasoning_content: false,
        actual_model: None,
        max_tokens_cap: None,
        responses_api_supported: None,
    }
}

fn upstream_context(state: &ServeState) -> UpstreamRequestContext {
    UpstreamRequestContext {
        client: state.client.clone(),
        upstream_base_url: state.config.upstream_base_url.clone(),
        upstream_api_key: state.config.upstream_api_key.clone(),
        is_copilot: state.config.is_copilot,
        is_openrouter: state.config.is_openrouter,
        copilot_tokens: state.copilot_tokens.clone(),
    }
}

async fn handle_chat_anthropic(
    body: &Value,
    client_wants_stream: bool,
    state: &ServeState,
) -> Result<RouterResponse> {
    send_anthropic_chat(body, client_wants_stream, &upstream_context(state)).await
}

async fn handle_chat_gemini(
    body: &mut Value,
    client_wants_stream: bool,
    state: &ServeState,
) -> Result<RouterResponse> {
    send_gemini_chat(body, client_wants_stream, &upstream_context(state)).await
}

async fn handle_chat_openai(
    body: &mut Value,
    client_wants_stream: bool,
    state: &ServeState,
) -> Result<RouterResponse> {
    send_openai_chat(body, client_wants_stream, &upstream_context(state)).await
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

            match *body {
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
                StreamingBody::Responses {
                    source,
                    mut converter,
                } => {
                    match *source {
                        StreamingBody::Upstream(mut upstream) => {
                            while let Some(chunk) = upstream.chunk().await? {
                                let mapped = converter.push_bytes(&chunk)?;
                                if !mapped.is_empty() {
                                    write_chunk(socket, mapped.as_bytes()).await?;
                                }
                            }
                        }
                        StreamingBody::Anthropic {
                            mut upstream,
                            converter: mut openai_converter,
                        } => {
                            while let Some(chunk) = upstream.chunk().await? {
                                let openai = openai_converter.push_bytes(&chunk)?;
                                if !openai.is_empty() {
                                    let mapped = converter.push_bytes(openai.as_bytes())?;
                                    if !mapped.is_empty() {
                                        write_chunk(socket, mapped.as_bytes()).await?;
                                    }
                                }
                            }
                            let openai_tail = openai_converter.finish()?;
                            if !openai_tail.is_empty() {
                                let mapped = converter.push_bytes(openai_tail.as_bytes())?;
                                if !mapped.is_empty() {
                                    write_chunk(socket, mapped.as_bytes()).await?;
                                }
                            }
                        }
                        StreamingBody::Gemini {
                            mut upstream,
                            converter: mut openai_converter,
                        } => {
                            while let Some(chunk) = upstream.chunk().await? {
                                let openai = openai_converter.push_bytes(&chunk)?;
                                if !openai.is_empty() {
                                    let mapped = converter.push_bytes(openai.as_bytes())?;
                                    if !mapped.is_empty() {
                                        write_chunk(socket, mapped.as_bytes()).await?;
                                    }
                                }
                            }
                            let openai_tail = openai_converter.finish()?;
                            if !openai_tail.is_empty() {
                                let mapped = converter.push_bytes(openai_tail.as_bytes())?;
                                if !mapped.is_empty() {
                                    write_chunk(socket, mapped.as_bytes()).await?;
                                }
                            }
                        }
                        StreamingBody::Responses { .. } => {
                            anyhow::bail!("nested responses stream sources are not supported");
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

    let formatted = format_http_chunk(chunk);
    if formatted.is_empty() {
        return Ok(());
    }
    socket.write_all(&formatted).await?;
    Ok(())
}

fn convert_chat_response_for_responses_route(
    chat_response: RouterResponse,
    client_wants_stream: bool,
    original_model: &str,
) -> Result<RouterResponse> {
    match chat_response {
        RouterResponse::Buffered {
            status,
            content_type,
            body,
        } => {
            if status >= 400 {
                return Ok(RouterResponse::buffered(status, &content_type, body));
            }

            if client_wants_stream {
                let sse = if content_type.contains("text/event-stream") {
                    convert_chat_sse_to_responses_sse(std::str::from_utf8(&body)?, original_model)?
                } else {
                    let chat_json: Value = serde_json::from_slice(&body)?;
                    convert_chat_response_to_responses_sse(&chat_json, false, original_model)
                };
                Ok(RouterResponse::buffered(
                    200,
                    "text/event-stream",
                    sse.into_bytes(),
                ))
            } else {
                let chat_json: Value = serde_json::from_slice(&body)?;
                let response_json =
                    convert_chat_response_to_responses_json(&chat_json, original_model)?;
                Ok(RouterResponse::buffered(
                    200,
                    "application/json",
                    serde_json::to_vec(&response_json)?,
                ))
            }
        }
        RouterResponse::Streaming {
            status,
            content_type: _,
            body,
        } => {
            if !client_wants_stream {
                anyhow::bail!(
                    "internal error: responses route received streaming body for non-streaming request"
                );
            }

            Ok(RouterResponse::Streaming {
                status,
                content_type: "text/event-stream".to_string(),
                body: Box::new(StreamingBody::Responses {
                    source: body,
                    converter: OpenAIToResponsesStreamConverter::new(original_model),
                }),
            })
        }
    }
}

fn format_http_chunk(chunk: &[u8]) -> Vec<u8> {
    if chunk.is_empty() {
        return Vec::new();
    }

    let mut formatted = format!("{:X}\r\n", chunk.len()).into_bytes();
    formatted.extend_from_slice(chunk);
    formatted.extend_from_slice(b"\r\n");
    formatted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::session_store::ApiKey;
    use http::Response as HttpResponse;
    use serde_json::json;

    fn test_key() -> ApiKey {
        ApiKey::new_with_protocol(
            "abc".to_string(),
            "test".to_string(),
            "https://example.com/v1".to_string(),
            None,
            "secret".to_string(),
        )
    }

    fn test_state(protocol: ProviderProtocol) -> ServeState {
        ServeState {
            config: Arc::new(ServeRouterConfig {
                upstream_base_url: "https://example.com/v1".to_string(),
                upstream_api_key: "secret".to_string(),
                upstream_protocol: protocol,
                is_copilot: false,
                is_openrouter: false,
            }),
            client: router_http_client(),
            key: test_key(),
            copilot_tokens: None,
            active_protocol: Arc::new(AtomicU8::new(protocol.to_u8())),
        }
    }

    fn mock_reqwest_response(
        status: u16,
        content_type: &str,
        body: impl Into<String>,
    ) -> reqwest::Response {
        HttpResponse::builder()
            .status(status)
            .header("content-type", content_type)
            .body(body.into())
            .unwrap()
            .into()
    }

    #[test]
    fn convert_chat_response_for_responses_route_maps_buffered_json() {
        let chat = json!({
            "choices": [{
                "message": {"role": "assistant", "content": "Hello from router"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        });

        let response = convert_chat_response_for_responses_route(
            RouterResponse::buffered(200, "application/json", serde_json::to_vec(&chat).unwrap()),
            false,
            "gpt-4o",
        )
        .unwrap();

        match response {
            RouterResponse::Buffered {
                status,
                content_type,
                body,
            } => {
                let json: Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(status, 200);
                assert_eq!(content_type, "application/json");
                assert_eq!(json["object"], "response");
                assert_eq!(json["output"][0]["content"][0]["text"], "Hello from router");
            }
            RouterResponse::Streaming { .. } => panic!("expected buffered response"),
        }
    }

    #[test]
    fn convert_chat_response_for_responses_route_maps_streaming_sse() {
        let chat_sse = concat!(
            "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\n\n",
            "data: [DONE]\n\n",
        );

        let response = convert_chat_response_for_responses_route(
            RouterResponse::buffered(200, "text/event-stream", chat_sse.as_bytes().to_vec()),
            true,
            "gpt-4o",
        )
        .unwrap();

        match response {
            RouterResponse::Buffered {
                status,
                content_type,
                body,
            } => {
                let sse = String::from_utf8(body).unwrap();
                assert_eq!(status, 200);
                assert_eq!(content_type, "text/event-stream");
                assert!(sse.contains("event: response.created"));
                assert!(sse.contains("\"delta\":\"Hel\""));
                assert!(sse.contains("event: response.completed"));
            }
            RouterResponse::Streaming { .. } => panic!("expected buffered SSE"),
        }
    }

    #[test]
    fn convert_chat_response_for_responses_route_rejects_streaming_non_stream_requests() {
        let response = convert_chat_response_for_responses_route(
            RouterResponse::Streaming {
                status: 200,
                content_type: "text/event-stream".to_string(),
                body: Box::new(StreamingBody::Upstream(mock_reqwest_response(
                    200,
                    "text/event-stream",
                    "data: [DONE]\n\n",
                ))),
            },
            false,
            "gpt-4o",
        );

        assert!(response.is_err());
    }

    #[test]
    fn responses_router_config_uses_active_protocol() {
        let state = test_state(ProviderProtocol::Google);
        let config = responses_router_config(&state);

        assert_eq!(config.target_protocol, ProviderProtocol::Google);
        assert_eq!(config.target_base_url, "https://example.com/v1");
        assert_eq!(config.api_key, "secret");
    }

    #[test]
    fn upstream_context_copies_router_flags() {
        let state = ServeState {
            config: Arc::new(ServeRouterConfig {
                upstream_base_url: "https://openrouter.ai/api/v1".to_string(),
                upstream_api_key: "secret".to_string(),
                upstream_protocol: ProviderProtocol::Openai,
                is_copilot: false,
                is_openrouter: true,
            }),
            client: router_http_client(),
            key: test_key(),
            copilot_tokens: None,
            active_protocol: Arc::new(AtomicU8::new(ProviderProtocol::Openai.to_u8())),
        };

        let context = upstream_context(&state);
        assert!(context.is_openrouter);
        assert!(!context.is_copilot);
        assert_eq!(context.upstream_base_url, "https://openrouter.ai/api/v1");
    }

    #[test]
    fn format_http_chunk_adds_hex_prefix_and_trailer() {
        assert_eq!(format_http_chunk(b"hello"), b"5\r\nhello\r\n");
        assert!(format_http_chunk(b"").is_empty());
    }
}
