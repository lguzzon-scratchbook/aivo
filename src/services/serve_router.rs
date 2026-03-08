//! Serve Router — exposes a local OpenAI-compatible HTTP API.
//!
//! Clients send OpenAI-format requests; this router transforms them to whatever
//! protocol the active upstream provider requires, forwards them, and returns
//! OpenAI-format responses.

use anyhow::Result;
use serde_json::{Value, json};
use std::sync::Arc;

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
    convert_gemini_to_openai_chat_response, convert_openai_chat_to_gemini_request,
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

impl ServeRouter {
    pub fn new(config: ServeRouterConfig, key: ApiKey) -> Self {
        Self { config, key }
    }

    /// Binds to the port eagerly (propagates "address already in use" immediately),
    /// then spawns the accept loop in the background and returns the join handle.
    pub async fn start_background(
        self,
        port: u16,
    ) -> Result<tokio::task::JoinHandle<Result<()>>> {
        let listener =
            tokio::net::TcpListener::bind(format!("127.0.0.1:{}", port)).await?;

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
                let path_no_query = path.split('?').next().unwrap_or(&path).to_string();

                let response_str = match path_no_query.as_str() {
                    "/v1/models" | "/models" => match handle_models(&state).await {
                        Ok(r) => r,
                        Err(e) => http_utils::http_error_response(500, &e.to_string()),
                    },
                    "/v1/chat/completions" => {
                        if !request.starts_with("POST ") {
                            http_utils::http_response(
                                405,
                                "application/json",
                                r#"{"error":{"message":"Method not allowed"}}"#,
                            )
                        } else {
                            match handle_chat(&request, &state).await {
                                Ok(r) => r,
                                Err(e) => http_utils::http_error_response(500, &e.to_string()),
                            }
                        }
                    }
                    _ => http_utils::http_response(
                        404,
                        "application/json",
                        r#"{"error":{"message":"Not found"}}"#,
                    ),
                };

                let _ = socket.write_all(response_str.as_bytes()).await;
            });
        }
}

async fn handle_models(state: &ServeState) -> Result<String> {
    let models = fetch_models(&state.client, &state.key).await?;
    let data: Vec<Value> = models
        .into_iter()
        .map(|id| json!({"id": id, "object": "model", "created": 0, "owned_by": "aivo"}))
        .collect();
    let resp = json!({"object": "list", "data": data});
    Ok(http_utils::http_response(
        200,
        "application/json",
        &resp.to_string(),
    ))
}

async fn handle_chat(request: &str, state: &ServeState) -> Result<String> {
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
        ProviderProtocol::Openai => handle_chat_openai(&mut body, state).await,
    }
}

async fn handle_chat_anthropic(
    body: &Value,
    client_wants_stream: bool,
    state: &ServeState,
) -> Result<String> {
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
    // Force non-streaming upstream; convert to SSE below if client requested it
    anthropic_req["stream"] = json!(false);

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
    let resp_body = response.text().await?;

    if status >= 400 {
        return Ok(http_utils::http_response(
            status,
            "application/json",
            &resp_body,
        ));
    }

    let anthropic_resp: Value = serde_json::from_str(&resp_body)?;
    let openai_resp = convert_anthropic_to_openai_chat_response(&anthropic_resp, &fallback_model);

    if client_wants_stream {
        let sse = convert_openai_chat_response_to_sse(&openai_resp);
        Ok(http_utils::http_response(200, "text/event-stream", &sse))
    } else {
        Ok(http_utils::http_response(
            200,
            "application/json",
            &openai_resp.to_string(),
        ))
    }
}

async fn handle_chat_gemini(
    body: &mut Value,
    client_wants_stream: bool,
    state: &ServeState,
) -> Result<String> {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("gemini-2.5-pro")
        .to_string();

    // Force non-streaming upstream; convert to SSE below if client requested it
    body["stream"] = json!(false);

    let gemini_req = convert_openai_chat_to_gemini_request(
        body,
        &OpenAIToGeminiConfig {
            default_model: "gemini-2.5-pro",
        },
    );

    let url = build_google_generate_content_url(&state.config.upstream_base_url, &model);
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
    let resp_body = response.text().await?;

    if status >= 400 {
        return Ok(http_utils::http_response(
            status,
            "application/json",
            &resp_body,
        ));
    }

    let gemini_resp: Value = serde_json::from_str(&resp_body)?;
    let openai_resp = convert_gemini_to_openai_chat_response(&gemini_resp, &model);

    if client_wants_stream {
        let sse = convert_openai_chat_response_to_sse(&openai_resp);
        Ok(http_utils::http_response(200, "text/event-stream", &sse))
    } else {
        Ok(http_utils::http_response(
            200,
            "application/json",
            &openai_resp.to_string(),
        ))
    }
}

async fn handle_chat_openai(body: &mut Value, state: &ServeState) -> Result<String> {
    // Apply model name normalization for specific providers
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
    let resp_body = response.text().await?;

    Ok(http_utils::http_response(status, &content_type, &resp_body))
}
