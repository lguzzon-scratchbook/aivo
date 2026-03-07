//! CopilotRouter: HTTP proxy for routing Claude Code requests through GitHub Copilot.
//!
//! Receives Anthropic Messages API requests from Claude Code, converts them to
//! OpenAI Chat Completions format, forwards to the Copilot API, and converts
//! the response back to Anthropic format.

use anyhow::Result;
use serde_json::{Value, json};
use std::sync::Arc;

use crate::services::anthropic_chat_request::{
    AnthropicToOpenAIConfig, convert_anthropic_to_openai_request,
};
use crate::services::anthropic_chat_response::{
    OpenAIToAnthropicConfig, UsageValueMode, convert_openai_to_anthropic_message,
};
use crate::services::copilot_auth::{
    COPILOT_EDITOR_VERSION, COPILOT_INTEGRATION_ID, COPILOT_OPENAI_INTENT, CopilotTokenManager,
};
use crate::services::http_utils;
use crate::services::model_names;

#[derive(Clone)]
pub struct CopilotRouterConfig {
    pub github_token: String,
}

pub struct CopilotRouter {
    config: CopilotRouterConfig,
}

struct CopilotRouterState {
    token_manager: Arc<CopilotTokenManager>,
    client: reqwest::Client,
}

impl CopilotRouter {
    pub fn new(config: CopilotRouterConfig) -> Self {
        Self { config }
    }

    pub async fn start_background(&self) -> Result<(u16, tokio::task::JoinHandle<Result<()>>)> {
        let (listener, port) = http_utils::bind_local_listener().await?;
        let state = CopilotRouterState {
            token_manager: Arc::new(CopilotTokenManager::new(self.config.github_token.clone())),
            client: http_utils::router_http_client(),
        };
        let handle = tokio::spawn(async move {
            http_utils::run_text_router(listener, Arc::new(state), handle_copilot_request).await
        });
        Ok((port, handle))
    }
}

async fn handle_copilot_request(request: String, state: Arc<CopilotRouterState>) -> String {
    if http_utils::is_post_path(&request, &["/v1/messages", "/messages"]) {
        match handle_messages(&request, &state.token_manager, &state.client).await {
            Ok(r) => r,
            Err(e) => http_utils::http_error_response(500, &e.to_string()),
        }
    } else {
        http_utils::http_error_response(404, "Not found")
    }
}

async fn handle_messages(
    request: &str,
    tm: &Arc<CopilotTokenManager>,
    client: &reqwest::Client,
) -> Result<String> {
    let body_str = http_utils::extract_request_body(request)?;
    let body: Value = serde_json::from_str(body_str)?;

    let is_streaming = body
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);
    let model = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("claude-sonnet-4-20250514")
        .to_string();

    // Convert Anthropic Messages → OpenAI Chat Completions
    let openai_req = anthropic_to_openai(&body);

    // Get a valid Copilot token
    let (copilot_token, api_endpoint) = tm.get_token().await?;

    // Forward to Copilot API
    let url = format!("{}/chat/completions", api_endpoint.trim_end_matches('/'));

    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", copilot_token))
        .header("Content-Type", "application/json")
        .header("Editor-Version", COPILOT_EDITOR_VERSION)
        .header("Copilot-Integration-Id", COPILOT_INTEGRATION_ID)
        .header("Openai-Intent", COPILOT_OPENAI_INTENT)
        .json(&openai_req)
        .send()
        .await?;

    let status = resp.status().as_u16();
    let resp_body = resp.text().await?;

    if status != 200 {
        return Ok(http_utils::http_error_response(status, &resp_body));
    }

    let openai_resp: Value = serde_json::from_str(&resp_body)?;

    if is_streaming {
        // Convert to Anthropic SSE format
        let sse = openai_to_anthropic_sse(&openai_resp, &model);
        Ok(http_utils::http_response(200, "text/event-stream", &sse))
    } else {
        // Convert to Anthropic Messages response
        let anthropic_resp = openai_to_anthropic(&openai_resp, &model);
        let json = serde_json::to_string(&anthropic_resp)?;
        Ok(http_utils::http_json_response(200, &json))
    }
}

// --- Model name mapping ---

/// Re-export for tests and internal use.
fn copilot_model_name(model: &str) -> String {
    model_names::copilot_model_name(model)
}

// --- Request conversion: Anthropic Messages → OpenAI Chat Completions ---

fn anthropic_to_openai(body: &Value) -> Value {
    convert_anthropic_to_openai_request(
        body,
        &AnthropicToOpenAIConfig {
            default_model: "claude-sonnet-4-20250514",
            preserve_stream: false,
            model_transform: Some(copilot_model_name),
            include_reasoning_content: false,
            require_non_empty_reasoning_content: false,
            stringify_other_tool_result_content: false,
            fallback_tool_arguments_json: "",
        },
    )
}

// --- Response conversion: OpenAI Chat Completions → Anthropic Messages ---

fn openai_to_anthropic(resp: &Value, model: &str) -> Value {
    convert_openai_to_anthropic_message(
        resp,
        &OpenAIToAnthropicConfig {
            fallback_id: "msg_copilot",
            model,
            include_created: false,
            usage_value_mode: UsageValueMode::PreserveJson,
        },
    )
}

/// Converts an OpenAI response to Anthropic SSE event stream.
fn openai_to_anthropic_sse(resp: &Value, model: &str) -> String {
    let anthropic = openai_to_anthropic(resp, model);
    let mut events = String::new();

    let input_tokens = anthropic["usage"]["input_tokens"].as_i64().unwrap_or(0);
    let output_tokens = anthropic["usage"]["output_tokens"].as_i64().unwrap_or(0);

    // message_start
    events.push_str(&format!(
        "event: message_start\ndata: {}\n\n",
        json!({
            "type": "message_start",
            "message": {
                "id": anthropic["id"],
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0}
            }
        })
    ));

    // Emit each content block
    if let Some(content) = anthropic.get("content").and_then(|c| c.as_array()) {
        for (idx, block) in content.iter().enumerate() {
            let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("text");

            match block_type {
                "text" => {
                    let text = block.get("text").and_then(|t| t.as_str()).unwrap_or("");
                    // content_block_start
                    events.push_str(&format!(
                        "event: content_block_start\ndata: {}\n\n",
                        json!({"type": "content_block_start", "index": idx, "content_block": {"type": "text", "text": ""}})
                    ));
                    // content_block_delta
                    if !text.is_empty() {
                        events.push_str(&format!(
                            "event: content_block_delta\ndata: {}\n\n",
                            json!({"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": text}})
                        ));
                    }
                    // content_block_stop
                    events.push_str(&format!(
                        "event: content_block_stop\ndata: {}\n\n",
                        json!({"type": "content_block_stop", "index": idx})
                    ));
                }
                "tool_use" => {
                    // content_block_start
                    events.push_str(&format!(
                        "event: content_block_start\ndata: {}\n\n",
                        json!({
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {
                                "type": "tool_use",
                                "id": block["id"],
                                "name": block["name"],
                                "input": {}
                            }
                        })
                    ));
                    // content_block_delta with input_json_delta
                    let input_str = serde_json::to_string(&block["input"]).unwrap_or_default();
                    if input_str != "{}" {
                        events.push_str(&format!(
                            "event: content_block_delta\ndata: {}\n\n",
                            json!({"type": "content_block_delta", "index": idx, "delta": {"type": "input_json_delta", "partial_json": input_str}})
                        ));
                    }
                    // content_block_stop
                    events.push_str(&format!(
                        "event: content_block_stop\ndata: {}\n\n",
                        json!({"type": "content_block_stop", "index": idx})
                    ));
                }
                _ => {}
            }
        }
    }

    // message_delta
    events.push_str(&format!(
        "event: message_delta\ndata: {}\n\n",
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": anthropic["stop_reason"], "stop_sequence": null},
            "usage": {"output_tokens": output_tokens}
        })
    ));

    // message_stop
    events.push_str("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");

    events
}

// HTTP utilities are now provided by crate::services::http_utils

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_copilot_model_name_strips_date_and_converts_dots() {
        assert_eq!(
            copilot_model_name("claude-sonnet-4-20250514"),
            "claude-sonnet-4"
        );
        assert_eq!(
            copilot_model_name("claude-sonnet-4-6-20250603"),
            "claude-sonnet-4.6"
        );
        assert_eq!(
            copilot_model_name("claude-opus-4-6-20250210"),
            "claude-opus-4.6"
        );
        assert_eq!(
            copilot_model_name("claude-haiku-4-5-20250501"),
            "claude-haiku-4.5"
        );
    }

    #[test]
    fn test_copilot_model_name_converts_dots() {
        assert_eq!(copilot_model_name("claude-sonnet-4"), "claude-sonnet-4");
        assert_eq!(copilot_model_name("claude-sonnet-4-6"), "claude-sonnet-4.6");
        assert_eq!(copilot_model_name("claude-haiku-4-5"), "claude-haiku-4.5");
        assert_eq!(copilot_model_name("claude-opus-4-5"), "claude-opus-4.5");
        assert_eq!(copilot_model_name("gpt-4o"), "gpt-4o");
    }

    #[test]
    fn test_anthropic_to_openai_basic() {
        let body = json!({
            "model": "claude-sonnet-4",
            "max_tokens": 1024,
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "How are you?"}
            ]
        });
        let result = anthropic_to_openai(&body);
        let messages = result["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 4); // system + 3 messages
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are helpful.");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "Hello");
        assert_eq!(messages[2]["role"], "assistant");
        assert_eq!(messages[2]["content"], "Hi!");
        assert_eq!(result["max_tokens"], 1024);
        assert_eq!(result["stream"], false);
    }

    #[test]
    fn test_anthropic_to_openai_system_array() {
        let body = json!({
            "model": "claude-sonnet-4",
            "max_tokens": 1024,
            "system": [{"type": "text", "text": "System prompt."}],
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let result = anthropic_to_openai(&body);
        let messages = result["messages"].as_array().unwrap();
        assert_eq!(messages[0]["content"], "System prompt.");
    }

    #[test]
    fn test_anthropic_to_openai_tool_use() {
        let body = json!({
            "model": "claude-sonnet-4",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"location": "SF"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Sunny, 72°F"}
                ]}
            ]
        });
        let result = anthropic_to_openai(&body);
        let messages = result["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        // Assistant message with tool calls
        assert_eq!(messages[1]["tool_calls"][0]["id"], "toolu_1");
        assert_eq!(
            messages[1]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert_eq!(messages[1]["content"], "Let me check.");
        // Tool result
        assert_eq!(messages[2]["role"], "tool");
        assert_eq!(messages[2]["tool_call_id"], "toolu_1");
        assert_eq!(messages[2]["content"], "Sunny, 72°F");
    }

    #[test]
    fn test_anthropic_to_openai_tools() {
        let body = json!({
            "model": "claude-sonnet-4",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather info",
                "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}}
            }]
        });
        let result = anthropic_to_openai(&body);
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "get_weather");
        assert_eq!(tools[0]["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn test_anthropic_to_openai_stop_sequences() {
        let body = json!({
            "model": "claude-sonnet-4",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
            "stop_sequences": ["\n\nHuman:"]
        });
        let result = anthropic_to_openai(&body);
        assert_eq!(result["stop"][0], "\n\nHuman:");
    }

    #[test]
    fn test_openai_to_anthropic_uses_requested_model_and_preserves_usage_shape() {
        let resp = json!({
            "id": "chatcmpl-xxx",
            "model": "ignored-provider-model",
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": "5", "completion_tokens": 3, "total_tokens": 8}
        });
        let result = openai_to_anthropic(&resp, "claude-sonnet-4.6");
        assert_eq!(result["id"], "chatcmpl-xxx");
        assert_eq!(result["model"], "claude-sonnet-4.6");
        assert_eq!(result["content"][0]["text"], "Hello!");
        assert_eq!(result["usage"]["input_tokens"], "5");
        assert_eq!(result["usage"]["output_tokens"], 3);
    }

    #[test]
    fn test_openai_to_anthropic_sse() {
        let resp = json!({
            "id": "chatcmpl-xxx",
            "choices": [{"message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2}
        });
        let sse = openai_to_anthropic_sse(&resp, "claude-sonnet-4");
        assert!(sse.contains("event: message_start"));
        assert!(sse.contains("event: content_block_start"));
        assert!(sse.contains("event: content_block_delta"));
        assert!(sse.contains("\"text\":\"Hi!\""));
        assert!(sse.contains("event: content_block_stop"));
        assert!(sse.contains("event: message_delta"));
        assert!(sse.contains("event: message_stop"));
    }

    #[test]
    fn test_openai_to_anthropic_sse_tool_use() {
        let resp = json!({
            "id": "chatcmpl-xxx",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{\"path\":\"test.rs\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        });
        let sse = openai_to_anthropic_sse(&resp, "claude-sonnet-4");
        assert!(sse.contains("\"type\":\"tool_use\""));
        assert!(sse.contains("\"name\":\"read_file\""));
        assert!(sse.contains("input_json_delta"));
    }

    #[test]
    fn test_extract_body() {
        let req =
            "POST /v1/messages HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"key\":\"val\"}";
        assert_eq!(
            http_utils::extract_request_body(req).unwrap(),
            "{\"key\":\"val\"}"
        );
    }

    #[test]
    fn test_openai_to_anthropic_sse_multi_choice() {
        // SSE should also handle multi-choice correctly
        let resp = json!({
            "id": "chatcmpl-xxx",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {"content": "Checking...", "role": "assistant"}
                },
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "exec", "arguments": "{\"cmd\":\"ls\"}"}
                        }]
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        });
        let sse = openai_to_anthropic_sse(&resp, "claude-sonnet-4.6");
        assert!(sse.contains("Checking..."));
        assert!(sse.contains("\"type\":\"tool_use\""));
        assert!(sse.contains("\"name\":\"exec\""));
        assert!(sse.contains("\"stop_reason\":\"tool_use\""));
    }

    #[test]
    fn test_extract_body_missing_separator() {
        assert!(http_utils::extract_request_body("POST /v1/messages HTTP/1.1").is_err());
    }

    #[test]
    fn test_error_response() {
        let resp = http_utils::http_error_response(500, "test error");
        assert!(resp.contains("500"));
        assert!(resp.contains("test error"));
    }

    #[test]
    fn test_tool_choice_auto() {
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "test", "description": "test", "input_schema": {}}],
            "tool_choice": {"type": "auto"}
        });
        let req = anthropic_to_openai(&body);
        assert_eq!(req["tool_choice"], json!("auto"));
    }

    #[test]
    fn test_tool_choice_any() {
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "test", "description": "test", "input_schema": {}}],
            "tool_choice": {"type": "any"}
        });
        let req = anthropic_to_openai(&body);
        assert_eq!(req["tool_choice"], json!("required"));
    }

    #[test]
    fn test_tool_choice_specific_tool() {
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "read_file", "description": "read", "input_schema": {}}],
            "tool_choice": {"type": "tool", "name": "read_file"}
        });
        let req = anthropic_to_openai(&body);
        assert_eq!(
            req["tool_choice"],
            json!({"type": "function", "function": {"name": "read_file"}})
        );
    }

    #[test]
    fn test_tool_choice_not_present() {
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
        });
        let req = anthropic_to_openai(&body);
        assert!(req.get("tool_choice").is_none());
    }
}
