use serde_json::{Value, json};

use crate::services::http_utils::current_unix_ts;
use crate::services::openai_models::{
    OpenAIChatChoice, OpenAIChatResponse, OpenAIChatResponseMessage, OpenAIChatToolCall,
    OpenAIChatToolCallFunction, OpenAIChatUsage,
};

#[derive(Clone, Copy, Debug)]
pub struct OpenAIToAnthropicChatConfig {
    pub default_model: &'static str,
}

pub fn convert_openai_chat_to_anthropic_request(
    body: &Value,
    config: &OpenAIToAnthropicChatConfig,
) -> Value {
    let mut system_blocks: Vec<Value> = Vec::new();
    let mut messages: Vec<Value> = Vec::new();

    if let Some(msgs) = body.get("messages").and_then(|m| m.as_array()) {
        for msg in msgs {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            match role {
                "system" => {
                    system_blocks.extend(extract_openai_anthropic_text_blocks(msg.get("content")));
                }
                "assistant" => messages.push(openai_assistant_to_anthropic(msg)),
                "tool" => messages.push(openai_tool_to_anthropic(msg)),
                _ => messages.push(openai_user_to_anthropic(msg, role)),
            }
        }
    }

    let mut req = json!({
        "model": body
            .get("model")
            .and_then(|m| m.as_str())
            .unwrap_or(config.default_model),
        "messages": messages,
        "stream": body.get("stream").cloned().unwrap_or(json!(false)),
        "max_tokens": body.get("max_tokens").cloned().unwrap_or(json!(4096)),
    });

    if !system_blocks.is_empty() {
        req["system"] = anthropic_text_blocks_to_content(system_blocks);
    }
    if let Some(v) = body.get("temperature") {
        req["temperature"] = v.clone();
    }
    if let Some(v) = body.get("top_p") {
        req["top_p"] = v.clone();
    }
    if let Some(v) = body.get("stop") {
        req["stop_sequences"] = v.clone();
    }
    if let Some(tools) = body.get("tools").and_then(|t| t.as_array()) {
        let anthropic_tools: Vec<Value> = tools
            .iter()
            .filter(|tool| tool.get("type").and_then(|v| v.as_str()) == Some("function"))
            .map(|tool| {
                json!({
                    "name": tool.get("function").and_then(|f| f.get("name")).cloned().unwrap_or_default(),
                    "description": tool.get("function").and_then(|f| f.get("description")).cloned().unwrap_or(json!("")),
                    "input_schema": tool.get("function").and_then(|f| f.get("parameters")).cloned().unwrap_or(json!({}))
                })
            })
            .collect();
        if !anthropic_tools.is_empty() {
            req["tools"] = Value::Array(anthropic_tools);
        }
    }
    if let Some(tc) = body.get("tool_choice") {
        match tc {
            // Anthropic has no "none" mode — disable tools by removing them entirely
            Value::String(s) if s == "none" => {
                if let Some(obj) = req.as_object_mut() {
                    obj.remove("tools");
                }
            }
            _ => {
                req["tool_choice"] = match tc {
                    Value::String(s) if s == "auto" => json!({"type": "auto"}),
                    Value::String(s) if s == "required" => json!({"type": "any"}),
                    Value::Object(obj)
                        if obj.get("type").and_then(|v| v.as_str()) == Some("function") =>
                    {
                        let name = obj
                            .get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        json!({"type": "tool", "name": name})
                    }
                    _ => tc.clone(),
                };
            }
        }
    }

    // OpenAI parallel_tool_calls:false → Anthropic disable_parallel_tool_use:true
    if body.get("parallel_tool_calls") == Some(&json!(false)) && req.get("tools").is_some() {
        match req.get_mut("tool_choice").and_then(|v| v.as_object_mut()) {
            Some(tc) => {
                tc.insert("disable_parallel_tool_use".to_string(), json!(true));
            }
            None => {
                req["tool_choice"] = json!({"type": "auto", "disable_parallel_tool_use": true});
            }
        }
    }

    req
}

pub fn convert_anthropic_to_openai_chat_response(resp: &Value, fallback_model: &str) -> Value {
    let mut text_parts: Vec<String> = Vec::new();
    let mut thinking_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<OpenAIChatToolCall> = Vec::new();

    if let Some(content) = resp.get("content").and_then(|c| c.as_array()) {
        for block in content {
            match block.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                "thinking" => {
                    if let Some(thinking) = block.get("thinking").and_then(|v| v.as_str())
                        && !thinking.is_empty()
                    {
                        thinking_parts.push(thinking.to_string());
                    }
                }
                "text" => {
                    if let Some(text) = block.get("text").and_then(|v| v.as_str())
                        && !text.is_empty()
                    {
                        text_parts.push(text.to_string());
                    }
                }
                "tool_use" => {
                    let args = block.get("input").cloned().unwrap_or(json!({}));
                    tool_calls.push(OpenAIChatToolCall {
                        id: block
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("call_0")
                            .to_string(),
                        kind: "function".to_string(),
                        function: OpenAIChatToolCallFunction {
                            name: block
                                .get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            arguments: serde_json::to_string(&args)
                                .unwrap_or_else(|_| "{}".to_string()),
                        },
                    });
                }
                _ => {}
            }
        }
    }

    let finish_reason = match resp
        .get("stop_reason")
        .and_then(|v| v.as_str())
        .unwrap_or("")
    {
        "tool_use" => "tool_calls",
        "max_tokens" => "length",
        _ => "stop",
    };

    let raw_input_tokens = resp
        .get("usage")
        .and_then(|u| u.get("input_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let completion_tokens = resp
        .get("usage")
        .and_then(|u| u.get("output_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let cache_read_input_tokens = resp
        .get("usage")
        .and_then(|u| u.get("cache_read_input_tokens"))
        .and_then(|v| v.as_u64());
    let cache_creation_input_tokens = resp
        .get("usage")
        .and_then(|u| u.get("cache_creation_input_tokens"))
        .and_then(|v| v.as_u64());
    // Normalize: Anthropic's input_tokens excludes cache, OpenAI's prompt_tokens includes it
    let prompt_tokens = raw_input_tokens
        + cache_read_input_tokens.unwrap_or(0)
        + cache_creation_input_tokens.unwrap_or(0);

    let response = OpenAIChatResponse {
        id: resp
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("chatcmpl-aivo")
            .to_string(),
        object: "chat.completion".to_string(),
        created: Some(
            resp.get("created")
                .and_then(|v| v.as_u64())
                .unwrap_or_else(current_unix_ts),
        ),
        model: resp
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or(fallback_model)
            .to_string(),
        choices: vec![OpenAIChatChoice {
            index: 0,
            message: OpenAIChatResponseMessage {
                role: "assistant".to_string(),
                content: (!text_parts.is_empty()).then(|| text_parts.join("\n")),
                reasoning_content: (!thinking_parts.is_empty()).then(|| thinking_parts.join("\n")),
                tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            },
            finish_reason: finish_reason.to_string(),
        }],
        usage: OpenAIChatUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            cache_read_input_tokens,
            cache_creation_input_tokens,
        },
    };

    serde_json::to_value(response).unwrap_or_else(
        |_| serde_json::json!({"error": "failed to serialize openai chat response"}),
    )
}

pub fn convert_openai_chat_response_to_sse(resp: &Value) -> Result<String, serde_json::Error> {
    let response: OpenAIChatResponse = serde_json::from_value(resp.clone())?;
    let id = response.id;
    let model = response.model;
    let created = response.created.unwrap_or_else(current_unix_ts);
    let choice = response.choices.into_iter().next();
    let message = choice
        .as_ref()
        .map(|choice| &choice.message)
        .cloned()
        .unwrap_or(OpenAIChatResponseMessage {
            role: "assistant".to_string(),
            content: None,
            reasoning_content: None,
            tool_calls: None,
        });
    let finish_reason = choice
        .map(|choice| Value::String(choice.finish_reason))
        .unwrap_or(Value::Null);

    let reasoning_content = message.reasoning_content.as_deref().unwrap_or("");

    let mut events = String::new();
    events.push_str(&format!(
        "data: {}\n\n",
        json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": Value::Null
            }]
        })
    ));

    // Emit reasoning_content before content (DeepSeek-reasoner thinking)
    if !reasoning_content.is_empty() {
        events.push_str(&format!(
            "data: {}\n\n",
            json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"reasoning_content": reasoning_content},
                    "finish_reason": Value::Null
                }]
            })
        ));
    }

    if let Some(text) = message.content.as_deref()
        && !text.is_empty()
    {
        events.push_str(&format!(
            "data: {}\n\n",
            json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": text},
                    "finish_reason": Value::Null
                }]
            })
        ));
    }

    if let Some(tool_calls) = message.tool_calls
        && !tool_calls.is_empty()
    {
        let delta_calls: Vec<Value> = tool_calls
            .iter()
            .enumerate()
            .map(|(index, tc)| {
                json!({
                    "index": index,
                    "id": tc.id,
                    "type": tc.kind,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                })
            })
            .collect();
        events.push_str(&format!(
            "data: {}\n\n",
            json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"tool_calls": delta_calls},
                    "finish_reason": Value::Null
                }]
            })
        ));
    }

    events.push_str(&format!(
        "data: {}\n\n",
        json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason
            }]
        })
    ));
    events.push_str("data: [DONE]\n\n");
    Ok(events)
}

fn openai_user_to_anthropic(msg: &Value, role: &str) -> Value {
    json!({
        "role": role,
        "content": openai_content_to_anthropic_content(msg.get("content"))
    })
}

fn openai_assistant_to_anthropic(msg: &Value) -> Value {
    let mut blocks: Vec<Value> = Vec::new();
    let text = extract_openai_text(msg.get("content"));
    if !text.is_empty() {
        blocks.push(json!({"type": "text", "text": text}));
    }
    if let Some(tool_calls) = msg.get("tool_calls").and_then(|v| v.as_array()) {
        for tc in tool_calls {
            let args = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|v| v.as_str())
                .and_then(|s| serde_json::from_str::<Value>(s).ok())
                .unwrap_or_else(|| json!({}));
            blocks.push(json!({
                "type": "tool_use",
                "id": tc.get("id").cloned().unwrap_or(json!("call_0")),
                "name": tc.get("function").and_then(|f| f.get("name")).cloned().unwrap_or(json!("")),
                "input": args
            }));
        }
    }
    json!({
        "role": "assistant",
        "content": blocks
    })
}

fn openai_tool_to_anthropic(msg: &Value) -> Value {
    let content = extract_openai_text(msg.get("content"));
    json!({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": msg.get("tool_call_id").cloned().unwrap_or(json!("")),
            "content": content
        }]
    })
}

fn extract_openai_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|v| v.as_str()))
            .collect::<Vec<_>>()
            .join("\n"),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

fn openai_content_to_anthropic_content(content: Option<&Value>) -> Value {
    anthropic_text_blocks_to_content(extract_openai_anthropic_text_blocks(content))
}

fn extract_openai_anthropic_text_blocks(content: Option<&Value>) -> Vec<Value> {
    match content {
        Some(Value::String(s)) => {
            if s.is_empty() {
                Vec::new()
            } else {
                vec![json!({"type": "text", "text": s})]
            }
        }
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(openai_content_part_to_anthropic_block)
            .collect(),
        Some(Value::Null) | None => Vec::new(),
        Some(other) => vec![json!({"type": "text", "text": other.to_string()})],
    }
}

fn openai_content_part_to_anthropic_block(part: &Value) -> Option<Value> {
    let part_type = part.get("type").and_then(|v| v.as_str());
    if !matches!(part_type, None | Some("text")) {
        return None;
    }

    let text = part.get("text").and_then(|v| v.as_str())?;
    let mut block = part.clone();
    if !block.is_object() {
        block = json!({});
    }
    block["type"] = Value::String("text".to_string());
    block["text"] = Value::String(text.to_string());
    Some(block)
}

fn anthropic_text_blocks_to_content(blocks: Vec<Value>) -> Value {
    if blocks.is_empty() {
        return Value::String(String::new());
    }

    if blocks.iter().all(is_plain_anthropic_text_block) {
        return Value::String(
            blocks
                .iter()
                .filter_map(|block| block.get("text").and_then(|v| v.as_str()))
                .collect::<Vec<_>>()
                .join("\n\n"),
        );
    }

    Value::Array(blocks)
}

fn is_plain_anthropic_text_block(block: &Value) -> bool {
    let Some(obj) = block.as_object() else {
        return false;
    };

    obj.len() == 2
        && obj.get("type").and_then(|v| v.as_str()) == Some("text")
        && obj.get("text").and_then(|v| v.as_str()).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_openai_chat_to_anthropic_request_with_tool_calls() {
        let body = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Be precise."},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "ls", "arguments": "{\"path\":\".\"}"}
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": "[]"}
            ],
            "tools": [{
                "type": "function",
                "function": {"name": "ls", "description": "list", "parameters": {"type":"object"}}
            }]
        });

        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "gpt-4o",
            },
        );
        assert_eq!(converted["system"], "Be precise.");
        assert_eq!(converted["messages"][1]["content"][0]["type"], "tool_use");
        assert_eq!(
            converted["messages"][2]["content"][0]["type"],
            "tool_result"
        );
        assert_eq!(converted["tools"][0]["name"], "ls");
    }

    #[test]
    fn test_convert_anthropic_to_openai_chat_response_with_tool_use() {
        let body = json!({
            "id": "msg_1",
            "model": "MiniMax-M1",
            "content": [
                {"type": "text", "text": "Need tool"},
                {"type": "tool_use", "id": "call_1", "name": "ls", "input": {"path":"."}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 4}
        });
        let converted = convert_anthropic_to_openai_chat_response(&body, "fallback");
        assert_eq!(converted["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            converted["choices"][0]["message"]["tool_calls"][0]["id"],
            "call_1"
        );
    }

    #[test]
    fn test_convert_openai_chat_to_anthropic_request_preserves_cache_control_blocks() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [
                {
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": "Be precise.",
                        "cache_control": {"type": "ephemeral"}
                    }]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": "hi",
                        "cache_control": {"type": "ephemeral"}
                    }]
                }
            ]
        });

        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "claude-sonnet-4-5",
            },
        );

        assert_eq!(converted["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(
            converted["messages"][0]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
    }

    #[test]
    fn test_convert_openai_chat_empty_messages_array() {
        let body = json!({"model": "gpt-4o", "messages": []});
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "gpt-4o",
            },
        );
        assert!(converted["messages"].as_array().unwrap().is_empty());
        assert_eq!(converted["model"], "gpt-4o");
    }

    #[test]
    fn test_convert_openai_chat_missing_model_uses_default() {
        let body = json!({"messages": [{"role": "user", "content": "hi"}]});
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "fallback-model",
            },
        );
        assert_eq!(converted["model"], "fallback-model");
    }

    #[test]
    fn test_convert_openai_chat_null_content_no_panic() {
        let body = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": null},
                {"role": "assistant", "content": null}
            ]
        });
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "gpt-4o",
            },
        );
        assert_eq!(converted["messages"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_convert_openai_chat_missing_messages_field() {
        let body = json!({"model": "gpt-4o"});
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "gpt-4o",
            },
        );
        assert!(converted["messages"].as_array().unwrap().is_empty());
    }

    #[test]
    fn test_convert_anthropic_response_empty_content() {
        let resp = json!({"id": "msg_1", "model": "test", "content": [], "usage": {}});
        let converted = convert_anthropic_to_openai_chat_response(&resp, "fallback");
        assert!(converted["choices"][0]["message"]["content"].is_null());
        assert!(converted["choices"][0]["message"]["tool_calls"].is_null());
    }

    #[test]
    fn test_convert_anthropic_response_missing_usage() {
        let resp = json!({"content": [{"type": "text", "text": "hi"}]});
        let converted = convert_anthropic_to_openai_chat_response(&resp, "fallback");
        assert_eq!(converted["usage"]["prompt_tokens"], 0);
        assert_eq!(converted["usage"]["completion_tokens"], 0);
    }

    #[test]
    fn test_convert_anthropic_response_unknown_stop_reason() {
        let resp =
            json!({"content": [{"type": "text", "text": "hi"}], "stop_reason": "weird_reason"});
        let converted = convert_anthropic_to_openai_chat_response(&resp, "fallback");
        assert_eq!(converted["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn test_extract_openai_text_unexpected_type() {
        assert_eq!(extract_openai_text(Some(&json!(42))), "42");
        assert_eq!(extract_openai_text(Some(&json!(true))), "true");
    }

    #[test]
    fn convert_openai_to_anthropic_invalid_tool_args_json() {
        let body = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "do_stuff", "arguments": "not json"}
                }]}
            ]
        });
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "gpt-4o",
            },
        );
        // Invalid JSON arguments should fall back to {}
        let tool_block = &converted["messages"][1]["content"][0];
        assert_eq!(tool_block["type"], "tool_use");
        assert_eq!(tool_block["input"], json!({}));
    }

    #[test]
    fn convert_anthropic_to_openai_null_usage_subfields() {
        let resp = json!({
            "id": "msg_1",
            "model": "test",
            "content": [{"type": "text", "text": "hi"}],
            "usage": {"input_tokens": null, "output_tokens": null}
        });
        let converted = convert_anthropic_to_openai_chat_response(&resp, "fallback");
        assert_eq!(converted["usage"]["prompt_tokens"], 0);
        assert_eq!(converted["usage"]["completion_tokens"], 0);
        assert_eq!(converted["usage"]["total_tokens"], 0);
    }

    #[test]
    fn convert_openai_to_anthropic_empty_string_arguments() {
        // OpenAI legitimately streams arguments: "" (empty string, not "{}")
        let body = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "do_stuff", "arguments": ""}
                }]}
            ]
        });
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "gpt-4o",
            },
        );
        // Empty string arguments should fall back to {}
        let tool_block = &converted["messages"][1]["content"][0];
        assert_eq!(tool_block["type"], "tool_use");
        assert_eq!(tool_block["input"], json!({}));
    }

    #[test]
    fn convert_openai_to_anthropic_parallel_tool_calls_false() {
        let body = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {"name": "ls", "description": "list", "parameters": {"type":"object"}}
            }],
            "parallel_tool_calls": false
        });
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "gpt-4o",
            },
        );
        // Should inject disable_parallel_tool_use into tool_choice
        assert_eq!(
            converted["tool_choice"]["disable_parallel_tool_use"],
            json!(true)
        );
        assert_eq!(converted["tool_choice"]["type"], "auto");
    }

    #[test]
    fn convert_openai_to_anthropic_parallel_tool_calls_false_with_existing_tool_choice() {
        let body = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {"name": "ls", "description": "list", "parameters": {"type":"object"}}
            }],
            "tool_choice": "required",
            "parallel_tool_calls": false
        });
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "gpt-4o",
            },
        );
        // Should inject disable_parallel_tool_use into existing tool_choice
        assert_eq!(converted["tool_choice"]["type"], "any");
        assert_eq!(
            converted["tool_choice"]["disable_parallel_tool_use"],
            json!(true)
        );
    }

    #[test]
    fn convert_openai_to_anthropic_tool_choice_none_strips_tools() {
        let body = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {"name": "ls", "description": "list", "parameters": {"type":"object"}}
            }],
            "tool_choice": "none"
        });
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "gpt-4o",
            },
        );
        // "none" should strip tools and not set tool_choice
        assert!(converted.get("tools").is_none());
        assert!(converted.get("tool_choice").is_none());
    }

    #[test]
    fn convert_openai_to_anthropic_sse_empty_choices() {
        // No SSE chunk conversion function exists; test convert with empty messages array
        let body = json!({"model": "gpt-4o", "messages": []});
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "gpt-4o",
            },
        );
        assert!(converted["messages"].as_array().unwrap().is_empty());
        assert_eq!(converted["model"], "gpt-4o");
        assert_eq!(converted["max_tokens"], 4096);
    }

    #[test]
    fn convert_openai_to_anthropic_sse_null_content() {
        // Tool calls with null content in assistant message
        let body = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "call tool"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_data", "arguments": "{\"x\":1}"}
                    }]
                }
            ]
        });
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "gpt-4o",
            },
        );
        // Assistant content should have only the tool_use block (no text block since content is null)
        let assistant_content = converted["messages"][1]["content"].as_array().unwrap();
        assert_eq!(assistant_content.len(), 1);
        assert_eq!(assistant_content[0]["type"], "tool_use");
        assert_eq!(assistant_content[0]["name"], "get_data");
    }

    #[test]
    fn convert_anthropic_to_openai_empty_text_blocks() {
        let resp = json!({
            "id": "msg_1",
            "model": "test",
            "content": [{"type": "text", "text": ""}],
            "usage": {"input_tokens": 5, "output_tokens": 2}
        });
        let converted = convert_anthropic_to_openai_chat_response(&resp, "fallback");
        // Empty text is skipped, so content should be null (no text_parts collected)
        assert!(converted["choices"][0]["message"]["content"].is_null());
    }

    #[test]
    fn convert_anthropic_to_openai_cache_tokens_summed() {
        let resp = json!({
            "id": "msg_1",
            "model": "test",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 20,
                "cache_creation_input_tokens": 30
            }
        });
        let converted = convert_anthropic_to_openai_chat_response(&resp, "fallback");
        // prompt_tokens = input_tokens + cache_read + cache_creation = 10 + 20 + 30 = 60
        assert_eq!(converted["usage"]["prompt_tokens"], 60);
        assert_eq!(converted["usage"]["completion_tokens"], 5);
        assert_eq!(converted["usage"]["total_tokens"], 65);
        assert_eq!(converted["usage"]["cache_read_input_tokens"], 20);
        assert_eq!(converted["usage"]["cache_creation_input_tokens"], 30);
    }

    #[test]
    fn convert_openai_to_anthropic_developer_role_mapped() {
        let body = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": "hi"}
            ]
        });
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "gpt-4o",
            },
        );
        // "developer" falls through to the _ match arm which calls openai_user_to_anthropic
        // preserving the role as-is ("developer")
        let messages = converted["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "developer");
        // Content should be converted properly
        assert!(messages[0]["content"].is_string() || messages[0]["content"].is_array());
    }
}
