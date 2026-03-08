use serde_json::{Value, json};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, Debug)]
pub struct OpenAIToAnthropicChatConfig {
    pub default_model: &'static str,
}

pub fn convert_openai_chat_to_anthropic_request(
    body: &Value,
    config: &OpenAIToAnthropicChatConfig,
) -> Value {
    let mut system_parts: Vec<String> = Vec::new();
    let mut messages: Vec<Value> = Vec::new();

    if let Some(msgs) = body.get("messages").and_then(|m| m.as_array()) {
        for msg in msgs {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            match role {
                "system" => {
                    let text = extract_openai_text(msg.get("content"));
                    if !text.is_empty() {
                        system_parts.push(text);
                    }
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

    if !system_parts.is_empty() {
        req["system"] = Value::String(system_parts.join("\n\n"));
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
        req["tool_choice"] = match tc {
            Value::String(s) if s == "auto" => json!({"type": "auto"}),
            Value::String(s) if s == "required" => json!({"type": "any"}),
            Value::Object(obj) if obj.get("type").and_then(|v| v.as_str()) == Some("function") => {
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

    req
}

pub fn convert_anthropic_to_openai_chat_response(resp: &Value, fallback_model: &str) -> Value {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();

    if let Some(content) = resp.get("content").and_then(|c| c.as_array()) {
        for block in content {
            match block.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                "text" => {
                    if let Some(text) = block.get("text").and_then(|v| v.as_str())
                        && !text.is_empty()
                    {
                        text_parts.push(text.to_string());
                    }
                }
                "tool_use" => {
                    let args = block.get("input").cloned().unwrap_or(json!({}));
                    tool_calls.push(json!({
                        "id": block.get("id").cloned().unwrap_or(json!("call_0")),
                        "type": "function",
                        "function": {
                            "name": block.get("name").cloned().unwrap_or(json!("")),
                            "arguments": serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string())
                        }
                    }));
                }
                _ => {}
            }
        }
    }

    let content = if text_parts.is_empty() {
        Value::Null
    } else {
        Value::String(text_parts.join("\n"))
    };

    let finish_reason = match resp
        .get("stop_reason")
        .and_then(|v| v.as_str())
        .unwrap_or("")
    {
        "tool_use" => "tool_calls",
        "max_tokens" => "length",
        _ => "stop",
    };

    let mut message = json!({
        "role": "assistant",
        "content": content
    });
    if message["content"].is_null() {
        message.as_object_mut().unwrap().remove("content");
    }
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls);
    }

    let prompt_tokens = resp
        .get("usage")
        .and_then(|u| u.get("input_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let completion_tokens = resp
        .get("usage")
        .and_then(|u| u.get("output_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    json!({
        "id": resp.get("id").cloned().unwrap_or(json!("chatcmpl-aivo")),
        "object": "chat.completion",
        "created": resp
            .get("created")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(current_unix_ts),
        "model": resp.get("model").and_then(|v| v.as_str()).unwrap_or(fallback_model),
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    })
}

pub fn convert_openai_chat_response_to_sse(resp: &Value) -> String {
    let id = resp.get("id").cloned().unwrap_or(json!("chatcmpl-aivo"));
    let model = resp.get("model").cloned().unwrap_or(json!("unknown"));
    let created = resp
        .get("created")
        .cloned()
        .unwrap_or_else(|| json!(current_unix_ts()));
    let choice = resp
        .get("choices")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .cloned()
        .unwrap_or(json!({}));
    let message = choice.get("message").cloned().unwrap_or(json!({}));
    let finish_reason = choice.get("finish_reason").cloned().unwrap_or(Value::Null);

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

    if let Some(text) = message.get("content").and_then(|v| v.as_str())
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

    if let Some(tool_calls) = message.get("tool_calls").and_then(|v| v.as_array())
        && !tool_calls.is_empty()
    {
        let delta_calls: Vec<Value> = tool_calls
            .iter()
            .enumerate()
            .map(|(index, tc)| {
                json!({
                    "index": index,
                    "id": tc.get("id").cloned().unwrap_or(json!("call_0")),
                    "type": "function",
                    "function": tc.get("function").cloned().unwrap_or(json!({}))
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
    events
}

fn openai_user_to_anthropic(msg: &Value, role: &str) -> Value {
    let text = extract_openai_text(msg.get("content"));
    json!({
        "role": role,
        "content": if text.is_empty() { Value::String(String::new()) } else { Value::String(text) }
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

fn current_unix_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
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
}
