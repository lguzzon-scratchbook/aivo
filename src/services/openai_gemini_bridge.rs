use serde_json::{Value, json};
use std::collections::HashMap;

use crate::services::http_utils::current_unix_ts;

#[derive(Clone, Copy, Debug)]
pub struct OpenAIToGeminiConfig {
    pub default_model: &'static str,
}

pub fn openai_chat_model(body: &Value, default_model: &str) -> String {
    body.get("model")
        .and_then(|m| m.as_str())
        .unwrap_or(default_model)
        .to_string()
}

pub fn build_google_generate_content_url(base_url: &str, model: &str) -> String {
    build_google_content_url(base_url, model, false)
}

pub fn build_google_stream_generate_content_url(base_url: &str, model: &str) -> String {
    build_google_content_url(base_url, model, true)
}

fn build_google_content_url(base_url: &str, model: &str, stream: bool) -> String {
    let base = base_url.trim_end_matches('/');
    let suffix = if stream {
        ":streamGenerateContent?alt=sse"
    } else {
        ":generateContent"
    };
    if base.ends_with("/models") {
        format!("{}/{}{}", base, model, suffix)
    } else {
        format!("{}/models/{}{}", base, model, suffix)
    }
}

pub fn convert_openai_chat_to_gemini_request(body: &Value, config: &OpenAIToGeminiConfig) -> Value {
    let mut system_parts: Vec<String> = Vec::new();
    let mut contents: Vec<Value> = Vec::new();
    let mut tool_names_by_call_id: HashMap<String, String> = HashMap::new();

    if let Some(messages) = body.get("messages").and_then(|m| m.as_array()) {
        for message in messages {
            let role = message
                .get("role")
                .and_then(|r| r.as_str())
                .unwrap_or("user");

            match role {
                "system" => {
                    let text = extract_openai_text(message.get("content"));
                    if !text.is_empty() {
                        system_parts.push(text);
                    }
                }
                "assistant" => {
                    let mut parts: Vec<Value> = Vec::new();
                    let text = extract_openai_text(message.get("content"));
                    if !text.is_empty() {
                        parts.push(json!({ "text": text }));
                    }
                    if let Some(tool_calls) = message.get("tool_calls").and_then(|v| v.as_array()) {
                        for (index, tool_call) in tool_calls.iter().enumerate() {
                            let name = tool_call
                                .get("function")
                                .and_then(|f| f.get("name"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            let call_id = tool_call
                                .get("id")
                                .and_then(|v| v.as_str())
                                .filter(|id| !id.is_empty())
                                .map(ToOwned::to_owned)
                                .unwrap_or_else(|| format!("call_{index}"));
                            tool_names_by_call_id.insert(call_id.clone(), name.to_string());
                            let args = tool_call
                                .get("function")
                                .and_then(|f| f.get("arguments"))
                                .and_then(|v| v.as_str())
                                .and_then(|s| serde_json::from_str::<Value>(s).ok())
                                .unwrap_or_else(|| json!({}));
                            parts.push(json!({
                                "functionCall": {
                                    "id": call_id,
                                    "name": name,
                                    "args": args
                                }
                            }));
                        }
                    }
                    if !parts.is_empty() {
                        contents.push(json!({ "role": "model", "parts": parts }));
                    }
                }
                "tool" => {
                    let call_id = message
                        .get("tool_call_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let name = tool_names_by_call_id
                        .get(call_id)
                        .cloned()
                        .unwrap_or_default();
                    let response = parse_tool_content(message.get("content"));
                    contents.push(json!({
                        "role": "user",
                        "parts": [{
                            "functionResponse": {
                                "id": call_id,
                                "name": name,
                                "response": response
                            }
                        }]
                    }));
                }
                _ => {
                    let text = extract_openai_text(message.get("content"));
                    contents.push(json!({
                        "role": "user",
                        "parts": [{ "text": text }]
                    }));
                }
            }
        }
    }

    let mut request = json!({
        "contents": contents,
    });

    if !system_parts.is_empty() {
        request["systemInstruction"] = json!({
            "parts": [{
                "text": system_parts.join("\n\n")
            }]
        });
    }

    if let Some(tools) = body.get("tools").and_then(|t| t.as_array()) {
        let declarations: Vec<Value> = tools
            .iter()
            .filter(|tool| tool.get("type").and_then(|v| v.as_str()) == Some("function"))
            .map(|tool| {
                json!({
                    "name": tool.get("function").and_then(|f| f.get("name")).cloned().unwrap_or_default(),
                    "description": tool.get("function").and_then(|f| f.get("description")).cloned().unwrap_or(json!("")),
                    "parameters": tool.get("function").and_then(|f| f.get("parameters")).cloned().unwrap_or(json!({"type": "object", "properties": {}}))
                })
            })
            .collect();
        if !declarations.is_empty() {
            request["tools"] = json!([{
                "functionDeclarations": declarations
            }]);
        }
    }

    if let Some(choice) = body.get("tool_choice") {
        let function_calling_config = match choice {
            Value::String(value) if value == "auto" => Some(json!({ "mode": "AUTO" })),
            Value::String(value) if value == "required" => Some(json!({ "mode": "ANY" })),
            Value::Object(obj) if obj.get("type").and_then(|v| v.as_str()) == Some("function") => {
                obj.get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|v| v.as_str())
                    .map(|name| {
                        json!({
                            "mode": "ANY",
                            "allowedFunctionNames": [name]
                        })
                    })
            }
            _ => None,
        };
        if let Some(config_value) = function_calling_config {
            request["toolConfig"] = json!({ "functionCallingConfig": config_value });
        }
    }

    let mut generation = serde_json::Map::new();
    if let Some(v) = body.get("max_tokens") {
        generation.insert("maxOutputTokens".to_string(), v.clone());
    }
    if let Some(v) = body.get("temperature") {
        generation.insert("temperature".to_string(), v.clone());
    }
    if let Some(v) = body.get("top_p") {
        generation.insert("topP".to_string(), v.clone());
    }
    if !generation.is_empty() {
        request["generationConfig"] = Value::Object(generation);
    }

    if request["contents"].as_array().is_none_or(|c| c.is_empty()) {
        request["contents"] = json!([{
            "role": "user",
            "parts": [{ "text": "" }]
        }]);
    }

    let _ = config.default_model;
    request
}

pub fn convert_gemini_to_openai_chat_response(resp: &Value, fallback_model: &str) -> Value {
    let candidate = resp
        .get("candidates")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .cloned()
        .unwrap_or_else(|| json!({}));
    let parts = candidate
        .get("content")
        .and_then(|content| content.get("parts"))
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    for (index, part) in parts.iter().enumerate() {
        if let Some(text) = part.get("text").and_then(|v| v.as_str())
            && !text.is_empty()
        {
            text_parts.push(text.to_string());
        }
        if let Some(function_call) = part.get("functionCall") {
            let args = function_call
                .get("args")
                .cloned()
                .unwrap_or_else(|| json!({}));
            let id = function_call
                .get("id")
                .and_then(|v| v.as_str())
                .filter(|id| !id.is_empty())
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| format!("call_{index}"));
            tool_calls.push(json!({
                "id": id,
                "type": "function",
                "function": {
                    "name": function_call.get("name").cloned().unwrap_or(json!("")),
                    "arguments": serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string())
                }
            }));
        }
    }

    let finish_reason = match candidate
        .get("finishReason")
        .and_then(|v| v.as_str())
        .unwrap_or("STOP")
    {
        "MAX_TOKENS" => "length",
        "SAFETY" => "content_filter",
        _ if !tool_calls.is_empty() => "tool_calls",
        _ => "stop",
    };

    let mut message = json!({
        "role": "assistant",
        "content": if text_parts.is_empty() { Value::Null } else { Value::String(text_parts.join("\n")) }
    });
    if message["content"].is_null() {
        message.as_object_mut().unwrap().remove("content");
    }
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls);
    }

    let prompt_tokens = resp
        .get("usageMetadata")
        .and_then(|u| u.get("promptTokenCount"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let completion_tokens = resp
        .get("usageMetadata")
        .and_then(|u| u.get("candidatesTokenCount"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    json!({
        "id": resp.get("responseId").cloned().unwrap_or(json!("chatcmpl-aivo")),
        "object": "chat.completion",
        "created": current_unix_ts(),
        "model": fallback_model,
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

fn parse_tool_content(content: Option<&Value>) -> Value {
    match content {
        Some(Value::String(value)) => {
            serde_json::from_str(value).unwrap_or_else(|_| json!({ "content": value }))
        }
        Some(other) => other.clone(),
        None => json!({}),
    }
}

fn extract_openai_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(|v| v.as_str()))
            .collect::<Vec<_>>()
            .join("\n"),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_openai_chat_to_gemini_request_with_tools() {
        let body = json!({
            "model": "gemini-2.5-pro",
            "messages": [
                {"role": "system", "content": "Be precise."},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "ls", "arguments": "{\"path\":\".\"}"}
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": "{\"files\":[]}"}
            ],
            "tools": [{
                "type": "function",
                "function": {"name": "ls", "description": "list", "parameters": {"type":"object"}}
            }]
        });

        let converted = convert_openai_chat_to_gemini_request(
            &body,
            &OpenAIToGeminiConfig {
                default_model: "gemini-2.5-pro",
            },
        );
        assert_eq!(
            converted["systemInstruction"]["parts"][0]["text"],
            "Be precise."
        );
        assert_eq!(
            converted["contents"][1]["parts"][0]["functionCall"]["name"],
            "ls"
        );
        assert_eq!(
            converted["contents"][2]["parts"][0]["functionResponse"]["name"],
            "ls"
        );
    }

    #[test]
    fn test_convert_gemini_to_openai_chat_response_with_function_call() {
        let body = json!({
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "Need tool"},
                        {"functionCall": {"id": "call_1", "name": "ls", "args": {"path":"."}}}
                    ]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 4
            }
        });

        let converted = convert_gemini_to_openai_chat_response(&body, "gemini-2.5-pro");
        assert_eq!(converted["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            converted["choices"][0]["message"]["tool_calls"][0]["id"],
            "call_1"
        );
    }

    #[test]
    fn test_build_google_stream_generate_content_url() {
        assert_eq!(
            build_google_stream_generate_content_url(
                "https://generativelanguage.googleapis.com/v1beta",
                "gemini-2.5-pro"
            ),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:streamGenerateContent?alt=sse"
        );
        assert_eq!(
            build_google_stream_generate_content_url(
                "https://generativelanguage.googleapis.com/v1beta/models",
                "google/gemini-2.5-pro"
            ),
            "https://generativelanguage.googleapis.com/v1beta/models/google/gemini-2.5-pro:streamGenerateContent?alt=sse"
        );
    }
}
