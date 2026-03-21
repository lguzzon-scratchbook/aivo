use serde_json::{Value, json};
use std::collections::HashMap;

use crate::services::http_utils::current_unix_ts;
use crate::services::model_names::google_native_model_name;

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
    let model = google_native_model_name(model);
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
    let cache_read_input_tokens = resp
        .get("usageMetadata")
        .and_then(|u| u.get("cachedContentTokenCount"))
        .and_then(|v| v.as_u64());

    let mut usage = json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens
    });
    if let Some(value) = cache_read_input_tokens {
        usage["cache_read_input_tokens"] = json!(value);
    }

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
        "usage": usage
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
                "candidatesTokenCount": 4,
                "cachedContentTokenCount": 90
            }
        });

        let converted = convert_gemini_to_openai_chat_response(&body, "gemini-2.5-pro");
        assert_eq!(converted["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            converted["choices"][0]["message"]["tool_calls"][0]["id"],
            "call_1"
        );
        assert_eq!(converted["usage"]["cache_read_input_tokens"], 90);
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
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:streamGenerateContent?alt=sse"
        );
    }

    #[test]
    fn test_convert_openai_chat_empty_messages() {
        let body = json!({"model": "gemini-2.5-pro", "messages": []});
        let converted = convert_openai_chat_to_gemini_request(
            &body,
            &OpenAIToGeminiConfig {
                default_model: "gemini-2.5-pro",
            },
        );
        // Empty messages → fallback empty content injected
        let contents = converted["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
    }

    #[test]
    fn test_convert_openai_chat_missing_messages_field() {
        let body = json!({"model": "gemini-2.5-pro"});
        let converted = convert_openai_chat_to_gemini_request(
            &body,
            &OpenAIToGeminiConfig {
                default_model: "gemini-2.5-pro",
            },
        );
        // No messages at all → fallback empty content
        let contents = converted["contents"].as_array().unwrap();
        assert!(!contents.is_empty());
    }

    #[test]
    fn test_convert_openai_chat_null_content_no_panic() {
        let body = json!({
            "model": "gemini-2.5-pro",
            "messages": [{"role": "user", "content": null}]
        });
        let converted = convert_openai_chat_to_gemini_request(
            &body,
            &OpenAIToGeminiConfig {
                default_model: "gemini-2.5-pro",
            },
        );
        assert!(converted["contents"].is_array());
    }

    #[test]
    fn test_convert_gemini_response_empty_candidates() {
        let resp = json!({"candidates": []});
        let converted = convert_gemini_to_openai_chat_response(&resp, "gemini-2.5-pro");
        assert_eq!(converted["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn test_convert_gemini_response_no_candidates_field() {
        let resp = json!({});
        let converted = convert_gemini_to_openai_chat_response(&resp, "fallback-model");
        assert_eq!(converted["model"], "fallback-model");
        assert_eq!(converted["choices"][0]["finish_reason"], "stop");
        assert_eq!(converted["usage"]["prompt_tokens"], 0);
    }

    #[test]
    fn test_convert_gemini_response_safety_finish_reason() {
        let resp = json!({
            "candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}]
        });
        let converted = convert_gemini_to_openai_chat_response(&resp, "gemini");
        assert_eq!(converted["choices"][0]["finish_reason"], "content_filter");
    }

    #[test]
    fn test_openai_chat_model_missing_uses_default() {
        let body = json!({});
        assert_eq!(openai_chat_model(&body, "default-model"), "default-model");
    }

    #[test]
    fn test_build_google_url_trailing_slash() {
        let url =
            build_google_generate_content_url("https://example.com/v1beta/", "gemini-2.5-pro");
        assert!(url.contains("models/gemini-2.5-pro:generateContent"));
        assert!(!url.contains("//models"));
    }

    #[test]
    fn test_parse_tool_content_none() {
        assert_eq!(parse_tool_content(None), json!({}));
    }

    #[test]
    fn test_parse_tool_content_non_json_string() {
        let content = json!("not valid json");
        assert_eq!(
            parse_tool_content(Some(&content)),
            json!({"content": "not valid json"})
        );
    }

    #[test]
    fn convert_openai_to_gemini_invalid_tool_args_json() {
        let body = json!({
            "model": "gemini-2.5-pro",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "do_stuff", "arguments": "not json"}
                }]}
            ]
        });
        let converted = convert_openai_chat_to_gemini_request(
            &body,
            &OpenAIToGeminiConfig {
                default_model: "gemini-2.5-pro",
            },
        );
        // Invalid JSON arguments should fall back to {}
        let fc = &converted["contents"][1]["parts"][0]["functionCall"];
        assert_eq!(fc["name"], "do_stuff");
        assert_eq!(fc["args"], json!({}));
    }

    #[test]
    fn convert_gemini_to_openai_null_function_call_args() {
        // Missing args field entirely → defaults to {}
        let resp_missing = json!({
            "candidates": [{
                "content": {
                    "parts": [
                        {"functionCall": {"name": "do_stuff"}}
                    ]
                },
                "finishReason": "STOP"
            }]
        });
        let converted = convert_gemini_to_openai_chat_response(&resp_missing, "gemini-2.5-pro");
        let tool_call = &converted["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(tool_call["function"]["name"], "do_stuff");
        let args: Value =
            serde_json::from_str(tool_call["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args, json!({}));

        // Explicit null args → serialized as "null", doesn't panic
        let resp_null = json!({
            "candidates": [{
                "content": {
                    "parts": [
                        {"functionCall": {"name": "do_stuff", "args": null}}
                    ]
                },
                "finishReason": "STOP"
            }]
        });
        let converted_null = convert_gemini_to_openai_chat_response(&resp_null, "gemini-2.5-pro");
        let tool_call_null = &converted_null["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(tool_call_null["function"]["name"], "do_stuff");
        assert!(tool_call_null["function"]["arguments"].as_str().is_some());
    }

    #[test]
    fn convert_gemini_to_openai_max_tokens_finish_reason() {
        let resp = json!({
            "candidates": [{
                "content": {"parts": [{"text": "truncated output"}]},
                "finishReason": "MAX_TOKENS"
            }]
        });
        let converted = convert_gemini_to_openai_chat_response(&resp, "gemini-2.5-pro");
        assert_eq!(converted["choices"][0]["finish_reason"], "length");
    }

    #[test]
    fn convert_gemini_to_openai_null_usage_metadata() {
        let resp = json!({
            "candidates": [{
                "content": {"parts": [{"text": "hi"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": null,
                "candidatesTokenCount": null
            }
        });
        let converted = convert_gemini_to_openai_chat_response(&resp, "gemini-2.5-pro");
        assert_eq!(converted["usage"]["prompt_tokens"], 0);
        assert_eq!(converted["usage"]["completion_tokens"], 0);
        assert_eq!(converted["usage"]["total_tokens"], 0);
    }

    #[test]
    fn convert_gemini_to_openai_missing_function_call_id() {
        let resp = json!({
            "candidates": [{
                "content": {
                    "parts": [
                        {"functionCall": {"name": "my_func", "args": {"key": "val"}}}
                    ]
                },
                "finishReason": "STOP"
            }]
        });
        let converted = convert_gemini_to_openai_chat_response(&resp, "gemini-2.5-pro");
        let tool_call = &converted["choices"][0]["message"]["tool_calls"][0];
        // Missing id should generate a synthetic one like "call_0"
        assert_eq!(tool_call["id"], "call_0");
        assert_eq!(tool_call["function"]["name"], "my_func");
    }

    #[test]
    fn parse_tool_content_json_object_passthrough() {
        let content = json!("{\"result\": 42, \"status\": \"ok\"}");
        let parsed = parse_tool_content(Some(&content));
        // Valid JSON object string should be parsed, not double-wrapped
        assert_eq!(parsed["result"], 42);
        assert_eq!(parsed["status"], "ok");
        assert!(parsed.get("content").is_none());
    }

    #[test]
    fn convert_openai_to_gemini_tool_choice_specific_function() {
        let body = json!({
            "model": "gemini-2.5-pro",
            "messages": [{"role": "user", "content": "call X"}],
            "tools": [{
                "type": "function",
                "function": {"name": "X", "description": "do X", "parameters": {"type": "object"}}
            }],
            "tool_choice": {
                "type": "function",
                "function": {"name": "X"}
            }
        });
        let converted = convert_openai_chat_to_gemini_request(
            &body,
            &OpenAIToGeminiConfig {
                default_model: "gemini-2.5-pro",
            },
        );
        let fc_config = &converted["toolConfig"]["functionCallingConfig"];
        assert_eq!(fc_config["mode"], "ANY");
        assert_eq!(fc_config["allowedFunctionNames"][0], "X");
    }
}
