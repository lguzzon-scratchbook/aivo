use serde_json::{Value, json};

use crate::services::openai_models::OpenAIChatResponseView;

pub enum UsageValueMode {
    CoerceU64,
    PreserveJson,
}

pub struct OpenAIToAnthropicConfig<'a> {
    pub fallback_id: &'a str,
    pub model: &'a str,
    pub include_created: bool,
    pub usage_value_mode: UsageValueMode,
}

pub fn convert_openai_to_anthropic_message(
    resp: &Value,
    config: &OpenAIToAnthropicConfig<'_>,
) -> Value {
    let response: OpenAIChatResponseView =
        serde_json::from_value(resp.clone()).expect("openai chat response should be typed");

    let mut content: Vec<Value> = Vec::new();
    let mut final_finish_reason = "stop";

    for choice in &response.choices {
        let finish_reason = choice.finish_reason.as_deref().unwrap_or("stop");

        if finish_reason == "tool_calls" {
            final_finish_reason = "tool_calls";
        } else if final_finish_reason != "tool_calls" {
            final_finish_reason = finish_reason;
        }

        if let Some(text) = choice.message.content.as_deref()
            && !text.is_empty()
        {
            content.push(json!({"type": "text", "text": text}));
        }

        if let Some(tool_calls) = &choice.message.tool_calls {
            for tc in tool_calls {
                let input: Value =
                    serde_json::from_str(&tc.function.arguments).unwrap_or(json!({}));

                content.push(json!({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": input,
                }));
            }
        }
    }

    if content.is_empty() {
        content.push(json!({"type": "text", "text": ""}));
    }

    let mut anthropic_resp = json!({
        "id": response.id.as_deref().unwrap_or(config.fallback_id),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": config.model,
        "stop_reason": map_finish_reason(final_finish_reason),
        "stop_sequence": null,
        "usage": {
            "input_tokens": usage_value(resp, "prompt_tokens", &config.usage_value_mode),
            "output_tokens": usage_value(resp, "completion_tokens", &config.usage_value_mode),
        }
    });

    if config.include_created
        && let Some(created) = response.created
    {
        anthropic_resp["created"] = json!(created);
    }

    anthropic_resp
}

fn map_finish_reason(finish_reason: &str) -> &'static str {
    match finish_reason {
        "stop" => "end_turn",
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        "content_filter" => "end_turn",
        _ => "end_turn",
    }
}

fn usage_value(resp: &Value, key: &str, mode: &UsageValueMode) -> Value {
    match mode {
        UsageValueMode::CoerceU64 => json!(
            resp.get("usage")
                .and_then(|u| u.get(key))
                .and_then(|v| v.as_u64())
                .unwrap_or(0)
        ),
        UsageValueMode::PreserveJson => resp
            .get("usage")
            .and_then(|u| u.get(key))
            .cloned()
            .unwrap_or(json!(0)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_openai_to_anthropic_message_merges_choices_and_includes_created() {
        let resp = json!({
            "id": "chatcmpl-123",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Let me check."},
                    "finish_reason": "stop"
                },
                {
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
            "usage": {"prompt_tokens": 12, "completion_tokens": 7}
        });

        let result = convert_openai_to_anthropic_message(
            &resp,
            &OpenAIToAnthropicConfig {
                fallback_id: "msg_default",
                model: "gpt-4o",
                include_created: true,
                usage_value_mode: UsageValueMode::CoerceU64,
            },
        );

        let content = result["content"].as_array().unwrap();
        assert_eq!(result["id"], "chatcmpl-123");
        assert_eq!(result["model"], "gpt-4o");
        assert_eq!(result["created"], 1700000000);
        assert_eq!(result["stop_reason"], "tool_use");
        assert_eq!(result["usage"]["input_tokens"], 12);
        assert_eq!(result["usage"]["output_tokens"], 7);
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "Let me check.");
        assert_eq!(content[1]["type"], "tool_use");
        assert_eq!(content[1]["name"], "get_weather");
        assert_eq!(content[1]["input"]["city"], "Paris");
    }

    #[test]
    fn test_convert_openai_to_anthropic_message_preserves_usage_json_shape() {
        let resp = json!({
            "choices": [{
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": "5", "completion_tokens": 3}
        });

        let result = convert_openai_to_anthropic_message(
            &resp,
            &OpenAIToAnthropicConfig {
                fallback_id: "msg_copilot",
                model: "claude-sonnet-4",
                include_created: false,
                usage_value_mode: UsageValueMode::PreserveJson,
            },
        );

        assert_eq!(result["id"], "msg_copilot");
        assert_eq!(result["model"], "claude-sonnet-4");
        assert_eq!(result["usage"]["input_tokens"], "5");
        assert_eq!(result["usage"]["output_tokens"], 3);
        assert!(result.get("created").is_none());
    }

    #[test]
    fn test_convert_openai_to_anthropic_message_falls_back_for_empty_or_invalid_content() {
        let resp = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_invalid",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{not-json"}
                    }]
                },
                "finish_reason": "length"
            }]
        });

        let result = convert_openai_to_anthropic_message(
            &resp,
            &OpenAIToAnthropicConfig {
                fallback_id: "msg_default",
                model: "unknown",
                include_created: false,
                usage_value_mode: UsageValueMode::CoerceU64,
            },
        );

        let content = result["content"].as_array().unwrap();
        assert_eq!(result["stop_reason"], "max_tokens");
        assert_eq!(result["usage"]["input_tokens"], 0);
        assert_eq!(result["usage"]["output_tokens"], 0);
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "tool_use");
        assert_eq!(content[0]["id"], "call_invalid");
        assert_eq!(content[0]["name"], "read_file");
        assert_eq!(content[0]["input"], json!({}));
    }

    #[test]
    fn test_convert_openai_to_anthropic_message_adds_empty_text_when_no_content_blocks_exist() {
        let resp = json!({
            "choices": [{
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "content_filter"
            }]
        });

        let result = convert_openai_to_anthropic_message(
            &resp,
            &OpenAIToAnthropicConfig {
                fallback_id: "msg_default",
                model: "unknown",
                include_created: false,
                usage_value_mode: UsageValueMode::CoerceU64,
            },
        );

        let content = result["content"].as_array().unwrap();
        assert_eq!(result["stop_reason"], "end_turn");
        assert_eq!(content.len(), 1);
        assert_eq!(content[0], json!({"type": "text", "text": ""}));
    }
}
