//! Shared SSE accumulation logic for OpenAI-compatible streaming responses.
//!
//! Extracts common tool-call, text, and reasoning accumulation state machines
//! from `anthropic_to_openai_router` and `responses_chat_conversion` to eliminate
//! duplication and improve testability.

use serde_json::{Value, json};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Default)]
pub struct AccumulatedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Default)]
pub struct AccumulatedResponse {
    pub message_id: String,
    pub model: String,
    pub content: String,
    pub reasoning_content: String,
    pub tool_calls: Vec<AccumulatedToolCall>,
    pub finish_reason: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_input_tokens: Option<u64>,
    pub cache_creation_input_tokens: Option<u64>,
    pub saw_tool_use: bool,
}

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

fn uuid_simple() -> String {
    let count = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!(
        "{:x}{:x}{:x}",
        duration.as_secs(),
        duration.subsec_nanos(),
        count
    )
}

impl AccumulatedResponse {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a single SSE line (e.g., `data: {"choices":...}`) into the accumulator.
    /// Returns true if the stream is finished (`[DONE]`).
    pub fn process_line(&mut self, line: &str) -> bool {
        let Some(data) = line.trim().strip_prefix("data: ") else {
            return false;
        };

        if data.trim() == "[DONE]" {
            if self.finish_reason.is_empty() {
                self.finish_reason = if self.saw_tool_use || !self.tool_calls.is_empty() {
                    "tool_calls".to_string()
                } else {
                    "stop".to_string()
                };
            }
            return true;
        }

        let Ok(chunk) = serde_json::from_str::<Value>(data) else {
            return false;
        };

        self.extract_metadata(&chunk);
        self.extract_usage(&chunk);

        self.process_choices(&chunk);

        false
    }

    /// Extract `id` and `model` from an SSE chunk.
    fn extract_metadata(&mut self, chunk: &Value) {
        if let Some(id) = chunk.get("id").and_then(|v| v.as_str())
            && !id.is_empty()
        {
            self.message_id = id.to_string();
        }
        if let Some(model) = chunk.get("model").and_then(|v| v.as_str())
            && !model.is_empty()
        {
            self.model = model.to_string();
        }
    }

    /// Extract usage statistics from an SSE chunk.
    fn extract_usage(&mut self, chunk: &Value) {
        if let Some(usage) = chunk.get("usage") {
            if let Some(v) = usage.get("prompt_tokens").and_then(|v| v.as_u64()) {
                self.input_tokens = v;
            }
            if let Some(v) = usage.get("completion_tokens").and_then(|v| v.as_u64()) {
                self.output_tokens = v;
            }
            if let Some(v) = usage
                .get("cache_read_input_tokens")
                .and_then(|v| v.as_u64())
            {
                self.cache_read_input_tokens = Some(v);
            }
            if let Some(v) = usage
                .get("cache_creation_input_tokens")
                .and_then(|v| v.as_u64())
            {
                self.cache_creation_input_tokens = Some(v);
            }
        }
    }

    /// Process `choices[].delta` to extract content, reasoning, tool_calls, function_calls, and finish_reason.
    fn process_choices(&mut self, chunk: &Value) {
        if let Some(choices) = chunk.get("choices").and_then(|v| v.as_array()) {
            for choice in choices {
                let delta = choice.get("delta").and_then(|v| v.as_object());
                let Some(delta) = delta else { continue };

                if let Some(text) = delta.get("content").and_then(|v| v.as_str()) {
                    self.content.push_str(text);
                }
                if let Some(rc) = delta.get("reasoning_content").and_then(|v| v.as_str()) {
                    self.reasoning_content.push_str(rc);
                }

                if let Some(tcs) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                    for tc in tcs {
                        let idx = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                        while self.tool_calls.len() <= idx {
                            self.tool_calls.push(AccumulatedToolCall::default());
                        }
                        if let Some(id) = tc.get("id").and_then(|v| v.as_str())
                            && !id.is_empty()
                        {
                            self.tool_calls[idx].id = id.to_string();
                        }
                        if let Some(name) = tc
                            .get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|v| v.as_str())
                            && !name.is_empty()
                        {
                            self.tool_calls[idx].name.push_str(name);
                        }
                        if let Some(args) = tc
                            .get("function")
                            .and_then(|f| f.get("arguments"))
                            .and_then(|v| v.as_str())
                        {
                            self.tool_calls[idx].arguments.push_str(args);
                        }
                    }
                }

                if let Some(function_call) = delta.get("function_call") {
                    self.saw_tool_use = true;
                    while self.tool_calls.is_empty() {
                        self.tool_calls.push(AccumulatedToolCall::default());
                    }
                    if let Some(id) = function_call.get("id").and_then(|v| v.as_str())
                        && !id.is_empty()
                    {
                        self.tool_calls[0].id = id.to_string();
                    }
                    if let Some(name) = function_call.get("name").and_then(|v| v.as_str())
                        && !name.is_empty()
                    {
                        self.tool_calls[0].name.push_str(name);
                    }
                    if let Some(args) = function_call.get("arguments").and_then(|v| v.as_str()) {
                        self.tool_calls[0].arguments.push_str(args);
                    }
                }

                if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str())
                    && !fr.is_empty()
                {
                    self.finish_reason = fr.to_string();
                }
            }
        }
    }

    /// Convert the accumulated state into a generic Chat Completions JSON response.
    pub fn to_chat_json(&self) -> Value {
        if !self.tool_calls.is_empty() {
            let tcs: Vec<Value> = self
                .tool_calls
                .iter()
                .enumerate()
                .map(|(i, tc)| {
                    json!({
                        "id": if tc.id.is_empty() { format!("call_{}", i) } else { tc.id.clone() },
                        "type": "function",
                        "function": {
                            "name": &tc.name,
                            "arguments": &tc.arguments
                        }
                    })
                })
                .collect();

            let mut msg = json!({
                "role": "assistant",
                "content": null,
                "tool_calls": tcs
            });
            if !self.reasoning_content.is_empty() {
                msg["reasoning_content"] = json!(&self.reasoning_content);
            }

            json!({
                "id": &self.message_id,
                "model": &self.model,
                "choices": [{
                    "message": msg,
                    "finish_reason": if self.finish_reason.is_empty() { "tool_calls" } else { &self.finish_reason }
                }],
                "usage": self.usage_json()
            })
        } else {
            let mut msg = json!({
                "role": "assistant",
                "content": &self.content
            });
            if !self.reasoning_content.is_empty() {
                msg["reasoning_content"] = json!(&self.reasoning_content);
            }

            let finish_reason = if self.finish_reason.is_empty() {
                "stop"
            } else {
                &self.finish_reason
            };

            json!({
                "id": &self.message_id,
                "model": &self.model,
                "choices": [{
                    "message": msg,
                    "finish_reason": finish_reason
                }],
                "usage": self.usage_json()
            })
        }
    }

    /// Convert accumulated state into Anthropic SSE format.
    pub fn to_anthropic_sse(&self) -> String {
        let mut output = String::new();
        let mut usage = json!({
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens
        });
        if let Some(v) = self.cache_read_input_tokens {
            usage["cache_read_input_tokens"] = json!(v);
        }
        if let Some(v) = self.cache_creation_input_tokens {
            usage["cache_creation_input_tokens"] = json!(v);
        }

        let msg_start = json!({
            "type": "message_start",
            "message": {
                "id": self.message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": self.model,
                "stop_reason": null,
                "stop_sequence": null,
                "usage": usage
            }
        });
        output.push_str(&format!("event: message_start\ndata: {}\n\n", msg_start));

        let mut block_idx = 0;

        if !self.reasoning_content.is_empty() {
            let start = json!({"type": "content_block_start", "index": block_idx, "content_block": {"type": "thinking", "thinking": ""}});
            output.push_str(&format!("event: content_block_start\ndata: {}\n\n", start));
            let delta = json!({"type": "content_block_delta", "index": block_idx, "delta": {"type": "thinking_delta", "thinking": self.reasoning_content}});
            output.push_str(&format!("event: content_block_delta\ndata: {}\n\n", delta));
            let stop = json!({"type": "content_block_stop", "index": block_idx});
            output.push_str(&format!("event: content_block_stop\ndata: {}\n\n", stop));
            block_idx += 1;
        }

        if !self.content.is_empty() {
            let start = json!({"type": "content_block_start", "index": block_idx, "content_block": {"type": "text", "text": ""}});
            output.push_str(&format!("event: content_block_start\ndata: {}\n\n", start));
            let delta = json!({"type": "content_block_delta", "index": block_idx, "delta": {"type": "text_delta", "text": self.content}});
            output.push_str(&format!("event: content_block_delta\ndata: {}\n\n", delta));
            let stop = json!({"type": "content_block_stop", "index": block_idx});
            output.push_str(&format!("event: content_block_stop\ndata: {}\n\n", stop));
            block_idx += 1;
        }

        for tc in &self.tool_calls {
            let id = if tc.id.is_empty() {
                format!("toolu_{}", uuid_simple())
            } else {
                tc.id.clone()
            };
            let start = json!({"type": "content_block_start", "index": block_idx, "content_block": {"type": "tool_use", "id": id, "name": tc.name}});
            output.push_str(&format!("event: content_block_start\ndata: {}\n\n", start));
            if !tc.arguments.is_empty() {
                let delta = json!({"type": "content_block_delta", "index": block_idx, "delta": {"type": "input_json_delta", "partial_json": tc.arguments}});
                output.push_str(&format!("event: content_block_delta\ndata: {}\n\n", delta));
            }
            let stop = json!({"type": "content_block_stop", "index": block_idx});
            output.push_str(&format!("event: content_block_stop\ndata: {}\n\n", stop));
            block_idx += 1;
        }

        let stop_reason = if self.saw_tool_use || !self.tool_calls.is_empty() {
            "tool_use"
        } else {
            match self.finish_reason.as_str() {
                "tool_calls" => "tool_use",
                "length" => "max_tokens",
                _ => "end_turn",
            }
        };

        let msg_delta = json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": null},
            "usage": usage
        });
        output.push_str(&format!("event: message_delta\ndata: {}\n\n", msg_delta));

        let msg_stop = json!({"type": "message_stop"});
        output.push_str(&format!("event: message_stop\ndata: {}\n\n", msg_stop));

        output
    }

    fn usage_json(&self) -> Value {
        let mut usage = json!({
            "prompt_tokens": self.input_tokens,
            "completion_tokens": self.output_tokens
        });
        if let Some(v) = self.cache_read_input_tokens {
            usage["cache_read_input_tokens"] = json!(v);
        }
        if let Some(v) = self.cache_creation_input_tokens {
            usage["cache_creation_input_tokens"] = json!(v);
        }
        usage
    }
}

/// Accumulate an entire SSE string into a structured response.
pub fn accumulate_sse_to_response(text: &str) -> AccumulatedResponse {
    let mut acc = AccumulatedResponse::new();
    for line in text.lines() {
        if acc.process_line(line) {
            break; // [DONE] received
        }
    }
    if acc.finish_reason.is_empty() && !acc.content.is_empty() {
        acc.finish_reason = "stop".to_string();
    } else if acc.finish_reason.is_empty() && acc.saw_tool_use {
        acc.finish_reason = "tool_calls".to_string();
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulate_text_response() {
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\
                   data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\
                   data: [DONE]\n";
        let acc = accumulate_sse_to_response(sse);
        assert_eq!(acc.content, "Hello world");
        assert!(acc.finish_reason == "stop" || acc.finish_reason.is_empty());
        assert!(acc.tool_calls.is_empty());
    }

    #[test]
    fn test_accumulate_tool_call_response() {
        let sse = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_x\",\"type\":\"function\",\"function\":{\"name\":\"shell\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\
                   data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"cmd\\\":\\\"ls\\\"}\"}}]},\"finish_reason\":null}]}\n\
                   data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\
                   data: [DONE]\n";
        let acc = accumulate_sse_to_response(sse);
        assert_eq!(acc.tool_calls.len(), 1);
        assert_eq!(acc.tool_calls[0].id, "call_x");
        assert_eq!(acc.tool_calls[0].name, "shell");
        assert!(acc.tool_calls[0].arguments.contains("ls"));
    }

    #[test]
    fn test_to_anthropic_sse_text() {
        let sse = "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"hello \"},\"finish_reason\":null}]}\n\
data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"world\"},\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":4,\"cache_read_input_tokens\":90,\"cache_creation_input_tokens\":15}}\n\
data: [DONE]\n";
        let acc = accumulate_sse_to_response(sse);
        let anthropic = acc.to_anthropic_sse();
        assert!(anthropic.contains("event: message_start"));
        assert!(anthropic.contains("\"type\":\"text_delta\""));
        assert!(anthropic.contains("\"text\":\"hello world\""));
        assert!(anthropic.contains("\"stop_reason\":\"end_turn\""));
        assert!(anthropic.contains("\"cache_read_input_tokens\":90"));
        assert!(anthropic.contains("event: message_stop"));
    }
}
