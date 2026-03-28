/**
 * Request building for chat: construct HTTP request bodies for OpenAI and
 * Anthropic chat completion APIs, including multimodal attachment encoding.
 */
use anyhow::Result;

use crate::commands::chat::is_document_mime;
use crate::services::anthropic_route_pipeline::inject_cache_control_on_last_block;
use crate::services::session_store::{AttachmentStorage, MessageAttachment};

use super::chat::ChatMessage;

pub(crate) fn format_text_attachment_content(name: &str, content: &str) -> String {
    format!("[Attached file: {name}]\n{content}")
}

pub(crate) fn build_openai_chat_request(
    model: &str,
    messages: &[ChatMessage],
    stream: bool,
) -> Result<serde_json::Value> {
    let mut encoded_messages = Vec::with_capacity(messages.len());
    for message in messages {
        encoded_messages.push(build_openai_message(message)?);
    }

    let mut body = serde_json::json!({
        "model": model,
        "messages": encoded_messages,
        "stream": stream,
    });
    if stream {
        body["stream_options"] = serde_json::json!({"include_usage": true});
    }
    Ok(body)
}

/// Returns the inline data for a materialized attachment, or fails if it is still a FileRef.
fn require_inline(attachment: &MessageAttachment) -> Result<&str> {
    match &attachment.storage {
        AttachmentStorage::Inline { data } => Ok(data),
        AttachmentStorage::FileRef { path } => anyhow::bail!(
            "Attachment '{}' is unresolved. Expected inline data before sending.",
            path
        ),
    }
}

fn build_openai_message(message: &ChatMessage) -> Result<serde_json::Value> {
    if message.attachments.is_empty() {
        return Ok(serde_json::json!({
            "role": message.role,
            "content": message.content,
        }));
    }

    let mut parts = Vec::new();
    if !message.content.is_empty() {
        parts.push(serde_json::json!({
            "type": "text",
            "text": message.content,
        }));
    }

    for attachment in &message.attachments {
        let data = require_inline(attachment)?;
        if attachment.mime_type.starts_with("image/") {
            parts.push(serde_json::json!({
                "type": "image_url",
                "image_url": {
                    "url": format!("data:{};base64,{}", attachment.mime_type, data),
                },
            }));
        } else if is_document_mime(&attachment.mime_type) {
            parts.push(serde_json::json!({
                "type": "file",
                "file": {
                    "filename": attachment.name,
                    "file_data": format!("data:{};base64,{}", attachment.mime_type, data),
                },
            }));
        } else {
            parts.push(serde_json::json!({
                "type": "text",
                "text": format_text_attachment_content(&attachment.name, data),
            }));
        }
    }

    Ok(serde_json::json!({
        "role": message.role,
        "content": parts,
    }))
}

pub(crate) fn build_responses_request(
    model: &str,
    messages: &[ChatMessage],
    stream: bool,
) -> Result<serde_json::Value> {
    let mut input = Vec::new();
    let mut instructions_parts = Vec::new();

    for message in messages {
        if message.role == "system" {
            if !message.content.is_empty() {
                instructions_parts.push(message.content.as_str());
            }
            continue;
        }
        input.push(build_responses_input_item(message)?);
    }

    let mut body = serde_json::json!({
        "model": model,
        "input": input,
        "stream": stream,
    });

    if !instructions_parts.is_empty() {
        body["instructions"] = serde_json::Value::String(instructions_parts.join("\n\n"));
    }

    Ok(body)
}

fn build_responses_input_item(message: &ChatMessage) -> Result<serde_json::Value> {
    if message.attachments.is_empty() {
        return Ok(serde_json::json!({
            "type": "message",
            "role": message.role,
            "content": message.content,
        }));
    }

    let mut parts = Vec::new();
    if !message.content.is_empty() {
        parts.push(serde_json::json!({
            "type": "input_text",
            "text": message.content,
        }));
    }

    for attachment in &message.attachments {
        let data = require_inline(attachment)?;
        if attachment.mime_type.starts_with("image/") {
            parts.push(serde_json::json!({
                "type": "input_image",
                "image_url": format!("data:{};base64,{}", attachment.mime_type, data),
            }));
        } else if is_document_mime(&attachment.mime_type) {
            parts.push(serde_json::json!({
                "type": "input_file",
                "filename": attachment.name,
                "file_data": format!("data:{};base64,{}", attachment.mime_type, data),
            }));
        } else {
            parts.push(serde_json::json!({
                "type": "input_text",
                "text": format_text_attachment_content(&attachment.name, data),
            }));
        }
    }

    Ok(serde_json::json!({
        "type": "message",
        "role": message.role,
        "content": parts,
    }))
}

pub(crate) fn build_anthropic_request(
    model: &str,
    messages: &[ChatMessage],
    stream: bool,
) -> Result<serde_json::Value> {
    let mut system_parts = Vec::new();
    let mut encoded_messages = Vec::new();

    for message in messages {
        if message.role == "system" {
            if !message.content.is_empty() {
                system_parts.push(message.content.clone());
            }
            continue;
        }

        let role = if message.role == "assistant" {
            "assistant"
        } else {
            "user"
        };
        encoded_messages.push(serde_json::json!({
            "role": role,
            "content": build_anthropic_content(message)?,
        }));
    }

    let mut request = serde_json::json!({
        "model": model,
        "messages": encoded_messages,
        "max_tokens": 8096,
        "stream": stream,
    });
    if !system_parts.is_empty() {
        request["system"] = serde_json::json!([{
            "type": "text",
            "text": system_parts.join("\n\n"),
            "cache_control": {"type": "ephemeral"}
        }]);
    }

    // Add cache_control to the last user message for Anthropic prompt caching
    for msg in encoded_messages.iter_mut().rev() {
        if msg["role"] != "user" {
            continue;
        }
        if let Some(content) = msg.get_mut("content") {
            inject_cache_control_on_last_block(content);
        }
        break;
    }

    request["messages"] = serde_json::json!(encoded_messages);
    Ok(request)
}

fn build_anthropic_content(message: &ChatMessage) -> Result<serde_json::Value> {
    if message.attachments.is_empty() {
        return Ok(serde_json::Value::String(message.content.clone()));
    }

    let mut blocks = Vec::new();
    if !message.content.is_empty() {
        blocks.push(serde_json::json!({
            "type": "text",
            "text": message.content,
        }));
    }

    for attachment in &message.attachments {
        let data = require_inline(attachment)?;
        if attachment.mime_type.starts_with("image/") {
            blocks.push(serde_json::json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": attachment.mime_type,
                    "data": data,
                },
            }));
        } else if is_document_mime(&attachment.mime_type) {
            blocks.push(serde_json::json!({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": attachment.mime_type,
                    "data": data,
                },
            }));
        } else {
            blocks.push(serde_json::json!({
                "type": "text",
                "text": format_text_attachment_content(&attachment.name, data),
            }));
        }
    }

    Ok(serde_json::Value::Array(blocks))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::session_store::AttachmentStorage;

    #[test]
    fn test_build_openai_chat_request_encodes_file_and_image_attachments() {
        let request = build_openai_chat_request(
            "gpt-4o",
            &[ChatMessage {
                role: "user".to_string(),
                content: "Review these".to_string(),
                reasoning_content: None,
                attachments: vec![
                    MessageAttachment {
                        name: "notes.md".to_string(),
                        mime_type: "text/markdown".to_string(),
                        storage: AttachmentStorage::Inline {
                            data: "# hello".to_string(),
                        },
                    },
                    MessageAttachment {
                        name: "diagram.png".to_string(),
                        mime_type: "image/png".to_string(),
                        storage: AttachmentStorage::Inline {
                            data: "YWJj".to_string(),
                        },
                    },
                ],
            }],
            true,
        )
        .unwrap();

        let parts = request["messages"][0]["content"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "text");
        assert!(parts[1]["text"].as_str().unwrap().contains("notes.md"));
        assert_eq!(parts[2]["type"], "image_url");
        assert_eq!(parts[2]["image_url"]["url"], "data:image/png;base64,YWJj");
    }

    #[test]
    fn test_build_anthropic_request_encodes_image_attachment() {
        let request = build_anthropic_request(
            "claude-sonnet-4-5",
            &[ChatMessage {
                role: "user".to_string(),
                content: String::new(),
                reasoning_content: None,
                attachments: vec![MessageAttachment {
                    name: "diagram.png".to_string(),
                    mime_type: "image/png".to_string(),
                    storage: AttachmentStorage::Inline {
                        data: "YWJj".to_string(),
                    },
                }],
            }],
            false,
        )
        .unwrap();

        let blocks = request["messages"][0]["content"].as_array().unwrap();
        assert_eq!(blocks[0]["type"], "image");
        assert_eq!(blocks[0]["source"]["media_type"], "image/png");
        assert_eq!(blocks[0]["source"]["data"], "YWJj");
    }

    #[test]
    fn test_build_responses_request_basic() {
        let request = build_responses_request(
            "gpt-5.4",
            &[ChatMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
                reasoning_content: None,
                attachments: vec![],
            }],
            true,
        )
        .unwrap();

        assert_eq!(request["model"], "gpt-5.4");
        assert_eq!(request["stream"], true);
        assert_eq!(request["input"][0]["type"], "message");
        assert_eq!(request["input"][0]["role"], "user");
        assert_eq!(request["input"][0]["content"], "hello");
        assert!(request.get("instructions").is_none());
    }

    #[test]
    fn test_build_responses_request_with_system() {
        let request = build_responses_request(
            "gpt-5.4",
            &[
                ChatMessage {
                    role: "system".to_string(),
                    content: "You are helpful.".to_string(),
                    reasoning_content: None,
                    attachments: vec![],
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: "hi".to_string(),
                    reasoning_content: None,
                    attachments: vec![],
                },
            ],
            false,
        )
        .unwrap();

        assert_eq!(request["instructions"], "You are helpful.");
        assert_eq!(request["input"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_build_responses_request_with_attachments() {
        let request = build_responses_request(
            "gpt-5.4",
            &[ChatMessage {
                role: "user".to_string(),
                content: "Review this".to_string(),
                reasoning_content: None,
                attachments: vec![
                    MessageAttachment {
                        name: "notes.md".to_string(),
                        mime_type: "text/markdown".to_string(),
                        storage: AttachmentStorage::Inline {
                            data: "# hello".to_string(),
                        },
                    },
                    MessageAttachment {
                        name: "diagram.png".to_string(),
                        mime_type: "image/png".to_string(),
                        storage: AttachmentStorage::Inline {
                            data: "YWJj".to_string(),
                        },
                    },
                ],
            }],
            true,
        )
        .unwrap();

        let parts = request["input"][0]["content"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "input_text");
        assert!(parts[1]["text"].as_str().unwrap().contains("notes.md"));
        assert_eq!(parts[2]["type"], "input_image");
    }
}
