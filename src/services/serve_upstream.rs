use anyhow::Result;
use serde_json::{Value, json};
use std::sync::Arc;

use crate::services::copilot_auth::CopilotTokenManager;
use crate::services::http_utils;
use crate::services::model_names::{copilot_model_name, transform_model_for_openrouter};
use crate::services::openai_anthropic_bridge::{
    OpenAIToAnthropicChatConfig, convert_anthropic_to_openai_chat_response,
    convert_openai_chat_response_to_sse, convert_openai_chat_to_anthropic_request,
};
use crate::services::openai_gemini_bridge::{
    OpenAIToGeminiConfig, build_google_generate_content_url,
    build_google_stream_generate_content_url, convert_gemini_to_openai_chat_response,
    convert_openai_chat_to_gemini_request,
};
use crate::services::serve_responses::OpenAIToResponsesStreamConverter;
use crate::services::serve_stream_converters::{
    AnthropicToOpenAIStreamConverter, GeminiToOpenAIStreamConverter,
};

#[derive(Clone)]
pub(crate) struct UpstreamRequestContext {
    pub(crate) client: reqwest::Client,
    pub(crate) upstream_base_url: String,
    pub(crate) upstream_api_key: String,
    pub(crate) is_copilot: bool,
    pub(crate) is_openrouter: bool,
    pub(crate) copilot_tokens: Option<Arc<CopilotTokenManager>>,
}

pub(crate) enum RouterResponse {
    Buffered {
        status: u16,
        content_type: String,
        body: Vec<u8>,
    },
    Streaming {
        status: u16,
        content_type: String,
        body: Box<StreamingBody>,
    },
}

pub(crate) enum StreamingBody {
    Upstream(reqwest::Response),
    Anthropic {
        upstream: reqwest::Response,
        converter: AnthropicToOpenAIStreamConverter,
    },
    Gemini {
        upstream: reqwest::Response,
        converter: GeminiToOpenAIStreamConverter,
    },
    Responses {
        source: Box<StreamingBody>,
        converter: OpenAIToResponsesStreamConverter,
    },
}

impl RouterResponse {
    pub(crate) fn buffered(status: u16, content_type: &str, body: Vec<u8>) -> Self {
        Self::Buffered {
            status,
            content_type: content_type.to_string(),
            body,
        }
    }
}

pub(crate) async fn send_anthropic_chat(
    body: &Value,
    client_wants_stream: bool,
    context: &UpstreamRequestContext,
) -> Result<RouterResponse> {
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
    anthropic_req["stream"] = json!(client_wants_stream);

    let url = http_utils::build_target_url(&context.upstream_base_url, "/v1/messages");
    let response = context
        .client
        .post(&url)
        .header("x-api-key", context.upstream_api_key.as_str())
        .header("anthropic-version", "2023-06-01")
        .header("Content-Type", "application/json")
        .header("User-Agent", "aivo-serve/1.0")
        .json(&anthropic_req)
        .send()
        .await?;

    let status = response.status().as_u16();
    let content_type = http_utils::response_content_type(&response);

    if status >= 400 {
        return Ok(RouterResponse::buffered(
            status,
            &content_type,
            response.bytes().await?.to_vec(),
        ));
    }

    if client_wants_stream && content_type.contains("text/event-stream") {
        return Ok(RouterResponse::Streaming {
            status,
            content_type: "text/event-stream".to_string(),
            body: Box::new(StreamingBody::Anthropic {
                upstream: response,
                converter: AnthropicToOpenAIStreamConverter::new(&fallback_model),
            }),
        });
    }

    let resp_body = response.text().await?;
    let anthropic_resp: Value = serde_json::from_str(&resp_body)?;
    let openai_resp = convert_anthropic_to_openai_chat_response(&anthropic_resp, &fallback_model);

    if client_wants_stream {
        Ok(RouterResponse::buffered(
            200,
            "text/event-stream",
            convert_openai_chat_response_to_sse(&openai_resp).into_bytes(),
        ))
    } else {
        Ok(RouterResponse::buffered(
            200,
            "application/json",
            openai_resp.to_string().into_bytes(),
        ))
    }
}

pub(crate) async fn send_gemini_chat(
    body: &mut Value,
    client_wants_stream: bool,
    context: &UpstreamRequestContext,
) -> Result<RouterResponse> {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("gemini-2.5-pro")
        .to_string();

    let gemini_req = convert_openai_chat_to_gemini_request(
        body,
        &OpenAIToGeminiConfig {
            default_model: "gemini-2.5-pro",
        },
    );

    let url = if client_wants_stream {
        build_google_stream_generate_content_url(&context.upstream_base_url, &model)
    } else {
        build_google_generate_content_url(&context.upstream_base_url, &model)
    };
    let response = context
        .client
        .post(&url)
        .header("x-goog-api-key", context.upstream_api_key.as_str())
        .header("Content-Type", "application/json")
        .header("User-Agent", "aivo-serve/1.0")
        .json(&gemini_req)
        .send()
        .await?;

    let status = response.status().as_u16();
    let content_type = http_utils::response_content_type(&response);

    if status >= 400 {
        return Ok(RouterResponse::buffered(
            status,
            &content_type,
            response.bytes().await?.to_vec(),
        ));
    }

    if client_wants_stream && content_type.contains("text/event-stream") {
        return Ok(RouterResponse::Streaming {
            status,
            content_type: "text/event-stream".to_string(),
            body: Box::new(StreamingBody::Gemini {
                upstream: response,
                converter: GeminiToOpenAIStreamConverter::new(&model),
            }),
        });
    }

    let resp_body = response.text().await?;
    let gemini_resp: Value = serde_json::from_str(&resp_body)?;
    let openai_resp = convert_gemini_to_openai_chat_response(&gemini_resp, &model);

    if client_wants_stream {
        Ok(RouterResponse::buffered(
            200,
            "text/event-stream",
            convert_openai_chat_response_to_sse(&openai_resp).into_bytes(),
        ))
    } else {
        Ok(RouterResponse::buffered(
            200,
            "application/json",
            openai_resp.to_string().into_bytes(),
        ))
    }
}

pub(crate) async fn send_openai_chat(
    body: &mut Value,
    client_wants_stream: bool,
    context: &UpstreamRequestContext,
) -> Result<RouterResponse> {
    if context.is_openrouter {
        let normalized = body
            .get("model")
            .and_then(|v| v.as_str())
            .map(transform_model_for_openrouter);
        if let Some(n) = normalized {
            body["model"] = json!(n);
        }
    } else if context.is_copilot {
        let normalized = body
            .get("model")
            .and_then(|v| v.as_str())
            .map(copilot_model_name);
        if let Some(n) = normalized {
            body["model"] = json!(n);
        }
    }

    let url = http_utils::build_chat_completions_url(&context.upstream_base_url);
    let req = http_utils::authorized_openai_post(
        &context.client,
        &url,
        context.upstream_api_key.as_str(),
        context.copilot_tokens.as_deref(),
    )
    .await?;

    let response = req.json(&*body).send().await?;
    let status = response.status().as_u16();
    let content_type = http_utils::response_content_type(&response);

    if status >= 400 {
        return Ok(RouterResponse::buffered(
            status,
            &content_type,
            response.bytes().await?.to_vec(),
        ));
    }

    if client_wants_stream && content_type.contains("text/event-stream") {
        return Ok(RouterResponse::Streaming {
            status,
            content_type,
            body: Box::new(StreamingBody::Upstream(response)),
        });
    }

    let resp_body = response.text().await?;

    if client_wants_stream && let Ok(openai_resp) = serde_json::from_str::<Value>(&resp_body) {
        return Ok(RouterResponse::buffered(
            200,
            "text/event-stream",
            convert_openai_chat_response_to_sse(&openai_resp).into_bytes(),
        ));
    }

    Ok(RouterResponse::buffered(
        status,
        &content_type,
        resp_body.into_bytes(),
    ))
}
