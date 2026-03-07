//! Request patch pipeline for Aivo's Anthropic-compatible routing.
//!
//! This keeps provider-specific request quirks modular so routers stay focused on
//! transport and streaming.

use anyhow::Result;
use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::Value;

use crate::services::model_names::transform_model_for_provider;

pub struct RequestContext<'a> {
    pub upstream_base_url: &'a str,
}

pub trait RequestPatch: Send + Sync {
    fn patch_json(&self, _route: &str, _body: &mut Value, _ctx: &RequestContext<'_>) -> Result<()> {
        Ok(())
    }

    fn patch_headers(
        &self,
        _route: &str,
        _headers: &mut HeaderMap,
        _ctx: &RequestContext<'_>,
    ) -> Result<()> {
        Ok(())
    }
}

pub struct RouterPipeline {
    patches: Vec<Box<dyn RequestPatch>>,
}

impl RouterPipeline {
    pub fn new(patches: Vec<Box<dyn RequestPatch>>) -> Self {
        Self { patches }
    }

    pub fn for_openrouter() -> Self {
        Self::new(vec![
            Box::new(ModelNamePatch),
            Box::new(AnthropicVersionPatch),
        ])
    }

    pub fn patch_json(
        &self,
        route: &str,
        body: &mut Value,
        ctx: &RequestContext<'_>,
    ) -> Result<()> {
        for patch in &self.patches {
            patch.patch_json(route, body, ctx)?;
        }
        Ok(())
    }

    pub fn patch_headers(
        &self,
        route: &str,
        headers: &mut HeaderMap,
        ctx: &RequestContext<'_>,
    ) -> Result<()> {
        for patch in &self.patches {
            patch.patch_headers(route, headers, ctx)?;
        }
        Ok(())
    }
}

/// Normalizes provider model names (e.g. OpenRouter model prefix/version shape).
pub struct ModelNamePatch;

impl RequestPatch for ModelNamePatch {
    fn patch_json(&self, _route: &str, body: &mut Value, ctx: &RequestContext<'_>) -> Result<()> {
        if let Some(model) = body.get_mut("model")
            && let Some(model_str) = model.as_str()
        {
            *model = Value::String(transform_model_for_provider(
                ctx.upstream_base_url,
                model_str,
            ));
        }
        Ok(())
    }
}

/// Adds Anthropic API version header where required by Anthropic-format endpoints.
pub struct AnthropicVersionPatch;

impl RequestPatch for AnthropicVersionPatch {
    fn patch_headers(
        &self,
        route: &str,
        headers: &mut HeaderMap,
        _ctx: &RequestContext<'_>,
    ) -> Result<()> {
        if matches!(route, "messages" | "messages/count_tokens") {
            headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_name_patch_openrouter_transform() {
        let patch = ModelNamePatch;
        let mut body = serde_json::json!({"model":"claude-sonnet-4-6"});
        let ctx = RequestContext {
            upstream_base_url: "https://openrouter.ai/api/v1",
        };
        patch.patch_json("messages", &mut body, &ctx).unwrap();
        assert_eq!(body["model"], "anthropic/claude-sonnet-4.6");
    }

    #[test]
    fn test_model_name_patch_non_openrouter_passthrough() {
        let patch = ModelNamePatch;
        let mut body = serde_json::json!({"model":"claude-sonnet-4-6"});
        let ctx = RequestContext {
            upstream_base_url: "https://api.example.com/v1",
        };
        patch.patch_json("messages", &mut body, &ctx).unwrap();
        assert_eq!(body["model"], "claude-sonnet-4-6");
    }

    #[test]
    fn test_anthropic_version_patch_only_messages_routes() {
        let patch = AnthropicVersionPatch;
        let ctx = RequestContext {
            upstream_base_url: "https://openrouter.ai/api/v1",
        };

        let mut headers = HeaderMap::new();
        patch.patch_headers("messages", &mut headers, &ctx).unwrap();
        assert!(headers.get("anthropic-version").is_some());

        let mut headers = HeaderMap::new();
        patch
            .patch_headers("chat/completions", &mut headers, &ctx)
            .unwrap();
        assert!(headers.get("anthropic-version").is_none());
    }

    #[test]
    fn test_pipeline_applies_all_patches() {
        let pipeline = RouterPipeline::for_openrouter();
        let ctx = RequestContext {
            upstream_base_url: "https://openrouter.ai/api/v1",
        };
        let mut body = serde_json::json!({"model":"claude-haiku-4-5"});
        let mut headers = HeaderMap::new();

        pipeline.patch_json("messages", &mut body, &ctx).unwrap();
        pipeline
            .patch_headers("messages", &mut headers, &ctx)
            .unwrap();

        assert_eq!(body["model"], "anthropic/claude-haiku-4.5");
        assert_eq!(
            headers
                .get("anthropic-version")
                .and_then(|v| v.to_str().ok()),
            Some("2023-06-01")
        );
    }
}
