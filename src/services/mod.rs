//! Core services module for the aivo CLI.
//! Provides session management and AI tool launching.

pub mod ai_launcher;
pub mod anthropic_chat_request;
pub mod anthropic_chat_response;
pub mod anthropic_route_pipeline;
pub mod anthropic_router;
pub mod anthropic_to_openai_router;
pub mod api_key_store;
pub mod chat_session_store;
pub mod codex_model_map;
pub mod copilot_auth;
pub mod copilot_router;
pub mod directory_starts;
pub mod environment_injector;
pub mod gemini_router;
pub mod http_utils;
pub mod known_providers;
pub mod launch_args;
pub mod launch_runtime;
pub mod model_names;
pub mod models_cache;
pub mod ollama;
pub mod openai_anthropic_bridge;
pub mod openai_gemini_bridge;
pub mod openai_models;
pub mod path_search;
pub mod protocol_fallback;
pub mod provider_profile;
pub mod provider_protocol;
pub mod request_log;
pub mod responses_chat_conversion;
pub mod responses_to_chat_router;
pub mod serve_responses;
pub mod serve_router;
pub mod serve_stream_converters;
pub mod serve_upstream;
pub mod session_crypto;
pub mod session_store;
pub mod system_env;
pub mod usage_stats_store;

#[allow(unused_imports)]
pub use ai_launcher::{AILauncher, LaunchOptions, ToolConfig};
pub use anthropic_router::{AnthropicRouter, AnthropicRouterConfig};
pub use anthropic_to_openai_router::{AnthropicToOpenAIRouter, AnthropicToOpenAIRouterConfig};
pub use copilot_router::{CopilotRouter, CopilotRouterConfig};
pub use environment_injector::EnvironmentInjector;
pub use gemini_router::{GeminiRouter, GeminiRouterConfig};
pub use models_cache::ModelsCache;
pub use responses_to_chat_router::{ResponsesToChatRouter, ResponsesToChatRouterConfig};
#[allow(unused_imports)]
pub use serve_router::{ServeRouter, ServeRouterConfig};
#[allow(unused_imports)]
pub use session_store::{ApiKey, SessionStore};
