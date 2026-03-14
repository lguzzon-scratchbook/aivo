//! Core services module for the aivo CLI.
//! Provides session management and AI tool launching.

pub mod ai_launcher;
pub mod anthropic_chat_request;
pub mod anthropic_chat_response;
pub mod anthropic_route_pipeline;
pub mod anthropic_router;
pub mod codex_model_map;
pub mod codex_router;
pub mod copilot_auth;
pub mod copilot_router;
pub mod environment_injector;
pub mod gemini_router;
pub mod http_utils;
pub mod model_names;
pub mod models_cache;
pub mod openai_anthropic_bridge;
pub mod openai_gemini_bridge;
pub mod openai_router;
pub mod provider_profile;
pub mod provider_protocol;
pub mod serve_router;
pub mod session_store;
pub mod system_env;

#[allow(unused_imports)]
pub use ai_launcher::{AILauncher, LaunchOptions, ToolConfig};
pub use anthropic_router::{AnthropicRouter, AnthropicRouterConfig};
pub use codex_router::{CodexRouter, CodexRouterConfig};
pub use copilot_router::{CopilotRouter, CopilotRouterConfig};
pub use environment_injector::EnvironmentInjector;
pub use gemini_router::{GeminiRouter, GeminiRouterConfig};
pub use models_cache::ModelsCache;
pub use openai_router::{OpenAIRouter, OpenAIRouterConfig};
#[allow(unused_imports)]
pub use serve_router::{ServeRouter, ServeRouterConfig};
#[allow(unused_imports)]
pub use session_store::{ApiKey, SessionStore};
