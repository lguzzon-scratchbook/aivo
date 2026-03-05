//! Core services module for the aivo CLI.
//! Provides session management and AI tool launching.

pub mod ai_launcher;
pub mod claude_code_router;
pub mod codex_router;
pub mod copilot_auth;
pub mod copilot_router;
pub mod environment_injector;
pub mod gemini_router;
pub mod http_utils;
pub mod model_names;
pub mod models_cache;
pub mod openai_router;
pub mod session_store;

#[allow(unused_imports)]
pub use ai_launcher::{AILauncher, LaunchOptions, ToolConfig};
pub use claude_code_router::{ClaudeCodeRouter, RouterConfig};
pub use codex_router::{CodexRouter, CodexRouterConfig};
pub use copilot_router::{CopilotRouter, CopilotRouterConfig};
pub use environment_injector::EnvironmentInjector;
pub use gemini_router::{GeminiRouter, GeminiRouterConfig};
pub use models_cache::ModelsCache;
pub use openai_router::{OpenAIRouter, OpenAIRouterConfig};
#[allow(unused_imports)]
pub use session_store::{ApiKey, SessionStore};
