//! Core services module for the aivo CLI.
//! Provides session management and AI tool launching.

pub mod ai_launcher;
pub mod claude_code_router;
pub mod codex_router;
pub mod environment_injector;
pub mod gemini_router;
pub mod session_store;

#[allow(unused_imports)]
pub use ai_launcher::{AILauncher, LaunchOptions, ToolConfig};
pub use claude_code_router::{ClaudeCodeRouter, RouterConfig};
pub use codex_router::{CodexRouter, CodexRouterConfig};
pub use gemini_router::{GeminiRouter, GeminiRouterConfig};
pub use environment_injector::EnvironmentInjector;
#[allow(unused_imports)]
pub use session_store::{ApiKey, SessionStore};
