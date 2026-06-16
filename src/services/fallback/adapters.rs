/// Provider adapters for the fallback resolution system.
///
/// Each adapter wraps a real provider SDK or launcher as an `InvokeProvider`,
/// mapping their error taxonomy to `FallbackErrorCategory`.

use super::resolve::{InvokeProvider, InvokeResult};
use super::types::{FallbackErrorCategory, ProviderError};

/// Maps the `AILauncher::launch` method into an `InvokeProvider` so the
/// fallback subsystem can try multiple CLI-tool targets in sequence.
///
/// "Success" means the tool process exited with code 0.
/// Any non-zero exit or launch failure advances to the next fallback target.
pub struct AILauncherInvoker {
    launcher: crate::services::ai_launcher::AILauncher,
    key: crate::services::session_store::ApiKey,
    debug: bool,
    env: Option<std::collections::HashMap<String, String>>,
    base_args: Vec<String>,
}

impl AILauncherInvoker {
    pub fn new(
        launcher: crate::services::ai_launcher::AILauncher,
        key: crate::services::session_store::ApiKey,
    ) -> Self {
        Self {
            launcher,
            key,
            debug: false,
            env: None,
            base_args: Vec::new(),
        }
    }

    pub fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    pub fn with_env(mut self, env: std::collections::HashMap<String, String>) -> Self {
        self.env = Some(env);
        self
    }

    pub fn with_base_args(mut self, args: Vec<String>) -> Self {
        self.base_args = args;
        self
    }

    fn parse_tool_type(provider: &str) -> Option<crate::services::ai_launcher::AIToolType> {
        match provider.to_lowercase().as_str() {
            "claude" | "anthropic" => Some(crate::services::ai_launcher::AIToolType::Claude),
            "codex" | "opencode" | "pi" => {
                crate::services::ai_launcher::AIToolType::parse(provider)
            }
            "gemini" | "google" => Some(crate::services::ai_launcher::AIToolType::Gemini),
            _ => crate::services::ai_launcher::AIToolType::parse(provider),
        }
    }
}

impl InvokeProvider for AILauncherInvoker {
    type Response = i32;

    fn invoke(&self, provider: &str, model: &str) -> InvokeResult<i32> {
        let tool = match Self::parse_tool_type(provider) {
            Some(t) => t,
            None => {
                return Err(ProviderError::new(
                    format!("Unknown provider '{}'. Expected claude, codex, gemini, opencode, or pi", provider),
                    FallbackErrorCategory::Other,
                ));
            }
        };

        let options = crate::services::ai_launcher::LaunchOptions {
            tool,
            args: self.base_args.clone(),
            model: Some(model.to_string()),
            debug: self.debug,
            env: self.env.clone(),
            key_override: Some(self.key.clone()),
        };

        // `launch` is async — we're in a sync trait method. We must create a
        // mini runtime to drive it. This is acceptable for CLI usage where
        // the fallback loop runs on a desktop-class machine with fast process
        // spawns.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                ProviderError::new(
                    format!("Failed to create runtime: {}", e),
                    FallbackErrorCategory::Internal,
                )
            })?;

        match rt.block_on(self.launcher.launch(&options)) {
            Ok(exit_code) if exit_code == 0 => Ok(exit_code),
            Ok(exit_code) => Err(classify_launch_error(provider, model, exit_code, None)),
            Err(e) => Err(classify_launch_error(
                provider,
                model,
                -1,
                Some(e.to_string()),
            )),
        }
    }
}

/// Classify a launch failure into a `ProviderError` with appropriate category.
fn classify_launch_error(
    provider: &str,
    model: &str,
    exit_code: i32,
    message: Option<String>,
) -> ProviderError {
    let msg = message.unwrap_or_else(|| format!("exit code {}", exit_code));
    let category = match exit_code {
        130 | 131 | 132 | 134 | 136 | 139 => {
            // Signal-based exits (SIGINT, SIGTERM, SIGABRT, etc.)
            FallbackErrorCategory::Network
        }
        _ => {
            let lower = msg.to_lowercase();
            // Check message content first for semantic classification
            if lower.contains("auth") || lower.contains("key") || lower.contains("credential")
            {
                FallbackErrorCategory::Auth
            } else if lower.contains("timeout") || lower.contains("timed out") {
                FallbackErrorCategory::Timeout
            } else if lower.contains("rate") || lower.contains("quota") {
                FallbackErrorCategory::RateLimit
            } else if lower.contains("network") || lower.contains("connection") {
                FallbackErrorCategory::Network
            } else if exit_code == -1 || lower.contains("not found") || lower.contains("no such") {
                FallbackErrorCategory::ModelNotFound
            } else {
                FallbackErrorCategory::Other
            }
        }
    };
    ProviderError::new(
        format!(
            "Launch failed for {}/{}: {}",
            provider, model, msg
        ),
        category,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tool_type() {
        assert_eq!(
            AILauncherInvoker::parse_tool_type("claude"),
            Some(crate::services::ai_launcher::AIToolType::Claude)
        );
        assert_eq!(
            AILauncherInvoker::parse_tool_type("anthropic"),
            Some(crate::services::ai_launcher::AIToolType::Claude)
        );
        assert_eq!(
            AILauncherInvoker::parse_tool_type("gemini"),
            Some(crate::services::ai_launcher::AIToolType::Gemini)
        );
        assert_eq!(
            AILauncherInvoker::parse_tool_type("google"),
            Some(crate::services::ai_launcher::AIToolType::Gemini)
        );
        assert_eq!(
            AILauncherInvoker::parse_tool_type("codex"),
            Some(crate::services::ai_launcher::AIToolType::Codex)
        );
        assert_eq!(AILauncherInvoker::parse_tool_type("unknown"), None);
    }

    #[test]
    fn test_classify_launch_error_exit_codes() {
        let err = classify_launch_error("claude", "sonnet", 1, None);
        assert_eq!(err.category, FallbackErrorCategory::Other);
        assert!(err.message.contains("exit code 1"));

        let err = classify_launch_error("claude", "sonnet", 139, None);
        assert_eq!(err.category, FallbackErrorCategory::Network);

        let err = classify_launch_error("claude", "sonnet", 130, None);
        assert_eq!(err.category, FallbackErrorCategory::Network);
    }

    #[test]
    fn test_classify_launch_error_messages() {
        let err = classify_launch_error(
            "codex",
            "gpt-4o",
            -1,
            Some("not found: codex".to_string()),
        );
        assert_eq!(err.category, FallbackErrorCategory::ModelNotFound);

        let err = classify_launch_error(
            "gemini",
            "gemini-pro",
            -1,
            Some("auth failed".to_string()),
        );
        assert_eq!(err.category, FallbackErrorCategory::Auth);

        let err = classify_launch_error(
            "claude",
            "opus",
            -1,
            Some("rate limit exceeded".to_string()),
        );
        assert_eq!(err.category, FallbackErrorCategory::RateLimit);
    }

    #[test]
    fn test_classify_launch_error_unknown_provider() {
        // Test the InvokeProvider's error path for unknown provider
        let err = AILauncherInvoker::parse_tool_type("invalid_provider_xyz");
        assert!(err.is_none());
    }
}
