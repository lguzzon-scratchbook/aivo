/**
 * Centralized error handling module for the aivo CLI.
 * Defines error types, exit codes, and error classification utilities.
 */
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitCode {
    Success,
    UserError,
    NetworkError,
    AuthError,
    ToolExit(i32),
}

impl ExitCode {
    pub fn code(self) -> i32 {
        match self {
            ExitCode::Success => 0,
            ExitCode::UserError => 1,
            ExitCode::NetworkError => 2,
            ExitCode::AuthError => 3,
            ExitCode::ToolExit(n) => n,
        }
    }
}

impl From<ExitCode> for i32 {
    fn from(code: ExitCode) -> Self {
        code.code()
    }
}

impl fmt::Display for ExitCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    User,
    Network,
    Auth,
}

/// CLI error with category for exit code mapping.
#[derive(Debug)]
#[allow(dead_code)]
pub struct CLIError {
    message: String,
    category: ErrorCategory,
    details: Option<String>,
    suggestion: Option<String>,
}

#[allow(dead_code)]
impl CLIError {
    pub fn new(
        message: impl Into<String>,
        category: ErrorCategory,
        details: Option<impl Into<String>>,
        suggestion: Option<impl Into<String>>,
    ) -> Self {
        Self {
            message: message.into(),
            category,
            details: details.map(|d| d.into()),
            suggestion: suggestion.map(|s| s.into()),
        }
    }

    /// Returns the error category for exit code mapping.
    pub fn category(&self) -> ErrorCategory {
        self.category
    }

    /// Returns the exit code for this error.
    pub fn exit_code(&self) -> ExitCode {
        match self.category {
            ErrorCategory::User => ExitCode::UserError,
            ErrorCategory::Network => ExitCode::NetworkError,
            ErrorCategory::Auth => ExitCode::AuthError,
        }
    }
}

impl fmt::Display for CLIError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(ref details) = self.details {
            write!(f, "\n  {}", details)?;
        }
        if let Some(ref suggestion) = self.suggestion {
            write!(f, "\n  Suggestion: {}", suggestion)?;
        }
        Ok(())
    }
}

impl std::error::Error for CLIError {}

/// Classifies an error into a category based on message patterns.
#[allow(dead_code)]
pub fn classify_error(error: &dyn std::error::Error) -> ErrorCategory {
    let msg = error.to_string().to_lowercase();
    if msg.contains("connection")
        || msg.contains("timeout")
        || msg.contains("dns")
        || msg.contains("network")
    {
        ErrorCategory::Network
    } else if msg.contains("auth") || msg.contains("unauthorized") || msg.contains("401") {
        ErrorCategory::Auth
    } else {
        ErrorCategory::User
    }
}

#[allow(dead_code)]
pub fn get_exit_code(error: &dyn std::error::Error) -> ExitCode {
    match classify_error(error) {
        ErrorCategory::User => ExitCode::UserError,
        ErrorCategory::Network => ExitCode::NetworkError,
        ErrorCategory::Auth => ExitCode::AuthError,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exit_code_values() {
        assert_eq!(ExitCode::Success.code(), 0);
        assert_eq!(ExitCode::UserError.code(), 1);
        assert_eq!(ExitCode::NetworkError.code(), 2);
        assert_eq!(ExitCode::AuthError.code(), 3);
        assert_eq!(ExitCode::ToolExit(130).code(), 130);
    }

    #[test]
    fn test_is_network_error() {
        let network_err =
            std::io::Error::new(std::io::ErrorKind::ConnectionRefused, "connection refused");
        assert_eq!(classify_error(&network_err), ErrorCategory::Network);

        let not_network = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        assert_eq!(classify_error(&not_network), ErrorCategory::User);
    }

    #[test]
    fn test_cli_error_creation() {
        let err = CLIError::new(
            "test error",
            ErrorCategory::User,
            None::<String>,
            None::<String>,
        );
        assert_eq!(err.to_string(), "test error");
    }

    #[test]
    fn test_cli_error_with_details_and_suggestion() {
        let err = CLIError::new(
            "Key not found",
            ErrorCategory::User,
            Some("No key matching 'foo' was found"),
            Some("Run 'aivo keys list' to see available keys"),
        );
        let display = err.to_string();
        assert!(display.contains("Key not found"));
        assert!(display.contains("No key matching 'foo' was found"));
        assert!(display.contains("Run 'aivo keys list'"));
    }

    #[test]
    fn test_cli_error_exit_code() {
        let err = CLIError::new(
            "auth failed",
            ErrorCategory::Auth,
            None::<String>,
            None::<String>,
        );
        assert_eq!(err.exit_code(), ExitCode::AuthError);
    }

    // Additional tests from tests/errors_test.rs
    #[test]
    fn test_cli_error_with_actionable_suggestion() {
        let err = CLIError::new(
            "Failed to connect to OpenRouter",
            ErrorCategory::Network,
            Some("HTTP 403: Invalid API key"),
            Some("Check your key with `aivo keys cat <id>` or add a new key with `aivo keys add`"),
        );
        let display = err.to_string();
        assert!(display.contains("Failed to connect"));
        assert!(display.contains("403"));
        assert!(
            display.contains("aivo keys cat"),
            "Error should suggest the 'keys cat' command"
        );
        assert!(
            display.contains("aivo keys add"),
            "Error should suggest the 'keys add' command"
        );
    }

    #[test]
    fn test_cli_error_no_details_or_suggestion() {
        let err = CLIError::new(
            "Simple error",
            ErrorCategory::User,
            None::<String>,
            None::<String>,
        );
        let display = err.to_string();
        assert_eq!(display, "Simple error");
    }

    #[test]
    fn test_classify_error_network_timeout_by_message() {
        let err = std::io::Error::other("request timeout");
        let category = classify_error(&err);
        assert_eq!(
            category,
            ErrorCategory::Network,
            "Should detect 'timeout' in message"
        );
    }

    #[test]
    fn test_get_exit_code() {
        let err = std::io::Error::other("test");
        let code = get_exit_code(&err);
        assert_eq!(code, ExitCode::UserError);
    }

    #[test]
    fn test_exit_code_display() {
        assert_eq!(format!("{}", ExitCode::Success), "0");
        assert_eq!(format!("{}", ExitCode::UserError), "1");
        assert_eq!(format!("{}", ExitCode::NetworkError), "2");
        assert_eq!(format!("{}", ExitCode::AuthError), "3");
        assert_eq!(format!("{}", ExitCode::ToolExit(130)), "130");
    }
}
