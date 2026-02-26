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
    #[allow(dead_code)]
    Network,
    Auth,
}

/// CLI error with category for exit code mapping.
#[derive(Debug, thiserror::Error)]
pub struct CLIError {
    message: String,
    #[allow(dead_code)]
    category: ErrorCategory,
}

impl CLIError {
    pub fn new(
        message: impl Into<String>,
        category: ErrorCategory,
        _details: Option<impl Into<String>>,
        _suggestion: Option<impl Into<String>>,
    ) -> Self {
        Self {
            message: message.into(),
            category,
        }
    }
}

impl fmt::Display for CLIError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

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
        let err = CLIError::new("test error", ErrorCategory::User, None::<String>, None::<String>);
        assert_eq!(err.to_string(), "test error");
    }
}
