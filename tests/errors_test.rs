use aivo::errors::{classify_error, get_exit_code, CLIError, ErrorCategory, ExitCode};

#[test]
fn test_exit_code_values() {
    assert_eq!(ExitCode::Success.code(), 0);
    assert_eq!(ExitCode::UserError.code(), 1);
    assert_eq!(ExitCode::NetworkError.code(), 2);
    assert_eq!(ExitCode::AuthError.code(), 3);
    assert_eq!(ExitCode::ToolExit(130).code(), 130);
}

#[test]
fn test_cli_error_creation() {
    let err = CLIError::new(
        "test error",
        ErrorCategory::Network,
        Some("details"),
        Some("suggestion"),
    );

    assert_eq!(err.to_string(), "test error");
}

#[test]
fn test_classify_error_network() {
    let err = std::io::Error::new(std::io::ErrorKind::ConnectionRefused, "connection refused");
    let category = classify_error(&err);
    assert_eq!(category, ErrorCategory::Network);
}

#[test]
fn test_get_exit_code() {
    let err = std::io::Error::new(std::io::ErrorKind::Other, "test");
    let code = get_exit_code(&err);
    assert_eq!(code, ExitCode::UserError);
}
