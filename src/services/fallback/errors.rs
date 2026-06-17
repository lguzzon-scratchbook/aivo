//! Error types for the fallback mechanism.
//!
//! Separated from `types` to improve modularity and reduce compilation coupling.

use std::fmt;

/// Validation error for fallback configuration.
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub message: String,
}

impl ValidationError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ValidationError {}

/// Structured error returned when all fallback targets are exhausted.
///
/// Implements spec §6: caller receives standard provider error with attempt history.
#[derive(Debug, Clone)]
pub struct FallbackExhaustedError {
    pub fallback_id: String,
    pub attempt_count: usize,
    pub last_error_category: String, // e.g., "auth", "network", "provider_reject"
    pub last_error_message: String,
}

impl FallbackExhaustedError {
    #[allow(dead_code)]
    pub fn new(
        fallback_id: impl Into<String>,
        attempt_count: usize,
        last_error_category: impl Into<String>,
        last_error_message: impl Into<String>,
    ) -> Self {
        Self {
            fallback_id: fallback_id.into(),
            attempt_count,
            last_error_category: last_error_category.into(),
            last_error_message: last_error_message.into(),
        }
    }
}

impl fmt::Display for FallbackExhaustedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "All {} fallback targets for '{}' exhausted. Last error: {} ({})",
            self.attempt_count, self.fallback_id, self.last_error_message, self.last_error_category
        )
    }
}

impl std::error::Error for FallbackExhaustedError {}
