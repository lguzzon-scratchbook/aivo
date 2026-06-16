/// Data model types for the fallback mechanism.
///
/// Implements §2 of the fallback spec:
/// <https://github.com/yuanchuan/aivo/blob/develop/docs/actions/reqs/req-provider-and-model-fallback-mechanism.md>
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A concrete provider + model invocation target.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProviderModelPair {
    pub provider: String,
    pub model: String,
}

/// A reference to another named fallback.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FallbackReference {
    #[serde(rename = "fallbackId")]
    pub fallback_id: String,
}

/// A single entry in a fallback sequence.
///
/// Must be either a `ProviderModelPair` or a `FallbackReference`.
/// No third entry type is permitted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Entry {
    ProviderModelPair(ProviderModelPair),
    FallbackReference(FallbackReference),
}

impl Entry {
    pub fn is_provider_model_pair(&self) -> bool {
        matches!(self, Entry::ProviderModelPair(_))
    }

    pub fn is_fallback_reference(&self) -> bool {
        matches!(self, Entry::FallbackReference(_))
    }
}

/// A named fallback definition with an ordered sequence of targets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FallbackDefinition {
    /// Unique identity for this fallback.
    pub id: String,
    /// Optional human-readable label.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Optional cumulative timeout for full resolution (milliseconds).
    #[serde(rename = "timeoutMs", skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    /// Ordered list of targets (must be non-empty).
    pub sequence: Vec<Entry>,
}

/// Registry mapping fallback IDs to their definitions.
pub type Registry = HashMap<String, FallbackDefinition>;

/// Pre-flattened ordered list of concrete targets for a fallback.
pub type FlatTargetList = Vec<ProviderModelPair>;

/// Immutable index of all pre-flattened fallbacks, keyed by fallback ID.
pub type FlattenedIndex = HashMap<String, FlatTargetList>;

/// Error categories for fallback attempt classification (§5.2).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum FallbackErrorCategory {
    RateLimit,
    Timeout,
    Auth,
    ModelNotFound,
    Network,
    Internal,
    Other,
}

impl std::fmt::Display for FallbackErrorCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FallbackErrorCategory::RateLimit => write!(f, "rate_limit"),
            FallbackErrorCategory::Timeout => write!(f, "timeout"),
            FallbackErrorCategory::Auth => write!(f, "auth"),
            FallbackErrorCategory::ModelNotFound => write!(f, "model_not_found"),
            FallbackErrorCategory::Network => write!(f, "network"),
            FallbackErrorCategory::Internal => write!(f, "internal"),
            FallbackErrorCategory::Other => write!(f, "other"),
        }
    }
}

/// Error produced during fallback resolution.
#[derive(Debug, Clone)]
pub struct ProviderError {
    pub message: String,
    pub category: FallbackErrorCategory,
}

impl ProviderError {
    pub fn new(message: impl Into<String>, category: FallbackErrorCategory) -> Self {
        Self {
            message: message.into(),
            category,
        }
    }
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.category, self.message)
    }
}

impl std::error::Error for ProviderError {}

/// Record of a single attempt during fallback resolution.
#[derive(Debug, Clone)]
pub struct AttemptRecord {
    pub provider: String,
    pub model: String,
    pub error: ProviderError,
}

/// Categorized summary of errors across all attempts.
#[derive(Debug, Clone)]
pub struct ErrorSummary {
    pub categories: Vec<FallbackErrorCategory>,
}

impl std::fmt::Display for ErrorSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let parts: Vec<String> = self.categories.iter().map(|c| c.to_string()).collect();
        write!(f, "{}", parts.join(", "))
    }
}

/// Error returned when all fallback targets are exhausted.
#[derive(Debug, Clone)]
pub struct FallbackExhaustedError {
    pub fallback_id: String,
    pub attempts: Vec<AttemptRecord>,
    pub last_error: ProviderError,
    pub summary: ErrorSummary,
}

impl FallbackExhaustedError {
    pub fn caller_error(&self) -> ProviderError {
        ProviderError::new(
            format!(
                "All {} targets exhausted for fallback '{}'. Error categories: {}. Last error: {}",
                self.attempts.len(),
                self.fallback_id,
                self.summary,
                self.last_error.message,
            ),
            self.last_error.category.clone(),
        )
    }
}

impl std::fmt::Display for FallbackExhaustedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Fallback '{}' exhausted: {} attempts, categories: {}",
            self.fallback_id,
            self.attempts.len(),
            self.summary,
        )
    }
}

impl std::error::Error for FallbackExhaustedError {}

/// Validation error for configuration.
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

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ValidationError {}

/// Maximum nesting depth for flattening (§5.1).
pub const MAX_FLATTEN_DEPTH: usize = 64;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entry_discrimination() {
        let pmp = Entry::ProviderModelPair(ProviderModelPair {
            provider: "anthropic".into(),
            model: "claude-sonnet-4-6".into(),
        });
        assert!(pmp.is_provider_model_pair());
        assert!(!pmp.is_fallback_reference());

        let fr = Entry::FallbackReference(FallbackReference {
            fallback_id: "auto".into(),
        });
        assert!(!fr.is_provider_model_pair());
        assert!(fr.is_fallback_reference());
    }

    #[test]
    fn test_definition_roundtrip() {
        let def = FallbackDefinition {
            id: "test".into(),
            description: Some("Test fallback".into()),
            timeout_ms: Some(30_000),
            sequence: vec![Entry::ProviderModelPair(ProviderModelPair {
                provider: "openai".into(),
                model: "gpt-4o".into(),
            })],
        };
        assert_eq!(def.id, "test");
        assert_eq!(def.sequence.len(), 1);
    }

    #[test]
    fn test_error_categories_display() {
        assert_eq!(
            FallbackErrorCategory::RateLimit.to_string(),
            "rate_limit"
        );
        assert_eq!(FallbackErrorCategory::Timeout.to_string(), "timeout");
        assert_eq!(FallbackErrorCategory::Auth.to_string(), "auth");
    }

    #[test]
    fn test_exhaustion_summary() {
        let err = FallbackExhaustedError {
            fallback_id: "auto".into(),
            attempts: vec![AttemptRecord {
                provider: "anthropic".into(),
                model: "claude-sonnet-4-6".into(),
                error: ProviderError::new("rate limited", FallbackErrorCategory::RateLimit),
            }],
            last_error: ProviderError::new("rate limited", FallbackErrorCategory::RateLimit),
            summary: ErrorSummary {
                categories: vec![FallbackErrorCategory::RateLimit],
            },
        };
        let caller = err.caller_error();
        assert!(caller.message.contains("1 targets exhausted"));
        assert!(caller.message.contains("auto"));
        assert!(caller.message.contains("rate_limit"));
    }
}
