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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Entry {
    ProviderModelPair(ProviderModelPair),
    FallbackReference(FallbackReference),
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

/// Maximum nesting depth for flattening (§5.1).
pub const MAX_FLATTEN_DEPTH: usize = 64;
