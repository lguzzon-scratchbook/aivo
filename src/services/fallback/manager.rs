/// Integration manager for the fallback subsystem.
///
/// Wires together configuration loading → validation → flattening →
/// runtime resolution, providing a single entry point for the rest of
/// the application.
use std::sync::Arc;

use super::flatten;
use super::types::{FlattenedIndex, ProviderModelPair, Registry, ValidationError};
use super::validate::validate_fallback_registry;

/// Manages fallback configuration and resolution lifecycle.
pub struct FallbackManager {
    /// Pre-flattened, immutable target index for runtime resolution.
    flat_index: Arc<FlattenedIndex>,
    /// Raw definitions (for serialization roundtrip).
    definitions: Registry,
}

impl FallbackManager {
    /// Create an empty fallback manager with no fallbacks configured.
    pub fn empty() -> Self {
        Self {
            flat_index: Arc::new(FlattenedIndex::new()),
            definitions: Registry::new(),
        }
    }

    /// Create a manager loaded from raw config definitions.
    ///
    /// Validates and flattens the definitions during construction.
    pub fn load(definitions: Registry) -> Result<Self, Vec<ValidationError>> {
        let flat_index = if definitions.is_empty() {
            FlattenedIndex::new()
        } else {
            validate_fallback_registry(&definitions)?;
            flatten::build_flattened_index(&definitions)
        };

        Ok(Self {
            flat_index: Arc::new(flat_index),
            definitions,
        })
    }

    /// Resolve a fallback alias to its flat list of concrete targets.
    ///
    /// Returns `None` if the alias is not a fallback.
    pub fn resolve_targets(&self, fallback_id: &str) -> Option<&[ProviderModelPair]> {
        self.flat_index.get(fallback_id).map(|v| v.as_slice())
    }
}

impl std::fmt::Debug for FallbackManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FallbackManager")
            .field("fallback_count", &self.definitions.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::fallback::types::{Entry, FallbackDefinition, ProviderModelPair};
    use std::collections::HashMap;

    fn pmp(provider: &str, model: &str) -> Entry {
        Entry::ProviderModelPair(ProviderModelPair {
            provider: provider.into(),
            model: model.into(),
        })
    }

    fn make_defs() -> Registry {
        let mut defs = HashMap::new();
        defs.insert(
            "auto".to_string(),
            FallbackDefinition {
                id: "auto".into(),
                description: None,
                timeout_ms: None,
                sequence: vec![pmp("anthropic", "claude-sonnet-4-6")],
            },
        );
        defs
    }

    #[test]
    fn test_empty_manager() {
        let mgr = FallbackManager::empty();
        assert!(mgr.resolve_targets("anything").is_none());
    }

    #[test]
    fn test_load_valid() {
        let mgr = FallbackManager::load(make_defs()).unwrap();
        assert_eq!(mgr.resolve_targets("auto").unwrap().len(), 1);
        assert!(mgr.resolve_targets("nonexistent").is_none());
    }

    #[test]
    fn test_load_invalid() {
        let mut defs = HashMap::new();
        defs.insert(
            "empty".to_string(),
            FallbackDefinition {
                id: "empty".into(),
                description: None,
                timeout_ms: None,
                sequence: vec![],
            },
        );
        let result = FallbackManager::load(defs);
        assert!(result.is_err());
    }

    #[test]
    fn test_target_resolution_nonexistent() {
        let mgr = FallbackManager::empty();
        assert!(mgr.resolve_targets("nope").is_none());
    }

    #[test]
    fn test_manager_debug() {
        let mgr = FallbackManager::load(make_defs()).unwrap();
        let debug = format!("{:?}", mgr);
        assert!(debug.contains("fallback_count"));
    }
}
