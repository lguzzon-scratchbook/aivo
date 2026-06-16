/// Integration manager for the fallback subsystem.
///
/// Wires together configuration loading → validation → flattening →
/// runtime resolution, providing a single entry point for the rest of
/// the application.

use std::collections::HashMap;
use std::sync::Arc;

use super::flatten;
use super::metrics::{FallbackMetrics, TargetKey};
use super::reload::FallbackState;
use super::resolve::{InvokeProvider, ResolutionResult, resolve};
use super::types::{
    FallbackDefinition, FlattenedIndex, ProviderModelPair, Registry,
    ValidationError,
};
use super::validate::ValidateRegistry;

/// Wraps an `InvokeProvider` to record per-attempt metrics.
struct MetricsInvoker<'a, T, P: InvokeProvider<Response = T>> {
    inner: &'a P,
    metrics: &'a FallbackMetrics,
    fallback_id: &'a str,
}

impl<T, P: InvokeProvider<Response = T>> InvokeProvider for MetricsInvoker<'_, T, P> {
    type Response = T;

    fn invoke(&self, provider: &str, model: &str) -> super::resolve::InvokeResult<T> {
        let tk = TargetKey::new(self.fallback_id, provider, model);
        self.metrics.record_attempt(tk.clone());

        let start = std::time::Instant::now();
        let result = self.inner.invoke(provider, model);
        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

        match &result {
            Ok(_) => self.metrics.record_success(tk.clone()),
            Err(e) => self.metrics.record_error(tk.clone(), &e.category.to_string()),
        }
        self.metrics.record_duration(tk, duration_ms);

        result
    }
}

/// A provider+model target resolved from a fallback alias.
#[derive(Debug, Clone)]
pub struct ResolvedTarget {
    pub provider: String,
    pub model: String,
}

/// Manages fallback configuration and resolution lifecycle.
pub struct FallbackManager {
    /// The currently active fallback state (validated + flattened).
    state: FallbackState,
    /// Metrics collector.
    metrics: Arc<FallbackMetrics>,
    /// Definitions loaded from config (used for serialization roundtrip).
    definitions: Registry,
}

impl FallbackManager {
    /// Create an empty fallback manager with no fallbacks configured.
    pub fn empty() -> Self {
        Self {
            state: FallbackState::empty(),
            metrics: Arc::new(FallbackMetrics::new()),
            definitions: Registry::new(),
        }
    }

    /// Create a manager loaded from raw config definitions.
    ///
    /// Validates and flattens the definitions during construction.
    pub fn load(definitions: HashMap<String, FallbackDefinition>) -> Result<Self, Vec<ValidationError>> {
        let flat_index = if definitions.is_empty() {
            FlattenedIndex::new()
        } else {
            ValidateRegistry::validate(&definitions)?;
            flatten::build_flattened_index(&definitions)
        };

        Ok(Self {
            state: FallbackState {
                flat_index: Arc::new(flat_index),
                registry: Arc::new(definitions.clone()),
            },
            metrics: Arc::new(FallbackMetrics::new()),
            definitions,
        })
    }

    /// Get the metrics collector.
    pub fn metrics(&self) -> &FallbackMetrics {
        &self.metrics
    }

    /// Get the current flat index.
    pub fn flat_index(&self) -> &FlattenedIndex {
        self.state.flat_index()
    }

    /// Get the raw definitions.
    pub fn definitions(&self) -> &Registry {
        &self.definitions
    }

    /// Check if a given model alias is a fallback alias.
    pub fn is_fallback(&self, alias: &str) -> bool {
        self.definitions.contains_key(alias)
    }

    /// Resolve a fallback alias to its flat list of concrete targets.
    ///
    /// Returns `None` if the alias is not a fallback.
    pub fn resolve_targets(&self, fallback_id: &str) -> Option<&[ProviderModelPair]> {
        self.state.flat_index().get(fallback_id).map(|v| v.as_slice())
    }

    /// Run full fallback resolution using a custom invoker.
    ///
    /// Records per-attempt metrics via a `MetricsInvoker` wrapper. On
    /// exhaustion, increments the exhaustion counter.
    pub fn resolve_with<T>(
        &self,
        fallback_id: &str,
        invoker: &impl InvokeProvider<Response = T>,
        timeout_ms: Option<u64>,
    ) -> ResolutionResult<T> {
        let metrics_invoker = MetricsInvoker {
            inner: invoker,
            metrics: &self.metrics,
            fallback_id,
        };

        let result = resolve(fallback_id, self.state.flat_index(), &metrics_invoker, timeout_ms);

        if let ResolutionResult::Exhausted(_) = &result {
            self.metrics.record_exhaustion(fallback_id);
        }

        result
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
    use crate::services::fallback::types::{Entry, ProviderModelPair};

    fn pmp(provider: &str, model: &str) -> Entry {
        Entry::ProviderModelPair(ProviderModelPair {
            provider: provider.into(),
            model: model.into(),
        })
    }

    #[test]
    fn test_empty_manager() {
        let mgr = FallbackManager::empty();
        assert!(!mgr.is_fallback("anything"));
        assert!(mgr.flat_index().is_empty());
    }

    #[test]
    fn test_load_valid() {
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

        let mgr = FallbackManager::load(defs).unwrap();
        assert!(mgr.is_fallback("auto"));
        assert!(!mgr.is_fallback("nonexistent"));
        assert_eq!(mgr.resolve_targets("auto").unwrap().len(), 1);
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
        let mgr = FallbackManager::empty();
        let debug = format!("{:?}", mgr);
        assert!(debug.contains("fallback_count"));
    }
}
