/// Configuration reload for fallback mechanism (§8).
///
/// Manages the active fallback registry and flat index with atomic swap.
/// Concurrent reloads are serialized via a mutex.

use std::sync::{Arc, Mutex, RwLock};

use super::flatten;
use super::types::{FallbackDefinition, FlattenedIndex, Registry, ValidationError};
use super::validate::ValidateRegistry;

/// Active fallback state visible to the rest of the system.
#[derive(Debug, Clone)]
pub struct FallbackState {
    /// Pre-flattened, immutable target index for runtime resolution.
    pub(crate) flat_index: Arc<FlattenedIndex>,
    /// Raw registry (kept for inspection).
    pub(crate) registry: Arc<Registry>,
}

impl FallbackState {
    /// Create an empty fallback state with no fallbacks configured.
    pub fn empty() -> Self {
        Self {
            flat_index: Arc::new(FlattenedIndex::new()),
            registry: Arc::new(Registry::new()),
        }
    }

    /// Get the pre-flattened index for runtime resolution.
    pub fn flat_index(&self) -> &FlattenedIndex {
        &self.flat_index
    }

    /// Get the raw registry (for inspection/debugging).
    pub fn registry(&self) -> &Registry {
        &self.registry
    }
}

/// Manages atomic reload of fallback configuration.
///
/// Thread-safe: readers see a consistent snapshot via `current_state()`.
/// Writes are serialized via an internal mutex.
pub struct ReloadConfiguration {
    /// The currently active fallback state.
    current: RwLock<FallbackState>,
    /// Serializes concurrent reload attempts.
    reload_lock: Mutex<()>,
}

impl ReloadConfiguration {
    /// Create a new reload manager with an empty fallback state.
    pub fn new() -> Self {
        Self {
            current: RwLock::new(FallbackState::empty()),
            reload_lock: Mutex::new(()),
        }
    }

    /// Create a reload manager initialized from a registry.
    pub fn from_registry(registry: Registry) -> Result<Self, Vec<ValidationError>> {
        ValidateRegistry::validate(&registry)?;
        let flat_index = flatten::build_flattened_index(&registry);
        Ok(Self {
            current: RwLock::new(FallbackState {
                flat_index: Arc::new(flat_index),
                registry: Arc::new(registry),
            }),
            reload_lock: Mutex::new(()),
        })
    }

    /// Atomically replace the active configuration.
    ///
    /// 1. Acquires reload lock (serializes concurrent reloads).
    /// 2. Validates the candidate registry.
    /// 3. Builds the flattened index.
    /// 4. Atomically swaps the active state.
    ///
    /// On failure, the previous configuration remains active.
    /// All errors are accumulated and returned together.
    pub fn reload(&self, new_registry: Registry) -> Result<(), Vec<ValidationError>> {
        let _guard = self.reload_lock.lock().map_err(|_| {
            vec![ValidationError::new("Failed to acquire reload lock")]
        })?;

        // Step 2: Full validation on candidate (§4)
        ValidateRegistry::validate(&new_registry)?;

        // Step 3: Build flattened index (§5.1)
        let new_flat_index = flatten::build_flattened_index(&new_registry);

        // Step 4: Atomic swap (§8)
        let new_state = FallbackState {
            flat_index: Arc::new(new_flat_index),
            registry: Arc::new(new_registry),
        };

        let mut current = self.current.write().map_err(|_| {
            vec![ValidationError::new("Failed to acquire write lock")]
        })?;
        *current = new_state;

        Ok(())
    }

    /// Get a snapshot of the current fallback state.
    ///
    /// Readers always see a consistent (old or new) state, never a
    /// half-constructed one.
    pub fn current_state(&self) -> FallbackState {
        let guard = self.current.read().expect("FallbackState read lock");
        guard.clone()
    }

    /// Get the current flat index (convenience).
    pub fn current_flat_index(&self) -> Arc<FlattenedIndex> {
        let guard = self.current.read().expect("FallbackState read lock");
        Arc::clone(&guard.flat_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::fallback::types::{Entry, ProviderModelPair};

    fn make_pmp(provider: &str, model: &str) -> Entry {
        Entry::ProviderModelPair(ProviderModelPair {
            provider: provider.into(),
            model: model.into(),
        })
    }

    fn make_registry(entries: Vec<(&str, Vec<Entry>)>) -> Registry {
        let mut reg = Registry::new();
        for (id, seq) in entries {
            reg.insert(
                id.to_string(),
                FallbackDefinition {
                    id: id.to_string(),
                    description: None,
                    timeout_ms: None,
                    sequence: seq,
                },
            );
        }
        reg
    }

    #[test]
    fn test_empty_state() {
        let rm = ReloadConfiguration::new();
        let state = rm.current_state();
        assert!(state.flat_index().is_empty());
        assert!(state.registry().is_empty());
    }

    #[test]
    fn test_valid_reload() {
        let rm = ReloadConfiguration::new();
        let registry = make_registry(vec![(
            "auto",
            vec![make_pmp("anthropic", "claude-sonnet-4-6")],
        )]);

        let result = rm.reload(registry);
        assert!(result.is_ok());

        let state = rm.current_state();
        assert_eq!(state.flat_index().len(), 1);
        assert!(state.flat_index().contains_key("auto"));
    }

    #[test]
    fn test_invalid_reload_preserves_old() {
        let rm = ReloadConfiguration::new();

        // First, load a valid config
        let valid = make_registry(vec![(
            "valid",
            vec![make_pmp("anthropic", "claude-sonnet-4-6")],
        )]);
        assert!(rm.reload(valid).is_ok());
        assert!(rm.current_state().flat_index().contains_key("valid"));

        // Now try an invalid config (empty sequence)
        let invalid = make_registry(vec![(
            "invalid",
            vec![], // empty sequence → validation error
        )]);
        let result = rm.reload(invalid);
        assert!(result.is_err());

        // The old config must still be active
        let state = rm.current_state();
        assert!(state.flat_index().contains_key("valid"));
        assert!(!state.flat_index().contains_key("invalid"));
    }

    #[test]
    fn test_all_errors_collected() {
        let rm = ReloadConfiguration::new();
        let mut registry = Registry::new();
        registry.insert(
            "broken-a".to_string(),
            FallbackDefinition {
                id: "broken-a".into(),
                description: None,
                timeout_ms: None,
                sequence: vec![],
            },
        );
        registry.insert(
            "broken-b".to_string(),
            FallbackDefinition {
                id: "broken-b".into(),
                description: None,
                timeout_ms: None,
                sequence: vec![
                    Entry::FallbackReference(super::super::types::FallbackReference {
                        fallback_id: "ghost".into(),
                    }),
                ],
            },
        );

        let result = rm.reload(registry);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 2, "Should have 2 errors (empty + unknown ref)");
    }

    #[test]
    fn test_concurrent_reads_see_consistent_state() {
        use std::thread;

        let rm = Arc::new(ReloadConfiguration::new());

        // Load initial config
        let init = make_registry(vec![(
            "initial",
            vec![make_pmp("p", "m")],
        )]);
        assert!(rm.reload(init).is_ok());

        let mut handles = vec![];
        for i in 0..10 {
            let rm = Arc::clone(&rm);
            let handle = thread::spawn(move || {
                let state = rm.current_state();
                // Every reader should see either the initial state
                // (no reload completed yet) or any valid state.
                assert!(state.flat_index().contains_key("initial") || state.flat_index().is_empty());
                i
            });
            handles.push(handle);
        }

        // All readers see a consistent state
        for h in handles {
            h.join().unwrap();
        }
    }
}
