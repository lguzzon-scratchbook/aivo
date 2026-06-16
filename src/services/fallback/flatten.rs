/// Eager flattening for fallback configurations (§5.1).
///
/// Every fallback's sequence is eagerly flattened at load time by recursively
/// expanding all `FallbackReference` entries into concrete `ProviderModelPair`
/// entries. The resulting flat index is immutable for the lifetime of the
/// configuration instance.
use std::collections::HashMap;

use super::types::{Entry, FlatTargetList, FlattenedIndex, MAX_FLATTEN_DEPTH, Registry};

/// Build the flattened index from a validated registry.
///
/// # Panics
///
/// Panics if flatten depth exceeds `MAX_FLATTEN_DEPTH`. This should never
/// happen because `ValidateRegistry` catches cycles, but the depth guard
/// prevents stack overflow on near-cycles.
pub fn build_flattened_index(registry: &Registry) -> FlattenedIndex {
    let mut index: FlattenedIndex = HashMap::with_capacity(registry.len());

    for fallback_id in registry.keys() {
        let flat = flatten(fallback_id, registry, 0);
        index.insert(fallback_id.clone(), flat);
    }

    index
}

/// Recursively flatten a single fallback into a list of concrete targets.
fn flatten(fallback_id: &str, registry: &Registry, depth: usize) -> FlatTargetList {
    assert!(
        depth <= MAX_FLATTEN_DEPTH,
        "Flatten depth exceeded for chain involving '{}'. Maximum nesting depth is {}.",
        fallback_id,
        MAX_FLATTEN_DEPTH,
    );

    let def = &registry[fallback_id];
    let mut result: FlatTargetList = Vec::new();

    for entry in &def.sequence {
        match entry {
            Entry::ProviderModelPair(pmp) => {
                result.push(pmp.clone());
            }
            Entry::FallbackReference(fr) => {
                let nested = flatten(&fr.fallback_id, registry, depth + 1);
                result.extend(nested);
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::fallback::types::{
        Entry, FallbackDefinition, FallbackReference, ProviderModelPair,
    };

    fn pmp(provider: &str, model: &str) -> Entry {
        Entry::ProviderModelPair(ProviderModelPair {
            provider: provider.into(),
            model: model.into(),
        })
    }

    fn rf(id: &str) -> Entry {
        Entry::FallbackReference(FallbackReference {
            fallback_id: id.into(),
        })
    }

    #[test]
    fn test_simple_flatten() {
        let mut registry = Registry::new();
        registry.insert(
            "auto".into(),
            FallbackDefinition {
                id: "auto".into(),
                description: None,
                sequence: vec![
                    pmp("anthropic", "claude-sonnet-4-6"),
                    pmp("openai", "gpt-4o"),
                ],
                timeout_ms: None,
            },
        );
        let index = build_flattened_index(&registry);
        let flat = index.get("auto").unwrap();
        assert_eq!(flat.len(), 2);
        assert_eq!(flat[0].provider, "anthropic");
        assert_eq!(flat[0].model, "claude-sonnet-4-6");
        assert_eq!(flat[1].provider, "openai");
        assert_eq!(flat[1].model, "gpt-4o");
    }

    #[test]
    fn test_nested_flatten() {
        let mut registry = Registry::new();
        registry.insert(
            "backup".into(),
            FallbackDefinition {
                id: "backup".into(),
                description: None,
                sequence: vec![pmp("openai", "gpt-4o-mini")],
                timeout_ms: None,
            },
        );
        registry.insert(
            "auto".into(),
            FallbackDefinition {
                id: "auto".into(),
                description: None,
                sequence: vec![pmp("anthropic", "claude-sonnet-4-6"), rf("backup")],
                timeout_ms: None,
            },
        );
        let index = build_flattened_index(&registry);
        let flat = index.get("auto").unwrap();
        assert_eq!(flat.len(), 2);
        assert_eq!(flat[0].provider, "anthropic");
        assert_eq!(flat[1].provider, "openai");
    }

    #[test]
    fn test_deep_nesting_preserves_order() {
        let mut registry = Registry::new();
        registry.insert(
            "c".into(),
            FallbackDefinition {
                id: "c".into(),
                description: None,
                sequence: vec![pmp("c-provider", "c-model")],
                timeout_ms: None,
            },
        );
        registry.insert(
            "b".into(),
            FallbackDefinition {
                id: "b".into(),
                description: None,
                sequence: vec![pmp("b-provider", "b-model"), rf("c")],
                timeout_ms: None,
            },
        );
        registry.insert(
            "a".into(),
            FallbackDefinition {
                id: "a".into(),
                description: None,
                sequence: vec![pmp("a-provider", "a-model"), rf("b")],
                timeout_ms: None,
            },
        );
        let index = build_flattened_index(&registry);
        let flat = index.get("a").unwrap();
        assert_eq!(flat.len(), 3);
        assert_eq!(flat[0].provider, "a-provider");
        assert_eq!(flat[1].provider, "b-provider");
        assert_eq!(flat[2].provider, "c-provider");
    }

    #[test]
    fn test_empty_fallback_in_nesting() {
        let mut registry = Registry::new();
        registry.insert(
            "empty".into(),
            FallbackDefinition {
                id: "empty".into(),
                description: None,
                sequence: vec![],
                timeout_ms: None,
            },
        );
        registry.insert(
            "auto".into(),
            FallbackDefinition {
                id: "auto".into(),
                description: None,
                sequence: vec![rf("empty")],
                timeout_ms: None,
            },
        );
        let index = build_flattened_index(&registry);
        let flat = index.get("auto").unwrap();
        assert!(flat.is_empty());
    }

    #[test]
    fn test_index_is_immutable_after_build() {
        let mut registry = Registry::new();
        registry.insert(
            "x".into(),
            FallbackDefinition {
                id: "x".into(),
                description: None,
                sequence: vec![pmp("p", "m")],
                timeout_ms: None,
            },
        );
        let index = build_flattened_index(&registry);
        // Modifying the registry after building must not affect the index
        registry
            .get_mut("x")
            .unwrap()
            .sequence
            .push(pmp("other", "model"));
        let flat = index.get("x").unwrap();
        assert_eq!(flat.len(), 1);
    }
}
