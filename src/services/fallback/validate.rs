/// Load-time validation for fallback configurations (§4).
///
/// Executes four sequential phases at config load or reload:
/// 1. Schema, entry type discrimination, and non-empty sequences
/// 2. Reference existence
/// 3. Cycle detection (iterative DFS)
/// 4. Atomic rejection or acceptance
use std::collections::HashMap;

use super::errors::ValidationError;
use super::types::{Entry, Registry};

/// Validate a complete fallback registry.
///
/// All validation errors are collected before rejection (§4 Phase 4).
pub fn validate_fallback_registry(registry: &Registry) -> Result<(), Vec<ValidationError>> {
    let mut errors: Vec<ValidationError> = Vec::new();

    // Phase 1: Schema, entry type discrimination, non-empty sequences
    phase1(registry, &mut errors);

    // Phase 2: Reference existence
    phase2(registry, &mut errors);

    // Phase 3: Cycle detection via iterative DFS
    phase3(registry, &mut errors);

    // Phase 4: Atomic rejection or acceptance
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn phase1(registry: &Registry, errors: &mut Vec<ValidationError>) {
    for fallback in registry.values() {
        if fallback.sequence.is_empty() {
            errors.push(ValidationError::new(format!(
                "Fallback '{}' has an empty sequence",
                fallback.id,
            )));
        }
    }
}

fn phase2(registry: &Registry, errors: &mut Vec<ValidationError>) {
    for fallback in registry.values() {
        for entry in &fallback.sequence {
            if let Entry::FallbackReference(fr) = entry
                && !registry.contains_key(&fr.fallback_id)
            {
                errors.push(ValidationError::new(format!(
                    "Fallback '{}' references unknown id '{}'",
                    fallback.id, fr.fallback_id,
                )));
            }
        }
    }
}

fn phase3(registry: &Registry, errors: &mut Vec<ValidationError>) {
    let mut visited: HashMap<String, bool> = HashMap::new();

    for fallback_id in registry.keys() {
        if visited.contains_key(fallback_id) {
            continue;
        }

        // Iterative DFS with two-phase stack entries
        let mut on_stack: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut stack: Vec<(String, bool)> = Vec::new();
        stack.push((fallback_id.clone(), false));

        while let Some((current_id, processed)) = stack.pop() {
            if processed {
                on_stack.remove(&current_id);
                visited.insert(current_id.clone(), true);
            } else if visited.contains_key(&current_id) && visited[&current_id] {
                continue;
            } else if on_stack.contains(&current_id) {
                errors.push(ValidationError::new(format!(
                    "Cycle detected involving '{}'",
                    current_id,
                )));
            } else {
                on_stack.insert(current_id.clone());
                stack.push((current_id.clone(), true));

                if let Some(def) = registry.get(&current_id) {
                    let children: Vec<String> = def
                        .sequence
                        .iter()
                        .filter_map(|e| {
                            if let Entry::FallbackReference(fr) = e {
                                Some(fr.fallback_id.clone())
                            } else {
                                None
                            }
                        })
                        .collect();

                    for child in children.into_iter().rev() {
                        stack.push((child, false));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::fallback::types::{
        Entry, FallbackDefinition, FallbackReference, ProviderModelPair,
    };

    fn make_pmp(provider: &str, model: &str) -> Entry {
        Entry::ProviderModelPair(ProviderModelPair {
            provider: provider.into(),
            model: model.into(),
        })
    }

    fn make_ref(id: &str) -> Entry {
        Entry::FallbackReference(FallbackReference {
            fallback_id: id.into(),
        })
    }

    #[test]
    fn test_valid_registry() {
        let mut registry = Registry::new();
        registry.insert(
            "auto".into(),
            FallbackDefinition {
                id: "auto".into(),
                description: Some("Auto fallback".into()),
                timeout_ms: None,
                sequence: vec![
                    make_pmp("anthropic", "claude-sonnet-4-6"),
                    make_pmp("openai", "gpt-4o"),
                ],
            },
        );
        let result = validate_fallback_registry(&registry);
        assert!(result.is_ok(), "Expected Ok, got Err: {:?}", result);
    }

    #[test]
    fn test_empty_sequence_rejected() {
        let mut registry = Registry::new();
        registry.insert(
            "empty".into(),
            FallbackDefinition {
                id: "empty".into(),
                description: None,
                timeout_ms: None,
                sequence: vec![],
            },
        );
        let result = validate_fallback_registry(&registry);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e.message.contains("empty sequence")));
    }

    #[test]
    fn test_unknown_reference() {
        let mut registry = Registry::new();
        registry.insert(
            "auto".into(),
            FallbackDefinition {
                id: "auto".into(),
                description: None,
                timeout_ms: None,
                sequence: vec![make_ref("nonexistent")],
            },
        );
        let result = validate_fallback_registry(&registry);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e.message.contains("unknown id")));
    }

    #[test]
    fn test_cycle_detected() {
        let mut registry = Registry::new();
        registry.insert(
            "a".into(),
            FallbackDefinition {
                id: "a".into(),
                description: None,
                timeout_ms: None,
                sequence: vec![make_ref("b")],
            },
        );
        registry.insert(
            "b".into(),
            FallbackDefinition {
                id: "b".into(),
                description: None,
                timeout_ms: None,
                sequence: vec![make_ref("a")],
            },
        );
        let result = validate_fallback_registry(&registry);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e.message.contains("Cycle detected")));
    }

    #[test]
    fn test_self_cycle() {
        let mut registry = Registry::new();
        registry.insert(
            "selfie".into(),
            FallbackDefinition {
                id: "selfie".into(),
                description: None,
                timeout_ms: None,
                sequence: vec![make_ref("selfie")],
            },
        );
        let result = validate_fallback_registry(&registry);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e.message.contains("Cycle detected")));
    }

    #[test]
    fn test_all_errors_collected() {
        let mut registry = Registry::new();
        registry.insert(
            "empty".into(),
            FallbackDefinition {
                id: "empty".into(),
                description: None,
                timeout_ms: None,
                sequence: vec![],
            },
        );
        registry.insert(
            "broken".into(),
            FallbackDefinition {
                id: "broken".into(),
                description: None,
                timeout_ms: None,
                sequence: vec![make_ref("ghost")],
            },
        );
        registry.insert(
            "oops".into(),
            FallbackDefinition {
                id: "oops".into(),
                description: None,
                timeout_ms: None,
                sequence: vec![make_ref("oops")],
            },
        );
        let result = validate_fallback_registry(&registry);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(
            errors.len() >= 3,
            "Expected 3+ errors, got {}",
            errors.len()
        );
    }

    #[test]
    fn test_nested_valid() {
        let mut registry = Registry::new();
        registry.insert(
            "primary".into(),
            FallbackDefinition {
                id: "primary".into(),
                description: None,
                timeout_ms: None,
                sequence: vec![make_pmp("anthropic", "claude-sonnet-4-6")],
            },
        );
        registry.insert(
            "auto".into(),
            FallbackDefinition {
                id: "auto".into(),
                description: None,
                timeout_ms: None,
                sequence: vec![make_ref("primary"), make_pmp("openai", "gpt-4o")],
            },
        );
        let result = validate_fallback_registry(&registry);
        assert!(result.is_ok(), "Expected Ok, got Err: {:?}", result);
    }
}
