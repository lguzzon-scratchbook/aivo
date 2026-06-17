/**
 * Integration tests for fallback resolution and CRUD operations.
 */
use std::collections::HashMap;

use aivo::services::fallback::{Entry, FallbackDefinition, ProviderModelPair};
use aivo::services::session_store::SessionStore;

fn make_pmp(provider: &str, model: &str) -> Entry {
    Entry::ProviderModelPair(ProviderModelPair {
        provider: provider.into(),
        model: model.into(),
    })
}

/// Creates a SessionStore with a tempdir for testing.
fn make_store() -> SessionStore {
    let dir = tempfile::TempDir::new().unwrap();
    SessionStore::with_path(dir.path().join("config.json"))
}

#[tokio::test]
async fn set_and_list_fallback() {
    let store = make_store();

    // Should start empty
    let fallbacks = store.get_fallbacks().await.unwrap();
    assert!(fallbacks.is_empty());

    // Set a fallback with two targets
    let mut defs = HashMap::new();
    defs.insert(
        "auto".to_string(),
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
    store.set_fallback("auto", &defs).await.unwrap();

    // Verify
    let loaded = store.get_fallbacks().await.unwrap();
    assert_eq!(loaded.len(), 1);
    let def = loaded.get("auto").unwrap();
    assert_eq!(def.sequence.len(), 2);
}

#[tokio::test]
async fn remove_fallback() {
    let store = make_store();

    // Set and verify
    let mut defs = HashMap::new();
    defs.insert(
        "test".to_string(),
        FallbackDefinition {
            id: "test".into(),
            description: None,
            timeout_ms: None,
            sequence: vec![make_pmp("p", "m")],
        },
    );
    store.set_fallback("test", &defs).await.unwrap();
    assert_eq!(store.get_fallbacks().await.unwrap().len(), 1);

    // Remove — rewrites the whole config without the removed key
    let mut remaining = store.get_fallbacks().await.unwrap();
    remaining.remove("test");
    store.set_fallback("test", &remaining).await.unwrap();
    assert!(store.get_fallbacks().await.unwrap().is_empty());
}

#[tokio::test]
async fn fallback_targets_resolved() {
    let store = make_store();

    // Set fallback
    let mut defs = HashMap::new();
    defs.insert(
        "auto".to_string(),
        FallbackDefinition {
            id: "auto".into(),
            description: None,
            timeout_ms: None,
            sequence: vec![make_pmp("p1", "m1"), make_pmp("p2", "m2")],
        },
    );
    store.set_fallback("auto", &defs).await.unwrap();

    // Get targets via fallback manager
    let mgr = store.fallback_manager().await.unwrap();
    let targets = mgr.resolve_targets("auto").unwrap();
    assert_eq!(targets.len(), 2);
    assert_eq!(targets[0].provider, "p1");
    assert_eq!(targets[0].model, "m1");
    assert_eq!(targets[1].provider, "p2");
    assert_eq!(targets[1].model, "m2");

    // Unknown fallback returns None
    assert!(mgr.resolve_targets("nonexistent").is_none());
}

#[tokio::test]
async fn fallback_validation_rejects_empty() {
    let store = make_store();

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

    let result = store.fallback_manager().await.unwrap();
    // Empty fallback is accepted (filtered during load with warning)
    // But validate_fallback_registry would reject it
    let validation = aivo::services::fallback::validate_fallback_registry(&defs);
    assert!(validation.is_err());
    let errors = validation.unwrap_err();
    assert!(errors.iter().any(|e| e.message.contains("empty sequence")));
    drop(result);
}

#[test]
fn test_check_timeout_exceeds() {
    use aivo::commands::fallback_resolve::check_timeout;
    use std::time::{Duration, Instant};
    let start = Instant::now() - Duration::from_millis(150);
    let result = check_timeout(start, Some(100));
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Fallback timeout exceeded")
    );
}

#[tokio::test]
async fn test_fallback_exhaustion_returns_structured_error() {
    let store = make_store();
    let mut defs = std::collections::HashMap::new();
    defs.insert(
        "fail".to_string(),
        aivo::services::fallback::FallbackDefinition {
            id: "fail".into(),
            description: None,
            timeout_ms: None,
            sequence: vec![make_pmp("unknown_provider", "m1")],
        },
    );
    store.set_fallback("fail", &defs).await.unwrap();

    let result =
        aivo::commands::fallback_resolve::resolve_fallback_targets(&store, "fail", None).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("exhausted"));
}
