use aivo::services::environment_injector::EnvironmentInjector;
use aivo::services::session_store::ApiKey;

fn test_key() -> ApiKey {
    ApiKey::new(
        "a1b2".to_string(),
        "test-key".to_string(),
        "http://localhost:8080".to_string(),
        "sk-test-key-12345".to_string(),
    )
}

#[test]
fn test_for_claude() {
    let injector = EnvironmentInjector::new();
    let key = test_key();
    let env = injector.for_claude(&key, None);

    assert_eq!(
        env.get("ANTHROPIC_BASE_URL").unwrap(),
        "http://localhost:8080"
    );
    assert_eq!(
        env.get("ANTHROPIC_API_KEY").unwrap(),
        "sk-test-key-12345"
    );
    assert_eq!(
        env.get("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC").unwrap(),
        "1"
    );
    assert!(env.get("ANTHROPIC_AUTH_TOKEN").is_none());
}

#[test]
fn test_for_claude_with_model() {
    let injector = EnvironmentInjector::new();
    let key = test_key();
    let env = injector.for_claude(&key, Some("claude-3-opus"));

    assert_eq!(env.get("ANTHROPIC_MODEL").unwrap(), "claude-3-opus");
    assert_eq!(
        env.get("ANTHROPIC_SMALL_FAST_MODEL").unwrap(),
        "claude-3-opus"
    );
}

#[test]
fn test_for_codex() {
    let injector = EnvironmentInjector::new();
    let key = test_key();
    let env = injector.for_codex(&key, None);

    assert_eq!(
        env.get("OPENAI_BASE_URL").unwrap(),
        "http://localhost:8080"
    );
    assert_eq!(
        env.get("OPENAI_API_KEY").unwrap(),
        "sk-test-key-12345"
    );
}

#[test]
fn test_for_gemini() {
    let injector = EnvironmentInjector::new();
    let key = test_key();
    let env = injector.for_gemini(&key);

    assert_eq!(
        env.get("GOOGLE_GEMINI_BASE_URL").unwrap(),
        "http://localhost:8080"
    );
    assert_eq!(
        env.get("GEMINI_API_KEY").unwrap(),
        "sk-test-key-12345"
    );
}

#[test]
fn test_merge() {
    let injector = EnvironmentInjector::new();
    let key = test_key();
    let tool_env = injector.for_claude(&key, None);
    let merged = injector.merge(&tool_env, None, false);

    assert!(merged.contains_key("ANTHROPIC_BASE_URL"));
    assert!(merged.contains_key("ANTHROPIC_API_KEY"));
}
