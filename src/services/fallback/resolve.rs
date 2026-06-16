/// Runtime resolution for fallback configurations (§5.2).
///
/// Resolves a fallback alias by sequentially attempting each target in the
/// pre-flattened index until one succeeds or all fail.

use super::types::{
    AttemptRecord, ErrorSummary, FallbackErrorCategory, FallbackExhaustedError, FlatTargetList,
    FlattenedIndex, ProviderError,
};

/// Result of a single provider invocation attempt.
pub type InvokeResult<T> = Result<T, ProviderError>;

/// Trait for provider invocation.
///
/// Each provider adapter implements this to map its SDK's error taxonomy to
/// the canonical `FallbackErrorCategory` types. The request is captured by
/// the invoker at construction time — `resolve` only provides the target
/// provider and model to try next.
pub trait InvokeProvider {
    /// The type of response to receive.
    type Response;

    /// Invoke a specific provider+model and return the response.
    ///
    /// Must map all provider SDK errors to `ProviderError` with the correct
    /// `FallbackErrorCategory`. Must NOT produce `Internal` errors — those are
    /// reserved for the fallback system itself.
    fn invoke(&self, provider: &str, model: &str) -> InvokeResult<Self::Response>;
}

/// Result of fallback resolution: either a successful response or exhaustion.
pub enum ResolutionResult<T> {
    Success(T),
    Exhausted(FallbackExhaustedError),
}

/// Run the fallback resolution procedure.
///
/// Iterates the pre-flattened target list in order, attempting each via
/// `invoker`. Advances on any provider error except `Internal`/`Panic`.
/// Enforces cumulative timeout if `timeout_ms` is set.
pub fn resolve<T, P>(
    fallback_id: &str,
    flat_index: &FlattenedIndex,
    invoker: &P,
    timeout_ms: Option<u64>,
) -> ResolutionResult<T>
where
    P: InvokeProvider<Response = T> + ?Sized,
{
    let targets: &FlatTargetList = match flat_index.get(fallback_id) {
        Some(t) => t,
        None => {
            return ResolutionResult::Exhausted(FallbackExhaustedError {
                fallback_id: fallback_id.to_string(),
                attempts: vec![],
                last_error: ProviderError::new(
                    format!("Unknown fallback alias '{}'", fallback_id),
                    FallbackErrorCategory::Other,
                ),
                summary: ErrorSummary {
                    categories: vec![],
                },
            });
        }
    };

    let start = std::time::Instant::now();
    let mut attempts: Vec<AttemptRecord> = Vec::new();
    let mut last_error: Option<ProviderError> = None;
    let mut categories: std::collections::BTreeSet<FallbackErrorCategory> =
        std::collections::BTreeSet::new();

    for (i, target) in targets.iter().enumerate() {
        // Check cumulative timeout before each attempt
        if let Some(timeout) = timeout_ms {
            if start.elapsed().as_millis() >= timeout as u128 {
                let elapsed = start.elapsed().as_millis();
                let err = ProviderError::new(
                    format!(
                        "Cumulative timeout exceeded for fallback '{}' after {} attempt(s) ({}ms >= {}ms timeout)",
                        fallback_id, i, elapsed, timeout,
                    ),
                    FallbackErrorCategory::Timeout,
                );
                last_error = Some(err);
                break;
            }
        }

        match invoker.invoke(&target.provider, &target.model) {
            Ok(response) => {
                return ResolutionResult::Success(response);
            }
            Err(err) => {
                // INTERNAL/PANIC errors signal system corruption, not
                // provider degradation. Must NOT advance (§5.2).
                if err.category == FallbackErrorCategory::Internal {
                    return ResolutionResult::Exhausted(FallbackExhaustedError {
                        fallback_id: fallback_id.to_string(),
                        attempts,
                        last_error: err,
                        summary: ErrorSummary {
                            categories: categories.into_iter().collect(),
                        },
                    });
                }

                #[cfg(debug_assertions)]
                {
                    eprintln!(
                        "[fallback] {} -> {}/{} failed: {}",
                        fallback_id, target.provider, target.model, err.message,
                    );
                }
                let category = err.category.clone();
                categories.insert(category);
                attempts.push(AttemptRecord {
                    provider: target.provider.clone(),
                    model: target.model.clone(),
                    error: err,
                });
                last_error = attempts.last().map(|a| a.error.clone());
            }
        }
    }

    let last_error = last_error.unwrap_or_else(|| {
        ProviderError::new("No targets available", FallbackErrorCategory::Other)
    });

    ResolutionResult::Exhausted(FallbackExhaustedError {
        fallback_id: fallback_id.to_string(),
        attempts,
        last_error,
        summary: ErrorSummary {
            categories: categories.into_iter().collect(),
        },
    })
}

/// Attempt a single provider+model pair and classify the error.
///
/// This is a convenience wrapper for the spec's `InvokeProvider` boundary.
/// Provider adapters should call this to convert their SDK errors into
/// canonical `ProviderError` values.
pub fn classify_error(message: impl Into<String>, status: Option<u16>) -> ProviderError {
    let msg = message.into();
    let category = match status {
        Some(429) => FallbackErrorCategory::RateLimit,
        Some(401) | Some(403) => FallbackErrorCategory::Auth,
        Some(404) | Some(405) | Some(415) | Some(501) => FallbackErrorCategory::ModelNotFound,
        Some(s) if s >= 500 => FallbackErrorCategory::Network,
        _ => {
            // Heuristic classification by message content
            let lower = msg.to_lowercase();
            if lower.contains("rate") || lower.contains("quota") || lower.contains("limit") {
                FallbackErrorCategory::RateLimit
            } else if lower.contains("timeout") || lower.contains("deadline") {
                FallbackErrorCategory::Timeout
            } else if lower.contains("auth") || lower.contains("key") || lower.contains("credential")
            {
                FallbackErrorCategory::Auth
            } else if lower.contains("not found") || lower.contains("unknown model") {
                FallbackErrorCategory::ModelNotFound
            } else if lower.contains("connection") || lower.contains("dns") {
                FallbackErrorCategory::Network
            } else {
                FallbackErrorCategory::Other
            }
        }
    };
    ProviderError::new(msg, category)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::fallback::types::ProviderModelPair;
    use std::collections::HashMap;

    struct FakeInvoker {
        /// Map of "(provider,model)" -> result or "error:category" to simulate failure
        responses: HashMap<String, Result<String, String>>,
    }

    impl FakeInvoker {
        fn new() -> Self {
            Self {
                responses: HashMap::new(),
            }
        }

        fn add_success(&mut self, provider: &str, model: &str, response: &str) {
            self.responses
                .insert(format!("{}/{}", provider, model), Ok(response.to_string()));
        }

        fn add_error(&mut self, provider: &str, model: &str, category: &str) {
            self.responses.insert(
                format!("{}/{}", provider, model),
                Err(category.to_string()),
            );
        }
    }

    impl InvokeProvider for FakeInvoker {
        type Response = String;

        fn invoke(&self, provider: &str, model: &str) -> InvokeResult<Self::Response> {
            let key = format!("{}/{}", provider, model);
            match self.responses.get(&key) {
                Some(Ok(resp)) => Ok(resp.clone()),
                Some(Err(category)) => Err(match category.as_str() {
                    "rate_limit" => ProviderError::new(category, FallbackErrorCategory::RateLimit),
                    "timeout" => ProviderError::new(category, FallbackErrorCategory::Timeout),
                    "auth" => ProviderError::new(category, FallbackErrorCategory::Auth),
                    "model_not_found" => {
                        ProviderError::new(category, FallbackErrorCategory::ModelNotFound)
                    }
                    "network" => ProviderError::new(category, FallbackErrorCategory::Network),
                    "internal" => ProviderError::new(category, FallbackErrorCategory::Internal),
                    _ => ProviderError::new(category, FallbackErrorCategory::Other),
                }),
                None => Err(ProviderError::new(
                    format!("unknown target {}/{}", provider, model),
                    FallbackErrorCategory::Other,
                )),
            }
        }
    }

    fn make_index(entries: Vec<(&str, &str)>) -> FlattenedIndex {
        let mut index = FlattenedIndex::new();
        index.insert(
            "auto".to_string(),
            entries
                .into_iter()
                .map(|(p, m)| ProviderModelPair {
                    provider: p.to_string(),
                    model: m.to_string(),
                })
                .collect(),
        );
        index
    }

    #[test]
    fn test_first_target_succeeds() {
        let mut invoker = FakeInvoker::new();
        invoker.add_success("anthropic", "claude-sonnet-4-6", "ok");
        let index = make_index(vec![("anthropic", "claude-sonnet-4-6")]);

        match resolve("auto", &index, &invoker, None) {
            ResolutionResult::Success(resp) => assert_eq!(resp, "ok"),
            _ => panic!("expected Success"),
        }
    }

    #[test]
    fn test_fallback_on_error() {
        let mut invoker = FakeInvoker::new();
        invoker.add_error("anthropic", "claude-sonnet-4-6", "rate_limit");
        invoker.add_success("openai", "gpt-4o", "ok");
        let index = make_index(vec![
            ("anthropic", "claude-sonnet-4-6"),
            ("openai", "gpt-4o"),
        ]);

        match resolve("auto", &index, &invoker, None) {
            ResolutionResult::Success(resp) => assert_eq!(resp, "ok"),
            _ => panic!("expected Success"),
        }
    }

    #[test]
    fn test_all_targets_exhausted() {
        let mut invoker = FakeInvoker::new();
        invoker.add_error("anthropic", "claude-sonnet-4-6", "rate_limit");
        invoker.add_error("openai", "gpt-4o", "timeout");
        let index = make_index(vec![
            ("anthropic", "claude-sonnet-4-6"),
            ("openai", "gpt-4o"),
        ]);

        match resolve("auto", &index, &invoker, None) {
            ResolutionResult::Exhausted(err) => {
                assert_eq!(err.attempts.len(), 2);
                assert!(err.summary.categories.contains(&FallbackErrorCategory::RateLimit));
                assert!(err.summary.categories.contains(&FallbackErrorCategory::Timeout));
            }
            _ => panic!("expected Exhausted"),
        }
    }

    #[test]
    fn test_unknown_fallback_id() {
        let invoker = FakeInvoker::new();
        let index = make_index(vec![]);
        match resolve("nonexistent", &index, &invoker, None) {
            ResolutionResult::Exhausted(err) => {
                assert!(err.last_error.message.contains("Unknown fallback alias"));
            }
            _ => panic!("expected Exhausted"),
        }
    }

    #[test]
    fn test_cumulative_timeout() {
        use std::time::Duration;

        // Slow invoker that takes 10ms per call
        struct SlowInvoker;
        impl InvokeProvider for SlowInvoker {
            type Response = String;
            fn invoke(&self, _provider: &str, _model: &str) -> InvokeResult<Self::Response> {
                std::thread::sleep(Duration::from_millis(10));
                Err(ProviderError::new("slow", FallbackErrorCategory::Network))
            }
        }

        let index = make_index(vec![("slow-a", "model-a"), ("slow-b", "model-b")]);

        // 5ms timeout — should fire after first 10ms call elapses
        match resolve("auto", &index, &SlowInvoker, Some(5)) {
            ResolutionResult::Exhausted(err) => {
                assert!(
                    err.last_error.message.contains("Cumulative timeout"),
                    "Expected timeout message, got: {}",
                    err.last_error.message,
                );
            }
            _ => panic!("expected Exhausted with timeout"),
        }
    }

    #[test]
    fn test_empty_targets_list() {
        let invoker = FakeInvoker::new();
        let index = make_index(vec![]);
        match resolve("auto", &index, &invoker, None) {
            ResolutionResult::Exhausted(err) => {
                assert_eq!(err.attempts.len(), 0);
            }
            _ => panic!("expected Exhausted"),
        }
    }

    #[test]
    fn test_error_summary_deduplicates_categories() {
        let mut invoker = FakeInvoker::new();
        invoker.add_error("p1", "m1", "rate_limit");
        invoker.add_error("p2", "m2", "rate_limit");
        invoker.add_error("p3", "m3", "auth");
        let index = make_index(vec![
            ("p1", "m1"),
            ("p2", "m2"),
            ("p3", "m3"),
        ]);

        match resolve("auto", &index, &invoker, None) {
            ResolutionResult::Exhausted(err) => {
                assert_eq!(err.summary.categories.len(), 2);
                assert!(err.summary.categories.contains(&FallbackErrorCategory::RateLimit));
                assert!(err.summary.categories.contains(&FallbackErrorCategory::Auth));
            }
            _ => panic!("expected Exhausted"),
        }
    }

    #[test]
    fn test_classify_error_by_status() {
        let err = classify_error("too many", Some(429));
        assert_eq!(err.category, FallbackErrorCategory::RateLimit);

        let err = classify_error("unauthorized", Some(401));
        assert_eq!(err.category, FallbackErrorCategory::Auth);

        let err = classify_error("not found", Some(404));
        assert_eq!(err.category, FallbackErrorCategory::ModelNotFound);

        let err = classify_error("server error", Some(503));
        assert_eq!(err.category, FallbackErrorCategory::Network);
    }

    #[test]
    fn test_classify_error_by_message() {
        let err = classify_error("rate limit exceeded", None);
        assert_eq!(err.category, FallbackErrorCategory::RateLimit);

        let err = classify_error("connection refused", None);
        assert_eq!(err.category, FallbackErrorCategory::Network);

        let err = classify_error("internal server error", None);
        assert_eq!(err.category, FallbackErrorCategory::Other);

        let err = classify_error("random error", None);
        assert_eq!(err.category, FallbackErrorCategory::Other);
    }
}
