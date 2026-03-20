use std::sync::atomic::{AtomicU8, Ordering};

use super::provider_protocol::{ProviderProtocol, fallback_protocols, is_protocol_mismatch};

/// Outcome of a single protocol attempt in the fallback loop.
pub enum AttemptOutcome<T> {
    /// Protocol matched and returned a usable result.
    Success(T),
    /// Protocol responded with a non-mismatch error — stop trying other protocols.
    ProviderError { status: u16, body: String },
    /// Protocol mismatch (wrong endpoint) — try the next candidate.
    Mismatch { status: u16, body: String },
}

/// Returns the ordered list of protocol candidates: active first, then fallbacks.
pub fn protocol_candidates(active_protocol: &AtomicU8) -> Vec<ProviderProtocol> {
    let current = ProviderProtocol::from_u8(active_protocol.load(Ordering::Relaxed));
    std::iter::once(current)
        .chain(fallback_protocols(current))
        .collect()
}

/// If this was a fallback attempt (attempt > 0), store the winning protocol and log.
pub fn commit_protocol_switch(
    active_protocol: &AtomicU8,
    protocol: ProviderProtocol,
    attempt: usize,
) {
    if attempt > 0 {
        active_protocol.store(protocol.to_u8(), Ordering::Relaxed);
        eprintln!("  • Protocol auto-switched to {}", protocol.as_str());
    }
}

/// Classify an HTTP response into an attempt outcome.
///
/// - `Some(T)` → `Success(T)`
/// - `None` + protocol-mismatch status → `Mismatch`
/// - `None` + other status → `ProviderError`
pub fn classify_attempt<T>(
    status: u16,
    response_text: String,
    success: Option<T>,
) -> AttemptOutcome<T> {
    match success {
        Some(val) => AttemptOutcome::Success(val),
        None if is_protocol_mismatch(status) => AttemptOutcome::Mismatch {
            status,
            body: response_text,
        },
        None => AttemptOutcome::ProviderError {
            status,
            body: response_text,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_attempt_success() {
        match classify_attempt(200, String::new(), Some(42)) {
            AttemptOutcome::Success(v) => assert_eq!(v, 42),
            _ => panic!("expected Success"),
        }
    }

    #[test]
    fn classify_attempt_mismatch() {
        match classify_attempt::<()>(404, "not found".into(), None) {
            AttemptOutcome::Mismatch { status, .. } => assert_eq!(status, 404),
            _ => panic!("expected Mismatch"),
        }
    }

    #[test]
    fn classify_attempt_provider_error() {
        match classify_attempt::<()>(401, "unauthorized".into(), None) {
            AttemptOutcome::ProviderError { status, .. } => assert_eq!(status, 401),
            _ => panic!("expected ProviderError"),
        }
    }

    #[test]
    fn protocol_candidates_starts_with_current() {
        let active = AtomicU8::new(ProviderProtocol::Google.to_u8());
        let candidates = protocol_candidates(&active);
        assert_eq!(candidates[0], ProviderProtocol::Google);
        assert!(candidates.len() > 1);
        assert!(!candidates[1..].contains(&ProviderProtocol::Google));
    }

    #[test]
    fn commit_switch_stores_on_fallback() {
        let active = AtomicU8::new(ProviderProtocol::Openai.to_u8());
        commit_protocol_switch(&active, ProviderProtocol::Google, 1);
        assert_eq!(
            ProviderProtocol::from_u8(active.load(Ordering::Relaxed)),
            ProviderProtocol::Google
        );
    }

    #[test]
    fn commit_switch_noop_on_first_attempt() {
        let active = AtomicU8::new(ProviderProtocol::Openai.to_u8());
        commit_protocol_switch(&active, ProviderProtocol::Google, 0);
        assert_eq!(
            ProviderProtocol::from_u8(active.load(Ordering::Relaxed)),
            ProviderProtocol::Openai
        );
    }

    #[test]
    fn classify_attempt_mismatch_405() {
        match classify_attempt::<()>(405, "method not allowed".into(), None) {
            AttemptOutcome::Mismatch { status, .. } => assert_eq!(status, 405),
            _ => panic!("expected Mismatch for 405"),
        }
    }

    #[test]
    fn classify_attempt_mismatch_415() {
        match classify_attempt::<()>(415, "unsupported media type".into(), None) {
            AttemptOutcome::Mismatch { status, .. } => assert_eq!(status, 415),
            _ => panic!("expected Mismatch for 415"),
        }
    }

    #[test]
    fn classify_attempt_provider_error_400() {
        match classify_attempt::<()>(400, "bad request".into(), None) {
            AttemptOutcome::ProviderError { status, body } => {
                assert_eq!(status, 400);
                assert_eq!(body, "bad request");
            }
            _ => panic!("expected ProviderError for 400"),
        }
    }

    #[test]
    fn classify_attempt_provider_error_403() {
        match classify_attempt::<()>(403, "forbidden".into(), None) {
            AttemptOutcome::ProviderError { status, .. } => assert_eq!(status, 403),
            _ => panic!("expected ProviderError for 403"),
        }
    }

    #[test]
    fn classify_attempt_provider_error_500() {
        match classify_attempt::<()>(500, "server error".into(), None) {
            AttemptOutcome::ProviderError { status, .. } => assert_eq!(status, 500),
            _ => panic!("expected ProviderError for 500"),
        }
    }

    #[test]
    fn classify_attempt_provider_error_502() {
        match classify_attempt::<()>(502, "bad gateway".into(), None) {
            AttemptOutcome::ProviderError { status, .. } => assert_eq!(status, 502),
            _ => panic!("expected ProviderError for 502"),
        }
    }

    #[test]
    fn classify_attempt_provider_error_503() {
        match classify_attempt::<()>(503, "unavailable".into(), None) {
            AttemptOutcome::ProviderError { status, .. } => assert_eq!(status, 503),
            _ => panic!("expected ProviderError for 503"),
        }
    }

    #[test]
    fn classify_attempt_success_ignores_status() {
        // When success is Some, status is irrelevant
        match classify_attempt(500, "error body".into(), Some("ok")) {
            AttemptOutcome::Success(v) => assert_eq!(v, "ok"),
            _ => panic!("expected Success even with error status"),
        }
    }

    #[test]
    fn protocol_candidates_anthropic_starts_with_anthropic() {
        let active = AtomicU8::new(ProviderProtocol::Anthropic.to_u8());
        let candidates = protocol_candidates(&active);
        assert_eq!(candidates[0], ProviderProtocol::Anthropic);
        assert!(!candidates[1..].contains(&ProviderProtocol::Anthropic));
    }

    #[test]
    fn protocol_candidates_openai_starts_with_openai() {
        let active = AtomicU8::new(ProviderProtocol::Openai.to_u8());
        let candidates = protocol_candidates(&active);
        assert_eq!(candidates[0], ProviderProtocol::Openai);
        assert!(!candidates[1..].contains(&ProviderProtocol::Openai));
    }

    #[test]
    fn commit_switch_stores_on_later_attempt() {
        let active = AtomicU8::new(ProviderProtocol::Openai.to_u8());
        commit_protocol_switch(&active, ProviderProtocol::Anthropic, 2);
        assert_eq!(
            ProviderProtocol::from_u8(active.load(Ordering::Relaxed)),
            ProviderProtocol::Anthropic
        );
    }
}
