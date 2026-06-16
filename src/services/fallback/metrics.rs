/// Operational metrics for the fallback mechanism (§9.1).
///
/// Lightweight atomic counters and histograms using lock-free primitives.
/// No external dependencies — designed to be consumed by a metrics exporter
/// or logged to structured output.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Key for per-target metrics.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct TargetKey {
    pub fallback_id: String,
    pub provider: String,
    pub model: String,
}

impl TargetKey {
    pub fn new(fallback_id: &str, provider: &str, model: &str) -> Self {
        Self {
            fallback_id: fallback_id.to_string(),
            provider: provider.to_string(),
            model: model.to_string(),
        }
    }
}

impl std::fmt::Display for TargetKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}/{}", self.fallback_id, self.provider, self.model)
    }
}

/// Per-target metrics counters.
#[derive(Debug)]
pub struct TargetMetrics {
    pub attempts: AtomicU64,
    pub successes: AtomicU64,
    pub errors_rate_limit: AtomicU64,
    pub errors_timeout: AtomicU64,
    pub errors_auth: AtomicU64,
    pub errors_network: AtomicU64,
    pub errors_model_not_found: AtomicU64,
    pub errors_other: AtomicU64,
}

impl TargetMetrics {
    fn new() -> Self {
        Self {
            attempts: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            errors_rate_limit: AtomicU64::new(0),
            errors_timeout: AtomicU64::new(0),
            errors_auth: AtomicU64::new(0),
            errors_network: AtomicU64::new(0),
            errors_model_not_found: AtomicU64::new(0),
            errors_other: AtomicU64::new(0),
        }
    }
}

/// Snapshot of a target's metrics for reporting.
#[derive(Debug, Clone)]
pub struct TargetMetricsSnapshot {
    pub attempts: u64,
    pub successes: u64,
    pub errors_rate_limit: u64,
    pub errors_timeout: u64,
    pub errors_auth: u64,
    pub errors_network: u64,
    pub errors_model_not_found: u64,
    pub errors_other: u64,
}

/// A single duration observation.
#[derive(Debug, Clone)]
pub struct DurationSample {
    pub target: TargetKey,
    pub duration_ms: f64,
}

/// Thread-safe fallback metrics collector.
#[derive(Debug)]
pub struct FallbackMetrics {
    targets: Mutex<HashMap<TargetKey, TargetMetrics>>,
    exhaustion_count: Mutex<HashMap<String, AtomicU64>>,
    auth_warning_count: AtomicU64,
    /// Duration samples: stored in-memory, rotated on read.
    durations: Mutex<Vec<DurationSample>>,
}

impl FallbackMetrics {
    /// Create a new metrics collector.
    pub fn new() -> Self {
        Self {
            targets: Mutex::new(HashMap::new()),
            exhaustion_count: Mutex::new(HashMap::new()),
            auth_warning_count: AtomicU64::new(0),
            durations: Mutex::new(Vec::new()),
        }
    }

    /// Record an attempt.
    pub fn record_attempt(&self, target: TargetKey) {
        self.with_target(&target, |m| { m.attempts.fetch_add(1, Ordering::Relaxed); });
    }

    /// Record a success.
    pub fn record_success(&self, target: TargetKey) {
        self.with_target(&target, |m| { m.successes.fetch_add(1, Ordering::Relaxed); });
    }

    /// Record an error with its category.
    pub fn record_error(&self, target: TargetKey, category: &str) {
        self.with_target(&target, |m| {
            let counter: &AtomicU64 = match category {
                "rate_limit" => &m.errors_rate_limit,
                "timeout" => &m.errors_timeout,
                "auth" => &m.errors_auth,
                "network" => &m.errors_network,
                "model_not_found" => &m.errors_model_not_found,
                _ => &m.errors_other,
            };
            counter.fetch_add(1, Ordering::Relaxed);
        });

        if category == "auth" {
            self.auth_warning_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record the duration of a resolution attempt.
    pub fn record_duration(&self, target: TargetKey, duration_ms: f64) {
        let mut durations = self.durations.lock().unwrap();
        durations.push(DurationSample { target, duration_ms });
    }

    /// Record that a fallback was exhausted.
    pub fn record_exhaustion(&self, fallback_id: &str) {
        let mut map = self.exhaustion_count.lock().unwrap();
        let counter = map
            .entry(fallback_id.to_string())
            .or_insert_with(|| AtomicU64::new(0));
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the auth warning count.
    pub fn auth_warning_count(&self) -> u64 {
        self.auth_warning_count.load(Ordering::Relaxed)
    }

    /// Get total exhaustion count for a fallback.
    pub fn exhaustion_count(&self, fallback_id: &str) -> u64 {
        let map = self.exhaustion_count.lock().unwrap();
        map.get(fallback_id)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Get a snapshot of all target metrics.
    pub fn target_snapshots(&self) -> Vec<(TargetKey, TargetMetricsSnapshot)> {
        let map = self.targets.lock().unwrap();
        map.iter()
            .map(|(key, metrics)| {
                let snapshot = TargetMetricsSnapshot {
                    attempts: metrics.attempts.load(Ordering::Relaxed),
                    successes: metrics.successes.load(Ordering::Relaxed),
                    errors_rate_limit: metrics.errors_rate_limit.load(Ordering::Relaxed),
                    errors_timeout: metrics.errors_timeout.load(Ordering::Relaxed),
                    errors_auth: metrics.errors_auth.load(Ordering::Relaxed),
                    errors_network: metrics.errors_network.load(Ordering::Relaxed),
                    errors_model_not_found: metrics.errors_model_not_found.load(Ordering::Relaxed),
                    errors_other: metrics.errors_other.load(Ordering::Relaxed),
                };
                (key.clone(), snapshot)
            })
            .collect()
    }

    /// Drain all recorded durations.
    pub fn drain_durations(&self) -> Vec<DurationSample> {
        let mut durations = self.durations.lock().unwrap();
        std::mem::take(&mut *durations)
    }

    /// Apply a function to the target metrics, creating it if needed.
    fn with_target<F>(&self, target: &TargetKey, f: F)
    where
        F: FnOnce(&TargetMetrics),
    {
        let mut map = self.targets.lock().unwrap();
        let metrics = map
            .entry(target.clone())
            .or_insert_with(TargetMetrics::new);
        f(metrics);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attempts_and_successes() {
        let m = FallbackMetrics::new();
        let tk = TargetKey::new("auto", "anthropic", "claude-sonnet-4-6");

        m.record_attempt(tk.clone());
        m.record_attempt(tk.clone());
        m.record_success(tk.clone());

        let snapshots = m.target_snapshots();
        assert_eq!(snapshots.len(), 1);
        let (_, s) = &snapshots[0];
        assert_eq!(s.attempts, 2);
        assert_eq!(s.successes, 1);
    }

    #[test]
    fn test_error_categories() {
        let m = FallbackMetrics::new();
        let tk = TargetKey::new("auto", "openai", "gpt-4o");

        m.record_error(tk.clone(), "rate_limit");
        m.record_error(tk.clone(), "auth");
        m.record_error(tk.clone(), "network");

        let snapshots = m.target_snapshots();
        let (_, s) = &snapshots[0];
        assert_eq!(s.errors_rate_limit, 1);
        assert_eq!(s.errors_auth, 1);
        assert_eq!(s.errors_network, 1);
    }

    #[test]
    fn test_exhaustion_count() {
        let m = FallbackMetrics::new();
        m.record_exhaustion("auto");
        m.record_exhaustion("auto");
        assert_eq!(m.exhaustion_count("auto"), 2);
    }

    #[test]
    fn test_auth_warning() {
        let m = FallbackMetrics::new();
        let tk = TargetKey::new("auto", "anthropic", "claude-sonnet-4-6");
        m.record_error(tk, "auth");
        // Auth warning should activate on auth errors
        // (auth_warning_count tracks ALL auth-prompted advancements)
        assert_eq!(m.auth_warning_count(), 1);
    }

    #[test]
    fn test_multiple_targets() {
        let m = FallbackMetrics::new();
        let t1 = TargetKey::new("auto", "a", "m1");
        let t2 = TargetKey::new("auto", "b", "m2");

        m.record_attempt(t1);
        m.record_attempt(t2);
        m.record_success(TargetKey::new("auto", "b", "m2"));

        let snapshots = m.target_snapshots();
        assert_eq!(snapshots.len(), 2);
    }

    #[test]
    fn test_duration_recording() {
        let m = FallbackMetrics::new();
        let tk = TargetKey::new("auto", "a", "m");
        m.record_duration(tk, 150.0);

        let durations = m.drain_durations();
        assert_eq!(durations.len(), 1);
        assert!((durations[0].duration_ms - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_target_key_display() {
        let tk = TargetKey::new("auto", "anthropic", "claude-sonnet-4-6");
        assert_eq!(tk.to_string(), "auto/anthropic/claude-sonnet-4-6");
    }
}
