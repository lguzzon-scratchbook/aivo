//! Sticky shutdown signal.
//!
//! `tokio::sync::Notify::notify_waiters()` only wakes tasks that are already
//! parked at the moment of the call — subsequent subscribers receive nothing
//! and wait for the next call. That bites the share/tunnel path, where a
//! Ctrl+C broadcast and a transcript-update broadcast can fire on the same
//! tick. If the transcript-update wake wins the `select!`, the handler loops
//! back, subscribes to shutdown again, and never wakes.
//!
//! `ShutdownSignal` wraps an `AtomicBool` with a `Notify`: `fire()` flips the
//! flag and broadcasts; `wait()` returns immediately whenever the flag is
//! already set, regardless of when the caller subscribed.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::Notify;

#[derive(Clone, Default)]
pub struct ShutdownSignal {
    inner: Arc<Inner>,
}

#[derive(Default)]
struct Inner {
    fired: AtomicBool,
    notify: Notify,
}

impl ShutdownSignal {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fire(&self) {
        self.inner.fired.store(true, Ordering::SeqCst);
        self.inner.notify.notify_waiters();
    }

    pub fn is_fired(&self) -> bool {
        self.inner.fired.load(Ordering::SeqCst)
    }

    /// Resolves once `fire()` has been (or has already been) called. The
    /// double check around `notified()` is the standard pattern for avoiding
    /// a missed wake when `fire()` lands between our two atomic reads.
    pub async fn wait(&self) {
        if self.is_fired() {
            return;
        }
        let notified = self.inner.notify.notified();
        if self.is_fired() {
            return;
        }
        notified.await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn wait_returns_immediately_after_fire() {
        let s = ShutdownSignal::new();
        s.fire();
        tokio::time::timeout(Duration::from_millis(50), s.wait())
            .await
            .expect("wait should resolve immediately");
        assert!(s.is_fired());
    }

    #[tokio::test]
    async fn wait_returns_when_fire_happens_later() {
        let s = ShutdownSignal::new();
        let s2 = s.clone();
        let h = tokio::spawn(async move { s2.wait().await });
        tokio::time::sleep(Duration::from_millis(20)).await;
        s.fire();
        tokio::time::timeout(Duration::from_millis(100), h)
            .await
            .expect("wait should resolve after fire")
            .unwrap();
    }

    #[tokio::test]
    async fn late_subscriber_sees_prior_fire() {
        // The whole point of this type: a subscriber created AFTER fire() must
        // still resolve immediately. `Notify::notify_waiters()` alone does not
        // satisfy this contract.
        let s = ShutdownSignal::new();
        s.fire();
        let s2 = s.clone();
        tokio::time::timeout(Duration::from_millis(50), async move { s2.wait().await })
            .await
            .expect("late subscriber should resolve immediately");
    }

    #[tokio::test]
    async fn many_concurrent_waiters_all_resolve() {
        let s = ShutdownSignal::new();
        let mut handles = Vec::new();
        for _ in 0..16 {
            let s2 = s.clone();
            handles.push(tokio::spawn(async move { s2.wait().await }));
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
        s.fire();
        for h in handles {
            tokio::time::timeout(Duration::from_millis(100), h)
                .await
                .expect("all waiters should resolve")
                .unwrap();
        }
    }
}
