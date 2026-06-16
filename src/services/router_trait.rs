use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8};
use anyhow::Result;

/// Standard result from starting any background router.
///
/// - `port`: the localhost port the router bound to.
/// - `active_protocol`: learned upstream protocol (None for simple routers
///   that do no protocol detection).
/// - `request_succeeded`: set to `true` after any non-error upstream response.
///   Used by `persist_runtime_discoveries` to gate protocol pinning.
/// - `responses_api_support`: tri-state Responses API support detection
///   (0=unknown, 1=supported, 2=not supported). Only meaningful for
///   `ResponsesToChatRouter`; `None` for all others.
/// - `handle`: the tokio JoinHandle for the router's background listener task.
///   The consumer must spawn this to poll it for unexpected exits.
#[allow(dead_code)]
pub struct RouterStart {
    pub port: u16,
    pub active_protocol: Option<Arc<AtomicU8>>,
    pub request_succeeded: Option<Arc<AtomicBool>>,
    pub responses_api_support: Option<Arc<AtomicU8>>,
    pub handle: tokio::task::JoinHandle<Result<()>>,
}

/// Unified interface for background router startup.
///
/// Each router type knows how to construct its internal state and bind a
/// localhost listener.  The `start` method bundles the standard setup,
/// returning a `RouterStart` that the dispatch layer can process uniformly.
#[allow(dead_code)]
#[allow(async_fn_in_trait)]
pub trait Router: Send + Sync {
    /// The router-specific configuration type (e.g. `AnthropicRouterConfig`).
    type Config;

    /// Start the router with the given config, optionally consulting `env`
    /// for runtime overrides (e.g. starter flags).
    async fn start(
        config: Self::Config,
        env: &HashMap<String, String>,
    ) -> Result<RouterStart>;
}
