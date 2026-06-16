/// Provider/model fallback mechanism.
///
/// Implements the fallback specification:
/// <https://github.com/yuanchuan/aivo/blob/develop/docs/actions/reqs/req-provider-and-model-fallback-mechanism.md>
///
/// A fallback is a named model alias that resolves to an ordered list of
/// provider/model targets. Selecting a fallback is indistinguishable from
/// selecting a concrete model from the caller's perspective.
pub mod types;
pub mod validate;
pub mod flatten;
pub mod resolve;
pub mod reload;
pub mod metrics;
pub mod manager;
pub mod adapters;

pub use resolve::{InvokeProvider, resolve, classify_error};
pub use types::*;
pub use validate::ValidateRegistry;
pub use manager::FallbackManager;
