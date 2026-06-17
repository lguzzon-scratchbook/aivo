pub mod flatten;
pub mod manager;
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

pub use manager::FallbackManager;
pub use types::*;
pub use types::FallbackExhaustedError;
pub use validate::validate_fallback_registry;
