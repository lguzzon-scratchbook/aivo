/// Placeholder loopback URL used during environment injection.
/// The AI launcher replaces this with the actual random port after binding.
pub const PLACEHOLDER_LOOPBACK_URL: &str = "http://127.0.0.1:0";

/// Standard JSON content type header value.
pub const CONTENT_TYPE_JSON: &str = "application/json";

/// Placeholder model value meaning "let the tool use its own default."
pub const MODEL_DEFAULT_PLACEHOLDER: &str = "__default__";

/// Display label shown in the model picker for the default/skip option.
pub const MODEL_DEFAULT_DISPLAY: &str = "(leave it to the tool)";

/// Default provider for new users who have no API keys configured.
/// The sentinel base URL is resolved to the real URL before HTTP calls.
pub const AIVO_STARTER_SENTINEL: &str = "aivo-starter";
pub const AIVO_STARTER_REAL_URL: &str = "https://api.getaivo.dev";
pub const AIVO_STARTER_MODEL: &str = "aivo/starter";
pub const AIVO_STARTER_KEY_NAME: &str = "aivo";
pub const AIVO_STARTER_EMPTY_SECRET: &str = "";

/// Signing key for starter endpoint request authentication.
/// Not a secret — raises the bar from "copy a URL" to "implement the protocol."
pub const AIVO_STARTER_SIGNING_KEY: &str =
    "39de0d498e4c6fe7f28f7ccc9956e8e34978188a7d2e122fe3c512fe22863f35";
