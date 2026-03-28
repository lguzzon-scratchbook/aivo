/// Placeholder loopback URL used during environment injection.
/// The AI launcher replaces this with the actual random port after binding.
pub const PLACEHOLDER_LOOPBACK_URL: &str = "http://127.0.0.1:0";

/// Standard JSON content type header value.
pub const CONTENT_TYPE_JSON: &str = "application/json";

/// Placeholder model value meaning "let the tool use its own default."
pub const MODEL_DEFAULT_PLACEHOLDER: &str = "__default__";

/// Display label shown in the model picker for the default/skip option.
pub const MODEL_DEFAULT_DISPLAY: &str = "(leave it to the tool)";
