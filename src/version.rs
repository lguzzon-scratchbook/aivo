//! Version management for the aivo CLI.
//! Version is embedded at build time from Cargo.toml.

/// The version of the aivo CLI, embedded at build time.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_format() {
        // Version should be in semver format (x.y.z)
        assert!(VERSION.contains('.'), "Version should contain dots");
    }
}
