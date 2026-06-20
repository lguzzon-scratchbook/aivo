//! Legacy marker for Gemini Google-OAuth key entries.
//!
//! aivo used to support signing in to the `gemini` CLI with a Google account
//! (an Installed-App OAuth flow whose tokens were projected into a shadow
//! `GEMINI_CLI_HOME`). That flow has been removed — see issue #12 / Google's
//! transition of the Gemini CLI to the Antigravity CLI — so no new
//! `gemini-oauth` keys can be created.
//!
//! This sentinel is retained only so a `gemini-oauth` entry left in a user's
//! store from an earlier version is still *recognized*: its credential stays
//! redacted in `keys cat`, it's excluded from key exports, and it's rejected
//! at launch with a clear "re-add with an API key" message — rather than being
//! mistaken for a plain API key with a bogus `gemini-oauth` base URL.

/// Sentinel stored in `ApiKey.base_url` to identify legacy Gemini OAuth
/// entries created before the OAuth sign-in flow was removed.
pub const GEMINI_OAUTH_SENTINEL: &str = "gemini-oauth";
