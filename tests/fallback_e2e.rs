/**
 * Full E2E test for the fallback retry mechanism.
 *
 * Tests that aivo's fallback correctly falls through from a failing provider
 * (hyper:deepseek-v4-flash → 401) to a succeeding one
 * (kiloGateway:kilo-auto/small → exit 0).
 *
 * Validates:
 *   1. Fallback definition is created and resolved
 *   2. First target fails, fallback is triggered ("fallback target 1/2 failed")
 *   3. Second target succeeds (exit 0, "Using key: kiloGateway")
 *
 * Requirements:
 *   - The `claude` binary must be installed and on PATH
 *   - HYPER_OPENAI_BASE_URL and HYPER_API_KEY env vars (or .env file in repo root)
 *   - KILO_OPENAI_BASE_URL and KILO_API_KEY env vars (or .env file)
 *   - Network access to both providers
 *
 * Run with:
 *   cargo test --test fallback_e2e -- --ignored
 *
 * Or with env vars already set:
 *   source .env && cargo test --test fallback_e2e -- --ignored
 */

use std::collections::HashMap;
use std::path::Path;

use assert_cmd::Command;
use tempfile::TempDir;

// ── Env var loading ───────────────────────────────────────────────────────

/// Required environment variables for the test.
const REQUIRED_VARS: &[&str] = &[
    "HYPER_OPENAI_BASE_URL",
    "HYPER_API_KEY",
    "KILO_OPENAI_BASE_URL",
    "KILO_API_KEY",
];

/// Load env vars from process environment or from `.env` in the repo root.
/// Tries process env first, falls back to `.env` parsing for each var.
fn load_required_env() -> HashMap<String, String> {
    let mut vars = HashMap::new();

    // Try process env first
    for name in REQUIRED_VARS {
        if let Ok(val) = std::env::var(name) {
            vars.insert(name.to_string(), val);
        }
    }

    // Fill any missing from .env file
    if vars.len() < REQUIRED_VARS.len() {
        let env_path = find_env_path();
        if let Some(content) = env_path.and_then(|p| std::fs::read_to_string(p).ok()) {
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('#') {
                    continue;
                }
                if let Some((key, value)) = trimmed.split_once('=') {
                    let key = key.trim();
                    if vars.contains_key(key) {
                        continue; // already set from environment
                    }
                    // Only expand simple variable references like $VAR
                    let value = expand_env_var(value.trim(), &content);
                    vars.insert(key.to_string(), value);
                }
            }
        }
    }

    vars
}

/// Naive variable expansion: replaces `$VAR` or `${VAR}` with values from the
/// full env file content. Only handles single references, not chained ones.
fn expand_env_var(value: &str, full_content: &str) -> String {
    let result = value.to_string();

    // Build a map of simple KEY=VALUE pairs from the file for expansion
    let mut file_vars: HashMap<&str, String> = HashMap::new();
    for line in full_content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = trimmed.split_once('=') {
            file_vars.insert(k.trim(), v.trim().to_string());
        }
    }

    // Replace $VAR (but not $ followed by space/newline)
    let mut expanded = String::new();
    let mut chars = result.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '$' {
            // Look ahead: ${VAR} or $VAR
            let mut var_name = String::new();
            if chars.peek() == Some(&'{') {
                chars.next(); // skip {
                while let Some(&c) = chars.peek() {
                    if c == '}' {
                        chars.next(); // skip }
                        break;
                    }
                    var_name.push(c);
                    chars.next();
                }
            } else {
                while let Some(&c) = chars.peek() {
                    if c.is_alphanumeric() || c == '_' {
                        var_name.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
            }
            if !var_name.is_empty() {
                if let Some(val) = file_vars.get(var_name.as_str()) {
                    expanded.push_str(val);
                }
                // If not found, leave $VAR as-is (shell would leave it too)
            }
        } else {
            expanded.push(ch);
        }
    }

    expanded
}

/// Find the repo root by looking for Cargo.toml starting from the test dir
/// or the CARGO_MANIFEST_DIR env var.
fn find_env_path() -> Option<String> {
    // When run via cargo test, CARGO_MANIFEST_DIR is set
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let env_file = Path::new(&manifest).join(".env");
        if env_file.exists() {
            return Some(env_file.to_string_lossy().to_string());
        }
    }

    // Fallback: walk up from current dir
    if let Ok(cwd) = std::env::current_dir() {
        let mut dir = cwd.as_path();
        loop {
            let env_file = dir.join(".env");
            if env_file.exists() {
                return Some(env_file.to_string_lossy().to_string());
            }
            match dir.parent() {
                Some(parent) => dir = parent,
                None => break,
            }
        }
    }

    None
}

/// Check if the `claude` binary is available on PATH.
fn claude_binary_available() -> bool {
    std::env::var_os("PATH")
        .as_ref()
        .and_then(|path| {
            std::env::split_paths(path).find_map(|dir| {
                let candidate = dir.join("claude");
                if candidate.is_file() {
                    Some(())
                } else {
                    None
                }
            })
        })
        .is_some()
}


// ── Test ──────────────────────────────────────────────────────────────────

#[ignore]
#[test]
fn fallback_retry_e2e() {
    // ── Prerequisites check ─────────────────────────────────────────────
    let mut issues: Vec<String> = Vec::new();

    // Check each required env var individually
    let env_vars = load_required_env();
    for var in REQUIRED_VARS {
        if !env_vars.contains_key::<str>(var) {
            issues.push(format!(
                "  - Env var {var} is not set and not found in `.env`.\n    \
                 Source .env: source .env"
            ));
        }
    }

    // Check claude binary
    if !claude_binary_available() {
        issues.push(
            "  - `claude` binary not found on $PATH.\n    \
             Install: npm install -g @anthropic-ai/claude-code"
                .to_string(),
        );
    }


    // Report and skip if prerequisites are missing
    if !issues.is_empty() {
        eprintln!("\n  {}", "─".repeat(56));
        eprintln!("  ⚠  E2E prereqs not met — skipping fallback_retry_e2e");
        eprintln!("  {}", "─".repeat(56));
        for issue in &issues {
            for (i, line) in issue.lines().enumerate() {
                if i == 0 {
                    eprintln!("{line}");
                } else {
                    eprintln!("  {line}");
                }
            }
        }
        eprintln!("  {}", "─".repeat(56));
        eprintln!("  To run:");
        eprintln!("    source .env && cargo test --test fallback_e2e -- --ignored");
        eprintln!("  {}\n", "─".repeat(56));
        return;
    }

    // ── Setup isolated config ────────────────────────────────────────────
    let home_dir = TempDir::new().expect("failed to create temp dir");

    // Helper to build a fresh aivo command with isolated HOME
    let aivo = || {
        let mut cmd = Command::cargo_bin("aivo").expect("aivo binary not found");
        cmd.env("HOME", home_dir.path());
        cmd
    };

    // ── Step: Add keys ───────────────────────────────────────────────────
    // Add hyper key
    aivo()
        .arg("keys")
        .arg("add")
        .arg("--name")
        .arg("hyper")
        .arg("--base-url")
        .arg(&env_vars["HYPER_OPENAI_BASE_URL"])
        .arg("--key")
        .arg(&env_vars["HYPER_API_KEY"])
        .assert()
        .success();

    // Add kiloGateway key
    aivo()
        .arg("keys")
        .arg("add")
        .arg("--name")
        .arg("kiloGateway")
        .arg("--base-url")
        .arg(&env_vars["KILO_OPENAI_BASE_URL"])
        .arg("--key")
        .arg(&env_vars["KILO_API_KEY"])
        .assert()
        .success();

    // ── Step: Create fallback ────────────────────────────────────────────
    const FALLBACK_ID: &str = "e2e-test-fallback";
    aivo()
        .arg("fallback")
        .arg("--set")
        .arg(FALLBACK_ID)
        .arg("--")
        .arg("hyper:deepseek-v4-flash")
        .arg("kiloGateway:kilo-auto/small")
        .assert()
        .success();

    // ── Step: Run claude with fallback model ─────────────────────────────
    // The fallback ID is used as --model. The first target (hyper) should
    // fail (401), triggering the fallback to the second (kiloGateway).
    let assert = aivo()
        .arg("claude")
        .arg("--model")
        .arg(FALLBACK_ID)
        .arg("-p")
        .arg("Write out only OK")
        .timeout(std::time::Duration::from_secs(60)) // 60s — allows for slow API
        .assert()
        .success();

    // ── Step: Validate output ────────────────────────────────────────────
    let stderr_bytes = assert.get_output().stderr.clone();
    let stderr = String::from_utf8_lossy(&stderr_bytes);

    // Check that the first target failed (fallback was triggered)
    assert!(
        stderr.contains("fallback target 1/2 failed")
            || stderr.contains("fallback target 1 of 2 failed"),
        "Expected fallback trigger message in stderr.\nStderr:\n{stderr}"
    );

    // Check that the second target (kiloGateway) was used
    assert!(
        stderr.contains("Using key: kiloGateway"),
        "Expected kiloGateway to be the active key.\nStderr:\n{stderr}"
    );

    // ── Cleanup ──────────────────────────────────────────────────────────
    // Remove fallback definition
    let _ = aivo()
        .arg("fallback")
        .arg("--rm")
        .arg(FALLBACK_ID)
        .assert()
        .try_success();

    // Remove keys (best-effort cleanup)
    let _ = aivo()
        .arg("keys")
        .arg("rm")
        .arg("hyper")
        .assert()
        .try_success();

    let _ = aivo()
        .arg("keys")
        .arg("rm")
        .arg("kiloGateway")
        .assert()
        .try_success();

    // TempDir drops automatically, cleaning up the config directory
}
